"""CalendarService: Google Calendar as a constraint + approved event creation.

Owner of `calendar_connections`, `calendar_event_cache`, `calendar_event_links`
(spec FR-13, FR-03, FR-07, FR-15, acceptance 6/12/18; CLAUDE.md rule 7).

Reading
    A periodic job (`calendar.sync`) copies the events of the calendars the owner
    picked for reading, for [now - sync_past_days, now + sync_future_days], into
    `calendar_event_cache`. `busy` comes from transparency, the owner's own RSVP
    (declined = free), the event type (working location = free) and is False for
    blocks this bot created (they are the owner's own plan, not a meeting).
    All-day events use the owner's timezone for their day bounds. A failed sync
    keeps the old cache and marks the data stale; the planner and the gate keep
    using the cache and show `STALE_NOTE`. `busy_intervals()` is the API for them.

Writing (insert only)
    `propose_calendar_block()` turns a work block into a CREATE_CALENDAR_EVENT
    action, which the executor always turns into a pending approval card showing
    exactly which block, when and in which calendar. The applier runs when that card
    is approved: inside the transaction it records a `calendar_event_links` row
    (state creating) and returns an after-commit hook that inserts the event. The
    business-level check refuses an insert unless an APPLIED approval for exactly
    that payload exists. The insert is idempotent: a deterministic event id (409 on
    a repeat) plus our operation key in the event's private extended property, which
    the reconcile job searches before retrying after a crash.

    There is no code path that changes or removes an existing Google event;
    UPDATE/DELETE_CALENDAR_EVENT stay ALWAYS_FORBIDDEN in the executor.

Network calls never run inside a DB transaction (`_no_tx` guards every entry).
Which calendars to read/write is open question 6: nothing is pre-selected; until
the owner picks read calendars nothing is synced and the bot keeps asking in
/calendar.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Any, TypeVar

from pydantic import ValidationError
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.config import EnvSettings
from procbot.context import AppContext
from procbot.crypto import SecretBoxError
from procbot.db.models import (
    CalendarConnection,
    CalendarEventCache,
    CalendarEventLink,
    PendingApproval,
    Setting,
)
from procbot.domain.actions import ActionType, CreateCalendarEvent
from procbot.domain.enums import Actor, ApprovalStatus, MessageCategory
from procbot.services import actions, leases
from procbot.services.audit import audit
from procbot.services.calendar_google import (
    OP_PROPERTY,
    AuthExpired,
    AuthRevoked,
    CalendarApiError,
    Conflict,
    GoogleClient,
    TokenSet,
    event_id_for,
    full_scope,
    parse_rfc3339,
)
from procbot.services.jobs import PeriodicJob, register_job
from procbot.services.profile import get_setting, set_setting
from procbot.timeutil import start_of_local_day, to_local

log = logging.getLogger(__name__)

T = TypeVar("T")

SYNC_JOB = "calendar.sync"
RECONCILE_JOB = "calendar.reconcile"
RECONCILE_INTERVAL_SECONDS = 300.0
# A block that could not be inserted for this long is given up (and reported once).
INSERT_GIVE_UP = timedelta(hours=24)
INSERT_LEASE_SECONDS = 120.0

SETTING_CALENDARS = "calendar.calendars"  # [{id, name, primary, access_role, time_zone}]
SETTING_WRITE_DECLINED = "calendar.write_declined"  # owner chose "do not create events"
SETTING_PENDING = "calendar.oauth_pending"  # one in-flight authorization (calendar_oauth)

STALE_NOTE = "⚠️ Данные календаря могут быть неактуальны."
REAUTH_TEXT = "Нужно заново подключить Google Calendar: /calendar"

WEEKDAYS_RU = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")


class ConnState(StrEnum):
    ACTIVE = "active"
    ERROR = "error"  # tokens rejected: the owner must connect again
    DISABLED = "disabled"  # disconnected by the owner


class LinkState(StrEnum):
    CREATING = "creating"
    CREATED = "created"
    FAILED = "failed"


class CalendarNotConnected(Exception):
    """No usable tokens (never connected, disconnected or needs re-authorization)."""


# --------------------------------------------------------------------------- scopes

_S = full_scope
READ_LIST_SCOPES = frozenset(
    _S(s) for s in ("calendar.calendarlist.readonly", "calendar.calendarlist", "calendar.readonly")
) | {_S("calendar")}
READ_EVENT_SCOPES = frozenset(
    _S(s)
    for s in (
        "calendar.events.readonly",
        "calendar.events",
        "calendar.readonly",
        "calendar",
        "calendar.events.owned",
        "calendar.events.owned.readonly",
    )
)
# write scope -> calendar access roles it can insert into
WRITE_SCOPE_ROLES: dict[str, frozenset[str]] = {
    _S("calendar.events.owned"): frozenset({"owner"}),
    _S("calendar.events"): frozenset({"owner", "writer"}),
    _S("calendar"): frozenset({"owner", "writer"}),
}


def writable_roles(scopes: Sequence[str]) -> frozenset[str]:
    roles: set[str] = set()
    for s in scopes:
        roles |= WRITE_SCOPE_ROLES.get(full_scope(s), frozenset())
    return frozenset(roles)


def can_read_events(scopes: Sequence[str]) -> bool:
    return any(full_scope(s) in READ_EVENT_SCOPES for s in scopes)


def can_list_calendars(scopes: Sequence[str]) -> bool:
    return any(full_scope(s) in READ_LIST_SCOPES for s in scopes)


# --------------------------------------------------------------------------- service


def google_configured(env: EnvSettings) -> bool:
    return bool(env.google_client_id.strip() and env.google_client_secret.strip())


def _no_tx(ctx: AppContext) -> None:
    if ctx.db.in_transaction():
        raise RuntimeError("Google Calendar must not be called inside a DB transaction")


class CalendarService:
    """Holds the HTTP client and in-process locks. Lives in `ctx.calendar`."""

    def __init__(self, ctx: AppContext, *, transport: Any = None) -> None:
        self.ctx = ctx
        self.client = GoogleClient(
            client_id=ctx.env.google_client_id,
            client_secret=ctx.env.google_client_secret,
            transport=transport,
        )
        self._token_lock = asyncio.Lock()
        self.sync_lock = asyncio.Lock()

    async def access_token(self, *, force_refresh: bool = False) -> str:
        """A valid access token; refreshes (network) outside any transaction."""
        ctx = self.ctx
        _no_tx(ctx)
        async with self._token_lock:
            async with ctx.db.transaction() as s:
                conn = await get_connection(s)
                tokens = _tokens(ctx, conn)
                if conn is None or tokens is None:
                    raise CalendarNotConnected()
                conn_id = conn.id
            now = ctx.clock.now()
            if not force_refresh and not tokens.expiring(now) and tokens.access_token:
                return tokens.access_token
            if not tokens.refresh_token:
                await mark_reauth(ctx, "no refresh token")
                raise CalendarNotConnected()
            try:
                new = await self.client.refresh(tokens.refresh_token, now=now)
            except AuthRevoked:
                await mark_reauth(ctx, "refresh rejected")
                raise CalendarNotConnected() from None
            new.refresh_token = new.refresh_token or tokens.refresh_token
            new.scopes = new.scopes or tokens.scopes
            async with ctx.db.transaction() as s:
                conn = await s.get(CalendarConnection, conn_id)
                if conn is not None and conn.encrypted_token is not None:
                    conn.encrypted_token = ctx.secrets.encrypt(new.to_json())
                    conn.updated_at = ctx.clock.now()
            return new.access_token

    async def call(self, fn: Callable[[str], Awaitable[T]]) -> T:
        """Run `fn(access_token)`; on 401 refresh once and retry."""
        token = await self.access_token()
        try:
            return await fn(token)
        except AuthExpired:
            token = await self.access_token(force_refresh=True)
            return await fn(token)


def get_service(ctx: AppContext) -> CalendarService | None:
    """The configured service, or None when the Google client is not set in env."""
    svc = ctx.calendar
    if isinstance(svc, CalendarService):
        return svc
    if not google_configured(ctx.env):
        return None
    svc = CalendarService(ctx)
    ctx.calendar = svc
    return svc


# --------------------------------------------------------------------------- connection


async def get_connection(session: AsyncSession) -> CalendarConnection | None:
    stmt = (
        select(CalendarConnection)
        .where(CalendarConnection.provider == "google")
        .order_by(CalendarConnection.id)
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


def _tokens(ctx: AppContext, conn: CalendarConnection | None) -> TokenSet | None:
    if conn is None or conn.encrypted_token is None or conn.status == ConnState.DISABLED:
        return None
    try:
        return TokenSet.from_json(ctx.secrets.decrypt(conn.encrypted_token))
    except (SecretBoxError, ValueError):
        log.error("calendar token cannot be decrypted (wrong MASTER_KEY?)")
        return None


async def save_tokens(ctx: AppContext, tokens: TokenSet) -> int:
    """Store freshly authorized tokens (encrypted) and activate the connection."""
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        conn = await get_connection(s)
        if conn is None:
            conn = CalendarConnection(provider="google", created_at=now, updated_at=now)
            s.add(conn)
        old = _tokens(ctx, conn)
        if tokens.refresh_token is None and old is not None:
            tokens.refresh_token = old.refresh_token
        conn.encrypted_token = ctx.secrets.encrypt(tokens.to_json())
        conn.scopes = list(tokens.scopes)
        conn.status = ConnState.ACTIVE
        conn.last_error = None
        conn.updated_at = now
        await s.flush()
        await audit(
            s,
            now=now,
            actor=Actor.USER,
            action="calendar.connected",
            entity="calendar_connection",
            entity_id=conn.id,
            details={"scopes": list(tokens.scopes), "refresh_token": bool(tokens.refresh_token)},
        )
        return conn.id


async def mark_reauth(ctx: AppContext, reason: str) -> None:
    """Tokens were rejected: drop them, keep the cache (stale) and tell the owner once."""
    now = ctx.clock.now()
    conn_id: int | None = None
    async with ctx.db.transaction() as s:
        conn = await get_connection(s)
        if conn is None or conn.status == ConnState.DISABLED:
            return
        conn.status = ConnState.ERROR
        conn.encrypted_token = None
        conn.last_error = REAUTH_TEXT
        conn.updated_at = now
        conn_id = conn.id
        await audit(
            s,
            now=now,
            actor=Actor.SYSTEM,
            action="calendar.reauth_needed",
            entity="calendar_connection",
            entity_id=conn.id,
            details={"reason": reason},
        )
    await notify(
        ctx,
        "📅 Google Calendar перестал принимать доступ бота. Планирую по сохранённому "
        "расписанию; данные календаря могут быть неактуальны.\n" + REAUTH_TEXT,
        dedup_key=f"calendar:reauth:{conn_id}:{now.date().isoformat()}",
    )


@dataclass(frozen=True)
class CalendarInfo:
    id: str
    name: str
    primary: bool = False
    access_role: str = "reader"
    time_zone: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "primary": self.primary,
            "access_role": self.access_role,
            "time_zone": self.time_zone,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CalendarInfo:
        return cls(
            id=str(d["id"]),
            name=str(d.get("name") or d["id"]),
            primary=bool(d.get("primary")),
            access_role=str(d.get("access_role") or "reader"),
            time_zone=d.get("time_zone"),
        )


@dataclass(frozen=True)
class CalendarStatus:
    configured: bool  # Google OAuth client present in the environment
    connected: bool  # usable tokens stored
    needs_reauth: bool
    scopes: list[str]
    read_calendar_ids: list[str]
    write_calendar_id: str | None
    write_declined: bool
    can_write: bool  # write calendar chosen + write scope granted + connected
    last_sync_at: datetime | None
    last_error: str | None
    stale: bool
    calendars: list[CalendarInfo] = field(default_factory=list)

    def name_of(self, calendar_id: str | None) -> str:
        for c in self.calendars:
            if c.id == calendar_id:
                return c.name
        return calendar_id or "—"


def _is_stale(ctx: AppContext, conn: CalendarConnection | None) -> bool:
    if conn is None or conn.status == ConnState.DISABLED or not conn.read_calendar_ids:
        return False
    if conn.status == ConnState.ERROR or conn.last_error:
        return True
    if conn.last_sync_at is None:
        return True
    limit = timedelta(minutes=ctx.config.calendar.stale_after_minutes)
    return ctx.clock.now() - conn.last_sync_at > limit


async def available_calendars(session: AsyncSession) -> list[CalendarInfo]:
    raw = await get_setting(session, SETTING_CALENDARS, None) or []
    out: list[CalendarInfo] = []
    for d in raw:
        if isinstance(d, dict) and d.get("id"):
            out.append(CalendarInfo.from_dict(d))
    return out


async def connection_status(session: AsyncSession, ctx: AppContext) -> CalendarStatus:
    conn = await get_connection(session)
    configured = google_configured(ctx.env)
    connected = (
        conn is not None and conn.encrypted_token is not None and conn.status == ConnState.ACTIVE
    )
    scopes = list(conn.scopes or []) if conn is not None else []
    write_id = conn.write_calendar_id if conn is not None else None
    cals = await available_calendars(session)
    can_write = False
    if connected and configured and write_id:
        roles = writable_roles(scopes)
        role = next((c.access_role for c in cals if c.id == write_id), None)
        can_write = role in roles if role is not None else bool(roles)
    return CalendarStatus(
        configured=configured,
        connected=connected,
        needs_reauth=conn is not None and conn.status == ConnState.ERROR,
        scopes=scopes,
        read_calendar_ids=list(conn.read_calendar_ids or []) if conn is not None else [],
        write_calendar_id=write_id,
        write_declined=bool(await get_setting(session, SETTING_WRITE_DECLINED, False)),
        can_write=can_write,
        last_sync_at=conn.last_sync_at if conn is not None else None,
        last_error=conn.last_error if conn is not None else None,
        stale=_is_stale(ctx, conn),
        calendars=cals,
    )


async def busy_intervals(
    session: AsyncSession, ctx: AppContext, start: datetime, end: datetime
) -> tuple[list[tuple[datetime, datetime]], bool]:
    """Busy calendar time inside [start, end): merged, sorted, clipped intervals plus
    `stale` (True = the cache may be outdated: show STALE_NOTE, still use the data).

    Reads the local cache only (no network), so it is safe inside a transaction.
    """
    rows = (
        await session.execute(
            select(CalendarEventCache.start_at, CalendarEventCache.end_at)
            .where(
                CalendarEventCache.busy.is_(True),
                CalendarEventCache.end_at > start,
                CalendarEventCache.start_at < end,
            )
            .order_by(CalendarEventCache.start_at)
        )
    ).all()
    merged: list[tuple[datetime, datetime]] = []
    for s_at, e_at in rows:
        s_at, e_at = max(s_at, start), min(e_at, end)
        if e_at <= s_at:
            continue
        if merged and s_at <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e_at))
        else:
            merged.append((s_at, e_at))
    return merged, _is_stale(ctx, await get_connection(session))


async def is_stale(session: AsyncSession, ctx: AppContext) -> bool:
    return _is_stale(ctx, await get_connection(session))


# --------------------------------------------------------------------------- calendar choice


def _calendar_info(item: dict[str, Any]) -> CalendarInfo:
    return CalendarInfo(
        id=str(item["id"]),
        name=str(item.get("summaryOverride") or item.get("summary") or item["id"])[:100],
        primary=bool(item.get("primary")),
        access_role=str(item.get("accessRole") or "reader"),
        time_zone=item.get("timeZone"),
    )


async def refresh_calendar_list(ctx: AppContext) -> list[CalendarInfo]:
    """Fetch the owner's calendar list (network) and remember it for the pickers.
    Selections pointing at calendars that disappeared are dropped."""
    svc = get_service(ctx)
    if svc is None:
        raise CalendarNotConnected()
    _no_tx(ctx)
    items = await svc.call(svc.client.list_calendars)
    cals = [_calendar_info(i) for i in items if i.get("id")]
    cals.sort(key=lambda c: (not c.primary, c.name.lower()))
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        await set_setting(s, SETTING_CALENDARS, [c.to_dict() for c in cals], now=now)
        conn = await get_connection(s)
        if conn is not None:
            ids = {c.id for c in cals}
            keep = [c for c in conn.read_calendar_ids or [] if c in ids]
            if keep != list(conn.read_calendar_ids or []):
                await _set_read_ids(s, conn, keep)
            if conn.write_calendar_id and conn.write_calendar_id not in ids:
                conn.write_calendar_id = None
            conn.updated_at = now
    return cals


async def _set_read_ids(session: AsyncSession, conn: CalendarConnection, ids: list[str]) -> None:
    removed = set(conn.read_calendar_ids or []) - set(ids)
    conn.read_calendar_ids = list(ids)
    if removed:
        await session.execute(
            delete(CalendarEventCache).where(
                CalendarEventCache.connection_id == conn.id,
                CalendarEventCache.calendar_id.in_(removed),
            )
        )


async def toggle_read_calendar(
    session: AsyncSession, ctx: AppContext, calendar_id: str
) -> bool | None:
    """Owner picked/unpicked a calendar for reading. Returns the new state, None if
    the calendar is unknown or there is no connection."""
    conn = await get_connection(session)
    cals = {c.id for c in await available_calendars(session)}
    if conn is None or calendar_id not in cals:
        return None
    ids = list(conn.read_calendar_ids or [])
    selected = calendar_id not in ids
    ids = [*ids, calendar_id] if selected else [c for c in ids if c != calendar_id]
    await _set_read_ids(session, conn, ids)
    conn.updated_at = ctx.clock.now()
    await audit(
        session,
        now=ctx.clock.now(),
        actor=Actor.USER,
        action="calendar.read_selected" if selected else "calendar.read_unselected",
        entity="calendar_connection",
        entity_id=conn.id,
        details={"calendar_id": calendar_id},
    )
    return selected


async def set_write_calendar(
    session: AsyncSession, ctx: AppContext, calendar_id: str | None
) -> bool:
    """Owner chose the write calendar (None = "do not create events")."""
    conn = await get_connection(session)
    if conn is None:
        return False
    now = ctx.clock.now()
    if calendar_id is not None:
        cal = next((c for c in await available_calendars(session) if c.id == calendar_id), None)
        if cal is None or cal.access_role not in writable_roles(conn.scopes or []):
            return False
    conn.write_calendar_id = calendar_id
    conn.updated_at = now
    await set_setting(session, SETTING_WRITE_DECLINED, calendar_id is None, now=now)
    await audit(
        session,
        now=now,
        actor=Actor.USER,
        action="calendar.write_selected",
        entity="calendar_connection",
        entity_id=conn.id,
        details={"calendar_id": calendar_id},
    )
    return True


def writable_calendars(status: CalendarStatus) -> list[CalendarInfo]:
    roles = writable_roles(status.scopes)
    return [c for c in status.calendars if c.access_role in roles]


async def _delete_setting(session: AsyncSession, key: str) -> None:
    row = await session.get(Setting, key)
    if row is not None:
        await session.delete(row)


async def disconnect(ctx: AppContext) -> bool:
    """Delete tokens, choices and cached events; then revoke at Google (best effort).

    Returns True when Google confirmed the revocation. Links to events the bot
    created stay (history); the events themselves are not touched."""
    _no_tx(ctx)
    now = ctx.clock.now()
    tokens: TokenSet | None = None
    async with ctx.db.transaction() as s:
        conn = await get_connection(s)
        if conn is not None:
            tokens = _tokens(ctx, conn)
            conn.encrypted_token = None
            conn.status = ConnState.DISABLED
            conn.scopes = []
            conn.read_calendar_ids = []
            conn.write_calendar_id = None
            conn.last_sync_at = None
            conn.last_error = None
            conn.updated_at = now
            await s.execute(
                delete(CalendarEventCache).where(CalendarEventCache.connection_id == conn.id)
            )
            await audit(
                s,
                now=now,
                actor=Actor.USER,
                action="calendar.disconnected",
                entity="calendar_connection",
                entity_id=conn.id,
                details={},
            )
        for key in (SETTING_CALENDARS, SETTING_WRITE_DECLINED, SETTING_PENDING):
            await _delete_setting(s, key)
    svc = get_service(ctx)
    if tokens is None or svc is None:
        return False
    return await svc.client.revoke(tokens.refresh_token or tokens.access_token)


# --------------------------------------------------------------------------- sync


@dataclass
class SyncReport:
    skipped: str | None = None  # not_configured / not_connected / no_calendars / busy
    calendars: int = 0
    events: int = 0
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.skipped is None and not self.errors


@dataclass(frozen=True)
class CachedEvent:
    external_id: str
    start_at: datetime
    end_at: datetime
    all_day: bool
    busy: bool
    summary: str | None


def parse_event(item: dict[str, Any], tz: str | None) -> CachedEvent | None:
    """Google event -> cache row data; None for cancelled / malformed events."""
    if item.get("status") == "cancelled" or not item.get("id"):
        return None
    start = item.get("start") or {}
    end = item.get("end") or {}
    try:
        if start.get("dateTime"):
            s_at = parse_rfc3339(start["dateTime"])
            e_at = parse_rfc3339(end.get("dateTime") or start["dateTime"])
            all_day = False
        elif start.get("date"):
            s_day = date.fromisoformat(start["date"])
            e_day = date.fromisoformat(end["date"]) if end.get("date") else s_day
            if e_day <= s_day:
                e_day = s_day + timedelta(days=1)  # end date is exclusive
            s_at, e_at = start_of_local_day(s_day, tz), start_of_local_day(e_day, tz)
            all_day = True
        else:
            return None
    except (ValueError, KeyError, TypeError):
        return None
    if e_at < s_at:
        return None
    busy = item.get("transparency", "opaque") != "transparent"
    for a in item.get("attendees") or []:
        if isinstance(a, dict) and a.get("self") and a.get("responseStatus") == "declined":
            busy = False
    if item.get("eventType") == "workingLocation":
        busy = False
    private = (item.get("extendedProperties") or {}).get("private") or {}
    if OP_PROPERTY in private:
        busy = False  # a work block this bot created: the owner's own plan
    summary = item.get("summary")
    return CachedEvent(
        external_id=str(item["id"])[:300],
        start_at=s_at,
        end_at=e_at,
        all_day=all_day,
        busy=busy,
        summary=None if summary is None else str(summary)[:500],
    )


async def sync_calendars(ctx: AppContext) -> SyncReport:
    """Refresh the cache for the selected read calendars (network, then one tx)."""
    _no_tx(ctx)
    svc = get_service(ctx)
    if svc is None:
        return SyncReport(skipped="not_configured")
    if svc.sync_lock.locked():
        return SyncReport(skipped="busy")
    async with svc.sync_lock:
        return await _sync(ctx, svc)


async def _sync(ctx: AppContext, svc: CalendarService) -> SyncReport:
    from procbot.services.tasks import owner_tz

    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        conn = await get_connection(s)
        if conn is None or conn.status != ConnState.ACTIVE or conn.encrypted_token is None:
            return SyncReport(skipped="not_connected")
        ids = list(conn.read_calendar_ids or [])
        conn_id = conn.id
        tz = await owner_tz(s)
    if not ids:
        return SyncReport(skipped="no_calendars")
    cfg = ctx.config.calendar
    t_min = now - timedelta(days=cfg.sync_past_days)
    t_max = now + timedelta(days=cfg.sync_future_days)
    report = SyncReport(calendars=len(ids))
    fetched: dict[str, dict[str, CachedEvent]] = {}
    for cid in ids:
        try:
            items = await svc.call(
                lambda tok, cid=cid: svc.client.list_events(tok, cid, t_min, t_max)
            )
        except CalendarNotConnected:
            report.errors[cid] = "нет доступа"
            break
        except CalendarApiError as exc:
            report.errors[cid] = exc.short()
            continue
        events: dict[str, CachedEvent] = {}
        for item in items:
            ev = parse_event(item, tz)
            if ev is not None:
                events[ev.external_id] = ev
        fetched[cid] = events
    done = ctx.clock.now()
    async with ctx.db.transaction() as s:
        conn = await s.get(CalendarConnection, conn_id)
        if conn is None or conn.status == ConnState.DISABLED:
            return SyncReport(skipped="not_connected")
        current = set(conn.read_calendar_ids or [])
        await s.execute(
            delete(CalendarEventCache).where(
                CalendarEventCache.connection_id == conn_id,
                CalendarEventCache.calendar_id.not_in(current),
            )
        )
        for cid, events in fetched.items():
            if cid not in current:
                continue
            await s.execute(
                delete(CalendarEventCache).where(
                    CalendarEventCache.connection_id == conn_id,
                    CalendarEventCache.calendar_id == cid,
                )
            )
            for ev in events.values():
                s.add(
                    CalendarEventCache(
                        connection_id=conn_id,
                        calendar_id=cid,
                        external_id=ev.external_id,
                        start_at=ev.start_at,
                        end_at=ev.end_at,
                        all_day=ev.all_day,
                        busy=ev.busy,
                        summary=ev.summary,
                        synced_at=done,
                    )
                )
                report.events += 1
        if report.errors:
            conn.last_error = "; ".join(f"{k}: {v}" for k, v in report.errors.items())[:500]
            if conn.status == ConnState.ERROR:
                conn.last_error = REAUTH_TEXT
        else:
            conn.last_sync_at = done
            conn.last_error = None
        conn.updated_at = done
    if report.errors:
        log.warning("calendar sync failed", extra={"count": len(report.errors)})
    return report


async def _sync_job(ctx: AppContext) -> None:
    await sync_calendars(ctx)


# --------------------------------------------------------------------------- notices


async def notify(ctx: AppContext, text: str, *, dedup_key: str, reply_markup: Any = None) -> None:
    """Owner notice through the Outbound Gate (SYSTEM_NOTICE); never raises."""
    try:
        from procbot.services.gate import deliver

        await deliver(
            ctx,
            category=MessageCategory.SYSTEM_NOTICE,
            text=text,
            reply_markup=reply_markup,
            dedup_key=dedup_key[:200],
        )
    except Exception:
        log.exception("calendar notice not delivered")


# --------------------------------------------------------------------------- creating events


def fmt_block(start: datetime, end: datetime, tz: str | None) -> str:
    """ "26.10 (пн) 10:00–11:30" in the owner's timezone (both dates if it spans days)."""
    ls, le = to_local(start, tz), to_local(end, tz)
    day = f"{ls:%d.%m} ({WEEKDAYS_RU[ls.weekday()]})"
    ends_at_midnight = le.date() == ls.date() + timedelta(days=1) and le.hour == le.minute == 0
    if ls.date() == le.date() or ends_at_midnight:
        return f"{day} {ls:%H:%M}–{le:%H:%M}"
    return f"{day} {ls:%H:%M} – {le:%d.%m} {le:%H:%M}"


def _payload_hash(action: CreateCalendarEvent) -> str:
    raw = actions.dump_action(action)
    return hashlib.sha256(repr(sorted(raw.items())).encode()).hexdigest()[:16]


async def _validate_block(
    session: AsyncSession, ctx: AppContext, action: CreateCalendarEvent
) -> CalendarStatus:
    if not action.title.strip():
        raise actions.ActionError("У события нет названия.")
    if action.start_at.tzinfo is None or action.end_at.tzinfo is None:
        raise actions.ActionError("Время события без часового пояса.")
    if action.end_at <= action.start_at:
        raise actions.ActionError("Конец события раньше начала.")
    if action.end_at - action.start_at > timedelta(hours=ctx.config.calendar.max_block_hours):
        raise actions.ActionError("Слишком длинный блок для события в календаре.")
    if action.end_at <= ctx.clock.now():
        raise actions.ActionError("Это время уже прошло.")
    st = await connection_status(session, ctx)
    if not st.connected:
        raise actions.ActionError("Google Calendar не подключён: /calendar")
    if not st.can_write:
        raise actions.ActionError("Календарь для создания событий не выбран: /calendar")
    return st


async def describe_create_event(
    session: AsyncSession, ctx: AppContext, action: CreateCalendarEvent
) -> dict[str, Any]:
    """Card text: exactly which block, when and where will be created."""
    from procbot.services.tasks import owner_tz

    st = await _validate_block(session, ctx, action)
    tz = await owner_tz(session)
    when = fmt_block(action.start_at, action.end_at, tz)
    cal = st.name_of(st.write_calendar_id)
    title = " ".join(action.title.split())
    lines = [f"📅 {when}", f"Календарь: {cal}"]
    busy, stale = await busy_intervals(session, ctx, action.start_at, action.end_at)
    if busy:
        lines.append("⚠️ В это время в календаре уже есть занятость.")
    if stale:
        lines.append(STALE_NOTE)
    lines.append("Существующие события не меняются; создаётся только этот блок.")
    return {
        "summary": f"Создать событие в календаре: «{title}» {when} в «{cal}»",
        "lines": lines,
        "calendar_id": st.write_calendar_id,
    }


def _same_action(row: PendingApproval, action: CreateCalendarEvent) -> bool:
    try:
        return actions.parse_action(row.action_payload) == action
    except ValidationError:
        return False


async def _pending_matches(
    session: AsyncSession, ctx: AppContext, action: CreateCalendarEvent
) -> list[PendingApproval]:
    now = ctx.clock.now()
    rows = (
        await session.execute(
            select(PendingApproval)
            .where(
                PendingApproval.action_type == ActionType.CREATE_CALENDAR_EVENT.value,
                PendingApproval.status == ApprovalStatus.PENDING,
            )
            .order_by(PendingApproval.id)
        )
    ).scalars()
    return [
        r for r in rows if _same_action(r, action) and (r.expires_at is None or r.expires_at > now)
    ]


async def _link_for_block(
    session: AsyncSession, calendar_id: str, action: CreateCalendarEvent
) -> CalendarEventLink | None:
    """An earlier link for the very same block (one block = one event)."""
    rows = (
        await session.execute(
            select(CalendarEventLink).where(
                CalendarEventLink.calendar_id == calendar_id,
                CalendarEventLink.start_at == action.start_at,
                CalendarEventLink.end_at == action.end_at,
                CalendarEventLink.state.in_([LinkState.CREATING, LinkState.CREATED]),
            )
        )
    ).scalars()
    for link in rows:
        if link.approval_id is None:
            continue
        a = await session.get(PendingApproval, link.approval_id)
        if a is not None and _same_action(a, action):
            return link
    return None


async def apply_create_event(
    session: AsyncSession, ctx: AppContext, action: CreateCalendarEvent, actor: Actor
) -> actions.ApplyOutcome:
    """Applier for CREATE_CALENDAR_EVENT (runs in the approving transaction).

    Business-level permission check: there must be an approval card for exactly
    this payload; the network insert additionally requires that card to be
    APPLIED (checked again in the after-commit hook)."""
    st = await _validate_block(session, ctx, action)
    matches = await _pending_matches(session, ctx, action)
    if not matches:
        raise actions.ActionError(
            "Событие в календаре создаётся только после подтверждения карточки."
        )
    linked = {
        r[0]
        for r in (
            await session.execute(
                select(CalendarEventLink.approval_id).where(
                    CalendarEventLink.approval_id.in_([m.id for m in matches])
                )
            )
        ).all()
    }
    approval = next((m for m in matches if m.id not in linked), matches[0])
    calendar_id = str((approval.diff or {}).get("calendar_id") or st.write_calendar_id)
    if calendar_id != st.write_calendar_id:
        raise actions.ActionError(
            "Календарь для записи изменился после предложения — предложи блок заново."
        )
    same = await _link_for_block(session, calendar_id, action)
    if same is not None and same.approval_id != approval.id:
        return actions.ApplyOutcome(
            result={"calendar_link_id": same.id, "already": True, "state": same.state}
        )
    link = (
        await session.execute(
            select(CalendarEventLink).where(CalendarEventLink.approval_id == approval.id)
        )
    ).scalar_one_or_none()
    now = ctx.clock.now()
    if link is None:
        link = CalendarEventLink(
            operation_key=f"cal:a{approval.id}:{secrets.token_hex(6)}",
            approval_id=approval.id,
            plan_item_id=action.plan_item_id,
            task_id=action.task_id,
            calendar_id=calendar_id,
            state=LinkState.CREATING,
            start_at=action.start_at,
            end_at=action.end_at,
            created_at=now,
        )
        session.add(link)
    elif link.state == LinkState.CREATED:
        return actions.ApplyOutcome(
            result={"calendar_link_id": link.id, "already": True, "state": link.state}
        )
    else:
        link.state = LinkState.CREATING
        link.created_at = now
    await session.flush()
    link_id = link.id

    async def _insert() -> None:
        await insert_linked_event(ctx, link_id, first_attempt=True)

    return actions.ApplyOutcome(
        result={"calendar_link_id": link_id, "calendar_id": calendar_id},
        after_commit=_insert,
    )


async def _authorizing_approval(
    session: AsyncSession, link: CalendarEventLink
) -> tuple[PendingApproval, CreateCalendarEvent] | None:
    """The APPLIED approval that allows this insert, with its exact action."""

    def ok(a: PendingApproval | None) -> CreateCalendarEvent | None:
        if a is None or a.status != ApprovalStatus.APPLIED:
            return None
        try:
            act = actions.parse_action(a.action_payload)
        except ValidationError:
            return None
        if not isinstance(act, CreateCalendarEvent):
            return None
        if act.start_at != link.start_at or act.end_at != link.end_at:
            return None
        if (a.diff or {}).get("calendar_id", link.calendar_id) != link.calendar_id:
            return None
        return act

    if link.approval_id is not None:
        a = await session.get(PendingApproval, link.approval_id)
        act = ok(a)
        if a is not None and act is not None:
            return a, act
    # Two identical cards: the applier may have linked the twin of the card that was
    # actually approved. Accept an APPLIED twin that no other link uses.
    twins = (
        await session.execute(
            select(PendingApproval)
            .where(
                PendingApproval.action_type == ActionType.CREATE_CALENDAR_EVENT.value,
                PendingApproval.status == ApprovalStatus.APPLIED,
            )
            .order_by(PendingApproval.id)
        )
    ).scalars()
    for a in list(twins):
        act = ok(a)
        if act is None:
            continue
        used = (
            await session.execute(
                select(CalendarEventLink.id).where(
                    CalendarEventLink.approval_id == a.id, CalendarEventLink.id != link.id
                )
            )
        ).first()
        if used is None:
            link.approval_id = a.id
            return a, act
    return None


def event_body(
    action: CreateCalendarEvent, link: CalendarEventLink, tz: str | None
) -> dict[str, Any]:
    zone_name = tz or "UTC"
    return {
        "id": event_id_for(link.operation_key),
        "summary": " ".join(action.title.split())[:500],
        "description": "Рабочий блок, созданный procbot по твоему подтверждению.\n"
        f"procbot-op: {link.operation_key}",
        "start": {"dateTime": to_local(link.start_at, tz).isoformat(), "timeZone": zone_name},
        "end": {"dateTime": to_local(link.end_at, tz).isoformat(), "timeZone": zone_name},
        "extendedProperties": {"private": {OP_PROPERTY: link.operation_key}},
    }


async def insert_linked_event(ctx: AppContext, link_id: int, *, first_attempt: bool) -> str:
    """Insert the event for an approved link (after commit / reconcile job).

    Returns created / reconciled / conflict / skipped / refused / busy / retry / failed.
    """
    from procbot.services.tasks import owner_tz

    _no_tx(ctx)
    lease = f"calendar.insert:{link_id}"
    if not await leases.acquire(ctx, lease, ttl_seconds=INSERT_LEASE_SECONDS):
        return "busy"
    try:
        async with ctx.db.transaction() as s:
            link = await s.get(CalendarEventLink, link_id)
            if link is None or link.state != LinkState.CREATING:
                return "skipped"
            auth = await _authorizing_approval(s, link)
            if auth is None:
                link.state = LinkState.FAILED
                await audit(
                    s,
                    now=ctx.clock.now(),
                    actor=Actor.SYSTEM,
                    action="calendar.insert_refused",
                    entity="calendar_event_link",
                    entity_id=link.id,
                    details={"reason": "no applied approval for this exact block"},
                )
                return "refused"
            _, action = auth
            body = event_body(action, link, await owner_tz(s))
            key, cal_id = link.operation_key, link.calendar_id
            start, end, created_at = link.start_at, link.end_at, link.created_at
        svc = get_service(ctx)
        how = "created"
        ext_id: str | None = None
        error: CalendarApiError | None = None
        try:
            if svc is None:
                raise CalendarNotConnected()
            if not first_attempt:
                found = await svc.call(
                    lambda tok: svc.client.list_events(
                        tok,
                        cal_id,
                        start - timedelta(days=1),
                        end + timedelta(days=1),
                        private_property=(OP_PROPERTY, key),
                        show_deleted=True,
                    )
                )
                if found:
                    ext_id, how = str(found[0].get("id")), "reconciled"
            if ext_id is None:
                try:
                    made = await svc.call(lambda tok: svc.client.insert_event(tok, cal_id, body))
                    ext_id = str(made.get("id") or body["id"])
                except Conflict:
                    ext_id, how = str(body["id"]), "conflict"
        except CalendarNotConnected:
            error = CalendarApiError("not connected")  # transient: may be reconnected
        except CalendarApiError as exc:
            error = exc
        now = ctx.clock.now()
        if error is not None:
            permanent = (
                error.status is not None
                and not error.transient
                and not isinstance(error, AuthExpired)
            )
            final = permanent or now - created_at > INSERT_GIVE_UP
            if not final:
                return "retry"
            await _fail_link(ctx, link_id, action.title, error.short())
            return "failed"
        async with ctx.db.transaction() as s:
            link = await s.get(CalendarEventLink, link_id)
            if link is None:
                return "skipped"
            link.state = LinkState.CREATED
            link.external_event_id = ext_id
            await audit(
                s,
                now=now,
                actor=Actor.USER,
                action="calendar.event_created",
                entity="calendar_event_link",
                entity_id=link.id,
                details={"how": how, "calendar_id": cal_id, "approval_id": link.approval_id},
            )
        return how
    finally:
        await leases.release(ctx, lease)


async def _fail_link(ctx: AppContext, link_id: int, title: str, reason: str) -> None:
    async with ctx.db.transaction() as s:
        link = await s.get(CalendarEventLink, link_id)
        if link is None or link.state != LinkState.CREATING:
            return
        link.state = LinkState.FAILED
        await audit(
            s,
            now=ctx.clock.now(),
            actor=Actor.SYSTEM,
            action="calendar.insert_failed",
            entity="calendar_event_link",
            entity_id=link.id,
            details={"reason": reason},
        )
    await notify(
        ctx,
        f"📅 Не удалось создать событие «{' '.join(title.split())[:100]}» в календаре "
        f"({reason}). Ничего не создано; можно предложить блок заново.",
        dedup_key=f"calendar:insert_failed:{link_id}",
    )


async def reconcile_pending_inserts(ctx: AppContext) -> dict[int, str]:
    """Job: finish links left in `creating` (crash between commit and insert,
    transient errors). Searches by the private property first: no duplicates."""
    _no_tx(ctx)
    async with ctx.db.transaction() as s:
        ids = list(
            (
                await s.execute(
                    select(CalendarEventLink.id)
                    .where(CalendarEventLink.state == LinkState.CREATING)
                    .order_by(CalendarEventLink.id)
                )
            ).scalars()
        )
    return {i: await insert_linked_event(ctx, i, first_attempt=False) for i in ids}


async def _reconcile_job(ctx: AppContext) -> None:
    await reconcile_pending_inserts(ctx)


async def propose_calendar_block(
    session: AsyncSession,
    ctx: AppContext,
    *,
    title: str,
    start_at: datetime,
    end_at: datetime,
    task_id: int | None = None,
    plan_item_id: int | None = None,
    origin: Actor = Actor.SYSTEM,
    op_key: str | None = None,
    group_key: str | None = None,
) -> actions.ExecResult:
    """Offer "create this block in the calendar" as an approval card (never applied
    directly: origin USER is downgraded to SYSTEM so the card is always shown).

    Returns the executor result: PENDING_APPROVAL with `approval_id` (render it with
    handlers.approvals.render_approval_card) or INVALID with a Russian `error`
    (e.g. calendar not connected / no write calendar). An identical pending card is
    reused instead of creating a second one."""
    if origin == Actor.USER:
        origin = Actor.SYSTEM
    action = CreateCalendarEvent(
        title=" ".join(title.split())[:500] or title,
        start_at=start_at,
        end_at=end_at,
        task_id=task_id,
        plan_item_id=plan_item_id,
    )
    existing = await _pending_matches(session, ctx, action)
    if existing:
        return actions.ExecResult(
            actions.ExecStatus.PENDING_APPROVAL,
            ActionType.CREATE_CALENDAR_EVENT,
            existing[0].op_key,
            approval_id=existing[0].id,
            repeated=True,
        )
    key = op_key or f"calendar:propose:{_payload_hash(action)}:{secrets.token_hex(4)}"
    return await actions.execute(
        session, ctx, action, origin=origin, op_key=key, group_key=group_key
    )


async def can_offer_events(session: AsyncSession, ctx: AppContext) -> bool:
    """True when "create in calendar" buttons make sense (connected + write calendar)."""
    return (await connection_status(session, ctx)).can_write


# --------------------------------------------------------------------------- wiring


def register(ctx: AppContext) -> None:
    """Applier for CREATE_CALENDAR_EVENT + sync and reconcile jobs.

    No applier is (or may be) registered for UPDATE/DELETE_CALENDAR_EVENT."""
    if ctx.calendar is None and google_configured(ctx.env):
        ctx.calendar = CalendarService(ctx)
    actions.register_applier(
        ctx,
        ActionType.CREATE_CALENDAR_EVENT,
        apply_create_event,
        describe=describe_create_event,
    )
    register_job(
        ctx,
        PeriodicJob(
            name=SYNC_JOB,
            interval_seconds=ctx.config.calendar.sync_interval_minutes * 60,
            run=_sync_job,
            run_on_start=True,
        ),
    )
    register_job(
        ctx,
        PeriodicJob(
            name=RECONCILE_JOB,
            interval_seconds=RECONCILE_INTERVAL_SECONDS,
            run=_reconcile_job,
            run_on_start=True,
        ),
    )


__all__ = [
    "STALE_NOTE",
    "CalendarInfo",
    "CalendarNotConnected",
    "CalendarService",
    "CalendarStatus",
    "SyncReport",
    "busy_intervals",
    "can_offer_events",
    "connection_status",
    "disconnect",
    "get_service",
    "insert_linked_event",
    "is_stale",
    "propose_calendar_block",
    "reconcile_pending_inserts",
    "sync_calendars",
]
