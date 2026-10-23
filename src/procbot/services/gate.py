"""Outbound Gate: the ONLY code that sends bot-initiated messages (FR-07, hard rule 4).

Checks, in order (architecture, section "Фоновый планировщик и привратник"):

1. the category is one of the six initiative categories, a timer signal, a
   system notice or the folded digest, and it is not disabled in config;
2. notifications are enabled in the profile (timer signals the user set up for
   a running session still go through);
3. not inside the owner's quiet hours (local windows, DST-correct). Importance
   never overrides quiet hours. Timer signals obey quiet hours / manual pause only
   if ``gate.timer_signals_respect_quiet_hours``;
4. no manual pause (``profile.pause_until``);
5. the calendar is not busy (``calendar_event_cache`` rows with busy=True overlapping
   the moment), if ``gate.respect_calendar_busy``; timer signals ignore it;
6. no active/paused work session (acceptance 14) - everything except timer signals
   is blocked while the user works;
7. the strict sliding limit: at most ``gate.max_reminders`` counted messages in any
   ``gate.window_minutes`` (counted centrally in ``outbound_messages``);
8. the dedup key was not used yet.

Which categories count toward the limit (open question 3): reminder categories
(PRE_TASK, START_DELAY, REPLAN) ALWAYS count; TIMER_SIGNAL and SYSTEM_NOTICE never
count; every other category uses ``GateConfig.policy(c).counts_toward_limit`` and
``None`` (not decided) is treated as "counts" - the conservative choice.

Atomic reservation: the limit is re-checked and the ``outbound_messages`` row is
inserted (send_state=sending) inside ONE ``BEGIN IMMEDIATE`` transaction, so two
concurrent tasks or processes can never exceed the limit. Telegram is called only
after that commit; the result is recorded in a second transaction. A crash between
send and confirm leaves the row in ``sending``; recovery marks it ``maybe_sent`` and
it is never re-sent blindly (it keeps counting toward the window).
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from aiogram.types import InlineKeyboardMarkup
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.config import GateConfig
from procbot.context import AppContext
from procbot.db.models import (
    CalendarEventCache,
    ConversationMessage,
    OutboundMessage,
    WorkSession,
)
from procbot.domain.enums import (
    INITIATIVE_CATEGORIES,
    REMINDER_CATEGORIES,
    Actor,
    MessageCategory,
    SendState,
    SessionStatus,
)
from procbot.services.audit import audit
from procbot.services.profile import get_profile, quiet_windows
from procbot.telegram.messenger import Markup, MessengerError
from procbot.timeutil import LocalWindow, active_window_end

log = logging.getLogger(__name__)

ALLOWED_CATEGORIES: frozenset[MessageCategory] = INITIATIVE_CATEGORIES | {
    MessageCategory.TIMER_SIGNAL,
    MessageCategory.SYSTEM_NOTICE,
    MessageCategory.DIGEST,
}
NEVER_COUNTED: frozenset[MessageCategory] = frozenset(
    {MessageCategory.TIMER_SIGNAL, MessageCategory.SYSTEM_NOTICE}
)

# Technical re-evaluation cadence while a work session runs: the session's end is
# not known in advance, so a blocked reminder is looked at again after this long.
# It is not a user-facing reminder interval (those are open question 3).
SESSION_RECHECK = timedelta(minutes=10)
# How far ahead calendar busy blocks are loaded to compute the next allowed time.
CALENDAR_HORIZON = timedelta(days=2)
_COUNTED_STATES = (SendState.SENDING, SendState.SENT, SendState.MAYBE_SENT)


class BlockReason(StrEnum):
    CATEGORY_NOT_ALLOWED = "category_not_allowed"
    CATEGORY_DISABLED = "category_disabled"
    NOTIFICATIONS_DISABLED = "notifications_disabled"
    QUIET_HOURS = "quiet_hours"
    PAUSED = "paused"
    CALENDAR_BUSY = "calendar_busy"
    WORK_SESSION = "work_session"
    RATE_LIMIT = "rate_limit"
    DUPLICATE = "duplicate"


class Outcome(StrEnum):
    ALLOWED = "allowed"  # check(): may be sent now
    BLOCKED = "blocked"
    SENT = "sent"  # deliver(): Telegram confirmed
    DUPLICATE = "duplicate"  # deliver(): dedup key already sent / maybe sent / in flight
    FAILED = "failed"  # deliver(): Telegram error (see `permanent`)


@dataclass(frozen=True)
class GateDecision:
    outcome: Outcome
    reason: BlockReason | None = None
    # When blocked: the earliest moment all known blocks are lifted (None = no such
    # moment, e.g. the category is disabled). For FAILED: None.
    retry_at: datetime | None = None
    outbound_id: int | None = None
    telegram_message_id: int | None = None
    error: str | None = None
    permanent: bool = False

    @property
    def allowed(self) -> bool:
        return self.outcome in (Outcome.ALLOWED, Outcome.SENT)

    @property
    def final(self) -> bool:
        """Blocked with no moment when it could pass (drop instead of rescheduling)."""
        return self.outcome == Outcome.BLOCKED and self.retry_at is None


def counts_toward_limit(category: MessageCategory, config: GateConfig) -> bool:
    if category in REMINDER_CATEGORIES:
        return True
    if category in NEVER_COUNTED:
        return False
    flag = config.policy(category).counts_toward_limit
    return True if flag is None else flag  # open question 3: None = conservative


def _as_category(category: MessageCategory | str) -> MessageCategory | None:
    try:
        return MessageCategory(category)
    except ValueError:
        return None


# --------------------------------------------------------------------------- state


@dataclass
class GateState:
    """Snapshot of everything the gate looks at, loaded once per decision."""

    loaded_at: datetime
    config: GateConfig
    tz: str | None = None
    quiet: list[LocalWindow] = field(default_factory=list)
    pause_until: datetime | None = None
    notifications_enabled: bool = True
    chat_id: int | None = None
    busy: list[tuple[datetime, datetime]] = field(default_factory=list)
    # Active/paused work session: blocks until this moment (re-evaluated then).
    session_block_until: datetime | None = None
    # created_at of counted outbound rows inside the window, ascending.
    counted: list[datetime] = field(default_factory=list)


async def load_state(session: AsyncSession, ctx: AppContext, now: datetime) -> GateState:
    cfg = ctx.config.gate
    profile = await get_profile(session)
    st = GateState(loaded_at=now, config=cfg)
    if profile is not None:
        st.tz = profile.timezone
        st.quiet = quiet_windows(profile)
        st.pause_until = profile.pause_until
        st.notifications_enabled = bool(profile.notifications_enabled)
        st.chat_id = profile.chat_id
    if cfg.respect_calendar_busy:
        rows = (
            await session.execute(
                select(CalendarEventCache.start_at, CalendarEventCache.end_at).where(
                    CalendarEventCache.busy.is_(True),
                    CalendarEventCache.end_at > now,
                    CalendarEventCache.start_at < now + CALENDAR_HORIZON,
                )
            )
        ).all()
        st.busy = sorted((r[0], r[1]) for r in rows)
    active = (
        await session.execute(
            select(WorkSession.id)
            .where(WorkSession.status.in_([SessionStatus.ACTIVE, SessionStatus.PAUSED]))
            .limit(1)
        )
    ).first()
    if active is not None:
        st.session_block_until = now + SESSION_RECHECK
    window = timedelta(minutes=cfg.window_minutes)
    counted = (
        (
            await session.execute(
                select(OutboundMessage.created_at)
                .where(
                    OutboundMessage.counts_toward_limit.is_(True),
                    OutboundMessage.created_at > now - window,
                    OutboundMessage.send_state.in_([str(s) for s in _COUNTED_STATES]),
                )
                .order_by(OutboundMessage.created_at)
            )
        )
        .scalars()
        .all()
    )
    st.counted = list(counted)
    return st


def evaluate(state: GateState, category: MessageCategory | str, at: datetime) -> GateDecision:
    """Pure decision for `category` at moment `at` (>= state.loaded_at), checks 1-7."""
    cfg = state.config
    cat = _as_category(category)
    if cat is None or cat not in ALLOWED_CATEGORIES:
        return GateDecision(Outcome.BLOCKED, BlockReason.CATEGORY_NOT_ALLOWED)
    if not cfg.policy(cat).enabled:
        return GateDecision(Outcome.BLOCKED, BlockReason.CATEGORY_DISABLED)
    is_timer = cat == MessageCategory.TIMER_SIGNAL
    if not state.notifications_enabled and not is_timer:
        return GateDecision(Outcome.BLOCKED, BlockReason.NOTIFICATIONS_DISABLED)
    respects_quiet = not is_timer or cfg.timer_signals_respect_quiet_hours
    if respects_quiet:
        end = active_window_end(at, state.quiet, state.tz)
        if end is not None:
            return GateDecision(Outcome.BLOCKED, BlockReason.QUIET_HOURS, retry_at=end)
        if state.pause_until is not None and state.pause_until > at:
            return GateDecision(Outcome.BLOCKED, BlockReason.PAUSED, retry_at=state.pause_until)
    if not is_timer and cfg.respect_calendar_busy:
        busy_end = _busy_end(state.busy, at)
        if busy_end is not None:
            return GateDecision(Outcome.BLOCKED, BlockReason.CALENDAR_BUSY, retry_at=busy_end)
    if not is_timer and state.session_block_until is not None and at < state.session_block_until:
        return GateDecision(
            Outcome.BLOCKED, BlockReason.WORK_SESSION, retry_at=state.session_block_until
        )
    if counts_toward_limit(cat, cfg):
        free_at = _slot_free_at(state, at)
        if free_at is not None or cfg.max_reminders <= 0:
            return GateDecision(Outcome.BLOCKED, BlockReason.RATE_LIMIT, retry_at=free_at)
    return GateDecision(Outcome.ALLOWED)


def _busy_end(busy: list[tuple[datetime, datetime]], at: datetime) -> datetime | None:
    end: datetime | None = None
    moment = at
    changed = True
    while changed:  # follow back-to-back / overlapping events
        changed = False
        for s, e in busy:
            if s <= moment < e and (end is None or e > end):
                end = e
                moment = e
                changed = True
    return end


def _slot_free_at(state: GateState, at: datetime) -> datetime | None:
    """None if a counted message may be sent at `at`, else when a slot frees up."""
    cfg = state.config
    if cfg.max_reminders <= 0:
        return None
    window = timedelta(minutes=cfg.window_minutes)
    times = state.counted
    # rows that still count at `at`: created_at > at - window
    first_in = bisect_right(times, at - window)
    in_window = times[first_in:]
    if len(in_window) < cfg.max_reminders:
        return None
    # the slot frees when enough of the oldest rows leave the window
    return in_window[len(in_window) - cfg.max_reminders] + window


def next_allowed_in(
    state: GateState, category: MessageCategory | str, after: datetime
) -> datetime | None:
    """Earliest moment >= `after` when `category` passes checks 1-7 (None = never)."""
    t = after
    for _ in range(64):
        d = evaluate(state, category, t)
        if d.outcome == Outcome.ALLOWED:
            return t
        if d.retry_at is None:
            return None
        t = d.retry_at if d.retry_at > t else t + timedelta(minutes=1)
    return t  # pragma: no cover - pathological chain of windows


def _with_retry(
    state: GateState, category: MessageCategory | str, d: GateDecision, at: datetime
) -> GateDecision:
    if d.outcome != Outcome.BLOCKED or d.retry_at is None:
        return d
    return GateDecision(
        Outcome.BLOCKED, d.reason, retry_at=next_allowed_in(state, category, d.retry_at)
    )


# --------------------------------------------------------------------------- public API


async def check(
    ctx: AppContext, category: MessageCategory | str, at: datetime | None = None
) -> GateDecision:
    """Cheap pre-check before rendering. `retry_at` is the next allowed moment."""
    now = ctx.clock.now()
    at = at or now
    async with ctx.db.transaction() as s:
        state = await load_state(s, ctx, now)
    return _with_retry(state, category, evaluate(state, category, at), at)


async def check_in(
    session: AsyncSession, ctx: AppContext, category: MessageCategory | str, at: datetime
) -> GateDecision:
    """Same as check() inside the caller's transaction."""
    state = await load_state(session, ctx, ctx.clock.now())
    return _with_retry(state, category, evaluate(state, category, at), at)


async def next_allowed_time(
    ctx: AppContext, category: MessageCategory | str, after: datetime | None = None
) -> datetime | None:
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        state = await load_state(s, ctx, now)
    return next_allowed_in(state, category, max(after or now, now))


def _markup_json(markup: Markup | None) -> dict[str, Any] | None:
    if markup is None:
        return None
    return markup.model_dump(mode="json", exclude_none=True)


async def deliver(
    ctx: AppContext,
    *,
    category: MessageCategory | str,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
    dedup_key: str,
    reminder_id: int | None = None,
) -> GateDecision:
    """Send a bot-initiated message if every rule allows it right now.

    Must not be called inside a transaction (Telegram is awaited).
    """
    if ctx.db.in_transaction():
        raise RuntimeError("gate.deliver must not be called inside a transaction")
    cat = _as_category(category)
    now = ctx.clock.now()

    # 1) atomic reservation: re-check + insert in one BEGIN IMMEDIATE transaction
    async with ctx.db.transaction() as s:
        state = await load_state(s, ctx, now)
        decision = evaluate(state, category, now)
        if decision.outcome != Outcome.ALLOWED:
            return _with_retry(state, category, decision, now)
        assert cat is not None
        existing = (
            await s.execute(select(OutboundMessage).where(OutboundMessage.dedup_key == dedup_key))
        ).scalar_one_or_none()
        counts = counts_toward_limit(cat, ctx.config.gate)
        if existing is not None:
            if existing.send_state not in (SendState.FAILED, SendState.CANCELLED):
                return GateDecision(
                    Outcome.DUPLICATE,
                    BlockReason.DUPLICATE,
                    outbound_id=existing.id,
                    telegram_message_id=existing.telegram_message_id,
                )
            row = existing  # a failed attempt released its slot; reuse the key
            row.send_state = SendState.SENDING
            row.category = str(cat)
            row.counts_toward_limit = counts
            row.text = text
            row.reply_markup = _markup_json(reply_markup)
            row.error = None
            row.created_at = now
            row.reminder_id = reminder_id
        else:
            row = OutboundMessage(
                category=str(cat),
                dedup_key=dedup_key,
                reminder_id=reminder_id,
                counts_toward_limit=counts,
                send_state=SendState.SENDING,
                text=text,
                reply_markup=_markup_json(reply_markup),
                created_at=now,
            )
            s.add(row)
        await s.flush()
        outbound_id = row.id
        chat_id = state.chat_id or ctx.owner_id

    # 2) network call outside any transaction
    try:
        message_id = await ctx.messenger.send_message(chat_id, text, reply_markup=reply_markup)
    except MessengerError as exc:
        done = ctx.clock.now()
        async with ctx.db.transaction() as s:
            row = await s.get(OutboundMessage, outbound_id)
            if row is not None:
                row.send_state = SendState.FAILED  # releases the slot
                row.error = str(exc)[:500]
            if exc.permanent:
                await audit(
                    s,
                    now=done,
                    actor=Actor.SYSTEM,
                    action="outbound.failed_permanent",
                    entity="outbound_message",
                    entity_id=outbound_id,
                    details={"category": str(cat), "error": str(exc)[:200]},
                )
        level = logging.ERROR if exc.permanent else logging.WARNING
        log.log(level, "outbound send failed (%s): %s", cat, exc)
        return GateDecision(
            Outcome.FAILED, outbound_id=outbound_id, error=str(exc), permanent=exc.permanent
        )

    # 3) confirm
    done = ctx.clock.now()
    async with ctx.db.transaction() as s:
        row = await s.get(OutboundMessage, outbound_id)
        if row is not None:
            row.send_state = SendState.SENT
            row.telegram_message_id = message_id
            row.sent_at = done
        s.add(
            ConversationMessage(
                direction="out",
                kind="text",
                telegram_message_id=message_id,
                text=text,
                created_at=done,
            )
        )
    return GateDecision(Outcome.SENT, outbound_id=outbound_id, telegram_message_id=message_id)


async def sweep_stuck_sending(ctx: AppContext, *, older_than: timedelta) -> int:
    """Rows stuck in `sending` for longer than `older_than` -> maybe_sent.

    They are never re-sent (we cannot know whether Telegram delivered them) and
    keep counting toward the sliding window. Younger rows may still be in flight
    in another instance and are left alone.
    """
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        rows = (
            (
                await s.execute(
                    select(OutboundMessage).where(
                        OutboundMessage.send_state == SendState.SENDING,
                        OutboundMessage.created_at <= now - older_than,
                    )
                )
            )
            .scalars()
            .all()
        )
        for r in rows:
            r.send_state = SendState.MAYBE_SENT
            r.error = "interrupted between send and confirm"
        if rows:
            await audit(
                s,
                now=now,
                actor=Actor.SYSTEM,
                action="outbound.maybe_sent",
                entity="outbound_message",
                details={"ids": [r.id for r in rows]},
            )
    return len(rows)
