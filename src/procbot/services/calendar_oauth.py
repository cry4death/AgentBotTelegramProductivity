"""Google OAuth 2.0 authorization-code flow with PKCE for the single owner.

Two redirect modes (Google retired the OOB copy-paste flow in 2022/2023):

- WEB: the bot has a public HTTPS URL. Google redirects the browser to
  `<public>/oauth/google/callback` (served by procbot.http_server, proxied to the
  bot's HTTP port). Needs an OAuth client of type "Web application" with exactly
  that redirect URI registered. Used when GOOGLE_REDIRECT_URI is a non-loopback
  URL, or derived from MINIAPP_PUBLIC_URL when GOOGLE_REDIRECT_URI is empty.
- MANUAL: no public URL. The redirect goes to a loopback address
  (http://127.0.0.1:<port>, allowed for OAuth clients of type "Desktop app");
  nothing listens there, the browser shows an error page, and the owner copies
  the address from the address bar into the chat (the bot deletes that message).

The `state` is random, single-use, bound to the owner's Telegram id and expires
after `calendar.oauth_state_ttl_minutes`; only its SHA-256 is stored. The PKCE
verifier is stored encrypted. Only one authorization is in flight at a time.
Tokens are exchanged with httpx outside any DB transaction and stored encrypted
(services.calendar.save_tokens).
"""

from __future__ import annotations

import hashlib
import hmac
import html
import logging
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any
from urllib.parse import parse_qs, urlparse

from procbot.config import EnvSettings
from procbot.context import AppContext
from procbot.crypto import SecretBoxError
from procbot.domain.enums import Actor
from procbot.services import calendar as cal
from procbot.services.audit import audit
from procbot.services.calendar_google import (
    AuthRevoked,
    CalendarApiError,
    build_auth_url,
    full_scope,
)
from procbot.services.profile import get_setting, set_setting

log = logging.getLogger(__name__)

CALLBACK_PATH = "/oauth/google/callback"
# Loopback redirect for MANUAL mode (Desktop-app OAuth clients accept any port).
LOOPBACK_REDIRECT = "http://127.0.0.1:8765/"
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}


class RedirectMode(StrEnum):
    WEB = "web"
    MANUAL = "manual"


def redirect_target(env: EnvSettings) -> tuple[RedirectMode, str]:
    uri = env.google_redirect_uri.strip()
    if uri:
        host = (urlparse(uri).hostname or "").lower()
        if host in _LOOPBACK_HOSTS:
            return RedirectMode.MANUAL, uri
        return RedirectMode.WEB, uri
    public = env.miniapp_public_url.strip()
    if public.startswith("https://"):
        return RedirectMode.WEB, public.rstrip("/") + CALLBACK_PATH
    return RedirectMode.MANUAL, LOOPBACK_REDIRECT


def requested_scopes(ctx: AppContext) -> list[str]:
    return [full_scope(s) for s in ctx.config.calendar.scopes]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class AuthStart:
    url: str
    mode: RedirectMode
    redirect_uri: str
    expires_at: datetime


@dataclass(frozen=True)
class ConnectResult:
    ok: bool
    message: str  # Russian, for the owner
    calendars: int = 0


async def start_authorization(ctx: AppContext) -> AuthStart:
    """Create a fresh state + PKCE verifier and return the Google consent URL.

    Replaces any earlier unfinished authorization (its state stops working)."""
    if not cal.google_configured(ctx.env):
        raise cal.CalendarNotConnected()
    mode, redirect_uri = redirect_target(ctx.env)
    state = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)  # 86 chars of [A-Za-z0-9_-]: valid PKCE
    now = ctx.clock.now()
    expires = now + timedelta(minutes=ctx.config.calendar.oauth_state_ttl_minutes)
    scopes = requested_scopes(ctx)
    async with ctx.db.transaction() as s:
        await set_setting(
            s,
            cal.SETTING_PENDING,
            {
                "state_sha256": _sha(state),
                "verifier": ctx.secrets.encrypt(verifier),
                "redirect_uri": redirect_uri,
                "mode": mode.value,
                "owner_id": ctx.owner_id,
                "expires_at": expires.isoformat(),
                "scopes": scopes,
            },
            now=now,
        )
        await audit(
            s,
            now=now,
            actor=Actor.USER,
            action="calendar.oauth_started",
            entity="calendar_connection",
            details={"mode": mode.value, "scopes": scopes},
        )
    url = build_auth_url(
        client_id=ctx.env.google_client_id,
        redirect_uri=redirect_uri,
        scopes=scopes,
        state=state,
        code_verifier=verifier,
    )
    return AuthStart(url=url, mode=mode, redirect_uri=redirect_uri, expires_at=expires)


def parse_pasted(text: str) -> tuple[str | None, str | None, str | None]:
    """(code, state, error) from a pasted redirect URL, its query string or a bare code."""
    raw = text.strip()
    if not raw:
        return None, None, None
    query = raw
    if "?" in raw:
        query = urlparse(raw).query or raw.split("?", 1)[1]
    if "=" in query and ("code=" in query or "error=" in query or "state=" in query):
        q = parse_qs(query.lstrip("?#"))

        def one(k: str) -> str | None:
            v = q.get(k)
            return v[0].strip() if v and v[0].strip() else None

        return one("code"), one("state"), one("error")
    if " " in raw or len(raw) < 10 or len(raw) > 512:
        return None, None, None
    return raw, None, None


async def complete_authorization(
    ctx: AppContext, *, code: str, state: str | None, require_state: bool
) -> ConnectResult:
    """Exchange the code (network), store tokens encrypted, fetch the calendar list."""
    cal._no_tx(ctx)
    svc = cal.get_service(ctx)
    if svc is None:
        return ConnectResult(False, "Google Calendar не настроен на сервере.")
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        pending: Any = await get_setting(s, cal.SETTING_PENDING, None)
        if not isinstance(pending, dict):
            return ConnectResult(False, "Нет начатого подключения. Начни заново: /calendar")
        if datetime.fromisoformat(pending["expires_at"]) <= now:
            await cal._delete_setting(s, cal.SETTING_PENDING)
            return ConnectResult(False, "Ссылка для подключения устарела. Начни заново: /calendar")
        if pending.get("owner_id") != ctx.owner_id:
            return ConnectResult(False, "Подключение начато не владельцем бота.")
        if state is None and require_state:
            return ConnectResult(False, "В ответе Google нет параметра state.")
        if state is not None and not hmac.compare_digest(_sha(state), pending["state_sha256"]):
            # do not consume the pending flow: a stray/forged request must not cancel it
            return ConnectResult(False, "Ответ не относится к текущему подключению (state).")
        try:
            verifier = ctx.secrets.decrypt(pending["verifier"])
        except SecretBoxError:
            return ConnectResult(False, "Не удалось прочитать данные подключения.")
        redirect_uri = str(pending["redirect_uri"])
        requested = list(pending.get("scopes") or [])
        await cal._delete_setting(s, cal.SETTING_PENDING)  # single use
    try:
        tokens = await svc.client.exchange_code(
            code, redirect_uri=redirect_uri, code_verifier=verifier, now=now
        )
    except AuthRevoked:
        return ConnectResult(
            False, "Google не принял код (истёк или уже использован). Начни заново: /calendar"
        )
    except CalendarApiError as exc:
        return ConnectResult(
            False, f"Не удалось связаться с Google ({exc.short()}). Попробуй ещё раз: /calendar"
        )
    tokens.scopes = [full_scope(x) for x in (tokens.scopes or requested)]
    if not cal.can_read_events(tokens.scopes) or not cal.can_list_calendars(tokens.scopes):
        await svc.client.revoke(tokens.refresh_token or tokens.access_token)
        return ConnectResult(
            False,
            "Google не выдал права на чтение календарей и событий (возможно, галочки сняты "
            "на экране согласия). Подключи ещё раз и оставь доступ к календарю: /calendar",
        )
    await cal.save_tokens(ctx, tokens)
    try:
        cals = await cal.refresh_calendar_list(ctx)
    except (CalendarApiError, cal.CalendarNotConnected):
        log.warning("calendar list not fetched after connect")
        cals = []
    msg = "✅ Google Calendar подключён."
    if not cal.writable_roles(tokens.scopes):
        msg += " Права на создание событий не выданы — бот будет только читать."
    if tokens.refresh_token is None:
        msg += (
            " Google не выдал refresh token: доступ пропадёт примерно через час — "
            "отключи бота в https://myaccount.google.com/permissions и подключи снова."
        )
    return ConnectResult(True, msg, calendars=len(cals))


async def cancel_on_error(ctx: AppContext, state: str | None) -> bool:
    """Google redirected with ?error=... : drop the pending flow if the state matches."""
    if state is None:
        return False
    async with ctx.db.transaction() as s:
        pending: Any = await get_setting(s, cal.SETTING_PENDING, None)
        if isinstance(pending, dict) and hmac.compare_digest(
            _sha(state), str(pending.get("state_sha256"))
        ):
            await cal._delete_setting(s, cal.SETTING_PENDING)
            return True
    return False


# --------------------------------------------------------------------------- web callback


def _page(title: str, text: str) -> str:
    return (
        "<!doctype html><html lang='ru'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)}</title></head>"
        "<body style='font-family:system-ui,sans-serif;max-width:32rem;margin:3rem auto;"
        f"padding:0 1rem'><h1>{html.escape(title)}</h1><p>{html.escape(text)}</p></body></html>"
    )


async def handle_web_callback(ctx: AppContext, params: Mapping[str, str]) -> tuple[int, str]:
    """GET /oauth/google/callback -> (HTTP status, HTML). Never echoes parameters."""
    state = params.get("state")
    if params.get("error"):
        await cancel_on_error(ctx, state)
        return 200, _page("Доступ не выдан", "Календарь не подключён. Вернитесь в Telegram.")
    code = params.get("code")
    if not code or not state:
        return 400, _page("Ошибка", "Неполный ответ Google.")
    res = await complete_authorization(ctx, code=code, state=state, require_state=True)
    if not res.ok:
        return 400, _page("Не получилось", res.message)
    await notify_connected(ctx, res)
    return 200, _page(
        "Готово", "Google Calendar подключён. Вернитесь в Telegram и выберите календари."
    )


async def notify_connected(ctx: AppContext, res: ConnectResult) -> None:
    """Tell the owner in Telegram (through the gate) to pick calendars."""
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    from procbot.telegram.callbacks import CalendarCb

    markup = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="📖 Выбрать календари", callback_data=CalendarCb(action="read").pack()
                )
            ]
        ]
    )
    await cal.notify(
        ctx,
        res.message + "\nТеперь выбери, какие календари читать и куда создавать события.",
        dedup_key=f"calendar:connected:{ctx.clock.now().isoformat()}",
        reply_markup=markup,
    )
