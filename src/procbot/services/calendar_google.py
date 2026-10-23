"""Minimal Google OAuth 2.0 + Calendar API v3 client over httpx (no googleapiclient).

Acceptance 18 / CLAUDE.md rule 7: this is the ONLY module that talks to Google, and
it exposes exactly these operations:

- OAuth: authorization URL, code exchange, token refresh, token revocation;
- Calendar read: list calendars, list events in a range, free/busy for a range;
- Calendar write: insert ONE new event.

There is deliberately no method that changes or removes an existing event, and
`_send` refuses any HTTP verb other than GET/POST and any POST outside an explicit
allow-list of endpoints (token, revoke, freeBusy, event insert). A test greps
this source to keep it that way.

The client never touches the database and never logs tokens, codes or bodies.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from urllib.parse import quote, urlencode

import httpx

log = logging.getLogger(__name__)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"
CALENDAR_API = "https://www.googleapis.com/calendar/v3"

SCOPE_PREFIX = "https://www.googleapis.com/auth/"

# Private extended property that carries our operation key on events we insert
# (reconciliation after a crash: search by it before inserting again).
OP_PROPERTY = "procbotOp"

HttpMethod = Literal["GET", "POST"]

# POST is allowed only to these endpoints (full-URL regexes).
_POST_ALLOWED = (
    re.compile(re.escape(GOOGLE_TOKEN_URL) + r"$"),
    re.compile(re.escape(GOOGLE_REVOKE_URL) + r"$"),
    re.compile(re.escape(CALENDAR_API) + r"/freeBusy$"),
    # events.insert: POST /calendars/{calendarId}/events (no event id in the path)
    re.compile(re.escape(CALENDAR_API) + r"/calendars/[^/]+/events$"),
)

_EVENT_FIELDS = (
    "nextPageToken,items(id,status,summary,start,end,transparency,eventType,"
    "attendees(self,responseStatus),extendedProperties/private)"
)
_CALENDAR_FIELDS = "nextPageToken,items(id,summary,summaryOverride,primary,accessRole,timeZone)"
MAX_PAGES = 20


# --------------------------------------------------------------------------- errors


class CalendarApiError(Exception):
    """A Google call failed. `transient` errors may be retried later."""

    def __init__(
        self, message: str, *, status: int | None = None, reason: str | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.reason = reason

    @property
    def transient(self) -> bool:
        if self.status is None:  # network / timeout
            return True
        if self.status in (429, 500, 502, 503, 504):
            return True
        return self.status == 403 and self.reason in {"rateLimitExceeded", "userRateLimitExceeded"}

    def short(self) -> str:
        if self.status is None:
            return "нет связи с Google"
        return f"HTTP {self.status}" + (f" ({self.reason})" if self.reason else "")


class AuthExpired(CalendarApiError):
    """401 from the API: the access token is no longer valid (refresh and retry)."""


class AuthRevoked(CalendarApiError):
    """invalid_grant on refresh/exchange: the owner must connect the calendar again."""


class Conflict(CalendarApiError):
    """409: an event with this id already exists (our earlier insert went through)."""


# --------------------------------------------------------------------------- tokens


@dataclass
class TokenSet:
    access_token: str
    refresh_token: str | None
    expires_at: datetime | None
    scopes: list[str] = field(default_factory=list)
    token_type: str = "Bearer"

    def to_json(self) -> str:
        return json.dumps(
            {
                "access_token": self.access_token,
                "refresh_token": self.refresh_token,
                "expires_at": None if self.expires_at is None else self.expires_at.isoformat(),
                "scopes": self.scopes,
                "token_type": self.token_type,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> TokenSet:
        data = json.loads(raw)
        exp = data.get("expires_at")
        return cls(
            access_token=str(data.get("access_token") or ""),
            refresh_token=data.get("refresh_token"),
            expires_at=None if not exp else datetime.fromisoformat(exp),
            scopes=list(data.get("scopes") or []),
            token_type=str(data.get("token_type") or "Bearer"),
        )

    def expiring(self, now: datetime, margin: timedelta = timedelta(seconds=60)) -> bool:
        return self.expires_at is not None and self.expires_at - margin <= now

    def __repr__(self) -> str:  # never print secrets
        return f"TokenSet(scopes={self.scopes!r}, expires_at={self.expires_at!r})"


def full_scope(scope: str) -> str:
    return scope if scope.startswith("https://") else SCOPE_PREFIX + scope


# --------------------------------------------------------------------------- PKCE


def pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def build_auth_url(
    *,
    client_id: str,
    redirect_uri: str,
    scopes: Sequence[str],
    state: str,
    code_verifier: str,
) -> str:
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(full_scope(s) for s in scopes),
        "state": state,
        "code_challenge": pkce_challenge(code_verifier),
        "code_challenge_method": "S256",
        # A refresh token is needed for background sync; prompt=consent makes
        # Google return it again on a reconnect.
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params, quote_via=quote)}"


def event_id_for(op_key: str) -> str:
    """Deterministic event id (base32hex, lowercase: Google allows a-v and 0-9).

    Inserting twice with the same id yields 409 instead of a duplicate event."""
    digest = hashlib.sha256(op_key.encode()).digest()
    return "pb" + base64.b32hexencode(digest).decode("ascii").rstrip("=").lower()


def parse_rfc3339(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:  # pragma: no cover - Google always sends an offset
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _rfc3339(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- client


class GoogleClient:
    """Stateless HTTP client. `transport` is injectable (httpx.MockTransport in tests)."""

    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.client_id = client_id
        self._client_secret = client_secret
        self._transport = transport
        self._timeout = timeout

    # ----------------------------------------------------------------- plumbing

    async def _send(
        self,
        method: HttpMethod,
        url: str,
        *,
        token: str | None = None,
        params: Mapping[str, Any] | None = None,
        form: Mapping[str, str] | None = None,
        body: Mapping[str, Any] | None = None,
    ) -> httpx.Response:
        if method not in ("GET", "POST"):  # pragma: no cover - typing already forbids it
            raise ValueError(f"HTTP method {method} is not allowed for Google Calendar")
        if method == "POST" and not any(p.match(url) for p in _POST_ALLOWED):
            raise ValueError("POST to this Google endpoint is not allowed")
        headers = {"Accept": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=self._timeout
            ) as client:
                return await client.request(
                    method, url, params=params, data=form, json=body, headers=headers
                )
        except httpx.HTTPError as exc:
            raise CalendarApiError(f"network error: {type(exc).__name__}") from None

    @staticmethod
    def _error(resp: httpx.Response) -> CalendarApiError:
        reason: str | None = None
        try:
            data = resp.json()
        except ValueError:
            data = {}
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, str):  # OAuth endpoints: {"error": "invalid_grant", ...}
            reason = err
        elif isinstance(err, dict):
            errors = err.get("errors") or []
            if errors and isinstance(errors[0], dict):
                reason = errors[0].get("reason")
            reason = reason or err.get("status")
        msg = f"google http {resp.status_code}"
        if resp.status_code == 401:
            return AuthExpired(msg, status=401, reason=reason)
        if resp.status_code == 409:
            return Conflict(msg, status=409, reason=reason)
        if reason in ("invalid_grant", "unauthorized_client"):
            return AuthRevoked(msg, status=resp.status_code, reason=reason)
        return CalendarApiError(msg, status=resp.status_code, reason=reason)

    async def _json(self, resp: httpx.Response) -> dict[str, Any]:
        if resp.status_code >= 400:
            raise self._error(resp)
        try:
            data = resp.json()
        except ValueError:
            raise CalendarApiError("invalid JSON from Google", status=resp.status_code) from None
        return data if isinstance(data, dict) else {}

    def _token_set(self, data: Mapping[str, Any], now: datetime) -> TokenSet:
        access = data.get("access_token")
        if not access:
            raise CalendarApiError("no access_token in token response")
        expires_in = data.get("expires_in")
        return TokenSet(
            access_token=str(access),
            refresh_token=data.get("refresh_token"),
            expires_at=None if expires_in is None else now + timedelta(seconds=int(expires_in)),
            scopes=str(data.get("scope") or "").split(),
            token_type=str(data.get("token_type") or "Bearer"),
        )

    # ----------------------------------------------------------------- OAuth

    async def exchange_code(
        self, code: str, *, redirect_uri: str, code_verifier: str, now: datetime
    ) -> TokenSet:
        resp = await self._send(
            "POST",
            GOOGLE_TOKEN_URL,
            form={
                "grant_type": "authorization_code",
                "code": code,
                "client_id": self.client_id,
                "client_secret": self._client_secret,
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            },
        )
        if resp.status_code == 400:
            err = self._error(resp)
            raise AuthRevoked(str(err), status=400, reason=err.reason)
        return self._token_set(await self._json(resp), now)

    async def refresh(self, refresh_token: str, *, now: datetime) -> TokenSet:
        resp = await self._send(
            "POST",
            GOOGLE_TOKEN_URL,
            form={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.client_id,
                "client_secret": self._client_secret,
            },
        )
        if resp.status_code in (400, 401):
            err = self._error(resp)
            raise AuthRevoked(str(err), status=resp.status_code, reason=err.reason)
        return self._token_set(await self._json(resp), now)

    async def revoke(self, token: str) -> bool:
        """Best effort: True when Google confirmed the revocation."""
        try:
            resp = await self._send("POST", GOOGLE_REVOKE_URL, form={"token": token})
        except CalendarApiError:
            return False
        return resp.status_code == 200

    # ----------------------------------------------------------------- read

    async def list_calendars(self, token: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        page: str | None = None
        for _ in range(MAX_PAGES):
            params: dict[str, Any] = {"maxResults": 250, "fields": _CALENDAR_FIELDS}
            if page:
                params["pageToken"] = page
            data = await self._json(
                await self._send(
                    "GET", f"{CALENDAR_API}/users/me/calendarList", token=token, params=params
                )
            )
            out.extend(i for i in data.get("items") or [] if isinstance(i, dict))
            page = data.get("nextPageToken")
            if not page:
                break
        return out

    async def list_events(
        self,
        token: str,
        calendar_id: str,
        time_min: datetime,
        time_max: datetime,
        *,
        private_property: tuple[str, str] | None = None,
        show_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        """Expanded (singleEvents) events overlapping [time_min, time_max)."""
        out: list[dict[str, Any]] = []
        page: str | None = None
        for _ in range(MAX_PAGES):
            params: dict[str, Any] = {
                "timeMin": _rfc3339(time_min),
                "timeMax": _rfc3339(time_max),
                "singleEvents": "true",
                "orderBy": "startTime",
                "maxResults": 250,
                "showDeleted": "true" if show_deleted else "false",
                "fields": _EVENT_FIELDS,
            }
            if private_property is not None:
                params["privateExtendedProperty"] = f"{private_property[0]}={private_property[1]}"
            if page:
                params["pageToken"] = page
            data = await self._json(
                await self._send("GET", _events_url(calendar_id), token=token, params=params)
            )
            out.extend(i for i in data.get("items") or [] if isinstance(i, dict))
            page = data.get("nextPageToken")
            if not page:
                break
        return out

    async def freebusy(
        self, token: str, calendar_ids: Sequence[str], time_min: datetime, time_max: datetime
    ) -> dict[str, list[tuple[datetime, datetime]]]:
        """Busy intervals per calendar (read-only query, POST by API design)."""
        data = await self._json(
            await self._send(
                "POST",
                f"{CALENDAR_API}/freeBusy",
                token=token,
                body={
                    "timeMin": _rfc3339(time_min),
                    "timeMax": _rfc3339(time_max),
                    "items": [{"id": c} for c in calendar_ids],
                },
            )
        )
        out: dict[str, list[tuple[datetime, datetime]]] = {}
        for cid, info in (data.get("calendars") or {}).items():
            busy = info.get("busy") or [] if isinstance(info, dict) else []
            out[cid] = [(parse_rfc3339(b["start"]), parse_rfc3339(b["end"])) for b in busy]
        return out

    # ----------------------------------------------------------------- write (insert only)

    async def insert_event(
        self, token: str, calendar_id: str, event: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Create ONE new event. The only write operation of this client."""
        return await self._json(
            await self._send("POST", _events_url(calendar_id), token=token, body=event)
        )


def _events_url(calendar_id: str) -> str:
    return f"{CALENDAR_API}/calendars/{quote(calendar_id, safe='')}/events"
