"""Telegram Mini App `initData` check (https://core.telegram.org/bots/webapps).

The page sends `Authorization: tma <initData>`. The data is signed by Telegram
with a key derived from the bot token; only the owner's user id is accepted.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timedelta
from urllib.parse import parse_qsl

# How long a signed initData stays usable after Telegram issued it.
MAX_AGE = timedelta(hours=24)


class InitDataError(Exception):
    pass


def _secret(bot_token: str) -> bytes:
    return hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()


def sign_init_data(fields: dict[str, str], bot_token: str) -> str:
    """Build a signed initData string (used by tests)."""
    from urllib.parse import urlencode

    check = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    digest = hmac.new(_secret(bot_token), check.encode(), hashlib.sha256).hexdigest()
    return urlencode({**fields, "hash": digest})


def verify_init_data(raw: str, *, bot_token: str, owner_id: int, now: datetime) -> int:
    """Return the owner's user id, or raise InitDataError."""
    pairs = dict(parse_qsl(raw, keep_blank_values=True, strict_parsing=False))
    received = pairs.pop("hash", None)
    if not received or not bot_token:
        raise InitDataError("no signature")
    check = "\n".join(f"{k}={v}" for k, v in sorted(pairs.items()))
    expected = hmac.new(_secret(bot_token), check.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, received):
        raise InitDataError("bad signature")
    try:
        issued = datetime.fromtimestamp(int(pairs.get("auth_date", "")), tz=now.tzinfo)
        user_id = int(json.loads(pairs.get("user", "{}"))["id"])
    except (ValueError, KeyError, TypeError) as exc:
        raise InitDataError("malformed initData") from exc
    if now - issued > MAX_AGE or issued - now > timedelta(minutes=5):
        raise InitDataError("expired")
    if user_id != owner_id:
        raise InitDataError("not the owner")
    return user_id
