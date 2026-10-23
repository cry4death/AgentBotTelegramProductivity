"""Structured JSON logging with correlation ids.

Rules (spec section 7, architecture "Наблюдаемость"):
- one JSON object per line on stderr;
- every record carries the current correlation id (set per Telegram update,
  per scheduler tick, per CLI command);
- secrets (bot token, API keys, Fernet keys) and message texts are never logged.
  A redaction filter is the last line of defence; code must not pass them anyway.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "procbot_correlation_id", default=None
)

# Extra attributes that may appear in records; anything else passed via `extra`
# is dropped unless it is in this allow-list (keeps message texts out of logs).
ALLOWED_EXTRA: frozenset[str] = frozenset(
    {
        "update_id",
        "job",
        "reminder_id",
        "task_id",
        "session_id",
        "approval_id",
        "category",
        "outcome",
        "duration_ms",
        "path",
        "status",
        "instance_id",
        "attempt",
        "count",
        "error_kind",
        "credential_label",
        "provider",
        "model",
    }
)

# Never emitted even if someone passes them explicitly.
FORBIDDEN_EXTRA: frozenset[str] = frozenset(
    {"text", "message_text", "prompt", "token", "api_key", "master_key", "secret"}
)

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{30,}\b"),  # Telegram bot token
    re.compile(r"\bbot\d{6,12}:[A-Za-z0-9_-]{30,}"),  # token inside API URLs
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),  # OpenAI / Anthropic style keys
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),  # Google API keys
    re.compile(r"\b[A-Za-z0-9_-]{43}=(?![A-Za-z0-9])"),  # Fernet keys
)

_STD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys() | {"message", "asctime"}
)


def redact(value: str) -> str:
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub("[REDACTED]", value)
    return value


def get_correlation_id() -> str | None:
    return _correlation_id.get()


def new_correlation_id(prefix: str = "c") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


@contextmanager
def correlation_scope(cid: str | None = None, *, prefix: str = "c") -> Iterator[str]:
    """Bind a correlation id for the duration of the block (task-local)."""
    value = cid or new_correlation_id(prefix)
    token = _correlation_id.set(value)
    try:
        yield value
    finally:
        _correlation_id.reset(token)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": redact(record.getMessage()),
        }
        cid = getattr(record, "correlation_id", None) or get_correlation_id()
        if cid:
            payload["correlation_id"] = cid
        for key, value in record.__dict__.items():
            if key in _STD_ATTRS or key == "correlation_id" or key in FORBIDDEN_EXTRA:
                continue
            if key in ALLOWED_EXTRA:
                payload[key] = (
                    value if isinstance(value, int | float | bool) else redact(str(value))
                )
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", *, stream: Any = None) -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())
    # aiogram's own per-update INFO lines are noise; keep warnings and errors.
    logging.getLogger("aiogram.event").setLevel(logging.WARNING)
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
    # SQLAlchemy echo would print parameters (= message texts); keep it quiet.
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)


class CorrelationMiddleware:
    """aiogram outer middleware: one correlation id per Telegram update."""

    async def __call__(
        self,
        handler: Callable[[Any, dict[str, Any]], Awaitable[Any]],
        event: Any,
        data: dict[str, Any],
    ) -> Any:
        update_id = getattr(event, "update_id", None)
        with correlation_scope(f"u{update_id}" if update_id is not None else None, prefix="u"):
            return await handler(event, data)
