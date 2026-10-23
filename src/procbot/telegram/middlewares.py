"""Input-layer middlewares: owner-only access and update de-duplication."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, Update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from procbot.context import AppContext
from procbot.db.models import ConversationMessage, ProcessedUpdate

log = logging.getLogger(__name__)

NOT_ALLOWED_TEXT = "Это личный бот, доступ закрыт."

Handler = Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]]


def update_user_id(update: Update) -> int | None:
    for attr in ("message", "edited_message", "callback_query", "inline_query", "my_chat_member"):
        obj = getattr(update, attr, None)
        if obj is not None and getattr(obj, "from_user", None) is not None:
            return obj.from_user.id
    return None


class AuthMiddleware(BaseMiddleware):
    """Outer middleware: only the configured owner reaches any handler.

    Strangers get a neutral refusal that reveals nothing about the owner's data.
    """

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx

    async def __call__(self, handler: Handler, event: TelegramObject, data: dict[str, Any]) -> Any:
        assert isinstance(event, Update)
        uid = update_user_id(event)
        owner = self.ctx.owner_id
        if owner and uid == owner:
            return await handler(event, data)
        log.warning("rejected update from non-owner", extra={"update_id": event.update_id})
        bot = data.get("bot")
        try:
            if event.message is not None and bot is not None:
                await bot.send_message(event.message.chat.id, NOT_ALLOWED_TEXT)
            elif event.callback_query is not None and bot is not None:
                await bot.answer_callback_query(event.callback_query.id, NOT_ALLOWED_TEXT)
        except Exception:
            pass
        return None


class DedupMiddleware(BaseMiddleware):
    """Outer middleware: a repeated update_id is ignored.

    The update id is recorded after the handler succeeds. Handlers that create
    data should also call `mark_processed(session, update_id, now)` inside their
    own transaction so the record is atomic with the change, and use
    `f"u{update_id}:..."` operation keys for anything they create.
    """

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx

    async def __call__(self, handler: Handler, event: TelegramObject, data: dict[str, Any]) -> Any:
        assert isinstance(event, Update)
        async with self.ctx.db.transaction() as s:
            seen = await s.get(ProcessedUpdate, event.update_id)
        if seen is not None:
            log.info("duplicate update ignored", extra={"update_id": event.update_id})
            return None
        data["update_id"] = event.update_id
        result = await handler(event, data)
        now = self.ctx.clock.now()
        async with self.ctx.db.transaction() as s:
            await mark_processed(s, event.update_id, now)
            await _log_incoming(s, event, now)
        return result


async def mark_processed(session: Any, update_id: int, now: Any) -> None:
    stmt = (
        sqlite_insert(ProcessedUpdate)
        .values(update_id=update_id, processed_at=now)
        .on_conflict_do_nothing(index_elements=["update_id"])
    )
    await session.execute(stmt)


async def _log_incoming(session: Any, update: Update, now: Any) -> None:
    """Conversation history (FR-12). Retention is open question 8: kept until deleted."""
    if update.message is not None:
        m = update.message
        kind = "voice" if m.voice else ("command" if (m.text or "").startswith("/") else "text")
        session.add(
            ConversationMessage(
                direction="in",
                kind=kind,
                telegram_message_id=m.message_id,
                update_id=update.update_id,
                text=m.text,
                created_at=now,
            )
        )
    elif update.callback_query is not None:
        session.add(
            ConversationMessage(
                direction="in",
                kind="callback",
                update_id=update.update_id,
                text=update.callback_query.data,
                created_at=now,
            )
        )
