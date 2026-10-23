"""Outbound Telegram port. Services never touch aiogram's Bot directly; they use
a Messenger so tests can record messages and the gate can control every send."""

from __future__ import annotations

from typing import Protocol

from aiogram import Bot
from aiogram.types import InlineKeyboardMarkup, ReplyKeyboardMarkup


class MessengerError(Exception):
    """Sending failed. `permanent` = retrying will not help (e.g. bot blocked)."""

    def __init__(self, message: str, *, permanent: bool = False) -> None:
        super().__init__(message)
        self.permanent = permanent


Markup = InlineKeyboardMarkup | ReplyKeyboardMarkup


class Messenger(Protocol):
    async def send_message(
        self, chat_id: int, text: str, reply_markup: Markup | None = None
    ) -> int:
        """Send a text message, return Telegram message_id. Raise MessengerError."""
        ...

    async def delete_message(self, chat_id: int, message_id: int) -> None: ...


class AiogramMessenger:
    def __init__(self, bot: Bot) -> None:
        self.bot = bot

    async def send_message(
        self, chat_id: int, text: str, reply_markup: Markup | None = None
    ) -> int:
        from aiogram.exceptions import (
            TelegramForbiddenError,
            TelegramNetworkError,
            TelegramRetryAfter,
            TelegramServerError,
        )

        try:
            msg = await self.bot.send_message(chat_id, text, reply_markup=reply_markup)
        except TelegramForbiddenError as exc:
            raise MessengerError(str(exc), permanent=True) from exc
        except (TelegramNetworkError, TelegramServerError, TelegramRetryAfter) as exc:
            raise MessengerError(str(exc)) from exc
        return msg.message_id

    async def delete_message(self, chat_id: int, message_id: int) -> None:
        try:
            await self.bot.delete_message(chat_id, message_id)
        except Exception:
            pass
