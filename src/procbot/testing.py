"""Test doubles shared by the test-suite: a recording Messenger, a mocked aiogram
session (no network), update builders and a context factory.

Lives in the package (not tests/) so every module's tests can import it.
"""

from __future__ import annotations

import itertools
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.methods import (
    EditMessageReplyMarkup,
    EditMessageText,
    GetFile,
    GetMe,
    SendMessage,
    TelegramMethod,
)
from aiogram.types import (
    CallbackQuery,
    Chat,
    File,
    InlineKeyboardMarkup,
    Message,
    Update,
    User,
    Voice,
)

from procbot.clock import FrozenClock
from procbot.config import AppConfig, EnvSettings
from procbot.context import AppContext
from procbot.crypto import SecretBox
from procbot.db.session import Database
from procbot.telegram.messenger import Markup, MessengerError

OWNER_ID = 1001
STRANGER_ID = 6666
BOT_ID = 4242
TEST_MASTER_KEY = "nIEqqyxTaHkyNLVcCwMfpNo1UjzXA2SxbJFk1bed0No="


@dataclass
class SentMessage:
    chat_id: int
    text: str
    reply_markup: Any = None
    message_id: int = 0


@dataclass
class FakeMessenger:
    """Records outbound (bot-initiated) messages. Can be told to fail."""

    sent: list[SentMessage] = field(default_factory=list)
    deleted: list[tuple[int, int]] = field(default_factory=list)
    fail_next: int = 0
    fail_permanent: bool = False
    _ids: itertools.count[int] = field(default_factory=lambda: itertools.count(5000))

    async def send_message(
        self, chat_id: int, text: str, reply_markup: Markup | None = None
    ) -> int:
        if self.fail_next > 0:
            self.fail_next -= 1
            raise MessengerError("simulated failure", permanent=self.fail_permanent)
        mid = next(self._ids)
        self.sent.append(SentMessage(chat_id, text, reply_markup, mid))
        return mid

    async def delete_message(self, chat_id: int, message_id: int) -> None:
        self.deleted.append((chat_id, message_id))

    def texts(self) -> list[str]:
        return [m.text for m in self.sent]


class MockedBotSession(BaseSession):
    """aiogram session that never touches the network and records every request."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[TelegramMethod[Any]] = []
        self.files: dict[str, bytes] = {}
        self._ids = itertools.count(9000)

    async def close(self) -> None:  # pragma: no cover - nothing to close
        return None

    async def make_request(
        self, bot: Bot, method: TelegramMethod[Any], timeout: int | None = None
    ) -> Any:
        self.requests.append(method)
        now = datetime.now(UTC)
        if isinstance(method, GetMe):
            return User(id=BOT_ID, is_bot=True, first_name="Bot", username="test_bot")
        if isinstance(method, SendMessage):
            return Message(
                message_id=next(self._ids),
                date=now,
                chat=Chat(id=int(method.chat_id), type="private"),
                text=method.text,
                reply_markup=method.reply_markup  # type: ignore[arg-type]
                if isinstance(method.reply_markup, InlineKeyboardMarkup)
                else None,
            )
        if isinstance(method, EditMessageText):
            return Message(
                message_id=int(method.message_id or 0),
                date=now,
                chat=Chat(id=int(method.chat_id or OWNER_ID), type="private"),
                text=method.text,
            )
        if isinstance(method, EditMessageReplyMarkup):
            return True
        if isinstance(method, GetFile):
            return File(
                file_id=method.file_id, file_unique_id=method.file_id, file_path=method.file_id
            )
        return True

    async def stream_content(
        self,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: int = 30,
        chunk_size: int = 65536,
        raise_for_status: bool = True,
    ) -> AsyncGenerator[bytes, None]:
        key = url.rsplit("/", 1)[-1]
        yield self.files.get(key, b"")

    # helpers -----------------------------------------------------------------
    def sent_texts(self) -> list[str]:
        out: list[str] = []
        for r in self.requests:
            if isinstance(r, SendMessage | EditMessageText) and r.text is not None:
                out.append(r.text)
        return out

    def sent_messages(self) -> list[SendMessage]:
        return [r for r in self.requests if isinstance(r, SendMessage)]

    def clear(self) -> None:
        self.requests.clear()


def make_bot() -> tuple[Bot, MockedBotSession]:
    session = MockedBotSession()
    bot = Bot(token="42:TEST", session=session)
    return bot, session


_update_ids = itertools.count(1)
_message_ids = itertools.count(1)


def user(uid: int = OWNER_ID) -> User:
    return User(id=uid, is_bot=False, first_name="Owner" if uid == OWNER_ID else "Stranger")


def message_update(
    text: str | None = None,
    *,
    uid: int = OWNER_ID,
    update_id: int | None = None,
    date: datetime | None = None,
    voice: Voice | None = None,
) -> Update:
    msg = Message(
        message_id=next(_message_ids),
        date=date or datetime.now(UTC),
        chat=Chat(id=uid, type="private"),
        from_user=user(uid),
        text=text,
        voice=voice,
    )
    return Update(update_id=update_id if update_id is not None else next(_update_ids), message=msg)


def callback_update(
    data: str,
    *,
    uid: int = OWNER_ID,
    update_id: int | None = None,
    message_text: str = "…",
) -> Update:
    msg = Message(
        message_id=next(_message_ids),
        date=datetime.now(UTC),
        chat=Chat(id=uid, type="private"),
        from_user=User(id=BOT_ID, is_bot=True, first_name="Bot"),
        text=message_text,
    )
    cq = CallbackQuery(
        id=str(next(_update_ids)), from_user=user(uid), chat_instance="ci", data=data, message=msg
    )
    return Update(
        update_id=update_id if update_id is not None else next(_update_ids), callback_query=cq
    )


def voice_update(file_id: str = "voice1", *, duration: int = 3, uid: int = OWNER_ID) -> Update:
    v = Voice(file_id=file_id, file_unique_id=file_id, duration=duration, mime_type="audio/ogg")
    return message_update(None, uid=uid, voice=v)


async def make_context(
    tmp_path: Path,
    *,
    now: datetime | None = None,
    config: AppConfig | None = None,
    messenger: FakeMessenger | None = None,
) -> AppContext:
    """A fully wired AppContext on a fresh SQLite file in tmp_path."""
    env = EnvSettings(
        telegram_bot_token="42:TEST",
        owner_telegram_id=OWNER_ID,
        master_key=TEST_MASTER_KEY,
        database_path=tmp_path / "test.db",
        backup_dir=tmp_path / "backups",
        lock_path=tmp_path / "procbot.lock",
        app_config_path=tmp_path / "missing-app.yaml",
        ai_config_path=tmp_path / "missing-ai.yaml",
    )
    db = Database.for_path(env.database_path)
    await db.create_all()
    return AppContext(
        env=env,
        config=config or AppConfig(),
        db=db,
        clock=FrozenClock(now or datetime(2026, 10, 26, 9, 0, tzinfo=UTC)),
        secrets=SecretBox(TEST_MASTER_KEY),
        messenger=messenger or FakeMessenger(),
    )
