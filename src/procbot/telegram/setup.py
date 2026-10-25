"""Bot UI set at startup: the command list and the Mini App menu button.

Both are scoped to the owner's private chat (nobody else can use the bot).
A failure is logged and never stops the bot: the same settings can be made
by hand in @BotFather.
"""

from __future__ import annotations

import logging

from aiogram import Bot
from aiogram.types import BotCommand, BotCommandScopeChat, MenuButtonWebApp, WebAppInfo

from procbot.config import EnvSettings

log = logging.getLogger(__name__)

MINIAPP_PATH = "/miniapp"
MENU_BUTTON_TEXT = "Открыть"

COMMANDS: tuple[tuple[str, str], ...] = (
    ("today", "План на сегодня"),
    ("add", "Добавить задачу"),
    ("tasks", "Список задач"),
    ("work", "Начать работу"),
    ("retro", "Записать сделанное задним числом"),
    ("week", "Неделя"),
    ("goals", "Цели"),
    ("habits", "Привычки"),
    ("report", "Отчёт"),
    ("memory", "Что бот помнит"),
    ("settings", "Настройки"),
    ("calendar", "Google Calendar"),
    ("ai", "Ключи ИИ"),
    ("export", "Выгрузить мои данные"),
    ("status", "Состояние бота"),
    ("help", "Справка"),
)


def miniapp_url(env: EnvSettings) -> str | None:
    """Public Mini App address, or None when no HTTPS address is configured
    (Telegram opens Mini Apps only over HTTPS)."""
    base = env.miniapp_public_url.strip().rstrip("/")
    if not base.startswith("https://"):
        return None
    return base if base.endswith(MINIAPP_PATH) else base + MINIAPP_PATH


async def configure_bot_ui(bot: Bot, env: EnvSettings) -> list[str]:
    """Set the commands and (with an HTTPS address) the menu button. Returns
    what was set, for the log and tests."""
    done: list[str] = []
    chat_id = env.owner_telegram_id
    if not chat_id:
        return done
    try:
        await bot.set_my_commands(
            [BotCommand(command=c, description=d) for c, d in COMMANDS],
            scope=BotCommandScopeChat(chat_id=chat_id),
        )
        done.append("commands")
    except Exception:
        log.warning("could not set the bot commands", exc_info=True)
    url = miniapp_url(env)
    if url is not None:
        try:
            await bot.set_chat_menu_button(
                chat_id=chat_id,
                menu_button=MenuButtonWebApp(text=MENU_BUTTON_TEXT, web_app=WebAppInfo(url=url)),
            )
            done.append("menu_button")
        except Exception:
            log.warning("could not set the Mini App menu button", exc_info=True)
    return done
