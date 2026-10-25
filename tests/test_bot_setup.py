"""Startup UI: the command list and the Mini App menu button for the owner."""

from __future__ import annotations

from aiogram.methods import SetChatMenuButton, SetMyCommands

from procbot.config import EnvSettings
from procbot.telegram.setup import COMMANDS, configure_bot_ui, miniapp_url
from procbot.testing import OWNER_ID, make_bot


def _env(url: str = "") -> EnvSettings:
    return EnvSettings(owner_telegram_id=OWNER_ID, miniapp_public_url=url)  # type: ignore[call-arg]


def test_miniapp_url_needs_https():
    assert miniapp_url(_env("")) is None
    assert miniapp_url(_env("http://bot.example")) is None
    assert miniapp_url(_env("https://bot.example/")) == "https://bot.example/miniapp"
    assert miniapp_url(_env("https://bot.example/miniapp")) == "https://bot.example/miniapp"


async def test_sets_commands_and_menu_button_for_the_owner_only():
    bot, session = make_bot()
    done = await configure_bot_ui(bot, _env("https://bot.example"))
    assert done == ["commands", "menu_button"]
    cmds = next(r for r in session.requests if isinstance(r, SetMyCommands))
    assert [c.command for c in cmds.commands] == [c for c, _ in COMMANDS]
    assert cmds.scope is not None and cmds.scope.chat_id == OWNER_ID  # type: ignore[union-attr]
    menu = next(r for r in session.requests if isinstance(r, SetChatMenuButton))
    assert menu.chat_id == OWNER_ID
    assert menu.menu_button.web_app.url == "https://bot.example/miniapp"  # type: ignore[union-attr]


async def test_without_https_only_commands():
    bot, session = make_bot()
    assert await configure_bot_ui(bot, _env("")) == ["commands"]
    assert not any(isinstance(r, SetChatMenuButton) for r in session.requests)


async def test_telegram_error_is_not_fatal():
    bot, session = make_bot()

    async def boom(*_a, **_k):
        raise RuntimeError("network")

    session.make_request = boom  # type: ignore[method-assign]
    assert await configure_bot_ui(bot, _env("https://bot.example")) == []
