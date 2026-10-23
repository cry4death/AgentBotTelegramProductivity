"""/ai handler: overview, keys (masked), enable/disable, test, spend, add via chat."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from sqlalchemy import select

from procbot.ai import Prompt
from procbot.ai.gateway import AIGateway
from procbot.ai.testing import make_fake_gateway
from procbot.context import AppContext
from procbot.db.models import AICredential, ConversationMessage, ConversationState
from procbot.telegram.callbacks import AIKeyCb
from procbot.testing import FakeMessenger, callback_update, message_update
from test_ai_gateway import BASE, ENV, cfg

pytestmark = pytest.mark.usefixtures("profile")

PASTED_KEY = "sk-or-v1-PASTEDSECRET-abcdef0123456789"


def cb(action: str, credential_id: int = 0, arg: str = "") -> str:
    return AIKeyCb(action=action, credential_id=credential_id, arg=arg).pack()


async def cred(ctx: AppContext, label: str) -> AICredential:
    async with ctx.db.transaction() as s:
        return (
            await s.execute(select(AICredential).where(AICredential.label == label))
        ).scalar_one()


async def test_ai_without_gateway(ctx: AppContext, dp: Any, bot_and_session: Any) -> None:
    bot, session = bot_and_session
    ctx.ai = None
    await dp.feed_update(bot, message_update("/ai"))
    assert "работает без ИИ" in session.sent_texts()[-1]


async def test_overview_keys_masked_and_toggle(
    ctx: AppContext, dp: Any, bot_and_session: Any
) -> None:
    bot, session = bot_and_session
    await make_fake_gateway(ctx, config=cfg(), environ=dict(ENV))
    await dp.feed_update(bot, message_update("/ai"))
    text = session.sent_texts()[-1]
    assert "Состояние: работает" in text and "Бюджет: не задан" in text

    await dp.feed_update(bot, callback_update(cb("keys")))
    text = session.sent_texts()[-1]
    assert "fa — fake, ****1111, активен" in text and "fb — fake, ****2222" in text
    everything = "\n".join(session.sent_texts())
    assert ENV["FA_KEY"] not in everything and ENV["FB_KEY"] not in everything

    fa = await cred(ctx, "fa")
    await dp.feed_update(bot, callback_update(cb("disable", fa.id)))
    assert (await cred(ctx, "fa")).status == "disabled"
    assert "отключён" in session.sent_texts()[-1]
    await dp.feed_update(bot, callback_update(cb("enable", fa.id)))
    assert (await cred(ctx, "fa")).status == "active"

    await dp.feed_update(bot, callback_update(cb("test", fa.id)))
    assert "ключ работает" in session.sent_texts()[-1]


async def test_models_and_spend_views(ctx: AppContext, dp: Any, bot_and_session: Any) -> None:
    bot, session = bot_and_session
    gw = await make_fake_gateway(ctx, config=cfg(), environ=dict(ENV))
    gw.fake.push("chat_reply", "ok")
    await gw.run("chat_reply", Prompt("hi"), initiator="t")
    await dp.feed_update(bot, callback_update(cb("models")))
    text = session.sent_texts()[-1]
    assert "fast" in text and "a1: m-a1 [fa]" in text and "stt1" in text
    await dp.feed_update(bot, callback_update(cb("spend", arg="day")))
    text = session.sent_texts()[-1]
    assert "Расходы ИИ сегодня" in text and "По задачам" in text and "chat_reply" in text


async def test_add_key_disabled_by_default(ctx: AppContext, dp: Any, bot_and_session: Any) -> None:
    bot, session = bot_and_session
    await make_fake_gateway(ctx, config=cfg(), environ=dict(ENV))
    await dp.feed_update(bot, callback_update(cb("add")))
    assert ".env" in session.sent_texts()[-1]
    await dp.feed_update(bot, callback_update(cb("prov", arg="openai")))
    assert ".env" in session.sent_texts()[-1]


async def test_add_key_via_chat_deletes_message_and_encrypts(
    ctx: AppContext, dp: Any, bot_and_session: Any, messenger: FakeMessenger
) -> None:
    bot, session = bot_and_session
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": [{"id": "model-x"}, {"id": "model-y"}]})

    config = cfg(
        credentials={
            **BASE["credentials"],
            "router": {
                "provider": "openai_compatible",
                "base_url": "https://router.example/api/v1",
                "api_key_env": "ROUTER_KEY",  # not set in env
            },
        },
        models={
            **BASE["models"],
            "r1": {"credential": "router", "model": "<r>", "structured_output": "native_schema"},
        },
        features={"allow_keys_via_chat": True},
    )

    async def notifier(text: str, key: str) -> None:
        return None

    gw = AIGateway(
        ctx, config, transport=httpx.MockTransport(handler), environ=dict(ENV), notifier=notifier
    )
    await gw.start()
    ctx.ai = gw

    await dp.feed_update(bot, callback_update(cb("add")))
    assert "Какой провайдер" in session.sent_texts()[-1]
    await dp.feed_update(bot, callback_update(cb("prov", arg="openai_compatible")))
    assert "base_url" in session.sent_texts()[-1]
    await dp.feed_update(bot, message_update("https://router.example/api/v1/"))
    assert "Пришли ключ" in session.sent_texts()[-1]
    key_msg = message_update(PASTED_KEY)
    assert key_msg.message is not None
    await dp.feed_update(bot, key_msg)

    assert (key_msg.message.chat.id, key_msg.message.message_id) in messenger.deleted
    reply = session.sent_texts()[-1]
    assert "openai_compatible-chat-1" in reply and "****6789" in reply
    assert "ключ работает, моделей доступно: 2" in reply
    assert seen[-1].headers["authorization"] == f"Bearer {PASTED_KEY}"
    assert str(seen[-1].url) == "https://router.example/api/v1/models"

    row = await cred(ctx, "openai_compatible-chat-1")
    assert row.source == "chat" and row.key_last4 == "6789"
    assert (
        PASTED_KEY not in row.encrypted_key and ctx.secrets.decrypt(row.encrypted_key) == PASTED_KEY
    )
    # The key never reaches replies or conversation history.
    assert all(PASTED_KEY not in t for t in session.sent_texts())
    async with ctx.db.transaction() as s:
        texts = [m.text or "" for m in (await s.execute(select(ConversationMessage))).scalars()]
        states = await s.execute(select(ConversationState))
        state_dump = json.dumps([r.data for r in states.scalars()], default=str)
    assert all(PASTED_KEY not in t for t in texts) and "[скрыто]" in texts
    assert PASTED_KEY not in state_dump
    # The chat key joins the pool of models on the same provider + base_url.
    assert "openai_compatible-chat-1" in gw.credentials.pool_for(gw.config.models["r1"])
    await gw.aclose()
