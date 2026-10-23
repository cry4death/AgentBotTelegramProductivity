"""Voice pipeline (FR-02, acceptance 5): voice -> transcript -> text pipeline;
the bot only ever answers with text."""

from __future__ import annotations

import pytest
from aiogram.methods import GetFile, SendAudio, SendMessage, SendVideoNote, SendVoice
from sqlalchemy import func, select

from procbot.ai.testing import FakeError, make_fake_gateway
from procbot.config import AISwitches, AppConfig
from procbot.db.models import AICall, ConversationMessage, PendingApproval, Task
from procbot.domain.enums import AIErrorKind
from procbot.testing import message_update, voice_update
from procbot.wiring import register_all


@pytest.fixture
def wired(ctx):
    return register_all(ctx)


@pytest.fixture
def bot(bot_and_session):
    return bot_and_session[0]


@pytest.fixture
def tg(bot_and_session):
    return bot_and_session[1]


async def count(ctx, model, *where) -> int:
    async with ctx.db.transaction() as s:
        stmt = select(func.count()).select_from(model)
        for w in where:
            stmt = stmt.where(w)
        return (await s.execute(stmt)).scalar_one()


def texts(tg) -> list[str]:
    return [r.text for r in tg.requests if isinstance(r, SendMessage)]


def assert_text_only(tg) -> None:
    assert not [r for r in tg.requests if isinstance(r, SendVoice | SendAudio | SendVideoNote)]


async def test_voice_is_transcribed_and_answered_with_text(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    tg.files["v1"] = b"OggS-fake-audio"
    gw = await make_fake_gateway(
        ctx,
        {
            "transcribe": "Завтра купить хлеб и позвонить маме",
            "route_message": {"intent": "new_tasks", "confidence": 0.9},
            "extract_tasks": {
                "candidates": [
                    {"title": "Купить хлеб", "when": "завтра"},
                    {"title": "Позвонить маме", "when": "завтра"},
                ]
            },
        },
    )
    upd = voice_update("v1", duration=4)
    await dp.feed_update(bot, upd)
    sent = texts(tg)
    assert sent[0] == "🎙 Распознал: «Завтра купить хлеб и позвонить маме»"
    assert "Купить хлеб" in sent[1] and "Позвонить маме" in sent[1]
    assert_text_only(tg)
    stt = gw.fake.calls_for("transcribe", kind="transcribe")
    assert len(stt) == 1 and stt[0].audio == b"OggS-fake-audio"
    # the transcript went through the same pipeline, as data
    assert "Завтра купить хлеб" in gw.fake.calls_for("extract_tasks")[0].user
    assert await count(ctx, Task) == 0
    assert await count(ctx, PendingApproval) == 2
    async with ctx.db.transaction() as s:
        row = (await s.execute(select(ConversationMessage))).scalar_one()
        stt_call = (await s.execute(select(AICall).where(AICall.task == "transcribe"))).scalar_one()
    assert row.kind == "voice" and row.transcript == "Завтра купить хлеб и позвонить маме"
    assert stt_call.outcome == "ok"
    # repeated delivery of the same update: nothing new
    await dp.feed_update(bot, upd)
    assert len(gw.fake.calls_for("transcribe", kind="transcribe")) == 1
    assert await count(ctx, PendingApproval) == 2


async def test_transcription_unavailable_asks_to_type(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    tg.files["v2"] = b"audio"
    await make_fake_gateway(ctx, {"transcribe": [FakeError(AIErrorKind.SERVER)] * 10})
    await dp.feed_update(bot, voice_update("v2"))
    assert texts(tg)[-1].startswith("Не получилось распознать голосовое")
    assert "текстом" in texts(tg)[-1]
    assert_text_only(tg)
    assert await count(ctx, PendingApproval) == 0


async def test_no_gateway_asks_to_type(wired, profile, dp, bot, tg) -> None:
    tg.files["v3"] = b"audio"
    await dp.feed_update(bot, voice_update("v3"))
    assert "текстом" in texts(tg)[-1]
    assert_text_only(tg)


async def test_voice_feature_switched_off(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    ctx.config = AppConfig(ai_features=AISwitches(voice=False))
    gw = await make_fake_gateway(ctx, {"transcribe": "текст"})
    await dp.feed_update(bot, voice_update("v4"))
    assert "выключены" in texts(tg)[-1]
    assert not gw.fake.calls_for("transcribe", kind="transcribe")
    assert not [r for r in tg.requests if isinstance(r, GetFile)]


async def test_too_long_voice_is_not_downloaded(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    gw = await make_fake_gateway(ctx, {"transcribe": "текст"})
    await dp.feed_update(bot, voice_update("v5", duration=3600))
    assert "слишком длинное" in texts(tg)[-1]
    assert not [r for r in tg.requests if isinstance(r, GetFile)]
    assert not gw.fake.calls_for("transcribe", kind="transcribe")


async def test_empty_transcript(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    tg.files["v6"] = b"audio"
    gw = await make_fake_gateway(ctx, {"transcribe": "   "})
    await dp.feed_update(bot, voice_update("v6"))
    assert "Не расслышал" in texts(tg)[-1]
    assert not gw.fake.calls_for("route_message")


async def test_voice_while_a_question_is_open(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    gw = await make_fake_gateway(ctx, {"transcribe": "текст"})
    await dp.feed_update(bot, message_update("/retro"))  # FSM waits for a choice
    await dp.feed_update(bot, voice_update("v7"))
    assert "напиши его текстом" in texts(tg)[-1]
    assert not gw.fake.calls_for("transcribe", kind="transcribe")


async def test_voice_chat_question_gets_text_reply(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    tg.files["v8"] = b"audio"
    await make_fake_gateway(
        ctx,
        {
            "transcribe": "как мне начать день",
            "route_message": {"intent": "question", "confidence": 0.9},
            "chat_reply": {"text": "Начни с самого короткого дела."},
        },
    )
    await dp.feed_update(bot, voice_update("v8"))
    assert texts(tg) == ["🎙 Распознал: «как мне начать день»", "Начни с самого короткого дела."]
    assert_text_only(tg)
