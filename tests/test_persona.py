"""Persona text (FR-08): style knobs independent, guardrails always present."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from procbot.ai.persona import (
    GUARDRAILS,
    SAFE_REPLY,
    PersonaKnobs,
    guard_reply,
    load_persona,
    persona_for,
    tone,
)
from procbot.services.profile import update_profile


def _profile(**kw):
    base = dict(style_preset="neutral", strictness=2, humor=1, profanity_allowed=False, verbosity=2)
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"style_preset": "coach", "strictness": 4, "humor": 0},
        {"style_preset": "ironic", "humor": 4, "profanity_allowed": True},
        {"style_preset": "soft", "strictness": 0, "verbosity": 0},
        {"style_preset": "unknown-preset", "strictness": 99, "verbosity": -3},
    ],
)
def test_guardrails_always_present(kw) -> None:
    text = persona_for(_profile(**kw))
    for rule in GUARDRAILS:
        assert rule in text
    assert "ложью, симуляцией" in text
    assert "Не ставь диагнозов" in text
    assert "предположения" in text
    assert "без унижений" in text.lower() or "Никаких унижений" in text


def test_knobs_change_style_independently() -> None:
    coach = persona_for(_profile(style_preset="coach", strictness=4))
    soft = persona_for(_profile(style_preset="soft", strictness=0))
    assert "Строгий тренер" in coach and "Строгость максимальная" in coach
    assert "Мягкий" in soft and "Строгость минимальная" in soft
    assert "Без ненормативной лексики" in persona_for(_profile())
    assert "Ненормативная лексика допустима" in persona_for(_profile(profanity_allowed=True))
    short = persona_for(_profile(verbosity=0))
    assert "предельно кратко" in short
    # out-of-range knobs are clamped, never crash
    assert PersonaKnobs.from_profile(_profile(strictness=99)).strictness == 4
    assert PersonaKnobs.from_profile(None) == PersonaKnobs()


async def test_load_persona_reads_profile(ctx, profile) -> None:
    async with ctx.db.transaction() as s:
        await update_profile(s, now=ctx.clock.now(), style_preset="ironic", humor=4)
    text = await load_persona(ctx)
    assert "Ироничный" in text and "Много юмора" in text
    assert GUARDRAILS[0] in text


def test_tone_helper() -> None:
    assert tone(_profile(style_preset="coach"), neutral="n", strict="s", soft="m") == "s"
    assert tone(_profile(style_preset="soft"), neutral="n", strict="s", soft="m") == "m"
    assert tone(_profile(), neutral="n", strict="s", soft="m") == "n"
    assert tone(None, neutral="n") == "n"


@pytest.mark.parametrize(
    "reply",
    [
        "Ты просто симулируешь усталость, хватит.",
        "Твоя усталость — это отговорка.",
        "Ты врёшь, что устал.",
        "У тебя явно депрессия.",
        "Ставлю диагноз: СДВГ.",
    ],
)
def test_guard_reply_blocks_accusations_and_diagnoses(reply) -> None:
    text, filtered = guard_reply(reply)
    assert filtered and text == SAFE_REPLY


@pytest.mark.parametrize(
    "reply",
    [
        "Похоже, ты устал — давай сократим план.",
        "Возможно, задача кажется слишком большой. Проверим: начни с 5 минут?",
        "Готово? Отметь в /tasks.",
    ],
)
def test_guard_reply_passes_normal_text(reply) -> None:
    assert guard_reply(reply) == (reply, False)
