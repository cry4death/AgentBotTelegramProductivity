"""Procrastination help (FR-04, acceptance 10): reason -> one next action, AI with
a template fallback, no judging tone, repeated press is a no-op."""

from __future__ import annotations

import pytest

from procbot.ai.testing import make_fake_gateway
from procbot.domain.actions import TaskDraft
from procbot.domain.enums import Actor, ObstacleReason
from procbot.services import interventions
from procbot.services import tasks as task_service
from procbot.wiring import register_all


@pytest.fixture
async def task_id(ctx, profile) -> int:
    register_all(ctx)
    async with ctx.db.transaction() as s:
        t = await task_service.create_task(
            s, ctx, TaskDraft(title="Написать отчёт"), actor=Actor.USER
        )
        return t.id


def test_tone_validator_blocks_judging_words():
    assert interventions.tone_errors("Ты просто ленишься, это отговорки")
    assert interventions.tone_errors("Похоже на СДВГ")
    assert interventions.tone_errors("Причина точно в страхе")
    assert not interventions.tone_errors(
        "Возможно, задача кажется большой. Открой файл на 5 минут."
    )


async def test_without_ai_template_suggestion_and_repeat_is_noop(ctx, task_id):
    first = await interventions.suggest_after_reason(
        ctx, task_id=task_id, reason=ObstacleReason.TOO_BIG
    )
    assert first is not None and not first.used_ai and first.text
    again = await interventions.suggest_after_reason(
        ctx, task_id=task_id, reason=ObstacleReason.TOO_BIG
    )
    assert again is not None and again.repeated
    assert again.intervention_id == first.intervention_id


async def test_judging_ai_answer_is_replaced_by_template(ctx, task_id):
    bad = {"kind": "tiny_step", "next_action": "Открой файл", "message": "Ты просто ленишься."}
    await make_fake_gateway(ctx, {"suggest_intervention": [bad] * 10})
    out = await interventions.suggest_after_reason(
        ctx, task_id=task_id, reason=ObstacleReason.UNPLEASANT
    )
    assert out is not None
    assert not interventions.tone_errors(out.text, out.next_action)
    assert "ленишься" not in out.text
