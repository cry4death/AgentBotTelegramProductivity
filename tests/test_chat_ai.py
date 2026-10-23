"""Free text through the dispatcher with the AI pipeline (fake gateway).

Acceptance 4 (multi-intent -> candidates, nothing created until confirmed, no
duplicate proposals on a repeated update), 19 (AIUnavailable -> stage A
fallback, task statuses untouched), routing to sessions / retro / chat."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from aiogram.methods import EditMessageText, SendMessage
from aiogram.types import InlineKeyboardMarkup
from sqlalchemy import func, select

from procbot.ai.persona import GUARDRAILS
from procbot.ai.testing import FakeError, make_fake_gateway
from procbot.db.models import ConversationMessage, PendingApproval, Task, WorkSession
from procbot.domain.actions import TaskDraft
from procbot.domain.enums import Actor, AIErrorKind, SessionMode, SessionStatus, TimeSource
from procbot.services import sessions as svc
from procbot.services import tasks as ts
from procbot.testing import callback_update, message_update
from procbot.wiring import register_all

ROUTE_TASKS = {"intent": "new_tasks", "confidence": 0.95}


@pytest.fixture
def wired(ctx):
    return register_all(ctx)


@pytest.fixture
def bot(bot_and_session):
    return bot_and_session[0]


@pytest.fixture
def tg(bot_and_session):
    return bot_and_session[1]


async def feed(dp, bot, upd) -> None:
    await dp.feed_update(bot, upd)


def last_send(tg) -> SendMessage:
    return tg.sent_messages()[-1]


def last_markup(tg) -> InlineKeyboardMarkup:
    for r in reversed(tg.requests):
        if isinstance(r, SendMessage | EditMessageText) and isinstance(
            r.reply_markup, InlineKeyboardMarkup
        ):
            return r.reply_markup
    raise AssertionError("no inline keyboard sent")


def button(markup: InlineKeyboardMarkup, prefix: str) -> str:
    for row in markup.inline_keyboard:
        for b in row:
            if b.text.startswith(prefix):
                assert b.callback_data is not None
                return b.callback_data
    raise AssertionError(
        f"no {prefix!r} in {[[b.text for b in r] for r in markup.inline_keyboard]}"
    )


async def count(ctx, model, *where) -> int:
    async with ctx.db.transaction() as s:
        stmt = select(func.count()).select_from(model)
        for w in where:
            stmt = stmt.where(w)
        return (await s.execute(stmt)).scalar_one()


async def snapshot_tasks(ctx) -> list[tuple]:
    async with ctx.db.transaction() as s:
        rows = (await s.execute(select(Task).order_by(Task.id))).scalars().all()
        return [
            (
                t.id,
                t.title,
                t.status,
                t.planned_date,
                t.planned_start_at,
                t.due_at,
                t.postpone_count,
            )
            for t in rows
        ]


async def add_task(ctx, title: str = "Отчёт") -> int:
    async with ctx.db.transaction() as s:
        t = await ts.create_task(s, ctx, TaskDraft(title=title), actor=Actor.USER)
        return t.id


# --------------------------------------------------------------------------- acceptance 4


async def test_multi_intent_phrase_gives_candidates_only(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    gw = await make_fake_gateway(
        ctx,
        {
            "route_message": ROUTE_TASKS,
            "extract_tasks": {
                "candidates": [
                    {"title": "Доделать сайт", "when": "завтра"},
                    {
                        "title": "Позаниматься проектом",
                        "when": "на выходных",
                        "uncertainties": ["priority", "scope"],
                    },
                ]
            },
        },
    )
    upd = message_update("Завтра доделать сайт, на выходных позаниматься проектом")
    await feed(dp, bot, upd)
    await feed(dp, bot, upd)  # repeated delivery
    assert len(gw.fake.calls_for("route_message")) == 1
    assert len(gw.fake.calls_for("extract_tasks")) == 1
    assert await count(ctx, Task) == 0
    assert await count(ctx, PendingApproval) == 2
    card = last_send(tg).text or ""
    assert "Доделать сайт" in card and "Позаниматься проектом" in card
    assert "❓ Неясно: срок / дата, приоритет, объём" in card
    assert "Без твоего подтверждения ничего не изменится" in card
    kb = last_markup(tg)
    button(kb, "✅ Создать все")
    await feed(dp, bot, callback_update(button(kb, "✅ Создать 1")))
    titles = [t[1] for t in await snapshot_tasks(ctx)]
    assert titles == ["Доделать сайт"]
    # uncertainty notes survive the card re-render after a press
    edited = [r for r in tg.requests if isinstance(r, EditMessageText)][-1]
    assert "❓ Неясно" in (edited.text or "")


async def test_reflection_only_gets_a_question(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    await make_fake_gateway(
        ctx,
        {
            "route_message": {"intent": "new_tasks", "confidence": 0.6},
            "extract_tasks": {
                "candidates": [],
                "is_reflection_only": True,
                "clarifying_question": "Это планы на конкретный день или просто мысли?",
            },
        },
    )
    await feed(dp, bot, message_update("думаю, что пора бы заняться здоровьем"))
    assert last_send(tg).text == "Это планы на конкретный день или просто мысли?"
    assert await count(ctx, PendingApproval) == 0 and await count(ctx, Task) == 0


# --------------------------------------------------------------------------- acceptance 19


async def test_ai_unavailable_falls_back_to_stage_a(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    tid = await add_task(ctx)
    before = await snapshot_tasks(ctx)
    await make_fake_gateway(ctx, {"route_message": [FakeError(AIErrorKind.SERVER)] * 10})
    await feed(dp, bot, message_update("молоко\nхлеб"))
    card = last_send(tg).text or ""
    assert "ИИ сейчас недоступен" in card
    assert "молоко" in card and "хлеб" in card
    assert await count(ctx, PendingApproval, PendingApproval.status == "pending") == 2
    assert await snapshot_tasks(ctx) == before
    async with ctx.db.transaction() as s:
        assert (await s.get(Task, tid)).status == "todo"  # type: ignore[union-attr]


async def test_extract_unavailable_falls_back(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    await make_fake_gateway(
        ctx,
        {"route_message": ROUTE_TASKS, "extract_tasks": ["not json"] * 10},
    )
    await feed(dp, bot, message_update("помыть окна"))
    assert "ИИ сейчас недоступен" in (last_send(tg).text or "")
    assert await count(ctx, Task) == 0
    async with ctx.db.transaction() as s:
        row = (await s.execute(select(PendingApproval))).scalar_one()
    assert row.origin == "system" and row.action_payload["task"]["title"] == "помыть окна"


async def test_chat_unavailable_falls_back(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    await make_fake_gateway(
        ctx,
        {
            "route_message": {"intent": "question", "confidence": 0.9},
            "chat_reply": [FakeError(AIErrorKind.SERVER)] * 10,
        },
    )
    await feed(dp, bot, message_update("что делать?"))
    assert "ИИ сейчас недоступен" in (last_send(tg).text or "")
    assert await count(ctx, Task) == 0


async def test_without_ai_config_no_note(wired, profile, dp, bot, tg) -> None:
    await feed(dp, bot, message_update("кофе"))
    assert "ИИ" not in (last_send(tg).text or "")
    assert await count(wired, PendingApproval) == 1


# --------------------------------------------------------------------------- chat


async def test_chat_reply_text_only_with_persona_and_history(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    await add_task(ctx, "Подготовить отчёт")
    gw = await make_fake_gateway(
        ctx,
        {
            "route_message": {"intent": "question", "confidence": 0.9},
            "chat_reply": {"text": "Начни с отчёта — 10 минут на план."},
        },
    )
    await feed(dp, bot, message_update("С чего начать день?"))
    assert last_send(tg).text == "Начни с отчёта — 10 минут на план."
    assert last_send(tg).reply_markup is None
    call = gw.fake.calls_for("chat_reply")[0]
    assert GUARDRAILS[3] in call.system
    assert "Подготовить отчёт" in call.user and "category=tasks" in call.user
    async with ctx.db.transaction() as s:
        rows = (
            (await s.execute(select(ConversationMessage).order_by(ConversationMessage.id)))
            .scalars()
            .all()
        )
    assert [(r.direction, r.text) for r in rows] == [
        ("in", "С чего начать день?"),
        ("out", "Начни с отчёта — 10 минут на план."),
    ]
    assert await count(ctx, PendingApproval) == 0


async def test_settings_and_unclear(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    await make_fake_gateway(
        ctx,
        {
            "route_message": [
                {"intent": "settings_request", "confidence": 0.9},
                {
                    "intent": "unclear",
                    "confidence": 0.9,
                    "clarifying_question": "Ты про какой отчёт?",
                },
            ]
        },
    )
    await feed(dp, bot, message_update("сделай поменьше напоминаний"))
    assert "/settings" in (last_send(tg).text or "")
    await feed(dp, bot, message_update("ну отчёт этот"))
    assert last_send(tg).text == "Ты про какой отчёт?"
    assert await count(ctx, PendingApproval) == 0


# --------------------------------------------------------------------------- sessions


async def test_session_pause_needs_a_tap(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    async with ctx.db.transaction() as s:
        ws = await svc.start_session(s, ctx, task_id=None, mode=SessionMode.FREE)
        sid = ws.id
    await make_fake_gateway(
        ctx,
        {
            "route_message": {
                "intent": "session_control",
                "session_action": "pause",
                "confidence": 0.9,
            }
        },
    )
    await feed(dp, bot, message_update("мне надо отойти минут на десять"))
    async with ctx.db.transaction() as s:
        assert (await s.get(WorkSession, sid)).status == SessionStatus.ACTIVE  # type: ignore[union-attr]
    await feed(dp, bot, callback_update(button(last_markup(tg), "⏸ Пауза")))
    async with ctx.db.transaction() as s:
        assert (await s.get(WorkSession, sid)).status == SessionStatus.PAUSED  # type: ignore[union-attr]


async def test_session_stop_without_session(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    await make_fake_gateway(
        ctx,
        {
            "route_message": {
                "intent": "session_control",
                "session_action": "stop",
                "confidence": 0.9,
            }
        },
    )
    await feed(dp, bot, message_update("всё, я закончил на сегодня с этим"))
    assert "нет активной" in (last_send(tg).text or "")


async def test_session_start_for_task_offers_button(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    tid = await add_task(ctx, "Отчёт")
    await make_fake_gateway(
        ctx,
        {
            "route_message": {
                "intent": "session_control",
                "session_action": "start",
                "task_id": tid,
                "confidence": 0.9,
            }
        },
    )
    await feed(dp, bot, message_update("давай займусь отчётом"))
    assert "Отчёт" in (last_send(tg).text or "")
    assert await count(ctx, WorkSession) == 0
    await feed(dp, bot, callback_update(button(last_markup(tg), "▶️ Начать")))
    assert "Выбери режим" in (last_send(tg).text or "")


# --------------------------------------------------------------------------- retro (scenario F)


async def test_retro_report_prefills_retro_flow(wired, profile, dp, bot, tg, clock) -> None:
    ctx = wired
    clock.set(datetime(2026, 10, 26, 16, 30, tzinfo=UTC))  # 17:30 Berlin
    await make_fake_gateway(
        ctx,
        {
            "route_message": {
                "intent": "retro_report",
                "confidence": 0.9,
                "retro_start": "14:00",
                "retro_end": "16:00",
                "retro_approximate": True,
            }
        },
    )
    await feed(dp, bot, message_update("я работал примерно с 2 до 4"))
    text = last_send(tg).text or ""
    assert "Понял так: 26.10 14:00–16:00" in text and "Над чем работал?" in text
    await feed(dp, bot, callback_update(button(last_markup(tg), "Без задачи")))
    assert "перерывы" in (last_send(tg).text or "")
    await feed(dp, bot, callback_update(button(last_markup(tg), "Без перерывов")))
    assert "уверен" in (last_send(tg).text or "")
    await feed(dp, bot, callback_update(button(last_markup(tg), "Примерно")))
    async with ctx.db.transaction() as s:
        ws = (await s.execute(select(WorkSession))).scalar_one()
    assert ws.time_source == TimeSource.APPROXIMATE and ws.confidence == "medium"
    assert ws.started_at == datetime(2026, 10, 26, 13, 0, tzinfo=UTC)
    assert ws.ended_at == datetime(2026, 10, 26, 15, 0, tzinfo=UTC)


async def test_retro_report_with_task_and_duration(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    tid = await add_task(ctx, "Отчёт")
    await make_fake_gateway(
        ctx,
        {
            "route_message": {
                "intent": "retro_report",
                "confidence": 0.9,
                "task_id": tid,
                "retro_duration_min": 90,
            }
        },
    )
    await feed(dp, bot, message_update("посидел над отчётом полтора часика"))
    assert "Во сколько примерно закончил" in (last_send(tg).text or "")
    await feed(dp, bot, callback_update(button(last_markup(tg), "Только что")))
    await feed(dp, bot, callback_update(button(last_markup(tg), "Без перерывов")))
    await feed(dp, bot, callback_update(button(last_markup(tg), "Не уверен")))
    async with ctx.db.transaction() as s:
        ws = (await s.execute(select(WorkSession))).scalar_one()
    assert ws.task_id == tid and ws.time_source == TimeSource.APPROXIMATE
    assert (ws.ended_at - ws.started_at).total_seconds() == 90 * 60  # type: ignore[operator]


async def test_plain_retro_command_still_asks_interval(wired, profile, dp, bot, tg) -> None:
    await feed(dp, bot, message_update("/retro"))
    await feed(dp, bot, callback_update(button(last_markup(tg), "Без задачи")))
    assert "Когда работал?" in (last_send(tg).text or "")


# --------------------------------------------------------------------------- postpone / obstacle


async def test_postpone_offers_task_buttons_without_changes(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    tid = await add_task(ctx, "Отчёт")
    before = await snapshot_tasks(ctx)
    await make_fake_gateway(
        ctx, {"route_message": {"intent": "postpone", "task_id": tid, "confidence": 0.9}}
    )
    await feed(dp, bot, message_update("отчёт сегодня точно не успею"))
    assert "Отчёт" in (last_send(tg).text or "")
    button(last_markup(tg), "📆 Перенести")
    assert await snapshot_tasks(ctx) == before
