"""Reports (FR-11): weekly review with/without AI, evening review, renderers,
scenario H (dispute -> memory corrected), handlers through the dispatcher."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from aiogram.methods import EditMessageText, SendMessage
from aiogram.types import InlineKeyboardMarkup
from sqlalchemy import select

from procbot.ai.context import ContextBuilder
from procbot.ai.testing import make_fake_gateway
from procbot.db.models import (
    Intervention,
    MemoryItem,
    MemorySource,
    PendingApproval,
    ReminderSchedule,
    Task,
    TaskEvent,
    WorkSession,
)
from procbot.domain.enums import MessageCategory
from procbot.services import analytics, approvals, memory, reports
from procbot.services import tasks as ts
from procbot.services.outbound import get_renderer
from procbot.telegram.keyboards import Menu
from procbot.testing import callback_update, message_update

TZ = "Europe/Berlin"
HYP = "Возможно, задачи без первого шага откладываются чаще"


@pytest.fixture
def wired(ctx, profile):
    ts.register(ctx)
    approvals.register(ctx)
    reports.register(ctx)
    return ctx


@pytest.fixture
def bot(bot_and_session):
    return bot_and_session[0]


@pytest.fixture
def tg(bot_and_session):
    return bot_and_session[1]


def markups(tg) -> list[InlineKeyboardMarkup]:
    return [
        r.reply_markup
        for r in tg.requests
        if isinstance(r, SendMessage | EditMessageText)
        and isinstance(r.reply_markup, InlineKeyboardMarkup)
    ]


def button(markup: InlineKeyboardMarkup, prefix: str) -> str:
    for row in markup.inline_keyboard:
        for b in row:
            if b.text.startswith(prefix):
                assert b.callback_data is not None
                return b.callback_data
    raise AssertionError(
        f"no {prefix!r} in {[[b.text for b in r] for r in markup.inline_keyboard]}"
    )


def find_button(tg, prefix: str) -> str:
    for mk in reversed(markups(tg)):
        try:
            return button(mk, prefix)
        except AssertionError:
            continue
    raise AssertionError(f"no button {prefix!r} sent")


async def add_task(ctx, title: str, **kw) -> int:
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        t = Task(title=title, created_at=now, updated_at=now, **kw)
        s.add(t)
        await s.flush()
        return t.id


def weekly_answer(summary="Неделя получилась ровной: задач завершено 1.", **kw):
    return {
        "summary": summary,
        "hypotheses": kw.get(
            "hypotheses",
            [{"content": HYP, "verification_hint": "Начинать каждую задачу с шага на 5 минут"}],
        ),
        "proposals": kw.get("proposals", []),
    }


async def mem_items(ctx) -> list[MemoryItem]:
    async with ctx.db.transaction() as s:
        return list((await s.execute(select(MemoryItem).order_by(MemoryItem.id))).scalars())


# --------------------------------------------------------------------------- weekly


async def test_weekly_with_ai_saves_hypotheses_and_proposals_as_approvals(wired):
    ctx = wired
    done = await add_task(
        ctx, "Готово", status="done", completed_at=ctx.clock.now() - timedelta(days=1)
    )
    open_id = await add_task(ctx, "Большой отчёт")
    gw = await make_fake_gateway(
        ctx,
        {
            "weekly_analysis": [
                weekly_answer(
                    proposals=[
                        {"text": "Начинать утро с самой неясной задачи", "kind": "advice"},
                        {
                            "text": "Перенести отчёт на среду",
                            "kind": "reschedule_task",
                            "task_id": open_id,
                            "planned_date": "2026-10-28",
                        },
                    ]
                )
            ]
        },
    )
    rep = await reports.build_weekly(
        ctx, first_day=date(2026, 10, 20), last_day=date(2026, 10, 26), op_prefix="t1"
    )
    assert done
    call = gw.fake.calls_for("weekly_analysis")[0]
    assert "weekly_numbers" in call.user and "Большой отчёт" in call.user
    # six directions + AI comment + hypothesis with a way to check it
    for title in analytics.DIRECTION_TITLES.values():
        assert title in rep.text
    assert "Комментарий ИИ" in rep.text and "задач завершено 1" in rep.text
    assert HYP in rep.text and "Как проверить: Начинать каждую задачу" in rep.text
    assert "Начинать утро с самой неясной задачи" in rep.text
    [item] = await mem_items(ctx)
    assert (item.kind, item.origin, item.content) == ("hypothesis", "ai_inferred", HYP)
    assert rep.new_hypothesis_ids == [item.id]
    # the reschedule became a pending approval, the task is unchanged
    assert len(rep.approval_ids) == 1
    async with ctx.db.transaction() as s:
        ap = await s.get(PendingApproval, rep.approval_ids[0])
        t = await s.get(Task, open_id)
    assert ap is not None and ap.status == "pending" and ap.origin == "ai"
    assert t is not None and t.planned_date is None
    assert button(rep.markup, "📝 Предложения (1)")
    assert button(rep.markup, "🙅 Не согласен с Г1")
    # repeat with the same op prefix: nothing duplicated
    gw.fake.push("weekly_analysis", weekly_answer())
    rep2 = await reports.build_weekly(
        ctx, first_day=date(2026, 10, 20), last_day=date(2026, 10, 26), op_prefix="t1"
    )
    assert len(await mem_items(ctx)) == 1 and rep2.new_hypothesis_ids == [item.id]


async def test_weekly_without_ai_is_template_only(wired):
    ctx = wired
    rep = await reports.build_weekly(
        ctx, first_day=date(2026, 10, 20), last_day=date(2026, 10, 26), op_prefix="t2"
    )
    assert rep.ai.status == "unavailable"
    assert "ИИ сейчас недоступен" in rep.text
    assert rep.text.count(analytics.INSUFFICIENT) >= 5
    assert analytics.NO_CAUSALITY in rep.text
    assert await mem_items(ctx) == []
    ctx.config.ai_features.weekly_analysis = False
    rep = await reports.build_weekly(
        ctx, first_day=date(2026, 10, 20), last_day=date(2026, 10, 26), op_prefix="t3"
    )
    assert rep.ai.status == "disabled" and "выключен" in rep.text


async def test_model_numbers_not_in_data_are_rejected(wired):
    ctx = wired
    gw = await make_fake_gateway(
        ctx,
        {
            "weekly_analysis": [
                weekly_answer(summary="Ты выполнил 87% плана и сэкономил 340 минут."),
                weekly_answer(summary="Данных пока мало, выводы делать рано.", hypotheses=[]),
            ]
        },
    )
    rep = await reports.build_weekly(
        ctx, first_day=date(2026, 10, 20), last_day=date(2026, 10, 26), op_prefix="t4"
    )
    calls = gw.fake.calls_for("weekly_analysis")
    assert len(calls) == 2 and calls[1].repair
    assert "87" not in rep.text and "Данных пока мало" in rep.text


async def test_weekly_renderer_replaces_stage_a(ctx, profile):
    from procbot.wiring import register_all

    register_all(ctx)
    renderer = get_renderer(ctx, MessageCategory.WEEKLY_REVIEW)
    assert renderer is reports.render_weekly_review
    assert get_renderer(ctx, MessageCategory.EVENING_REVIEW) is reports.render_evening_review
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        r = ReminderSchedule(
            category="weekly_review",
            due_at=now,
            dedup_key="weekly:2026-10-26",
            payload={"local_date": "2026-10-26"},
            created_at=now,
            updated_at=now,
        )
        s.add(r)
        await s.flush()
    out = await renderer(ctx, r)
    assert out is not None and "Итоги недели 20.10–26.10" in out.text
    assert "6. Гипотезы о закономерностях" in out.text


# --------------------------------------------------------------------------- evening


async def test_evening_review_asks_only_on_deviations(wired, dp, bot, tg, clock):
    ctx = wired
    clock.set(datetime(2026, 10, 26, 19, 0, tzinfo=UTC))  # 20:00 Berlin
    await add_task(
        ctx, "Сделано", status="done", completed_at=datetime(2026, 10, 26, 10, tzinfo=UTC)
    )
    text, kb = await reports.build_evening(ctx)
    assert "✅ Сделано: 1" in text and "Заметных отклонений" in text
    assert "🤔" not in str(kb.inline_keyboard)

    tid = await add_task(
        ctx, "Позвонить в банк", planned_start_at=datetime(2026, 10, 26, 9, tzinfo=UTC)
    )
    async with ctx.db.transaction() as s:
        s.add(
            WorkSession(
                task_id=None,
                mode="free",
                status="finished",
                started_at=datetime(2026, 10, 26, 13, tzinfo=UTC),
                ended_at=datetime(2026, 10, 26, 13, 40, tzinfo=UTC),
                time_source="retro",
                confidence="low",
                created_at=clock.now(),
                updated_at=clock.now(),
            )
        )
    text, kb = await reports.build_evening(ctx)
    assert "⏳ Не начато из запланированного: 1" in text
    assert "40 мин задним числом (ретроспективно, приблизительно)" in text
    assert "без подтверждённого начала: 1" in text
    assert "Что помешало?" in text

    # press "why" -> reason buttons -> recorded once
    await dp.feed_update(bot, callback_update(button(kb, "🤔")))
    data = find_button(tg, "🤔 Не понимал")
    await dp.feed_update(bot, callback_update(data))
    await dp.feed_update(bot, callback_update(data))
    async with ctx.db.transaction() as s:
        evs = list(
            (
                await s.execute(
                    select(TaskEvent).where(
                        TaskEvent.task_id == tid, TaskEvent.event_type == "obstacle_reported"
                    )
                )
            ).scalars()
        )
    assert len(evs) == 1 and evs[0].payload["category"] == "unclear" and evs[0].actor == "user"


async def test_report_menu_and_today_week_through_dispatcher(wired, dp, bot, tg):
    await dp.feed_update(bot, message_update(Menu.REPORT))
    assert "📊 Отчёты" in tg.sent_texts()[-1]
    mk = markups(tg)[-1]
    assert button(mk, "💰 Расходы ИИ").startswith("ak:spend")
    await dp.feed_update(bot, callback_update(button(mk, "📅 Сегодня")))
    assert "🌙 Итоги дня 26.10" in tg.sent_texts()[-1]
    await dp.feed_update(bot, callback_update(button(mk, "🗓 Неделя")))
    assert "Итоги недели 20.10–26.10" in tg.sent_texts()[-1]
    await dp.feed_update(bot, callback_update(button(mk, "📆 Свой период")))
    await dp.feed_update(bot, message_update("01.10-15.10"))
    last = tg.sent_texts()[-1]
    assert "Отчёт за 01.10.2026–15.10.2026" in last and "1. Выполнение задач" in last


# --------------------------------------------------------------------------- scenario H


async def test_scenario_h_dispute_refutes_and_corrects_memory(wired, dp, bot, tg):
    ctx = wired
    memory.register(ctx)
    await make_fake_gateway(ctx, {"weekly_analysis": [weekly_answer()]})
    await dp.feed_update(bot, message_update("/week"))
    week_text = tg.sent_texts()[-1]
    assert HYP in week_text
    [hyp] = await mem_items(ctx)

    blocks = await ContextBuilder(ctx, task="chat_reply").build(
        message="x", history=0, tasks=0, memory=True
    )
    assert HYP in "\n".join(b.text() for b in blocks)

    # "Не согласен" on the hypothesis -> refuted, leaves the model context
    await dp.feed_update(bot, callback_update(find_button(tg, "🙅 Не согласен с Г1")))
    assert "опровергнуто" in tg.sent_texts()[-1]
    async with ctx.db.transaction() as s:
        refuted = await s.get(MemoryItem, hyp.id)
    assert refuted is not None and (refuted.status, refuted.origin) == ("refuted", "user_refuted")
    blocks = await ContextBuilder(ctx, task="chat_reply").build(
        message="x", history=0, tasks=0, memory=True
    )
    assert HYP not in "\n".join(b.text() for b in blocks)

    # "как на самом деле" -> the owner's own statement, linked to the refuted one
    await dp.feed_update(bot, callback_update(find_button(tg, "✏️ Написать, как на самом деле")))
    await dp.feed_update(bot, message_update("Откладываю, когда задача неясна, а не без шага"))
    assert "вы сказали" in tg.sent_texts()[-1]
    items = await mem_items(ctx)
    new = items[-1]
    assert (new.kind, new.origin, new.status) == ("pattern", "user_said", "active")
    async with ctx.db.transaction() as s:
        srcs = {(x.source_type, x.source_id) for x in await memory.item_sources(s, new.id)}
    assert ("memory_item", hyp.id) in srcs and any(t == "update" for t, _ in srcs)
    blocks = await ContextBuilder(ctx, task="chat_reply").build(
        message="x", history=0, tasks=0, memory=True
    )
    assert "Откладываю, когда задача неясна" in "\n".join(b.text() for b in blocks)

    # general dispute of a conclusion -> stored as the owner's statement
    await dp.feed_update(bot, callback_update(find_button(tg, "💬 Не согласен с выводом")))
    await dp.feed_update(bot, message_update("На этой неделе я болел, задержки не от лени"))
    assert "записал как твоё утверждение" in tg.sent_texts()[-1]
    last = (await mem_items(ctx))[-1]
    assert last.origin == "user_said" and "болел" in last.content


async def test_weekly_proposals_card_through_dispatcher(wired, dp, bot, tg):
    ctx = wired
    open_id = await add_task(ctx, "Разобрать почту")
    await make_fake_gateway(
        ctx,
        {
            "weekly_analysis": [
                weekly_answer(
                    hypotheses=[],
                    proposals=[
                        {
                            "text": "Отложить почту на потом",
                            "kind": "defer_task",
                            "task_id": open_id,
                        }
                    ],
                )
            ]
        },
    )
    await dp.feed_update(bot, message_update("/week"))
    await dp.feed_update(bot, callback_update(find_button(tg, "📝 Предложения (1)")))
    assert "Предложения на следующую неделю" in tg.sent_texts()[-1]
    await dp.feed_update(bot, callback_update(find_button(tg, "✅ Принять")))
    async with ctx.db.transaction() as s:
        t = await s.get(Task, open_id)
    assert t is not None and t.status == "someday"


async def test_measured_pattern_is_announced_in_report(wired):
    ctx = wired
    tid = await add_task(ctx, "Отчёт")
    async with ctx.db.transaction() as s:
        for d in (21, 22, 23):
            s.add(
                Intervention(
                    task_id=tid,
                    kind="tiny_step",
                    reason_category="too_big",
                    reason_origin="user_said",
                    created_at=datetime(2026, 10, d, 9, tzinfo=UTC),
                )
            )
    rep = await reports.build_weekly(
        ctx, first_day=date(2026, 10, 20), last_day=date(2026, 10, 26), op_prefix="t5"
    )
    assert rep.measured == ["Часто отмечаемая причина задержек: «слишком большая задача»"]
    assert "Добавлено в память как измеренное" in rep.text
    async with ctx.db.transaction() as s:
        n = len((await s.execute(select(MemorySource))).scalars().all())
    assert n == 3
