"""Initiative seeders (open question 2, acceptance 23), stage-A renderers and the
reminder buttons (FR-04, acceptance 9/10)."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, time, timedelta

from sqlalchemy import select

from procbot.context import AppContext
from procbot.db.models import (
    Intervention,
    InterventionOutcome,
    ReminderSchedule,
    Task,
    TaskEvent,
    WorkSession,
)
from procbot.domain.enums import (
    InterventionResult,
    MessageCategory,
    ReminderState,
    SessionStatus,
    TaskStatus,
)
from procbot.services import initiative, reminders
from procbot.services.initiative import seed_initiative_reminders
from procbot.services.interventions_basic import (
    find_recent_intervention,
    record_intervention_outcome,
)
from procbot.services.profile import update_profile
from procbot.services.reminders import schedule_reminder
from procbot.services.scheduler import Scheduler
from procbot.telegram.callbacks import ReminderCb, SessionCb, TaskCb
from procbot.testing import callback_update

BERLIN = "Europe/Berlin"


def _wire(ctx: AppContext) -> None:
    reminders.register(ctx)
    initiative.register(ctx)


async def _set(ctx: AppContext, **fields) -> None:
    async with ctx.db.transaction() as s:
        await update_profile(s, now=ctx.clock.now(), **fields)


async def _pending(ctx: AppContext, category: MessageCategory) -> list[ReminderSchedule]:
    async with ctx.db.transaction() as s:
        return list(
            (
                await s.execute(
                    select(ReminderSchedule)
                    .where(
                        ReminderSchedule.category == category,
                        ReminderSchedule.state == ReminderState.PENDING,
                    )
                    .order_by(ReminderSchedule.due_at)
                )
            ).scalars()
        )


async def test_nothing_is_seeded_when_times_are_not_set(ctx, profile):
    assert await seed_initiative_reminders(ctx) == 0
    async with ctx.db.transaction() as s:
        assert (await s.execute(select(ReminderSchedule))).first() is None


async def test_morning_time_stays_local_across_autumn_dst(ctx, clock, messenger, profile):
    """Acceptance 23: 09:00 local before and after 2026-10-25 (CEST -> CET)."""
    _wire(ctx)
    await _set(ctx, morning_time=time(9, 0))
    clock.set(datetime(2026, 10, 24, 6, 0, tzinfo=UTC))  # 08:00 CEST
    await seed_initiative_reminders(ctx)
    [r] = await _pending(ctx, MessageCategory.MORNING_PLAN)
    assert r.due_at == datetime(2026, 10, 24, 7, 0, tzinfo=UTC)  # 09:00 CEST
    assert r.dedup_key == "morning:2026-10-24"
    sched = Scheduler(ctx)
    sent_at: list[datetime] = []
    t = clock.now()
    while t < datetime(2026, 10, 27, 12, 0, tzinfo=UTC):
        clock.set(t)
        before = len(messenger.sent)
        await sched.tick()
        if len(messenger.sent) > before:
            sent_at.append(t)
        t += timedelta(minutes=30)
    assert sent_at == [
        datetime(2026, 10, 24, 7, 0, tzinfo=UTC),  # CEST
        datetime(2026, 10, 25, 8, 0, tzinfo=UTC),  # CET from the DST night on
        datetime(2026, 10, 26, 8, 0, tzinfo=UTC),
        datetime(2026, 10, 27, 8, 0, tzinfo=UTC),
    ]


async def test_seeding_across_spring_dst_and_idempotency(ctx, clock, profile):
    await _set(ctx, morning_time=time(9, 0), evening_time=time(21, 30))
    clock.set(datetime(2026, 3, 28, 12, 0, tzinfo=UTC))
    await seed_initiative_reminders(ctx)
    await seed_initiative_reminders(ctx)
    [m] = await _pending(ctx, MessageCategory.MORNING_PLAN)
    [e] = await _pending(ctx, MessageCategory.EVENING_REVIEW)
    assert m.due_at == datetime(2026, 3, 29, 7, 0, tzinfo=UTC)  # 09:00 CEST
    assert e.due_at == datetime(2026, 3, 28, 20, 30, tzinfo=UTC)  # 21:30 CET
    assert m.dedup_key == "morning:2026-03-29"


async def test_changing_or_unsetting_time_replaces_planned_occurrence(ctx, clock, profile):
    clock.set(datetime(2026, 10, 26, 7, 30, tzinfo=UTC))  # 08:30 local
    await _set(ctx, morning_time=time(10, 0))
    await seed_initiative_reminders(ctx)
    [r] = await _pending(ctx, MessageCategory.MORNING_PLAN)
    assert r.dedup_key == "morning:2026-10-26"
    await _set(ctx, morning_time=time(8, 0))  # already passed today -> tomorrow
    await seed_initiative_reminders(ctx)
    [r] = await _pending(ctx, MessageCategory.MORNING_PLAN)
    assert r.dedup_key == "morning:2026-10-27"
    assert r.due_at == datetime(2026, 10, 27, 7, 0, tzinfo=UTC)
    await _set(ctx, morning_time=None)
    await seed_initiative_reminders(ctx)
    assert await _pending(ctx, MessageCategory.MORNING_PLAN) == []


async def test_weekly_needs_day_and_time(ctx, clock, profile):
    await _set(ctx, weekly_review_time=time(18, 0))
    await seed_initiative_reminders(ctx)
    assert await _pending(ctx, MessageCategory.WEEKLY_REVIEW) == []
    await _set(ctx, weekly_review_dow=6)  # Sunday
    await seed_initiative_reminders(ctx)
    [w] = await _pending(ctx, MessageCategory.WEEKLY_REVIEW)
    assert w.due_at == datetime(2026, 11, 1, 17, 0, tzinfo=UTC)  # Sun 18:00 CET


async def _render(ctx: AppContext, category: MessageCategory, day: date):
    _wire(ctx)
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        r = await schedule_reminder(
            s,
            now=now,
            category=category,
            due_at=now,
            dedup_key=f"{category}:{day}",
            payload={"local_date": day.isoformat()},
        )
    renderer = ctx.registry["renderers"][category]
    return await renderer(ctx, r)


async def test_morning_lists_todays_tasks(ctx, clock, profile):
    now = clock.now()  # 2026-10-26 10:00 local
    async with ctx.db.transaction() as s:
        s.add(
            Task(
                title="Созвон",
                planned_start_at=datetime(2026, 10, 26, 13, 0, tzinfo=UTC),
                created_at=now,
                updated_at=now,
            )
        )
        s.add(
            Task(
                title="Купить хлеб", planned_date=date(2026, 10, 26), created_at=now, updated_at=now
            )
        )
        s.add(
            Task(
                title="Завтрашнее", planned_date=date(2026, 10, 27), created_at=now, updated_at=now
            )
        )
    out = await _render(ctx, MessageCategory.MORNING_PLAN, date(2026, 10, 26))
    assert out is not None
    assert "14:00 — Созвон" in out.text and "Купить хлеб" in out.text
    assert "Завтрашнее" not in out.text


async def test_morning_with_empty_day(ctx, profile):
    out = await _render(ctx, MessageCategory.MORNING_PLAN, date(2026, 10, 26))
    assert out is not None and "ничего не запланировано" in out.text


async def test_evening_summary_done_started_not_started(ctx, clock, profile):
    now = clock.now()
    async with ctx.db.transaction() as s:
        done = Task(
            title="Готовое",
            status=TaskStatus.DONE,
            planned_date=date(2026, 10, 26),
            completed_at=now,
            created_at=now,
            updated_at=now,
        )
        started = Task(
            title="Начатое",
            status=TaskStatus.IN_PROGRESS,
            planned_date=date(2026, 10, 26),
            created_at=now,
            updated_at=now,
        )
        idle = Task(
            title="Несделанное", planned_date=date(2026, 10, 26), created_at=now, updated_at=now
        )
        s.add_all([done, started, idle])
        await s.flush()
        s.add(
            WorkSession(
                task_id=started.id,
                mode="free",
                status=SessionStatus.FINISHED,
                started_at=now,
                ended_at=now + timedelta(minutes=20),
                created_at=now,
                updated_at=now,
            )
        )
    clock.advance(hours=10)
    out = await _render(ctx, MessageCategory.EVENING_REVIEW, date(2026, 10, 26))
    assert out is not None
    assert "✅ Сделано: 1" in out.text and "Готовое" in out.text
    assert "Начато, но не закончено: 1" in out.text and "Начатое" in out.text
    assert "⏳ Не начато: 1" in out.text and "Несделанное" in out.text
    datas = [b.callback_data for row in out.reply_markup.inline_keyboard for b in row]  # type: ignore[union-attr]
    assert any(d.startswith("rm:evening") for d in datas)


async def test_weekly_is_minimal_and_says_full_analytics_later(ctx, profile):
    out = await _render(ctx, MessageCategory.WEEKLY_REVIEW, date(2026, 11, 1))
    assert out is not None
    assert "Завершено задач: 0" in out.text
    assert "появится" in out.text


async def test_pre_task_heads_up_has_start_and_later(ctx, clock, profile):
    _wire(ctx)
    now = clock.now()
    planned = now + timedelta(minutes=10)
    async with ctx.db.transaction() as s:
        t = Task(title="Тренировка", planned_start_at=planned, created_at=now, updated_at=now)
        s.add(t)
        await s.flush()
        r = await schedule_reminder(
            s,
            now=now,
            category=MessageCategory.PRE_TASK,
            due_at=now,
            dedup_key="pre",
            task_id=t.id,
            payload={"task_id": t.id, "planned_start_at": planned.isoformat()},
        )
    out = await ctx.registry["renderers"][MessageCategory.PRE_TASK](ctx, r)
    assert out is not None and "Через 10 мин" in out.text and "Тренировка" in out.text
    datas = [b.callback_data for row in out.reply_markup.inline_keyboard for b in row]
    assert SessionCb(action="pick", task_id=t.id).pack() in datas
    assert ReminderCb(action="later", reminder_id=r.id, task_id=t.id).pack() in datas


async def _start_delay_card(ctx: AppContext) -> tuple[int, int]:
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        t = Task(title="Отчёт", planned_start_at=now, created_at=now, updated_at=now)
        s.add(t)
        await s.flush()
        r = await schedule_reminder(
            s,
            now=now,
            category=MessageCategory.START_DELAY,
            due_at=now,
            dedup_key="d",
            task_id=t.id,
        )
        return t.id, r.id


async def test_reason_button_records_quick_intervention_and_offers_step(
    ctx, profile, dp, bot_and_session
):
    bot, session = bot_and_session
    tid, rid = await _start_delay_card(ctx)
    data = ReminderCb(action="reason", reminder_id=rid, task_id=tid, arg="unclear").pack()
    await dp.feed_update(bot, callback_update(data))
    await dp.feed_update(bot, callback_update(data))  # repeated press: no duplicate
    async with ctx.db.transaction() as s:
        ivs = (await s.execute(select(Intervention))).scalars().all()
        events = (
            (await s.execute(select(TaskEvent).where(TaskEvent.event_type == "obstacle_reported")))
            .scalars()
            .all()
        )
    assert len(ivs) == 1
    iv = ivs[0]
    assert (iv.depth, iv.reason_category, iv.reason_origin, iv.used_ai) == (
        "quick",
        "unclear",
        "user_said",
        False,
    )
    assert len(events) == 1
    replies = session.sent_messages()
    assert replies and "первый физический шаг" in (replies[0].text or "")
    kb = replies[0].reply_markup
    datas = [b.callback_data for row in kb.inline_keyboard for b in row]  # type: ignore[union-attr]
    assert SessionCb(action="pick", task_id=tid, arg="short_start").pack() in datas


async def test_reason_replies_are_hypotheses_not_accusations(ctx, profile, dp, bot_and_session):
    bot, session = bot_and_session
    tid, rid = await _start_delay_card(ctx)
    for reason in ("too_big", "low_energy", "unpleasant", "distractions", "priority_conflict"):
        data = ReminderCb(action="reason", reminder_id=rid, task_id=tid, arg=reason).pack()
        await dp.feed_update(bot, callback_update(data))
    texts = " ".join(m.text or "" for m in session.sent_messages()).lower()
    for banned in (r"\bлень\b", r"\bленив", r"\bленишься", "оправдани", "симулир", "диагноз"):
        assert re.search(banned, texts) is None, banned
    msgs = session.sent_messages()
    too_big = [b.callback_data for row in msgs[0].reply_markup.inline_keyboard for b in row]  # type: ignore[union-attr]
    assert TaskCb(action="subtasks", task_id=tid).pack() in too_big
    low = [b.callback_data for row in msgs[1].reply_markup.inline_keyboard for b in row]  # type: ignore[union-attr]
    assert any(d.startswith("rm:rest") for d in low)  # rest is a valid outcome
    assert any(d.startswith("tk:postpone") for d in low)


async def test_later_and_snooze_buttons(ctx, clock, profile, dp, bot_and_session):
    bot, session = bot_and_session
    tid, rid = await _start_delay_card(ctx)
    await dp.feed_update(
        bot, callback_update(ReminderCb(action="later", reminder_id=rid, task_id=tid).pack())
    )
    snooze = ReminderCb(action="snooze", reminder_id=rid, task_id=tid, arg="30").pack()
    await dp.feed_update(bot, callback_update(snooze))
    await dp.feed_update(bot, callback_update(snooze))
    texts = session.sent_texts()
    assert "Когда напомнить" in texts[0]
    assert "напомню около 10:30" in texts[1]
    assert "Уже отложено" in texts[2]
    async with ctx.db.transaction() as s:
        snoozed = (
            (
                await s.execute(
                    select(ReminderSchedule).where(ReminderSchedule.dedup_key.like("snz:%"))
                )
            )
            .scalars()
            .all()
        )
    assert len(snoozed) == 1 and snoozed[0].due_at == clock.now() + timedelta(minutes=30)


async def test_rest_button_is_recorded_without_pressure(ctx, profile, dp, bot_and_session):
    bot, session = bot_and_session
    tid, rid = await _start_delay_card(ctx)
    await dp.feed_update(
        bot, callback_update(ReminderCb(action="rest", reminder_id=rid, task_id=tid).pack())
    )
    async with ctx.db.transaction() as s:
        iv = (await s.execute(select(Intervention))).scalar_one()
    assert iv.kind == "rest"
    assert "нормальный выбор" in session.sent_texts()[0]


async def test_record_intervention_outcome_links_session(ctx, clock, profile):
    tid, rid = await _start_delay_card(ctx)
    now = clock.now()
    async with ctx.db.transaction() as s:
        iv = Intervention(
            task_id=tid, reminder_id=rid, kind="short_start", depth="quick", created_at=now
        )
        s.add(iv)
        ws = WorkSession(
            task_id=tid,
            mode="short_start",
            status=SessionStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
        s.add(ws)
        await s.flush()
        found = await find_recent_intervention(s, task_id=tid, since=now - timedelta(hours=1))
        assert found is not None and found.id == iv.id
        o1 = await record_intervention_outcome(
            s, intervention_id=iv.id, result=InterventionResult.STARTED, session_id=ws.id, now=now
        )
        o2 = await record_intervention_outcome(
            s, intervention_id=iv.id, result="started", session_id=ws.id, now=now
        )
        assert o1.id == o2.id
        assert (
            await find_recent_intervention(s, task_id=tid, since=now - timedelta(hours=1)) is None
        )
        n = len((await s.execute(select(InterventionOutcome))).scalars().all())
    assert n == 1
