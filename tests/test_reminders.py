"""Task reminders (pre_task / start_delay), their timing sources (open question 3),
snooze, and acceptance 13: silence alone never triggers messages."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta

from sqlalchemy import func, select

from procbot.config import AppConfig, ReminderConfig
from procbot.context import AppContext
from procbot.db.models import OutboundMessage, ReminderSchedule, Task
from procbot.domain.enums import MessageCategory, ReminderState, TaskStatus
from procbot.services import initiative, reminders
from procbot.services.profile import update_profile
from procbot.services.reminders import snooze_reminder, sync_task_reminders
from procbot.services.scheduler import Scheduler

NOW = datetime(2026, 10, 26, 9, 0, tzinfo=UTC)


async def _set(ctx: AppContext, **fields) -> None:
    async with ctx.db.transaction() as s:
        await update_profile(s, now=ctx.clock.now(), **fields)


async def _task(ctx: AppContext, **fields) -> int:
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        t = Task(title=fields.pop("title", "Задача"), created_at=now, updated_at=now, **fields)
        s.add(t)
        await s.flush()
        await sync_task_reminders(s, ctx, t)
        return t.id


async def _change(ctx: AppContext, task_id: int, **fields) -> None:
    async with ctx.db.transaction() as s:
        t = await s.get(Task, task_id)
        assert t is not None
        for k, v in fields.items():
            setattr(t, k, v)
        await sync_task_reminders(s, ctx, t)


async def _rows(ctx: AppContext, task_id: int | None = None) -> list[ReminderSchedule]:
    async with ctx.db.transaction() as s:
        stmt = select(ReminderSchedule).order_by(ReminderSchedule.id)
        if task_id is not None:
            stmt = stmt.where(ReminderSchedule.task_id == task_id)
        return list((await s.execute(stmt)).scalars())


def _live(rows: list[ReminderSchedule]) -> list[ReminderSchedule]:
    return [r for r in rows if r.state == ReminderState.PENDING]


async def test_no_timing_configured_means_no_task_reminders(ctx, profile):
    """Open question 3: no interval is invented; both types are off."""
    tid = await _task(ctx, planned_start_at=NOW + timedelta(hours=2))
    assert await _rows(ctx, tid) == []


async def test_profile_timing_schedules_pre_task_and_start_delay(ctx, profile):
    await _set(ctx, pre_task_lead_min=10, start_delay_min=15)
    start = NOW + timedelta(hours=2)
    tid = await _task(ctx, planned_start_at=start)
    rows = _live(await _rows(ctx, tid))
    by_cat = {r.category: r for r in rows}
    assert by_cat[MessageCategory.PRE_TASK].due_at == start - timedelta(minutes=10)
    assert by_cat[MessageCategory.PRE_TASK].dedup_key == f"pre:{tid}:{start.isoformat()}"
    delay = by_cat[MessageCategory.START_DELAY]
    assert delay.due_at == start + timedelta(minutes=15)
    assert delay.payload["planned_start_at"] == start.isoformat()
    assert delay.payload["task_id"] == tid


async def test_timing_precedence_task_then_profile_then_config(tmp_path, ctx, profile):
    ctx.config = AppConfig(
        reminders=ReminderConfig(pre_task_lead_minutes=5, start_delay_after_minutes=20)
    )
    start = NOW + timedelta(hours=3)
    t_cfg = await _task(ctx, planned_start_at=start)
    await _set(ctx, start_delay_min=7)
    t_prof = await _task(ctx, planned_start_at=start)
    t_task = await _task(ctx, planned_start_at=start, reminder_overrides={"pre_task_lead_min": 30})

    def due(rows, cat):
        return next(r.due_at for r in rows if r.category == cat)

    rows = await _rows(ctx, t_cfg)
    assert due(rows, MessageCategory.PRE_TASK) == start - timedelta(minutes=5)
    assert due(rows, MessageCategory.START_DELAY) == start + timedelta(minutes=20)
    rows = await _rows(ctx, t_prof)
    assert due(rows, MessageCategory.START_DELAY) == start + timedelta(minutes=7)
    rows = await _rows(ctx, t_task)
    assert due(rows, MessageCategory.PRE_TASK) == start - timedelta(minutes=30)
    assert due(rows, MessageCategory.START_DELAY) == start + timedelta(minutes=7)


async def test_replanning_replaces_reminders_cleanly(ctx, profile):
    await _set(ctx, pre_task_lead_min=10, start_delay_min=15)
    old = NOW + timedelta(hours=2)
    tid = await _task(ctx, planned_start_at=old)
    new = NOW + timedelta(hours=5)
    await _change(ctx, tid, planned_start_at=new)
    await _change(ctx, tid, planned_start_at=new)  # repeated sync is a no-op
    rows = await _rows(ctx, tid)
    assert len(rows) == 4
    live = _live(rows)
    assert {r.dedup_key for r in live} == {
        f"pre:{tid}:{new.isoformat()}",
        f"delay:{tid}:{new.isoformat()}",
    }
    cancelled = [r for r in rows if r.state == ReminderState.CANCELLED]
    assert {r.last_reason for r in cancelled} == {"replanned"}
    # back to the old time: the cancelled pair is revived, not duplicated
    await _change(ctx, tid, planned_start_at=old)
    rows = await _rows(ctx, tid)
    assert len(rows) == 4
    assert {r.dedup_key for r in _live(rows)} == {
        f"pre:{tid}:{old.isoformat()}",
        f"delay:{tid}:{old.isoformat()}",
    }


async def test_done_cancelled_someday_backlog_cancel_reminders(ctx, profile):
    await _set(ctx, pre_task_lead_min=10, start_delay_min=15)
    for change in (
        {"status": TaskStatus.DONE},
        {"status": TaskStatus.CANCELLED},
        {"status": TaskStatus.SOMEDAY},
        {"is_backlog": True},
        {"planned_start_at": None},
    ):
        tid = await _task(ctx, planned_start_at=NOW + timedelta(hours=1))
        assert len(_live(await _rows(ctx, tid))) == 2
        await _change(ctx, tid, **change)
        assert _live(await _rows(ctx, tid)) == []


async def test_past_moments_are_not_scheduled_and_deferred_rows_are_kept(ctx, clock, profile):
    await _set(ctx, pre_task_lead_min=10, start_delay_min=15)
    tid = await _task(ctx, planned_start_at=NOW + timedelta(minutes=5))
    rows = _live(await _rows(ctx, tid))
    assert [r.category for r in rows] == [MessageCategory.START_DELAY]  # pre moment passed
    # 30 min later the delay row (due 09:20, maybe deferred) survives an unrelated sync
    clock.advance(minutes=30)
    await _change(ctx, tid, title="Задача (переименована)")
    assert [r.category for r in _live(await _rows(ctx, tid))] == [MessageCategory.START_DELAY]


async def test_snooze_creates_one_new_reminder(ctx, clock, profile):
    await _set(ctx, start_delay_min=15)
    tid = await _task(ctx, planned_start_at=NOW + timedelta(hours=1))
    rid = _live(await _rows(ctx, tid))[0].id
    async with ctx.db.transaction() as s:
        row, created = await snooze_reminder(s, ctx, reminder_id=rid, minutes=30)
    assert created and row is not None
    assert row.due_at == NOW + timedelta(minutes=30)
    assert row.payload["snoozed_from"] == rid
    assert row.category == MessageCategory.START_DELAY
    async with ctx.db.transaction() as s:
        again, created2 = await snooze_reminder(s, ctx, reminder_id=rid, minutes=60)
    assert not created2 and again is not None and again.id == row.id


async def test_silence_for_days_without_plans_sends_nothing(ctx, clock, messenger, profile):
    """Acceptance 13: no plans, no times set -> days of silence produce no message."""
    reminders.register(ctx)
    initiative.register(ctx)
    await _task(ctx, title="В бэклоге", is_backlog=True)
    await _task(ctx, title="Без времени")
    sched = Scheduler(ctx)
    t = NOW
    while t < NOW + timedelta(days=4):
        clock.set(t)
        await sched.tick()
        t += timedelta(minutes=30)
    assert messenger.sent == []
    async with ctx.db.transaction() as s:
        assert (await s.execute(select(func.count(ReminderSchedule.id)))).scalar_one() == 0
        assert (await s.execute(select(func.count(OutboundMessage.id)))).scalar_one() == 0


async def test_silence_with_daily_times_only_sends_the_agreed_messages(
    ctx, clock, messenger, profile
):
    """With a morning time set, silence still adds nothing beyond one morning message
    per day (no "are you there?" series)."""
    reminders.register(ctx)
    initiative.register(ctx)
    await _set(ctx, morning_time=time(9, 0))
    sched = Scheduler(ctx)
    t = NOW + timedelta(minutes=1)
    while t < NOW + timedelta(days=3):
        clock.set(t)
        await sched.tick()
        t += timedelta(minutes=30)
    async with ctx.db.transaction() as s:
        cats = (await s.execute(select(OutboundMessage.category))).scalars().all()
    assert cats == [MessageCategory.MORNING_PLAN] * 3
