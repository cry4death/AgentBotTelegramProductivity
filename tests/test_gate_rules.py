"""Gate rules: quiet hours (incl. DST nights), manual pause, calendar busy, active
work session (acceptance 12, 14, 23) and "no avalanche" after a blocked period."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

import pytest
from sqlalchemy import select

from procbot.context import AppContext
from procbot.db.models import (
    CalendarConnection,
    CalendarEventCache,
    OutboundMessage,
    ReminderSchedule,
    Task,
    WorkSession,
)
from procbot.domain.enums import MessageCategory, ReminderState, SessionStatus
from procbot.services import gate, initiative, reminders
from procbot.services.gate import BlockReason, Outcome
from procbot.services.outbound import register_renderer
from procbot.services.profile import update_profile
from procbot.services.reminders import render_payload_text, schedule_reminder
from procbot.services.scheduler import Scheduler
from procbot.timeutil import resolve_local

BERLIN = "Europe/Berlin"
QUIET = [{"start": "23:00", "end": "08:00"}]


def _wire(ctx: AppContext) -> None:
    reminders.register(ctx)
    initiative.register(ctx)
    register_renderer(ctx, MessageCategory.TIMER_SIGNAL, render_payload_text)


async def _set(ctx: AppContext, **fields) -> None:
    async with ctx.db.transaction() as s:
        await update_profile(s, now=ctx.clock.now(), **fields)


@pytest.mark.parametrize(
    ("moment", "expected_end"),
    [
        # 2026-10-25: Berlin falls back 03:00 CEST -> 02:00 CET; 08:00 CET = 07:00 UTC
        (datetime(2026, 10, 24, 22, 30, tzinfo=UTC), datetime(2026, 10, 25, 7, 0, tzinfo=UTC)),
        (datetime(2026, 10, 25, 1, 30, tzinfo=UTC), datetime(2026, 10, 25, 7, 0, tzinfo=UTC)),
        # 2026-03-29: springs forward 02:00 CET -> 03:00 CEST; 08:00 CEST = 06:00 UTC
        (datetime(2026, 3, 28, 22, 30, tzinfo=UTC), datetime(2026, 3, 29, 6, 0, tzinfo=UTC)),
        (datetime(2026, 3, 29, 1, 30, tzinfo=UTC), datetime(2026, 3, 29, 6, 0, tzinfo=UTC)),
        # ordinary night (CET)
        (datetime(2026, 11, 2, 23, 30, tzinfo=UTC), datetime(2026, 11, 3, 7, 0, tzinfo=UTC)),
    ],
)
async def test_quiet_hours_block_until_local_end_across_dst(
    ctx, clock, profile, messenger, moment, expected_end
):
    await _set(ctx, quiet_hours=QUIET)
    clock.set(moment)
    d = await gate.check(ctx, MessageCategory.START_DELAY)
    assert d.outcome == Outcome.BLOCKED and d.reason == BlockReason.QUIET_HOURS
    assert d.retry_at == expected_end
    # importance never overrides quiet hours; nothing is sent
    sent = await gate.deliver(ctx, category=MessageCategory.START_DELAY, text="x", dedup_key="q")
    assert sent.outcome == Outcome.BLOCKED
    assert messenger.sent == []
    clock.set(expected_end)
    assert (await gate.check(ctx, MessageCategory.START_DELAY)).outcome == Outcome.ALLOWED
    clock.set(expected_end - timedelta(minutes=1))
    assert (await gate.check(ctx, MessageCategory.START_DELAY)).outcome == Outcome.BLOCKED


async def test_quiet_hours_start_is_local_time_on_dst_evening(ctx, clock, profile):
    await _set(ctx, quiet_hours=QUIET)
    # 2026-10-24 22:59 local (CEST) = 20:59 UTC -> allowed; 23:00 CEST = 21:00 UTC -> blocked
    clock.set(datetime(2026, 10, 24, 20, 59, tzinfo=UTC))
    assert (await gate.check(ctx, MessageCategory.PRE_TASK)).allowed
    clock.set(datetime(2026, 10, 24, 21, 0, tzinfo=UTC))
    assert not (await gate.check(ctx, MessageCategory.PRE_TASK)).allowed
    # a week later (CET) 23:00 local = 22:00 UTC
    clock.set(datetime(2026, 10, 31, 21, 30, tzinfo=UTC))
    assert (await gate.check(ctx, MessageCategory.PRE_TASK)).allowed


async def test_timer_signal_and_quiet_hours_flag(ctx, clock, profile):
    await _set(ctx, quiet_hours=QUIET)
    clock.set(datetime(2026, 11, 2, 23, 30, tzinfo=UTC))
    assert (await gate.check(ctx, MessageCategory.TIMER_SIGNAL)).allowed
    ctx.config.gate.timer_signals_respect_quiet_hours = True
    assert not (await gate.check(ctx, MessageCategory.TIMER_SIGNAL)).allowed


async def test_manual_pause_blocks_until_pause_end(ctx, clock, profile, messenger):
    now = clock.now()
    await _set(ctx, pause_until=now + timedelta(hours=2))
    d = await gate.deliver(ctx, category=MessageCategory.MORNING_PLAN, text="x", dedup_key="p")
    assert d.reason == BlockReason.PAUSED and d.retry_at == now + timedelta(hours=2)
    assert messenger.sent == []
    clock.advance(hours=2)
    assert (await gate.check(ctx, MessageCategory.MORNING_PLAN)).allowed


async def test_pause_then_quiet_hours_chain_to_next_allowed_time(ctx, clock, profile):
    await _set(ctx, quiet_hours=QUIET)
    clock.set(datetime(2026, 11, 2, 20, 0, tzinfo=UTC))  # 21:00 local
    await _set(ctx, pause_until=datetime(2026, 11, 2, 22, 30, tzinfo=UTC))  # 23:30 local
    d = await gate.check(ctx, MessageCategory.PRE_TASK)
    assert d.reason == BlockReason.PAUSED
    assert d.retry_at == datetime(2026, 11, 3, 7, 0, tzinfo=UTC)  # end of the quiet night


async def _busy(ctx: AppContext, start: datetime, end: datetime, ext: str = "e1") -> None:
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        conn = (await s.execute(select(CalendarConnection))).scalars().first()
        if conn is None:
            conn = CalendarConnection(created_at=now, updated_at=now)
            s.add(conn)
            await s.flush()
        s.add(
            CalendarEventCache(
                connection_id=conn.id,
                calendar_id="primary",
                external_id=ext,
                start_at=start,
                end_at=end,
                busy=True,
                synced_at=now,
            )
        )


async def test_calendar_busy_blocks_initiative_but_not_timer(ctx, clock, profile):
    now = clock.now()
    await _busy(ctx, now - timedelta(minutes=10), now + timedelta(minutes=50))
    await _busy(ctx, now + timedelta(minutes=50), now + timedelta(minutes=80), "e2")
    d = await gate.check(ctx, MessageCategory.PRE_TASK)
    assert d.reason == BlockReason.CALENDAR_BUSY
    assert d.retry_at == now + timedelta(minutes=80)  # back-to-back events followed
    assert (await gate.check(ctx, MessageCategory.TIMER_SIGNAL)).allowed
    ctx.config.gate.respect_calendar_busy = False
    assert (await gate.check(ctx, MessageCategory.PRE_TASK)).allowed


async def _task_with_reminder(
    ctx: AppContext, category: MessageCategory, due: datetime, title: str = "Отчёт"
) -> tuple[int, int]:
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        planned = due if category == MessageCategory.PRE_TASK else due - timedelta(minutes=10)
        if category == MessageCategory.PRE_TASK:
            planned = due + timedelta(minutes=10)
        t = Task(title=title, planned_start_at=planned, created_at=now, updated_at=now)
        s.add(t)
        await s.flush()
        r = await schedule_reminder(
            s,
            now=now,
            category=category,
            due_at=due,
            dedup_key=f"{category}:{t.id}",
            task_id=t.id,
            payload={"task_id": t.id, "planned_start_at": planned.isoformat()},
        )
        return t.id, r.id


async def test_active_session_blocks_reminders_but_timer_signal_passes(
    ctx, clock, profile, messenger
):
    """Acceptance 14: no progress checks during an active session; timer signals only."""
    _wire(ctx)
    now = clock.now()
    _, delay_id = await _task_with_reminder(ctx, MessageCategory.START_DELAY, now)
    _, pre_id = await _task_with_reminder(ctx, MessageCategory.PRE_TASK, now, "Звонок")
    async with ctx.db.transaction() as s:
        other = Task(title="Код", created_at=now, updated_at=now)
        s.add(other)
        await s.flush()
        ws = WorkSession(
            task_id=other.id,
            mode="free",
            status=SessionStatus.ACTIVE,
            started_at=now - timedelta(minutes=5),
            created_at=now,
            updated_at=now,
        )
        s.add(ws)
        await s.flush()
        await schedule_reminder(
            s,
            now=now,
            category=MessageCategory.TIMER_SIGNAL,
            due_at=now,
            dedup_key=f"timer:{ws.id}:end",
            session_id=ws.id,
            payload={"text": "⏰ Время сессии вышло"},
        )
    for cat in (MessageCategory.START_DELAY, MessageCategory.PRE_TASK, MessageCategory.REPLAN):
        d = await gate.check(ctx, cat)
        assert d.reason == BlockReason.WORK_SESSION
    await Scheduler(ctx).tick()
    assert messenger.texts() == ["⏰ Время сессии вышло"]
    async with ctx.db.transaction() as s:
        for rid in (delay_id, pre_id):
            r = await s.get(ReminderSchedule, rid)
            assert r is not None and r.state == ReminderState.PENDING
            assert r.last_reason == BlockReason.WORK_SESSION and r.due_at > now


async def test_paused_session_also_blocks(ctx, clock, profile):
    now = clock.now()
    async with ctx.db.transaction() as s:
        s.add(
            WorkSession(
                mode="pomodoro",
                status=SessionStatus.PAUSED,
                started_at=now,
                created_at=now,
                updated_at=now,
            )
        )
    assert (await gate.check(ctx, MessageCategory.EVENING_REVIEW)).reason == (
        BlockReason.WORK_SESSION
    )
    async with ctx.db.transaction() as s:
        ws = (await s.execute(select(WorkSession))).scalar_one()
        ws.status = SessionStatus.UNCONFIRMED  # an unconfirmed start is not "working"
    assert (await gate.check(ctx, MessageCategory.EVENING_REVIEW)).allowed


@pytest.mark.parametrize(
    "night",
    [date(2026, 10, 24), date(2026, 3, 28), date(2026, 11, 2)],
    ids=["dst-autumn", "dst-spring", "regular"],
)
async def test_no_avalanche_after_quiet_hours(ctx, clock, profile, messenger, night):
    """Reminders due during the night: nothing is sent at night; at the end of the
    quiet hours one digest of the stale ones plus the still relevant one."""
    _wire(ctx)
    await _set(ctx, quiet_hours=QUIET, start_delay_min=10)
    evening = resolve_local(night, time(22, 0), BERLIN)
    clock.set(evening)
    starts = [
        resolve_local(night, time(23, 20), BERLIN),
        resolve_local(night + timedelta(days=1), time(0, 20), BERLIN),
        resolve_local(night + timedelta(days=1), time(1, 20), BERLIN),
        resolve_local(night + timedelta(days=1), time(4, 20), BERLIN),
        resolve_local(night + timedelta(days=1), time(5, 20), BERLIN),
        resolve_local(night + timedelta(days=1), time(7, 0), BERLIN),
    ]
    async with ctx.db.transaction() as s:
        for i, st in enumerate(starts):
            t = Task(
                title=f"Ночная {i}", planned_start_at=st, created_at=evening, updated_at=evening
            )
            s.add(t)
            await s.flush()
            await reminders.sync_task_reminders(s, ctx, t)
    quiet_end = resolve_local(night + timedelta(days=1), time(8, 0), BERLIN)
    sched = Scheduler(ctx)
    t = evening
    while t < quiet_end:
        clock.set(t)
        await sched.tick()
        t += timedelta(minutes=10)
    assert messenger.sent == []  # quiet hours: nothing at all
    morning_sends: list[str] = []
    while t < quiet_end + timedelta(hours=2):
        clock.set(t)
        await sched.tick()
        t += timedelta(minutes=10)
    morning_sends = messenger.texts()
    async with ctx.db.transaction() as s:
        out = (await s.execute(select(OutboundMessage))).scalars().all()
    digests = [o for o in out if o.category == MessageCategory.DIGEST]
    assert len(digests) == 1
    assert all(o.created_at == quiet_end for o in out)  # sent right when allowed
    assert len(morning_sends) == 2  # 1 digest + the 07:00 task (still relevant)
    digest_text = digests[0].text
    for i in range(5):
        assert f"Ночная {i}" in digest_text
    assert "Ночная 5" not in digest_text
    assert any("Ночная 5" in x for x in morning_sends if x != digest_text)
