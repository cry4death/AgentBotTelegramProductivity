"""Scheduler loop, leases, retries, renderer contract and the start-delay question
(acceptance 9)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from procbot.context import AppContext
from procbot.db.models import AuditLog, OutboundMessage, ReminderSchedule, Task, WorkSession
from procbot.domain.enums import MessageCategory, ReminderState, SessionStatus, TaskStatus
from procbot.services import initiative, leases, reminders
from procbot.services.jobs import PeriodicJob, register_job
from procbot.services.outbound import Outgoing, register_renderer
from procbot.services.reminders import schedule_reminder
from procbot.services.scheduler import (
    MAX_SEND_ATTEMPTS,
    ReminderResult,
    Scheduler,
    process_reminder,
)
from procbot.telegram.callbacks import ReminderCb, SessionCb, TaskCb

NOW = datetime(2026, 10, 26, 9, 0, tzinfo=UTC)


def _wire(ctx: AppContext) -> None:
    reminders.register(ctx)
    initiative.register(ctx)


async def _delay_reminder(ctx: AppContext, title: str = "Написать отчёт") -> tuple[int, int]:
    now = ctx.clock.now()
    planned = now - timedelta(minutes=15)
    async with ctx.db.transaction() as s:
        t = Task(title=title, planned_start_at=planned, created_at=now, updated_at=now)
        s.add(t)
        await s.flush()
        r = await schedule_reminder(
            s,
            now=now,
            category=MessageCategory.START_DELAY,
            due_at=now,
            dedup_key=f"delay:{t.id}:{planned.isoformat()}",
            task_id=t.id,
            payload={"task_id": t.id, "planned_start_at": planned.isoformat()},
        )
        return t.id, r.id


async def _state(ctx: AppContext, rid: int) -> ReminderSchedule:
    async with ctx.db.transaction() as s:
        r = await s.get(ReminderSchedule, rid)
        assert r is not None
        return r


async def test_lease_is_exclusive_and_expires(ctx, clock):
    assert await leases.acquire(ctx, "x", ttl_seconds=60, owner="a")
    assert not await leases.acquire(ctx, "x", ttl_seconds=60, owner="b")
    assert not await leases.acquire(ctx, "x", ttl_seconds=60, owner="a")  # not shared
    assert not await leases.release(ctx, "x", owner="b")  # only the holder releases
    clock.advance(seconds=61)
    assert await leases.acquire(ctx, "x", ttl_seconds=60, owner="b")
    assert await leases.release(ctx, "x", owner="b")
    assert await leases.acquire(ctx, "x", ttl_seconds=60, owner="c")
    clock.advance(seconds=120)
    assert await leases.expire_stale(ctx) == 1


async def test_reminder_with_lease_held_elsewhere_is_skipped(ctx, clock, messenger, profile):
    _wire(ctx)
    _, rid = await _delay_reminder(ctx)
    assert await leases.acquire(ctx, f"reminder:{rid}", owner="other-instance")
    assert await process_reminder(ctx, rid) == ReminderResult.SKIPPED
    assert messenger.sent == []
    clock.advance(seconds=ctx.config.scheduler.lease_seconds + 1)
    assert await process_reminder(ctx, rid) == ReminderResult.SENT
    assert await process_reminder(ctx, rid) == ReminderResult.SKIPPED  # already done
    assert len(messenger.sent) == 1


async def test_start_delay_asks_short_question_with_quick_buttons(ctx, clock, messenger, profile):
    """Acceptance 9 / scenario C: first a short question with buttons, no deep dialog."""
    _wire(ctx)
    tid, rid = await _delay_reminder(ctx)
    await Scheduler(ctx).tick()
    assert len(messenger.sent) == 1
    msg = messenger.sent[0]
    assert "Написать отчёт" in msg.text
    assert len(msg.text) < 300
    buttons = [b for row in msg.reply_markup.inline_keyboard for b in row]
    datas = [b.callback_data for b in buttons]
    assert SessionCb(action="pick", task_id=tid).pack() in datas
    assert TaskCb(action="postpone", task_id=tid).pack() in datas
    reasons = {
        ReminderCb.unpack(d).arg
        for d in datas
        if d.startswith("rm:") and ReminderCb.unpack(d).action == "reason"
    }
    assert reasons == {
        "unclear",
        "too_big",
        "unpleasant",
        "distractions",
        "low_energy",
        "priority_conflict",
        "external",
        "other",
    }
    assert any(ReminderCb.unpack(d).action == "rest" for d in datas if d.startswith("rm:"))
    assert (await _state(ctx, rid)).state == ReminderState.DONE


async def test_start_delay_is_dropped_when_task_started_or_closed(ctx, clock, messenger, profile):
    _wire(ctx)
    now = clock.now()
    tid, rid = await _delay_reminder(ctx, "Уже начал")
    async with ctx.db.transaction() as s:
        s.add(
            WorkSession(
                task_id=tid,
                mode="free",
                status=SessionStatus.FINISHED,
                started_at=now - timedelta(minutes=10),
                ended_at=now - timedelta(minutes=1),
                created_at=now,
                updated_at=now,
            )
        )
    tid2, rid2 = await _delay_reminder(ctx, "Уже сделал")
    async with ctx.db.transaction() as s:
        t2 = await s.get(Task, tid2)
        assert t2 is not None
        t2.status = TaskStatus.DONE
    await Scheduler(ctx).tick()
    assert messenger.sent == []
    assert (await _state(ctx, rid)).state == ReminderState.DROPPED
    assert (await _state(ctx, rid2)).state == ReminderState.DROPPED


async def test_transient_failure_retries_with_backoff_then_sends(ctx, clock, messenger, profile):
    _wire(ctx)
    _, rid = await _delay_reminder(ctx)
    messenger.fail_next = 2
    assert await process_reminder(ctx, rid) == ReminderResult.RETRY
    r = await _state(ctx, rid)
    assert r.attempts == 1 and r.due_at == NOW + timedelta(seconds=30)
    clock.set(r.due_at)
    assert await process_reminder(ctx, rid) == ReminderResult.RETRY
    r = await _state(ctx, rid)
    assert r.attempts == 2 and r.due_at == NOW + timedelta(seconds=30 + 60)
    clock.set(r.due_at)
    assert await process_reminder(ctx, rid) == ReminderResult.SENT
    async with ctx.db.transaction() as s:
        rows = (await s.execute(select(OutboundMessage))).scalars().all()
    assert len(rows) == 1 and rows[0].send_state == "sent"  # one row, reused per attempt
    assert len(messenger.sent) == 1


async def test_transient_failures_give_up_after_limit(ctx, clock, messenger, profile):
    _wire(ctx)
    _, rid = await _delay_reminder(ctx)
    messenger.fail_next = 100
    for _ in range(MAX_SEND_ATTEMPTS - 1):
        assert await process_reminder(ctx, rid) == ReminderResult.RETRY
        clock.set((await _state(ctx, rid)).due_at)
    assert await process_reminder(ctx, rid) == ReminderResult.DROPPED


async def test_permanent_failure_drops_and_logs(ctx, clock, messenger, profile):
    _wire(ctx)
    _, rid = await _delay_reminder(ctx)
    messenger.fail_next = 1
    messenger.fail_permanent = True
    assert await process_reminder(ctx, rid) == ReminderResult.DROPPED
    async with ctx.db.transaction() as s:
        actions = (await s.execute(select(AuditLog.action))).scalars().all()
    assert "outbound.failed_permanent" in actions


async def test_renderer_none_or_missing_drops(ctx, clock, messenger, profile):
    async def nothing(_ctx, _r):
        return None

    register_renderer(ctx, MessageCategory.REPLAN, nothing)
    now = clock.now()
    async with ctx.db.transaction() as s:
        a = await schedule_reminder(
            s, now=now, category=MessageCategory.REPLAN, due_at=now, dedup_key="a"
        )
        b = await schedule_reminder(
            s, now=now, category=MessageCategory.WEEKLY_REVIEW, due_at=now, dedup_key="b"
        )
    await Scheduler(ctx).tick()
    assert (await _state(ctx, a.id)).state == ReminderState.DROPPED
    assert (await _state(ctx, b.id)).last_reason == "no renderer"
    assert messenger.sent == []


async def test_renderer_runs_outside_transactions(ctx, clock, messenger, profile):
    seen: list[bool] = []

    async def probe(c: AppContext, _r):
        seen.append(c.db.in_transaction())
        return Outgoing(text="план пересмотреть?")

    register_renderer(ctx, MessageCategory.REPLAN, probe)
    now = clock.now()
    async with ctx.db.transaction() as s:
        await schedule_reminder(
            s, now=now, category=MessageCategory.REPLAN, due_at=now, dedup_key="r"
        )
    await Scheduler(ctx).tick()
    assert seen == [False]
    assert messenger.texts() == ["план пересмотреть?"]


async def test_job_failure_does_not_break_tick_and_jobs_respect_interval(ctx, clock):
    calls: list[str] = []

    async def bad(_ctx):
        calls.append("bad")
        raise RuntimeError("boom")

    async def good(_ctx):
        calls.append("good")

    register_job(ctx, PeriodicJob(name="bad", interval_seconds=60, run=bad, run_on_start=True))
    register_job(ctx, PeriodicJob(name="good", interval_seconds=60, run=good))
    sched = Scheduler(ctx)
    report = await sched.tick()
    assert calls == ["bad"] and report.jobs_failed == ["bad"]
    clock.advance(seconds=30)
    await sched.tick()
    assert calls == ["bad"]
    clock.advance(seconds=31)
    await sched.tick()
    assert sorted(calls) == ["bad", "bad", "good"]


async def test_job_lease_prevents_parallel_run(ctx, clock):
    calls: list[str] = []

    async def job(_ctx):
        calls.append("run")

    register_job(ctx, PeriodicJob(name="j", interval_seconds=60, run=job, run_on_start=True))
    assert await leases.acquire(ctx, "job:j", owner="elsewhere")
    await Scheduler(ctx).tick()
    assert calls == []


async def test_run_forever_stops(ctx, clock):
    ctx.config.scheduler.tick_seconds = 0.01
    stop = asyncio.Event()
    sched = Scheduler(ctx)
    task = asyncio.create_task(sched.run_forever(stop))
    await asyncio.sleep(0.05)
    stop.set()
    await asyncio.wait_for(task, timeout=2)
