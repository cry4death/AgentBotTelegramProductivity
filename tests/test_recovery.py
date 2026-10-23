"""Startup recovery (scenario G): reminders survive a restart (acceptance 2), a crash
between send and confirm is never re-sent, missed reminders fold into one digest,
and recovery is idempotent."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, time, timedelta

from sqlalchemy import select

from procbot.context import AppContext
from procbot.db.models import JobLease, OutboundMessage, ReminderSchedule, Task
from procbot.db.session import Database
from procbot.domain.enums import MessageCategory, ReminderState, SendState
from procbot.services import gate, initiative, reminders
from procbot.services.gate import Outcome
from procbot.services.profile import update_profile
from procbot.services.recovery import run_recovery
from procbot.services.reminders import schedule_reminder, sync_task_reminders
from procbot.services.scheduler import outbound_key
from procbot.testing import FakeMessenger

NOW = datetime(2026, 10, 26, 9, 0, tzinfo=UTC)


def _wire(ctx: AppContext) -> None:
    reminders.register(ctx)
    initiative.register(ctx)


async def _restart(ctx: AppContext) -> AppContext:
    """Simulate a new process: fresh engine on the same file, fresh registry."""
    await ctx.db.dispose()
    new = dataclasses.replace(
        ctx,
        db=Database.for_path(ctx.env.database_path),
        instance_id="restarted",
        registry={},
        messenger=FakeMessenger(),
    )
    _wire(new)
    return new


async def _snapshot(ctx: AppContext) -> tuple[list[tuple], list[tuple]]:
    async with ctx.db.transaction() as s:
        rem = [
            (r.id, r.state, r.due_at, r.reschedule_count, r.dedup_key, r.payload)
            for r in (await s.execute(select(ReminderSchedule).order_by(ReminderSchedule.id)))
            .scalars()
            .all()
        ]
        out = [
            (o.id, o.send_state, o.dedup_key)
            for o in (await s.execute(select(OutboundMessage).order_by(OutboundMessage.id)))
            .scalars()
            .all()
        ]
    return rem, out


async def test_planned_reminders_survive_restart(ctx, clock, profile):
    async with ctx.db.transaction() as s:
        await update_profile(
            s, now=NOW, start_delay_min=10, pre_task_lead_min=5, morning_time=time(9, 0)
        )
        t = Task(
            title="Сайт", planned_start_at=NOW + timedelta(hours=3), created_at=NOW, updated_at=NOW
        )
        s.add(t)
        await s.flush()
        await sync_task_reminders(s, ctx, t)
    _wire(ctx)
    await initiative.seed_initiative_reminders(ctx)
    before, _ = await _snapshot(ctx)
    assert len(before) == 3
    ctx2 = await _restart(ctx)
    try:
        report = await run_recovery(ctx2)
        after, out = await _snapshot(ctx2)
        assert after == before  # nothing lost, nothing duplicated
        assert out == [] and ctx2.messenger.sent == []  # type: ignore[attr-defined]
        assert report.initiative_planned == 1
    finally:
        await ctx2.db.dispose()


async def test_crash_between_send_and_confirm_is_maybe_sent_not_resent(ctx, clock, profile):
    _wire(ctx)
    crash_at = NOW - timedelta(minutes=10)
    async with ctx.db.transaction() as s:
        r = await schedule_reminder(
            s,
            now=crash_at,
            category=MessageCategory.MORNING_PLAN,
            due_at=crash_at,
            dedup_key="morning:2026-10-26",
        )
        s.add(
            OutboundMessage(
                category=MessageCategory.MORNING_PLAN,
                dedup_key=outbound_key(r.id),
                reminder_id=r.id,
                counts_toward_limit=True,
                send_state=SendState.SENDING,
                text="Доброе утро!",
                created_at=crash_at,
            )
        )
        s.add(
            JobLease(
                name=f"reminder:{r.id}",
                owner="dead-process",
                acquired_at=crash_at,
                expires_at=crash_at + timedelta(minutes=2),
            )
        )
        rid = r.id
    ctx2 = await _restart(ctx)
    try:
        report = await run_recovery(ctx2)
        assert report.leases_released == 1
        assert report.marked_maybe_sent == 1
        assert ctx2.messenger.sent == []  # type: ignore[attr-defined]
        async with ctx2.db.transaction() as s:
            row = (await s.execute(select(OutboundMessage))).scalar_one()
            rem = await s.get(ReminderSchedule, rid)
        assert row.send_state == SendState.MAYBE_SENT
        assert rem is not None and rem.state == ReminderState.DONE
        # the maybe-sent message still counts toward the sliding window
        for i in range(3):
            d = await gate.deliver(
                ctx2, category=MessageCategory.PRE_TASK, text="x", dedup_key=f"x{i}"
            )
            assert d.outcome == Outcome.SENT
        d = await gate.deliver(ctx2, category=MessageCategory.PRE_TASK, text="y", dedup_key="y")
        assert d.outcome == Outcome.BLOCKED
    finally:
        await ctx2.db.dispose()


async def test_recent_sending_row_is_left_in_flight_and_not_resent(ctx, clock, profile):
    _wire(ctx)
    at = NOW - timedelta(seconds=20)
    async with ctx.db.transaction() as s:
        r = await schedule_reminder(
            s, now=at, category=MessageCategory.MORNING_PLAN, due_at=at, dedup_key="m"
        )
        s.add(
            OutboundMessage(
                category=MessageCategory.MORNING_PLAN,
                dedup_key=outbound_key(r.id),
                reminder_id=r.id,
                counts_toward_limit=True,
                send_state=SendState.SENDING,
                text="…",
                created_at=at,
            )
        )
    ctx2 = await _restart(ctx)
    try:
        report = await run_recovery(ctx2)
        assert report.marked_maybe_sent == 0 and report.in_flight == 1
        assert ctx2.messenger.sent == []  # type: ignore[attr-defined]
        clock.advance(minutes=5)
        report = await run_recovery(ctx2)
        assert report.marked_maybe_sent == 1
        assert ctx2.messenger.sent == []  # type: ignore[attr-defined]
    finally:
        await ctx2.db.dispose()


async def test_missed_reminders_after_downtime_fold_into_one_digest(ctx, clock, profile):
    async with ctx.db.transaction() as s:
        await update_profile(s, now=NOW, start_delay_min=10)
    _wire(ctx)
    # the bot was down 03:00-09:00 UTC; tasks were planned during the downtime
    clock.set(NOW - timedelta(hours=7))
    async with ctx.db.transaction() as s:
        for i, h in enumerate((3, 4, 5, 6, 7, 8)):
            t = Task(
                title=f"Пропущенная {i}",
                planned_start_at=NOW.replace(hour=h) - timedelta(minutes=10),
                created_at=clock.now(),
                updated_at=clock.now(),
            )
            s.add(t)
            await s.flush()
            await sync_task_reminders(s, ctx, t)
    clock.set(NOW)
    ctx2 = await _restart(ctx)
    try:
        report = await run_recovery(ctx2)
        sent = ctx2.messenger.sent  # type: ignore[attr-defined]
        # stale ones (due > 2h ago) -> one digest; the 2 recent ones are still relevant
        async with ctx2.db.transaction() as s:
            out = (await s.execute(select(OutboundMessage))).scalars().all()
        cats = sorted(o.category for o in out)
        assert cats.count(MessageCategory.DIGEST) == 1
        assert len(sent) <= ctx2.config.gate.max_reminders
        digest = next(o for o in out if o.category == MessageCategory.DIGEST)
        for i in range(4):
            assert f"Пропущенная {i}" in digest.text
        assert report.reminders.results.get("folded") == 4
        # idempotent: a second run changes nothing and sends nothing
        before = await _snapshot(ctx2)
        n = len(sent)
        await run_recovery(ctx2)
        assert await _snapshot(ctx2) == before
        assert len(ctx2.messenger.sent) == n  # type: ignore[attr-defined]
    finally:
        await ctx2.db.dispose()


async def test_recovery_twice_is_idempotent_on_empty_state(ctx, profile):
    _wire(ctx)
    first = await run_recovery(ctx)
    snap = await _snapshot(ctx)
    second = await run_recovery(ctx)
    assert await _snapshot(ctx) == snap
    assert first.marked_maybe_sent == second.marked_maybe_sent == 0
