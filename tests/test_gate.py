"""Outbound Gate: strict 4/60 sliding limit (acceptance 11), counting policy,
dedup, atomic reservation under concurrency (tasks and processes)."""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from procbot.config import AppConfig, CategoryPolicy, GateConfig
from procbot.context import AppContext
from procbot.db.models import ConversationMessage, OutboundMessage, ReminderSchedule, Task
from procbot.db.session import Database
from procbot.domain.enums import MessageCategory, ReminderState, SendState
from procbot.services import gate, initiative, reminders
from procbot.services.gate import BlockReason, Outcome
from procbot.services.reminders import schedule_reminder
from procbot.services.scheduler import Scheduler
from procbot.testing import FakeMessenger

T0 = datetime(2026, 10, 26, 9, 0, tzinfo=UTC)  # Monday, 10:00 Berlin (CET)


async def _deliver(ctx: AppContext, key: str, cat: MessageCategory = MessageCategory.PRE_TASK):
    return await gate.deliver(ctx, category=cat, text=f"msg {key}", dedup_key=key)


async def _counted(ctx: AppContext) -> int:
    async with ctx.db.transaction() as s:
        return (
            await s.execute(
                select(func.count(OutboundMessage.id)).where(
                    OutboundMessage.counts_toward_limit.is_(True),
                    OutboundMessage.send_state == SendState.SENT,
                )
            )
        ).scalar_one()


async def _due_start_delays(ctx: AppContext, n: int, due: datetime) -> list[int]:
    ids: list[int] = []
    async with ctx.db.transaction() as s:
        for i in range(n):
            planned = due - timedelta(minutes=10)
            t = Task(title=f"Задача {i}", planned_start_at=planned, created_at=due, updated_at=due)
            s.add(t)
            await s.flush()
            r = await schedule_reminder(
                s,
                now=due,
                category=MessageCategory.START_DELAY,
                due_at=due,
                dedup_key=f"delay:{t.id}",
                task_id=t.id,
                payload={"task_id": t.id, "planned_start_at": planned.isoformat()},
            )
            ids.append(r.id)
    return ids


def _wire(ctx: AppContext) -> None:
    reminders.register(ctx)
    initiative.register(ctx)


async def test_six_due_reminders_same_minute_exactly_four_sent(ctx, clock, messenger, profile):
    _wire(ctx)
    clock.set(T0)
    ids = await _due_start_delays(ctx, 6, T0)
    report = await Scheduler(ctx).tick()
    assert len(messenger.sent) == 4
    assert report.count(report_key("sent")) == 4
    assert report.count(report_key("rescheduled")) == 2
    async with ctx.db.transaction() as s:
        rows = [await s.get(ReminderSchedule, i) for i in ids]
    done = [r for r in rows if r and r.state == ReminderState.DONE]
    waiting = [r for r in rows if r and r.state == ReminderState.PENDING]
    assert len(done) == 4 and len(waiting) == 2
    for r in waiting:
        # rescheduled to when the sliding window frees a slot, not piled up
        assert r.due_at == T0 + timedelta(minutes=60)
        assert r.reschedule_count == 1
        assert r.last_reason == BlockReason.RATE_LIMIT
    # nothing more inside the window
    clock.set(T0 + timedelta(minutes=59, seconds=59))
    await Scheduler(ctx).tick()
    assert len(messenger.sent) == 4
    clock.set(T0 + timedelta(minutes=60))
    await Scheduler(ctx).tick()
    assert len(messenger.sent) == 6


def report_key(name: str):
    from procbot.services.scheduler import ReminderResult

    return ReminderResult(name)


async def test_sliding_window_boundaries(ctx, clock, profile):
    for minute in (0, 20, 40, 50):
        clock.set(T0 + timedelta(minutes=minute))
        assert (await _deliver(ctx, f"k{minute}")).outcome == Outcome.SENT
    clock.set(T0 + timedelta(minutes=55))
    d = await _deliver(ctx, "late")
    assert d.outcome == Outcome.BLOCKED and d.reason == BlockReason.RATE_LIMIT
    assert d.retry_at == T0 + timedelta(minutes=60)  # the 09:00 message leaves the window
    clock.set(T0 + timedelta(minutes=59, seconds=59))
    assert (await gate.check(ctx, MessageCategory.START_DELAY)).outcome == Outcome.BLOCKED
    clock.set(T0 + timedelta(minutes=60))  # exactly 60 min later: 09:00 no longer counts
    assert (await _deliver(ctx, "k60")).outcome == Outcome.SENT
    d = await _deliver(ctx, "k60b")
    assert d.outcome == Outcome.BLOCKED
    assert d.retry_at == T0 + timedelta(minutes=80)


async def test_any_60_minute_window_never_exceeds_four(ctx, clock, messenger, profile):
    """Try to send every 5 minutes for 3 hours: every sliding window holds <= 4."""
    sent_at: list[datetime] = []
    for i in range(36):
        clock.set(T0 + timedelta(minutes=5 * i))
        if (await _deliver(ctx, f"r{i}")).outcome == Outcome.SENT:
            sent_at.append(clock.now())
    assert len(sent_at) == 12  # 4 per hour
    for t in sent_at:
        in_window = [x for x in sent_at if t - timedelta(minutes=60) < x <= t]
        assert len(in_window) <= 4


async def test_concurrent_deliveries_with_gather_respect_limit(ctx, clock, messenger, profile):
    results = await asyncio.gather(*(_deliver(ctx, f"g{i}") for i in range(10)))
    assert sum(r.outcome == Outcome.SENT for r in results) == 4
    assert sum(r.outcome == Outcome.BLOCKED for r in results) == 6
    assert len(messenger.sent) == 4
    assert await _counted(ctx) == 4


async def test_two_processes_on_one_database_respect_limit(ctx, clock, messenger, profile):
    """Two Database objects on the same file simulate two processes."""
    db2 = Database.for_path(ctx.env.database_path)
    ctx2 = dataclasses.replace(ctx, db=db2, instance_id="other-process", registry={})
    try:
        calls = []
        for i in range(12):
            c = ctx if i % 2 == 0 else ctx2
            calls.append(_deliver(c, f"p{i}"))
        results = await asyncio.gather(*calls)
        assert sum(r.outcome == Outcome.SENT for r in results) == 4
        assert len(messenger.sent) == 4
    finally:
        await db2.dispose()


async def test_two_schedulers_never_process_a_reminder_twice(ctx, clock, messenger, profile):
    db2 = Database.for_path(ctx.env.database_path)
    ctx2 = dataclasses.replace(ctx, db=db2, instance_id="other-process", registry={})
    _wire(ctx)
    _wire(ctx2)
    try:
        clock.set(T0)
        ids = await _due_start_delays(ctx, 6, T0)
        await asyncio.gather(Scheduler(ctx).tick(), Scheduler(ctx2).tick())
        assert len(messenger.sent) == 4
        texts = messenger.texts()
        assert len(set(texts)) == 4  # no reminder was delivered twice
        async with ctx.db.transaction() as s:
            outbound = (await s.execute(select(OutboundMessage))).scalars().all()
            assert sorted(o.reminder_id for o in outbound if o.reminder_id) == sorted(
                {o.reminder_id for o in outbound if o.reminder_id}
            )
            states = [(await s.get(ReminderSchedule, i)).state for i in ids]  # type: ignore[union-attr]
        assert states.count(ReminderState.DONE) == 4
        assert states.count(ReminderState.PENDING) == 2
    finally:
        await db2.dispose()


async def test_failed_send_releases_slot(ctx, clock, messenger, profile):
    messenger.fail_next = 1
    d = await _deliver(ctx, "f0")
    assert d.outcome == Outcome.FAILED and not d.permanent
    for i in range(4):
        assert (await _deliver(ctx, f"f{i + 1}")).outcome == Outcome.SENT
    assert (await _deliver(ctx, "f9")).outcome == Outcome.BLOCKED
    # the failed key can be retried later (row reused, not duplicated)
    clock.advance(minutes=61)
    assert (await _deliver(ctx, "f0")).outcome == Outcome.SENT
    async with ctx.db.transaction() as s:
        n = (
            await s.execute(
                select(func.count(OutboundMessage.id)).where(OutboundMessage.dedup_key == "f0")
            )
        ).scalar_one()
    assert n == 1


async def test_counting_policy_per_category(tmp_path, profile, ctx, clock, messenger):
    for i in range(4):
        assert (await _deliver(ctx, f"c{i}")).outcome == Outcome.SENT
    # timer signals and system notices never count and are not limited
    assert (await _deliver(ctx, "t", MessageCategory.TIMER_SIGNAL)).outcome == Outcome.SENT
    assert (await _deliver(ctx, "n", MessageCategory.SYSTEM_NOTICE)).outcome == Outcome.SENT
    # open question 3: policy None -> counts (conservative) -> blocked now
    d = await _deliver(ctx, "m", MessageCategory.MORNING_PLAN)
    assert d.outcome == Outcome.BLOCKED and d.reason == BlockReason.RATE_LIMIT
    # explicitly configured as not counting -> allowed
    ctx.config = AppConfig(
        gate=GateConfig(
            categories={MessageCategory.MORNING_PLAN: CategoryPolicy(counts_toward_limit=False)}
        )
    )
    assert (await _deliver(ctx, "m", MessageCategory.MORNING_PLAN)).outcome == Outcome.SENT
    # reminder categories ALWAYS count, whatever the config says
    ctx.config = AppConfig(
        gate=GateConfig(
            categories={MessageCategory.START_DELAY: CategoryPolicy(counts_toward_limit=False)}
        )
    )
    assert gate.counts_toward_limit(MessageCategory.START_DELAY, ctx.config.gate)
    assert (await _deliver(ctx, "s", MessageCategory.START_DELAY)).outcome == Outcome.BLOCKED


async def test_unknown_and_disabled_categories(ctx, profile, messenger):
    d = await gate.deliver(ctx, category="chit_chat", text="hi", dedup_key="x")
    assert d.outcome == Outcome.BLOCKED and d.reason == BlockReason.CATEGORY_NOT_ALLOWED
    assert d.final
    ctx.config = AppConfig(
        gate=GateConfig(categories={MessageCategory.WEEKLY_REVIEW: CategoryPolicy(enabled=False)})
    )
    d = await _deliver(ctx, "w", MessageCategory.WEEKLY_REVIEW)
    assert d.reason == BlockReason.CATEGORY_DISABLED and d.final
    assert messenger.sent == []


async def test_notifications_disabled_blocks_all_but_timer(ctx, profile, messenger):
    from procbot.services.profile import update_profile

    async with ctx.db.transaction() as s:
        await update_profile(s, now=ctx.clock.now(), notifications_enabled=False)
    d = await _deliver(ctx, "a", MessageCategory.MORNING_PLAN)
    assert d.reason == BlockReason.NOTIFICATIONS_DISABLED and d.final
    assert (await _deliver(ctx, "b", MessageCategory.TIMER_SIGNAL)).outcome == Outcome.SENT


async def test_dedup_key_sends_once(ctx, profile, messenger):
    assert (await _deliver(ctx, "same")).outcome == Outcome.SENT
    d = await _deliver(ctx, "same")
    assert d.outcome == Outcome.DUPLICATE
    assert len(messenger.sent) == 1


async def test_deliver_inside_transaction_is_refused(ctx, profile):
    with pytest.raises(RuntimeError):
        async with ctx.db.transaction():
            await _deliver(ctx, "tx")


async def test_sent_message_is_logged_and_confirmed(ctx, profile, messenger):
    d = await _deliver(ctx, "log")
    async with ctx.db.transaction() as s:
        row = await s.get(OutboundMessage, d.outbound_id)
        conv = (await s.execute(select(ConversationMessage))).scalars().all()
    assert row is not None and row.send_state == SendState.SENT
    assert row.telegram_message_id == messenger.sent[0].message_id
    assert row.sent_at is not None
    assert [(c.direction, c.text) for c in conv] == [("out", "msg log")]


async def test_permanent_failure_is_audited(ctx, profile):
    from procbot.db.models import AuditLog

    fm = FakeMessenger(fail_next=1, fail_permanent=True)
    ctx.messenger = fm
    d = await _deliver(ctx, "perm")
    assert d.outcome == Outcome.FAILED and d.permanent
    async with ctx.db.transaction() as s:
        actions = (await s.execute(select(AuditLog.action))).scalars().all()
    assert "outbound.failed_permanent" in actions
