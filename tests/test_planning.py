"""Morning planner: acceptance 6 (no work over busy calendar time, tz, explained
priorities, approximate estimates) and 7 (no silent rescheduling)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from procbot.ai.testing import make_fake_gateway
from procbot.db.models import CalendarConnection, CalendarEventCache, PlanRevision, Task
from procbot.services import planning
from procbot.wiring import register_all

# ctx clock: 2026-10-26 09:00 UTC = 10:00 Europe/Berlin (profile fixture tz)
BUSY_START = datetime(2026, 10, 26, 10, 0, tzinfo=UTC)  # 11:00 local
BUSY_END = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)  # 13:00 local


async def _seed(ctx) -> list[int]:
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        conn = CalendarConnection(status="active", last_sync_at=now, created_at=now, updated_at=now)
        s.add(conn)
        await s.flush()
        s.add(
            CalendarEventCache(
                connection_id=conn.id,
                calendar_id="primary",
                external_id="meet1",
                start_at=BUSY_START,
                end_at=BUSY_END,
                busy=True,
                summary="Встреча",
                synced_at=now,
            )
        )
        tasks = [
            Task(
                title=f"Задача {i}",
                planned_date=now.date(),
                estimate_minutes=90 if i < 2 else None,  # last one has no estimate
                created_at=now,
                updated_at=now,
            )
            for i in range(3)
        ]
        s.add_all(tasks)
        await s.flush()
        return [t.id for t in tasks]


async def _items(ctx, revision_id: int):
    async with ctx.db.transaction() as s:
        return await planning.revision_items(s, revision_id)


def _assert_no_overlap(items) -> None:
    timed = [i for i in items if i.start_at and i.end_at and i.kind in ("task", "habit")]
    assert timed, "plan has no timed work"
    for i in timed:
        assert not planning.overlaps(i.start_at, i.end_at, BUSY_START, BUSY_END), i.title
        assert i.start_at >= datetime(2026, 10, 26, 9, 0, tzinfo=UTC)  # not before "now"


async def test_plan_without_ai_avoids_busy_time_and_explains(ctx, profile) -> None:
    register_all(ctx)
    await _seed(ctx)
    rev_id = await planning.propose(ctx)
    items = await _items(ctx, rev_id)
    _assert_no_overlap(items)
    async with ctx.db.transaction() as s:
        rev = await s.get(PlanRevision, rev_id)
        assert rev is not None and rev.status == "proposed"
        assert rev.explanation
        assert not rev.used_ai
    # a task without an estimate is laid out with an approximate (marked) duration
    assert any(i.estimate_is_approximate for i in items if i.kind == "task")


async def test_ai_plan_over_meeting_is_rejected_and_replaced(ctx, profile) -> None:
    register_all(ctx)
    ids = await _seed(ctx)
    bad = {
        "items": [
            {"kind": "task", "task_id": ids[0], "start": "11:00", "end": "12:30"},
        ],
        "priorities": "Сначала задача 0",
        "assumptions": [],
        "approximate_task_ids": [ids[0]],
        "unplanned": [],
    }
    await make_fake_gateway(ctx, {"plan_day": [bad] * 10})
    rev_id = await planning.propose(ctx)
    _assert_no_overlap(await _items(ctx, rev_id))


async def test_plan_is_only_a_proposal_until_approved(ctx, profile) -> None:
    register_all(ctx)
    ids = await _seed(ctx)
    rev_id = await planning.propose(ctx)
    async with ctx.db.transaction() as s:
        tasks = (await s.execute(select(Task).where(Task.id.in_(ids)))).scalars().all()
        assert all(t.planned_start_at is None for t in tasks)  # nothing moved yet
        rev = await s.get(PlanRevision, rev_id)
        assert rev is not None
        await planning.approve_revision(s, ctx, plan_id=rev.plan_id, revision_id=rev_id)
    async with ctx.db.transaction() as s:
        tasks = (await s.execute(select(Task).where(Task.id.in_(ids)))).scalars().all()
        planned = [t for t in tasks if t.planned_start_at is not None]
        assert planned
        for t in planned:
            end = t.planned_start_at + timedelta(minutes=t.estimate_minutes or 0)
            assert not planning.overlaps(t.planned_start_at, end, BUSY_START, BUSY_END)


async def test_morning_plan_is_sent_through_scheduler(ctx, profile, messenger, clock) -> None:
    from datetime import time

    from procbot.services.initiative import seed_initiative_reminders
    from procbot.services.profile import update_profile
    from procbot.services.scheduler import Scheduler

    register_all(ctx)
    await _seed(ctx)
    async with ctx.db.transaction() as s:
        await update_profile(s, now=clock.now(), morning_time=time(10, 5))
    await seed_initiative_reminders(ctx)
    clock.advance(minutes=6)  # 10:06 local
    await Scheduler(ctx).tick()
    assert len(messenger.sent) == 1
    text = messenger.sent[0].text
    assert "Встреча" in text or "план" in text.lower()
    async with ctx.db.transaction() as s:
        assert (await s.execute(select(PlanRevision))).scalars().first() is not None
