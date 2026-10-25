"""Recurring tasks: rule validation, next date, the next copy on "done" (once),
and "undo done" dropping an untouched copy."""

from __future__ import annotations

from datetime import date, time

import pytest
from sqlalchemy import select

from procbot.db.models import Task
from procbot.domain.actions import TaskDraft
from procbot.domain.enums import Actor, TaskStatus
from procbot.services import tasks as ts
from procbot.services.recurrence import RecurrenceError, next_date, parse_rule, rule_text
from procbot.timeutil import local_date, resolve_local, to_local


def test_parse_and_text():
    assert parse_rule(None) is None and parse_rule({"type": "none"}) is None
    assert parse_rule({"type": "daily"}) == {"type": "daily", "every": 1}
    assert parse_rule({"type": "weekly", "days": [4, 0, 0]}) == {"type": "weekly", "days": [0, 4]}
    assert rule_text({"type": "weekly", "days": [0, 1, 2, 3, 4]}) == "по будням"
    assert rule_text({"type": "monthly", "day": 5}) == "каждый месяц, 5-го"
    for bad in (
        {"type": "weekly", "days": []},
        {"type": "daily", "every": 0},
        {"type": "monthly", "day": 32},
        {"type": "x"},
        [1],
    ):
        with pytest.raises(RecurrenceError):
            parse_rule(bad)


def test_next_date_rules():
    mon = date(2026, 10, 26)
    assert next_date({"type": "daily", "every": 1}, base=mon, after=mon) == date(2026, 10, 27)
    # done late: daily skips to the first day after today, keeping the step
    assert next_date({"type": "daily", "every": 3}, base=mon, after=date(2026, 10, 31)) == date(
        2026, 11, 1
    )
    assert next_date({"type": "weekly", "days": [0, 3]}, base=mon, after=mon) == date(2026, 10, 29)
    # monthly on the 31st falls back to the last day of a short month
    assert next_date(
        {"type": "monthly", "day": 31}, base=date(2026, 10, 31), after=date(2026, 10, 31)
    ) == date(2026, 11, 30)
    assert next_date(
        {"type": "monthly", "day": 1}, base=date(2026, 10, 1), after=date(2026, 10, 5)
    ) == date(2026, 11, 1)


async def _make(ctx, **kw) -> int:
    async with ctx.db.transaction() as s:
        return (
            await ts.create_task(s, ctx, TaskDraft(title="Полить цветы", **kw), actor=Actor.USER)
        ).id


async def test_done_makes_one_next_copy_with_same_time(ctx, profile):
    async with ctx.db.transaction() as s:
        tz = await ts.owner_tz(s)
        today = await ts.local_today(s, ctx)
    start = resolve_local(today, time(9, 30), tz)
    tid = await _make(
        ctx,
        planned_start_at=start,
        estimate_minutes=10,
        recurrence={"type": "weekly", "days": [0, 3]},
    )
    async with ctx.db.transaction() as s:
        await ts.complete_task(s, ctx, tid, actor=Actor.USER)
        await ts.complete_task(s, ctx, tid, actor=Actor.USER)  # repeated press
    async with ctx.db.transaction() as s:
        rows = (await s.execute(select(Task).order_by(Task.id))).scalars().all()
        assert len(rows) == 2
        nxt = rows[1]
        assert nxt.status == TaskStatus.TODO and nxt.recurrence == {
            "type": "weekly",
            "days": [0, 3],
        }
        assert local_date(nxt.planned_start_at, tz) == date(2026, 10, 29)  # Monday -> Thursday
        assert to_local(nxt.planned_start_at, tz).time() == time(9, 30)
        assert nxt.estimate_minutes == 10 and nxt.created_by == "system"
        assert (await ts.next_occurrence_of(s, tid)).id == nxt.id
    # reopen + done again: still one copy
    async with ctx.db.transaction() as s:
        await ts.reopen_task(s, ctx, tid, actor=Actor.USER)
        await ts.complete_task(s, ctx, tid, actor=Actor.USER)
        assert len((await s.execute(select(Task))).scalars().all()) == 2


async def test_undo_done_removes_untouched_copy(ctx, profile):
    tid = await _make(ctx, planned_date=date(2026, 10, 26), recurrence={"type": "daily"})
    async with ctx.db.transaction() as s:
        await ts.complete_task(s, ctx, tid, actor=Actor.USER)
    async with ctx.db.transaction() as s:
        task = await ts.undo_complete(s, ctx, tid, actor=Actor.USER)
        assert task.status == TaskStatus.TODO
        assert [t.id for t in (await s.execute(select(Task))).scalars()] == [tid]
    async with ctx.db.transaction() as s:  # done again: the copy comes back
        await ts.complete_task(s, ctx, tid, actor=Actor.USER)
        nxt = await ts.next_occurrence_of(s, tid)
        assert nxt is not None and nxt.planned_date == date(2026, 10, 27)


async def test_undo_keeps_a_copy_already_changed(ctx, profile):
    tid = await _make(ctx, planned_date=date(2026, 10, 26), recurrence={"type": "daily"})
    async with ctx.db.transaction() as s:
        await ts.complete_task(s, ctx, tid, actor=Actor.USER)
        nxt = await ts.next_occurrence_of(s, tid)
        assert nxt is not None
        await ts.update_task(s, ctx, nxt.id, actor=Actor.USER, title="Полить фикус")
    async with ctx.db.transaction() as s:
        await ts.undo_complete(s, ctx, tid, actor=Actor.USER)
        assert len((await s.execute(select(Task))).scalars().all()) == 2


async def test_bad_rule_is_refused_and_subtasks_never_repeat(ctx, profile):
    with pytest.raises(ts.TaskError):
        await _make(ctx, recurrence={"type": "weekly", "days": []})
    parent = await _make(ctx)
    async with ctx.db.transaction() as s:
        sub = await ts.create_task(
            s,
            ctx,
            TaskDraft(title="шаг", parent_id=parent, recurrence={"type": "daily"}),
            actor=Actor.USER,
        )
        assert sub.recurrence is None
        await ts.update_task(
            s, ctx, parent, actor=Actor.USER, recurrence={"type": "monthly", "day": 1}
        )
        assert (await ts.require_task(s, parent)).recurrence == {"type": "monthly", "day": 1}
        await ts.update_task(s, ctx, parent, actor=Actor.USER, recurrence=None)
        assert (await ts.require_task(s, parent)).recurrence is None
