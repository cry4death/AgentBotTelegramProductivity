"""TaskService: creation, planning, postponing, statuses, subtasks, lists, journal."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import func, select

from procbot.db.models import AuditLog, Category, Task, TaskEvent
from procbot.db.session import Database
from procbot.domain.actions import SubtaskDraft, TaskDraft
from procbot.domain.enums import Actor, ObstacleReason, SessionOutcome, TaskEventType, TaskStatus
from procbot.services import tasks as ts
from procbot.services.dayview import build_day_view, render_day_view


async def _events(ctx, task_id: int) -> list[TaskEvent]:
    async with ctx.db.transaction() as s:
        return await ts.task_events(s, task_id)


async def _create(ctx, **kw) -> Task:
    async with ctx.db.transaction() as s:
        return await ts.create_task(s, ctx, TaskDraft(**kw), actor=Actor.USER)


async def test_create_resolves_category_context_and_journals(ctx, profile) -> None:
    t = await _create(ctx, title="  Купить   хлеб ", category="Дом", context="Улица")
    t2 = await _create(ctx, title="Молоко", category="дом")
    assert t.title == "Купить хлеб"
    assert t.category_id == t2.category_id is not None
    async with ctx.db.transaction() as s:
        assert (await s.execute(select(func.count(Category.id)))).scalar_one() == 1
    ev = await _events(ctx, t.id)
    assert [(e.event_type, e.actor) for e in ev] == [("created", "user")]
    assert t2.sort_order > t.sort_order


async def test_verification_default_follows_done_criteria(ctx, profile) -> None:
    plain = await _create(ctx, title="Помыть посуду")
    crit = await _create(ctx, title="Отчёт", done_criteria="Отправлен руководителю")
    explicit = await _create(ctx, title="X", done_criteria="Y", verification="button")
    assert plain.verification == "button" and not ts.requires_criteria_check(plain)
    assert crit.verification == "criteria" and ts.requires_criteria_check(crit)
    assert explicit.verification == "button"
    async with ctx.db.transaction() as s:
        t = await ts.update_task(s, ctx, plain.id, actor=Actor.USER, done_criteria="Сухо")
        assert t.verification == "criteria"
        t = await ts.update_task(s, ctx, plain.id, actor=Actor.USER, verification="button")
        assert t.verification == "button"


async def test_plan_reschedule_postpone_events_keep_old_and_new(ctx, profile) -> None:
    t = await _create(ctx, title="Сайт")
    start = datetime(2026, 10, 26, 13, 0, tzinfo=UTC)  # 14:00 Berlin
    async with ctx.db.transaction() as s:
        await ts.plan_task(s, ctx, t.id, actor=Actor.USER, planned_start_at=start)
        await ts.reschedule_task(
            s, ctx, t.id, actor=Actor.USER, planned_start_at=start + timedelta(hours=2)
        )
        task, ev = await ts.postpone_task(
            s, ctx, t.id, actor=Actor.USER, planned_date=date(2026, 10, 27), reason="устал"
        )
        await ts.report_obstacle(
            s,
            ctx,
            t.id,
            reason=ObstacleReason.LOW_ENERGY,
            actor=Actor.USER,
            related_event_id=ev.id,
        )
        assert task.postpone_count == 1
        assert task.planned_date == date(2026, 10, 27) and task.planned_start_at is None
    evs = await _events(ctx, t.id)
    kinds = [e.event_type for e in evs]
    assert kinds == ["created", "updated", "rescheduled", "postponed", "obstacle_reported"]
    resched = evs[2]
    assert resched.payload["old"]["planned_start_at"] == start.isoformat()
    assert resched.payload["new"]["planned_start_at"] == (start + timedelta(hours=2)).isoformat()
    assert evs[3].reason == "устал" and evs[3].payload["new"]["planned_date"] == "2026-10-27"
    assert evs[4].reason == "low_energy" and evs[4].payload["event_id"] == evs[3].id


async def test_status_transitions_partial_is_never_done(ctx, profile) -> None:
    t = await _create(ctx, title="Глава диплома", done_criteria="10 страниц")
    async with ctx.db.transaction() as s:
        p = await ts.record_progress(s, ctx, t.id, actor=Actor.USER, note="5 страниц")
        assert p.status == TaskStatus.IN_PROGRESS and p.completed_at is None
        assert p.progress_note == "5 страниц"
        d = await ts.complete_task(s, ctx, t.id, actor=Actor.USER, criteria_confirmed=True)
        assert d.status == TaskStatus.DONE and d.completed_at == ctx.clock.now()
        await ts.complete_task(s, ctx, t.id, actor=Actor.USER)  # no-op
        r = await ts.reopen_task(s, ctx, t.id, actor=Actor.USER)
        assert r.status == TaskStatus.IN_PROGRESS  # had progress
        c = await ts.cancel_task(s, ctx, t.id, actor=Actor.USER, reason="не нужно")
        assert c.status == TaskStatus.CANCELLED
        with pytest.raises(ts.TaskError):
            await ts.plan_task(s, ctx, t.id, actor=Actor.USER, planned_date=date(2026, 10, 27))
    kinds = [e.event_type for e in await _events(ctx, t.id)]
    assert kinds == ["created", "progressed", "completed", "reopened", "cancelled"]


async def test_someday_and_backlog(ctx, profile) -> None:
    t = await _create(ctx, title="Выучить испанский", planned_date=date(2026, 10, 26))
    async with ctx.db.transaction() as s:
        await ts.defer_to_someday(s, ctx, t.id, actor=Actor.USER)
        assert [x.id for x in await ts.list_tasks(s, ctx, ts.TaskView.SOMEDAY)] == [t.id]
        await ts.move_to_backlog(s, ctx, t.id, actor=Actor.USER)
        task = await ts.require_task(s, t.id)
        assert task.status == TaskStatus.TODO and task.is_backlog
        assert [x.id for x in await ts.list_tasks(s, ctx, ts.TaskView.BACKLOG)] == [t.id]


async def test_subtasks_create_and_replace_keeps_done(ctx, profile) -> None:
    parent = await _create(ctx, title="Сайт", due_date=date(2026, 11, 1))
    async with ctx.db.transaction() as s:
        a, b = await ts.create_subtasks(
            s,
            ctx,
            parent.id,
            [SubtaskDraft(title="Макет"), SubtaskDraft(title="Вёрстка")],
            actor=Actor.AI,
        )
        await ts.complete_task(s, ctx, a.id, actor=Actor.USER)
        _created, removed = await ts.replace_open_subtasks(
            s,
            ctx,
            parent.id,
            [SubtaskDraft(title="Шапка"), SubtaskDraft(title="Подвал")],
            actor=Actor.AI,
        )
        assert [r.id for r in removed] == [b.id]
        subs = await ts.list_subtasks(s, parent.id)
        assert [x.title for x in subs] == ["Макет", "Шапка", "Подвал"]
        assert (await ts.next_open_subtask(s, parent.id)).title == "Шапка"  # type: ignore[union-attr]
        p = await ts.require_task(s, parent.id)
        assert p.title == "Сайт" and p.due_date == date(2026, 11, 1)
        with pytest.raises(ts.TaskError):  # only one level of subtasks
            await ts.create_subtasks(s, ctx, a.id, [SubtaskDraft(title="x")], actor=Actor.AI)
    pev = [e.event_type for e in await _events(ctx, parent.id)]
    assert pev == ["created", "subtasks_changed", "subtasks_changed"]


async def test_reorder_changes_only_sort_order(ctx, profile) -> None:
    due = datetime(2026, 10, 27, 10, 0, tzinfo=UTC)
    a = await _create(ctx, title="A", due_at=due, due_is_hard=True)
    b = await _create(ctx, title="B", planned_start_at=due)
    c = await _create(ctx, title="C")
    async with ctx.db.transaction() as s:
        changed = await ts.reorder_tasks(s, ctx, [c.id, a.id], actor=Actor.AI)
        assert set(changed) == {a.id, c.id}
        rows = sorted(
            (await s.execute(select(Task).where(Task.parent_id.is_(None)))).scalars(),
            key=ts.sort_key,
        )
        assert [r.title for r in rows] == ["C", "B", "A"]
        ra = await ts.require_task(s, a.id)
        rb = await ts.require_task(s, b.id)
        assert ra.due_at == due and ra.due_is_hard and rb.planned_start_at == due


async def test_today_buckets_respect_owner_timezone(ctx, profile, clock) -> None:
    # now = 2026-10-26 09:00 UTC = 10:00 Berlin
    await _create(ctx, title="overdue", due_at=datetime(2026, 10, 26, 8, 0, tzinfo=UTC))
    await _create(ctx, title="late", planned_start_at=datetime(2026, 10, 26, 16, 0, tzinfo=UTC))
    await _create(ctx, title="early", planned_start_at=datetime(2026, 10, 26, 10, 0, tzinfo=UTC))
    await _create(ctx, title="untimed", planned_date=date(2026, 10, 26))
    await _create(ctx, title="due", due_date=date(2026, 10, 26))
    await _create(ctx, title="missed", planned_date=date(2026, 10, 25))
    # 23:30 UTC on the 26th is already the 27th in Berlin -> upcoming, not today
    await _create(
        ctx, title="tomorrow", planned_start_at=datetime(2026, 10, 26, 23, 30, tzinfo=UTC)
    )
    await _create(ctx, title="backlog")
    async with ctx.db.transaction() as s:
        b = await ts.today_buckets(s, ctx)
        upcoming = await ts.list_tasks(s, ctx, ts.TaskView.UPCOMING)
        view = await build_day_view(s, ctx)
    assert [t.title for t in b.overdue] == ["overdue"]
    assert [t.title for t in b.planned] == ["early", "late", "untimed"]
    assert [t.title for t in b.due_today] == ["due"]
    assert [t.title for t in b.missed] == ["missed"]
    assert b.backlog_count == 1
    assert [t.title for t in upcoming] == ["tomorrow"]
    text, kb = render_day_view(view)
    assert "Просрочено" in text and "11:00 early" in text and "17:00 late" in text
    assert "этапе B" in text
    assert kb.inline_keyboard[0][0].callback_data.startswith("ws:pick")  # type: ignore[union-attr]


async def test_done_list_recent_only(ctx, profile, clock) -> None:
    t = await _create(ctx, title="old")
    async with ctx.db.transaction() as s:
        await ts.complete_task(s, ctx, t.id, actor=Actor.USER)
    clock.advance(days=10)
    t2 = await _create(ctx, title="new")
    async with ctx.db.transaction() as s:
        await ts.complete_task(s, ctx, t2.id, actor=Actor.USER)
        done = await ts.list_tasks(s, ctx, ts.TaskView.DONE)
    assert [x.title for x in done] == ["new"]


async def test_record_work_outcome_rules_and_idempotency(ctx, profile) -> None:
    t = await _create(ctx, title="Код")
    async with ctx.db.transaction() as s:
        r = await ts.record_work_outcome(
            s, ctx, task_id=t.id, outcome=SessionOutcome.STARTED, actor=Actor.USER, session_id=7
        )
        assert r.status == TaskStatus.IN_PROGRESS
        await ts.record_work_outcome(
            s, ctx, task_id=t.id, outcome=SessionOutcome.STARTED, actor=Actor.USER, session_id=7
        )
        r = await ts.record_work_outcome(
            s,
            ctx,
            task_id=t.id,
            outcome=SessionOutcome.PROGRESSED,
            actor=Actor.USER,
            session_id=7,
            note="половина",
        )
        assert r.status == TaskStatus.IN_PROGRESS and r.progress_note == "половина"
        r = await ts.record_work_outcome(
            s, ctx, task_id=t.id, outcome=SessionOutcome.COMPLETED, actor=Actor.USER, session_id=8
        )
        assert r.status == TaskStatus.DONE and r.completed_at is not None
        await ts.record_work_outcome(
            s, ctx, task_id=t.id, outcome=SessionOutcome.COMPLETED, actor=Actor.USER, session_id=8
        )
    kinds = [e.event_type for e in await _events(ctx, t.id)]
    assert kinds == ["created", "started", "progressed", "completed"]


async def test_delete_keeps_audit_snapshot(ctx, profile) -> None:
    t = await _create(ctx, title="Удалить меня")
    async with ctx.db.transaction() as s:
        await ts.create_subtasks(s, ctx, t.id, [SubtaskDraft(title="шаг")], actor=Actor.USER)
    async with ctx.db.transaction() as s:
        await ts.delete_task(s, ctx, t.id, actor=Actor.USER)
    async with ctx.db.transaction() as s:
        assert (await s.execute(select(func.count(Task.id)))).scalar_one() == 0
        log = (
            await s.execute(select(AuditLog).where(AuditLog.action == "task.deleted"))
        ).scalar_one()
        assert log.details["task"]["title"] == "Удалить меня"
        assert len(log.details["subtasks"]) == 1


async def test_restart_persistence(ctx, profile, tmp_path) -> None:
    t = await _create(ctx, title="Пережить рестарт", planned_date=date(2026, 10, 27))
    async with ctx.db.transaction() as s:
        await ts.postpone_task(s, ctx, t.id, actor=Actor.USER, to_backlog=True, reason="x")
    await ctx.db.dispose()
    ctx.db = Database.for_path(ctx.env.database_path)
    async with ctx.db.transaction() as s:
        task = await ts.require_task(s, t.id)
        assert task.title == "Пережить рестарт" and task.is_backlog and task.postpone_count == 1
        kinds = [e.event_type for e in await ts.task_events(s, t.id)]
    assert kinds == ["created", "updated", "postponed"] or kinds == ["created", "postponed"]
    assert TaskEventType.POSTPONED in kinds
