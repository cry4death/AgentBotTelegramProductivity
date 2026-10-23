"""ActionExecutor: FR-15 permission matrix, constraints, idempotency, audit."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import func, select

from procbot.db.models import AuditLog, PendingApproval, Task
from procbot.domain.actions import (
    ActionType,
    AdjustReminders,
    ChooseIntervention,
    CreateSubtasks,
    CreateTask,
    DeleteCalendarEvent,
    ReorderTasks,
    RescheduleTask,
    SubtaskDraft,
    TaskDraft,
    UpdateCalendarEvent,
)
from procbot.domain.enums import Actor, TaskStatus
from procbot.services import actions, approvals
from procbot.services import tasks as ts
from procbot.services.actions import ApplyOutcome, ExecStatus


@pytest.fixture
def wired(ctx):
    ts.register(ctx)
    approvals.register(ctx)
    return ctx


async def _task(ctx, **kw) -> Task:
    async with ctx.db.transaction() as s:
        return await ts.create_task(s, ctx, TaskDraft(**kw), actor=Actor.USER)


async def _count(ctx, model) -> int:
    async with ctx.db.transaction() as s:
        return (await s.execute(select(func.count()).select_from(model))).scalar_one()


async def _audit_actions(ctx) -> list[str]:
    async with ctx.db.transaction() as s:
        return [a.action for a in (await s.execute(select(AuditLog))).scalars()]


async def test_needs_approval_from_ai_creates_card_not_task(wired, profile) -> None:
    ctx = wired
    async with ctx.db.transaction() as s:
        res = await actions.execute(
            s, ctx, CreateTask(task=TaskDraft(title="Новая")), origin=Actor.AI, op_key="k1"
        )
    assert res.status is ExecStatus.PENDING_APPROVAL and res.approval_id
    assert await _count(ctx, Task) == 0
    assert await _count(ctx, PendingApproval) == 1
    assert "action.pending" in await _audit_actions(ctx)


async def test_user_origin_applies_directly_and_op_key_is_idempotent(wired, profile) -> None:
    ctx = wired
    act = CreateTask(task=TaskDraft(title="Сам решил"))
    async with ctx.db.transaction() as s:
        r1 = await actions.execute(s, ctx, act, origin=Actor.USER, op_key="u1:add")
        r2 = await actions.execute(s, ctx, act, origin=Actor.USER, op_key="u1:add")
    assert r1.status is ExecStatus.APPLIED and not r1.repeated
    assert r2.status is ExecStatus.APPLIED and r2.repeated
    assert r2.result["task_id"] == r1.result["task_id"]
    assert await _count(ctx, Task) == 1


async def test_auto_subtasks_under_parent_keep_parent_intact(wired, profile) -> None:
    ctx = wired
    parent = await _task(ctx, title="Сайт", due_date=date(2026, 11, 1), due_is_hard=True)
    async with ctx.db.transaction() as s:
        res = await actions.execute(
            s,
            ctx,
            CreateSubtasks(parent_id=parent.id, subtasks=[SubtaskDraft(title="Открыть макет")]),
            origin=Actor.AI,
            op_key="ai:sub",
        )
    assert res.status is ExecStatus.APPLIED
    async with ctx.db.transaction() as s:
        p = await ts.require_task(s, parent.id)
        subs = await ts.list_subtasks(s, parent.id)
        events = [e.actor for e in await ts.task_events(s, subs[0].id)]
    assert (p.title, p.due_date, p.due_is_hard, p.status) == (
        "Сайт",
        date(2026, 11, 1),
        True,
        TaskStatus.TODO,
    )
    assert [x.title for x in subs] == ["Открыть макет"] and events == ["ai"]
    assert await _count(ctx, PendingApproval) == 0


async def test_auto_subtasks_under_done_parent_rejected(wired, profile) -> None:
    ctx = wired
    parent = await _task(ctx, title="Готово уже")
    async with ctx.db.transaction() as s:
        await ts.complete_task(s, ctx, parent.id, actor=Actor.USER)
    async with ctx.db.transaction() as s:
        res = await actions.execute(
            s,
            ctx,
            CreateSubtasks(parent_id=parent.id, subtasks=[SubtaskDraft(title="ещё")]),
            origin=Actor.AI,
            op_key="ai:sub2",
        )
        missing = await actions.execute(
            s,
            ctx,
            CreateSubtasks(parent_id=999, subtasks=[SubtaskDraft(title="ещё")]),
            origin=Actor.AI,
            op_key="ai:sub3",
        )
    assert res.status is ExecStatus.INVALID and "закрыта" in (res.error or "")
    assert missing.status is ExecStatus.INVALID
    assert await _count(ctx, Task) == 1
    assert "action.invalid" in await _audit_actions(ctx)


async def test_auto_reorder_and_adjust_reminders(wired, profile) -> None:
    ctx = wired
    due = datetime(2026, 10, 27, 9, 0, tzinfo=UTC)
    a = await _task(ctx, title="A", due_at=due)
    b = await _task(ctx, title="B", planned_start_at=due)
    async with ctx.db.transaction() as s:
        r1 = await actions.execute(
            s, ctx, ReorderTasks(task_ids=[b.id, a.id]), origin=Actor.AI, op_key="ai:ro"
        )
        r2 = await actions.execute(
            s,
            ctx,
            AdjustReminders(task_id=b.id, pre_task_lead_min=15),
            origin=Actor.AI,
            op_key="ai:rem",
        )
    assert r1.status is ExecStatus.APPLIED and r2.status is ExecStatus.APPLIED
    async with ctx.db.transaction() as s:
        ra, rb = await ts.require_task(s, a.id), await ts.require_task(s, b.id)
    assert rb.sort_order < ra.sort_order
    assert ra.due_at == due and rb.planned_start_at == due
    assert rb.reminder_overrides == {"pre_task_lead_min": 15}


async def test_forbidden_calendar_changes_dropped_even_for_user(wired, profile) -> None:
    ctx = wired
    async with ctx.db.transaction() as s:
        r1 = await actions.execute(
            s, ctx, UpdateCalendarEvent(external_event_id="e1"), origin=Actor.AI, op_key="f1"
        )
        r2 = await actions.execute(
            s, ctx, DeleteCalendarEvent(external_event_id="e1"), origin=Actor.USER, op_key="f2"
        )
    assert r1.status is ExecStatus.FORBIDDEN and r2.status is ExecStatus.FORBIDDEN
    assert await _count(ctx, PendingApproval) == 0
    assert (await _audit_actions(ctx)).count("action.forbidden") == 2


async def test_unsupported_and_invalid_payload(wired, profile) -> None:
    ctx = wired
    async with ctx.db.transaction() as s:
        r1 = await actions.execute(
            s,
            ctx,
            ChooseIntervention(kind="tiny_step", suggestion="открыть файл"),
            origin=Actor.AI,
            op_key="x1",
        )
        r2 = await actions.execute(
            s, ctx, {"type": "create_task", "task": {"title": ""}}, origin=Actor.AI, op_key="x2"
        )
        r3 = await actions.execute(
            s,
            ctx,
            {"type": "create_task", "task": {"title": "Из словаря"}},
            origin=Actor.AI,
            op_key="x3",
        )
    assert r1.status is ExecStatus.UNSUPPORTED
    assert r2.status is ExecStatus.INVALID
    assert r3.status is ExecStatus.PENDING_APPROVAL


async def test_reschedule_from_ai_needs_approval_user_applies(wired, profile) -> None:
    """Acceptance 7."""
    ctx = wired
    t = await _task(ctx, title="Утверждённая", planned_date=date(2026, 10, 26))
    move = RescheduleTask(task_id=t.id, planned_date=date(2026, 10, 30), reason="перегруз")
    async with ctx.db.transaction() as s:
        r_ai = await actions.execute(s, ctx, move, origin=Actor.AI, op_key="ai:mv")
        r_sys = await actions.execute(s, ctx, move, origin=Actor.SYSTEM, op_key="sys:mv")
    assert r_ai.status is ExecStatus.PENDING_APPROVAL
    assert r_sys.status is ExecStatus.PENDING_APPROVAL
    async with ctx.db.transaction() as s:
        assert (await ts.require_task(s, t.id)).planned_date == date(2026, 10, 26)
    async with ctx.db.transaction() as s:
        res = await approvals.approve(s, ctx, r_ai.approval_id)  # type: ignore[arg-type]
    assert res.status == "applied"
    async with ctx.db.transaction() as s:
        task = await ts.require_task(s, t.id)
        ev = (await ts.task_events(s, t.id))[-1]
    assert task.planned_date == date(2026, 10, 30)
    assert ev.event_type == "rescheduled" and ev.actor == "user" and ev.reason == "перегруз"
    # explicit user command applies directly
    async with ctx.db.transaction() as s:
        r_user = await actions.execute(
            s,
            ctx,
            RescheduleTask(task_id=t.id, planned_date=date(2026, 10, 31)),
            origin=Actor.USER,
            op_key="u:mv",
        )
    assert r_user.status is ExecStatus.APPLIED


async def test_after_commit_hook_returned_not_run(wired, profile) -> None:
    ctx = wired
    ran: list[str] = []

    async def hook() -> None:
        ran.append("x")

    async def applier(session, ctx, action, actor) -> ApplyOutcome:
        return ApplyOutcome({"ok": True}, after_commit=hook)

    actions.register_applier(ctx, ActionType.SAVE_HYPOTHESIS, applier)
    async with ctx.db.transaction() as s:
        res = await actions.execute(
            s,
            ctx,
            {"type": "save_hypothesis", "content": "по утрам проще"},
            origin=Actor.AI,
            op_key="h1",
        )
        assert ran == []
    assert await actions.run_after_commit(res.after_commit) == []
    assert ran == ["x"]


async def test_failed_applier_rolls_back_its_writes(wired, profile) -> None:
    ctx = wired

    async def applier(session, ctx, action, actor) -> ApplyOutcome:
        await ts.create_task(session, ctx, TaskDraft(title="half"), actor=actor)
        raise actions.ActionError("boom")

    actions.register_applier(ctx, ActionType.SAVE_HYPOTHESIS, applier)
    async with ctx.db.transaction() as s:
        res = await actions.execute(
            s, ctx, {"type": "save_hypothesis", "content": "x"}, origin=Actor.AI, op_key="h2"
        )
        await ts.create_task(s, ctx, TaskDraft(title="kept"), actor=Actor.USER)
    assert res.status is ExecStatus.INVALID
    async with ctx.db.transaction() as s:
        titles = [t.title for t in (await s.execute(select(Task))).scalars()]
    assert titles == ["kept"]
