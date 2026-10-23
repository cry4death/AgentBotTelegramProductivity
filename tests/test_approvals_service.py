"""ApprovalService: propose/approve/reject/expire, batches, double presses."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import func, select

from procbot.db.models import PendingApproval, Task
from procbot.domain.actions import CreateTask, TaskDraft
from procbot.domain.enums import Actor, ApprovalStatus
from procbot.services import actions, approvals
from procbot.services import tasks as ts
from procbot.services.jobs import registered_jobs


@pytest.fixture
def wired(ctx):
    ts.register(ctx)
    approvals.register(ctx)
    return ctx


async def _count_tasks(ctx) -> int:
    async with ctx.db.transaction() as s:
        return (await s.execute(select(func.count(Task.id)))).scalar_one()


async def _propose(ctx, title: str, op: str, group: str | None = None) -> int:
    async with ctx.db.transaction() as s:
        res = await actions.execute(
            s,
            ctx,
            CreateTask(task=TaskDraft(title=title)),
            origin=Actor.AI,
            op_key=op,
            group_key=group,
        )
    assert res.approval_id is not None
    return res.approval_id


async def test_propose_idempotent_and_payload_roundtrip(wired, profile) -> None:
    ctx = wired
    draft = TaskDraft(title="Отчёт", done_criteria="Отправлен", due_date=date(2026, 10, 30))
    async with ctx.db.transaction() as s:
        a = await approvals.propose(
            s, ctx, CreateTask(task=draft), origin=Actor.AI, op_key="op", diff={"summary": "x"}
        )
        b = await approvals.propose(
            s, ctx, CreateTask(task=draft), origin=Actor.AI, op_key="op", diff={"summary": "x"}
        )
        assert a.id == b.id
        assert a.expires_at == ctx.clock.now() + timedelta(hours=24)
        parsed = actions.parse_action(a.action_payload)
    assert parsed.task.title == "Отчёт" and parsed.task.due_date == date(2026, 10, 30)
    # verification was not set explicitly -> criteria default still applies after roundtrip
    assert "verification" not in parsed.task.model_fields_set
    async with ctx.db.transaction() as s:
        res = await approvals.approve(s, ctx, a.id)
        task = await ts.require_task(s, res.approval.result["task_id"])  # type: ignore[index]
    assert task.verification == "criteria"


async def test_double_approve_creates_one_task(wired, profile) -> None:
    ctx = wired
    aid = await _propose(ctx, "Один раз", "op1")
    async with ctx.db.transaction() as s:
        r1 = await approvals.approve(s, ctx, aid)
    async with ctx.db.transaction() as s:
        r2 = await approvals.approve(s, ctx, aid)
        r3 = await approvals.reject(s, ctx, aid)
    assert r1.changed and r1.status is ApprovalStatus.APPLIED
    assert not r2.changed and r2.status is ApprovalStatus.APPLIED
    assert not r3.changed and r3.status is ApprovalStatus.APPLIED
    assert await _count_tasks(ctx) == 1


async def test_reject_creates_nothing_and_cannot_be_applied_later(wired, profile) -> None:
    ctx = wired
    aid = await _propose(ctx, "Нет", "op2")
    async with ctx.db.transaction() as s:
        await approvals.reject(s, ctx, aid)
        r = await approvals.approve(s, ctx, aid)
    assert r.status is ApprovalStatus.REJECTED and not r.changed
    assert await _count_tasks(ctx) == 0


async def test_expired_card_cannot_be_applied(wired, profile, clock) -> None:
    ctx = wired
    a1 = await _propose(ctx, "Истечёт", "op3")
    a2 = await _propose(ctx, "Тоже", "op4")
    clock.advance(hours=25)
    job = next(j for j in registered_jobs(ctx) if j.name == approvals.EXPIRE_JOB)
    await job.run(ctx)
    async with ctx.db.transaction() as s:
        assert (await approvals.get_approval(s, a1)).status == ApprovalStatus.EXPIRED  # type: ignore[union-attr]
        r = await approvals.approve(s, ctx, a1)
    assert r.status is ApprovalStatus.EXPIRED and not r.changed
    # a2 expired by the job too; approving before the job would also refuse
    async with ctx.db.transaction() as s:
        r2 = await approvals.approve(s, ctx, a2)
    assert r2.status is ApprovalStatus.EXPIRED
    assert await _count_tasks(ctx) == 0


async def test_expiry_checked_on_press_even_without_job(wired, profile, clock) -> None:
    ctx = wired
    aid = await _propose(ctx, "Без джоба", "op5")
    clock.advance(hours=24, seconds=1)
    async with ctx.db.transaction() as s:
        r = await approvals.approve(s, ctx, aid)
    assert r.status is ApprovalStatus.EXPIRED and r.changed
    assert await _count_tasks(ctx) == 0


async def test_group_accept_all_and_per_item(wired, profile) -> None:
    ctx = wired
    ids = [await _propose(ctx, f"T{i}", f"g:{i}", group="g") for i in range(3)]
    async with ctx.db.transaction() as s:
        await approvals.reject(s, ctx, ids[0])
        res = await approvals.approve_group(s, ctx, ids[1])
        again = await approvals.approve_group(s, ctx, ids[1])
    assert [r.status for r in res] == ["rejected", "applied", "applied"]
    assert [r.changed for r in res] == [False, True, True]
    assert not any(r.changed for r in again)
    assert approvals.summarize(res) == "Применено: 2."
    async with ctx.db.transaction() as s:
        titles = sorted(t.title for t in (await s.execute(select(Task))).scalars())
    assert titles == ["T1", "T2"]


async def test_unsupported_payload_marks_failed(wired, profile) -> None:
    ctx = wired
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        row = PendingApproval(
            op_key="raw",
            action_type="create_goal",
            action_payload={"type": "create_goal", "title": "Цель"},
            diff={},
            status=ApprovalStatus.PENDING,
            origin="ai",
            created_at=now,
        )
        s.add(row)
        await s.flush()
        res = await approvals.approve(s, ctx, row.id)
    assert res.status is ApprovalStatus.FAILED and "не поддерживается" in res.message


async def test_apply_time_validation_failure_is_failed_not_applied(wired, profile) -> None:
    ctx = wired
    async with ctx.db.transaction() as s:
        parent = await ts.create_task(s, ctx, TaskDraft(title="P"), actor=Actor.USER)
        res = await actions.execute(
            s,
            ctx,
            CreateTask(task=TaskDraft(title="child", parent_id=parent.id)),
            origin=Actor.AI,
            op_key="c",
        )
        await ts.complete_task(s, ctx, parent.id, actor=Actor.USER)
        r = await approvals.approve(s, ctx, res.approval_id)  # type: ignore[arg-type]
    assert r.status is ApprovalStatus.FAILED
    assert await _count_tasks(ctx) == 1


async def test_pending_card_survives_restart(wired, profile) -> None:
    from procbot.db.session import Database

    ctx = wired
    aid = await _propose(ctx, "После рестарта", "op-restart")
    await ctx.db.dispose()
    ctx.db = Database.for_path(ctx.env.database_path)
    async with ctx.db.transaction() as s:
        res = await approvals.approve(s, ctx, aid)
    assert res.status is ApprovalStatus.APPLIED
    assert await _count_tasks(ctx) == 1
