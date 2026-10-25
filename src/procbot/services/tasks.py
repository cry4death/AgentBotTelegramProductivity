"""TaskService: tasks and subtasks, statuses, planning, postponements, journal.

Owner of the `tasks` and `task_events` tables. Rules:
- every change writes an append-only `task_events` row with its actor;
- after a change of planned time or status, reminders are re-synced
  (`services.reminders.sync_task_reminders`);
- a partial result is never "done" (FR-06, acceptance 17): PROGRESSED keeps the
  task `in_progress` and stores the note;
- which tasks are "important" is open question 4: no scale is invented. The user
  picks the verification mode per task: "button" (one tap closes) or "criteria"
  (the bot shows the definition of done and asks full / partial / no). Writing a
  definition of done switches the default to "criteria" unless the user chose
  the mode explicitly.
- ordering hint: `sort_order`, then `priority` where a LOWER number means more
  urgent (P1 before P2; no fixed scale), then creation order.

Also registers the appliers of all task actions in the ActionExecutor and offers
`propose_task_candidates`, the confirmation flow for tasks found in free text
(stage A splits lines; stage B swaps in AI extraction, same flow).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.models import (
    Category,
    Context,
    Goal,
    Milestone,
    PendingApproval,
    Project,
    Task,
    TaskEvent,
)
from procbot.domain.actions import (
    ActionType,
    AdjustReminders,
    ChangeDeadline,
    ChangeTaskStatus,
    CreateSubtasks,
    CreateTask,
    ReorderTasks,
    ReplaceSubtasks,
    RescheduleTask,
    SubtaskDraft,
    TaskDraft,
    UpdateTask,
)
from procbot.domain.enums import (
    Actor,
    EstimateSource,
    ObstacleReason,
    SessionOutcome,
    TaskEventType,
    TaskStatus,
)
from procbot.services import actions, approvals
from procbot.services.actions import ActionError, ApplyOutcome
from procbot.services.audit import audit
from procbot.services.profile import get_profile
from procbot.services.recurrence import RecurrenceError, next_date, parse_rule
from procbot.services.reminders import sync_task_reminders
from procbot.services.structure import get_or_create_category, get_or_create_context
from procbot.timeutil import fmt_local, local_date, local_day_bounds, resolve_local, to_local

OPEN_STATUSES: frozenset[str] = frozenset({TaskStatus.TODO, TaskStatus.IN_PROGRESS})
CLOSED_STATUSES: frozenset[str] = frozenset({TaskStatus.DONE, TaskStatus.CANCELLED})
VERIFICATION_MODES: frozenset[str] = frozenset({"button", "criteria"})
MAX_CANDIDATES = 15
RECENT_DONE_DAYS = 7

STATUS_LABELS: dict[str, str] = {
    TaskStatus.TODO: "к выполнению",
    TaskStatus.IN_PROGRESS: "в работе",
    TaskStatus.DONE: "готово",
    TaskStatus.SOMEDAY: "когда-нибудь",
    TaskStatus.CANCELLED: "отменена",
}


class TaskError(ActionError):
    """Task operation not possible; str() is a short Russian message."""


class TaskNotFound(TaskError):
    def __init__(self, task_id: int) -> None:
        super().__init__(f"Задача #{task_id} не найдена.")
        self.task_id = task_id


class TaskView(StrEnum):
    TODAY = "today"
    UPCOMING = "upcoming"
    BACKLOG = "backlog"
    SOMEDAY = "someday"
    DONE = "done"


# --------------------------------------------------------------------------- basics


async def get_task(session: AsyncSession, task_id: int) -> Task | None:
    return await session.get(Task, task_id)


async def require_task(session: AsyncSession, task_id: int) -> Task:
    task = await session.get(Task, task_id)
    if task is None:
        raise TaskNotFound(task_id)
    return task


async def owner_tz(session: AsyncSession) -> str:
    """Owner IANA timezone; UTC until onboarding set one."""
    profile = await get_profile(session)
    return (profile.timezone if profile is not None else None) or "UTC"


async def local_today(session: AsyncSession, ctx: AppContext) -> date:
    return local_date(ctx.clock.now(), await owner_tz(session))


def is_open(task: Task) -> bool:
    return task.status in OPEN_STATUSES


def requires_criteria_check(task: Task) -> bool:
    """True when "done" must be compared with the definition of done first."""
    return task.verification == "criteria"


def sort_key(task: Task) -> tuple[int, int, int]:
    prio = task.priority if task.priority is not None else 1 << 30
    return (task.sort_order, prio, task.id)


def _jsonable(v: Any) -> Any:
    if isinstance(v, datetime | date | time):
        return v.isoformat()
    return v


def _schedule_state(task: Task) -> dict[str, Any]:
    return {
        "planned_date": _jsonable(task.planned_date),
        "planned_start_at": _jsonable(task.planned_start_at),
        "is_backlog": task.is_backlog,
        "status": task.status,
    }


def _due_state(task: Task) -> dict[str, Any]:
    return {
        "due_at": _jsonable(task.due_at),
        "due_date": _jsonable(task.due_date),
        "due_is_hard": task.due_is_hard,
    }


def snapshot(task: Task) -> dict[str, Any]:
    keys = [
        "id",
        "title",
        "description",
        "parent_id",
        "goal_id",
        "project_id",
        "milestone_id",
        "category_id",
        "context_id",
        "status",
        "due_at",
        "due_date",
        "due_is_hard",
        "planned_start_at",
        "planned_date",
        "estimate_minutes",
        "priority",
        "importance",
        "done_criteria",
        "verification",
        "is_backlog",
        "sort_order",
        "postpone_count",
        "progress_note",
        "created_by",
    ]
    return {k: _jsonable(getattr(task, k)) for k in keys}


async def _event(
    session: AsyncSession,
    ctx: AppContext,
    task: Task,
    event_type: TaskEventType,
    actor: Actor,
    *,
    reason: str | None = None,
    payload: dict[str, Any] | None = None,
) -> TaskEvent:
    row = TaskEvent(
        task_id=task.id,
        event_type=str(event_type),
        actor=str(actor),
        reason=reason,
        payload=payload or {},
        created_at=ctx.clock.now(),
    )
    session.add(row)
    await session.flush()
    return row


async def _touch(session: AsyncSession, ctx: AppContext, task: Task, *, sync: bool) -> None:
    task.updated_at = ctx.clock.now()
    await session.flush()
    if sync:
        await sync_task_reminders(session, ctx, task)


async def task_events(session: AsyncSession, task_id: int) -> list[TaskEvent]:
    stmt = select(TaskEvent).where(TaskEvent.task_id == task_id).order_by(TaskEvent.id)
    return list((await session.execute(stmt)).scalars())


async def _next_sort_order(session: AsyncSession, parent_id: int | None) -> int:
    cond = Task.parent_id.is_(None) if parent_id is None else Task.parent_id == parent_id
    last = (await session.execute(select(func.max(Task.sort_order)).where(cond))).scalar()
    return (last or 0) + 1


def _estimate_source(actor: Actor) -> str:
    return EstimateSource.AI if actor == Actor.AI else EstimateSource.USER


def _clean_title(title: str) -> str:
    clean = " ".join(title.split()).strip()
    if not clean:
        raise TaskError("Пустое название задачи.")
    return clean[:500]


# --------------------------------------------------------------------------- create


async def create_task(
    session: AsyncSession,
    ctx: AppContext,
    draft: TaskDraft,
    *,
    actor: Actor,
    reason: str | None = None,
    event_payload: dict[str, Any] | None = None,
) -> Task:
    """Create a task (or a subtask when draft.parent_id is set)."""
    now = ctx.clock.now()
    tz = await owner_tz(session)
    if draft.parent_id is not None:
        parent = await require_task(session, draft.parent_id)
        _check_parent_open(parent)
    for model, ref in (
        (Goal, draft.goal_id),
        (Project, draft.project_id),
        (Milestone, draft.milestone_id),
    ):
        if ref is not None and await session.get(model, ref) is None:
            raise TaskError(f"Не найдено: {model.__tablename__} #{ref}.")
    category_id = (
        (await get_or_create_category(session, ctx, draft.category)).id if draft.category else None
    )
    context_id = (
        (await get_or_create_context(session, ctx, draft.context)).id if draft.context else None
    )
    verification = draft.verification
    if draft.done_criteria and "verification" not in draft.model_fields_set:
        verification = "criteria"
    planned_start = draft.planned_start_at
    planned_date = draft.planned_date
    if planned_start is not None:
        planned_date = local_date(planned_start, tz)
    task = Task(
        title=_clean_title(draft.title),
        description=draft.description,
        parent_id=draft.parent_id,
        goal_id=draft.goal_id,
        project_id=draft.project_id,
        milestone_id=draft.milestone_id,
        category_id=category_id,
        context_id=context_id,
        status=TaskStatus.TODO,
        due_at=draft.due_at,
        due_date=draft.due_date,
        due_is_hard=draft.due_is_hard,
        planned_start_at=planned_start,
        planned_date=planned_date,
        estimate_minutes=draft.estimate_minutes,
        estimate_source=_estimate_source(actor) if draft.estimate_minutes else None,
        priority=draft.priority,
        importance=draft.importance,
        done_criteria=draft.done_criteria,
        verification=verification,
        is_backlog=draft.is_backlog and planned_date is None,
        recurrence=_checked_rule(draft.recurrence) if draft.parent_id is None else None,
        sort_order=await _next_sort_order(session, draft.parent_id),
        postpone_count=0,
        reminder_overrides={},
        approved=True,
        created_by=str(actor),
        created_at=now,
        updated_at=now,
    )
    session.add(task)
    await session.flush()
    await _event(
        session,
        ctx,
        task,
        TaskEventType.CREATED,
        actor,
        reason=reason,
        payload={"task": snapshot(task), **(event_payload or {})},
    )
    await sync_task_reminders(session, ctx, task)
    return task


def _checked_rule(raw: Any) -> dict[str, Any] | None:
    try:
        return parse_rule(raw)
    except RecurrenceError as exc:
        raise TaskError(str(exc)) from exc


def _check_parent_open(parent: Task) -> None:
    if parent.parent_id is not None:
        raise TaskError("Подзадачи можно создавать только у основной задачи.")
    if parent.status in CLOSED_STATUSES:
        raise TaskError("Родительская задача уже закрыта — подзадачи не добавляются.")


# --------------------------------------------------------------------------- subtasks


async def list_subtasks(
    session: AsyncSession, parent_id: int, *, include_cancelled: bool = False
) -> list[Task]:
    stmt = select(Task).where(Task.parent_id == parent_id)
    if not include_cancelled:
        stmt = stmt.where(Task.status != TaskStatus.CANCELLED)
    rows = list((await session.execute(stmt)).scalars())
    return sorted(rows, key=sort_key)


async def create_subtasks(
    session: AsyncSession,
    ctx: AppContext,
    parent_id: int,
    drafts: Sequence[SubtaskDraft],
    *,
    actor: Actor,
    reason: str | None = None,
) -> list[Task]:
    """Add steps under an open top-level task. The parent itself is NOT changed
    (title, goal, deadline, scope stay as they are: FR-05, acceptance 8)."""
    parent = await require_task(session, parent_id)
    _check_parent_open(parent)
    if not drafts:
        raise TaskError("Нет подзадач для добавления.")
    created: list[Task] = []
    for d in drafts:
        created.append(
            await create_task(
                session,
                ctx,
                TaskDraft(
                    title=d.title,
                    estimate_minutes=d.estimate_minutes,
                    done_criteria=d.done_criteria,
                    parent_id=parent.id,
                ),
                actor=actor,
                reason=reason,
            )
        )
    await _event(
        session,
        ctx,
        parent,
        TaskEventType.SUBTASKS_CHANGED,
        actor,
        reason=reason,
        payload={"added": [t.id for t in created], "removed": []},
    )
    return created


async def replace_open_subtasks(
    session: AsyncSession,
    ctx: AppContext,
    parent_id: int,
    drafts: Sequence[SubtaskDraft],
    *,
    actor: Actor,
    reason: str | None = None,
) -> tuple[list[Task], list[Task]]:
    """Rebuild the open steps (finer or coarser). Done subtasks stay; replaced
    open ones are cancelled (kept for history), not deleted. Returns
    (created, replaced)."""
    parent = await require_task(session, parent_id)
    _check_parent_open(parent)
    if not drafts:
        raise TaskError("Нет подзадач для замены.")
    removed: list[Task] = []
    for sub in await list_subtasks(session, parent.id):
        if sub.status in (TaskStatus.TODO, TaskStatus.IN_PROGRESS, TaskStatus.SOMEDAY):
            sub.status = TaskStatus.CANCELLED
            await _event(
                session,
                ctx,
                sub,
                TaskEventType.CANCELLED,
                actor,
                reason=reason or "replaced",
                payload={"replaced": True},
            )
            await _touch(session, ctx, sub, sync=True)
            removed.append(sub)
    created: list[Task] = []
    for d in drafts:
        created.append(
            await create_task(
                session,
                ctx,
                TaskDraft(
                    title=d.title,
                    estimate_minutes=d.estimate_minutes,
                    done_criteria=d.done_criteria,
                    parent_id=parent.id,
                ),
                actor=actor,
                reason=reason,
            )
        )
    await _event(
        session,
        ctx,
        parent,
        TaskEventType.SUBTASKS_CHANGED,
        actor,
        reason=reason,
        payload={"added": [t.id for t in created], "removed": [t.id for t in removed]},
    )
    return created, removed


async def next_open_subtask(session: AsyncSession, parent_id: int) -> Task | None:
    for sub in await list_subtasks(session, parent_id):
        if is_open(sub):
            return sub
    return None


# --------------------------------------------------------------------------- fields


EDITABLE_FIELDS: frozenset[str] = frozenset(
    {
        "title",
        "description",
        "done_criteria",
        "verification",
        "estimate_minutes",
        "importance",
        "priority",
        "category",
        "context",
        "project_id",
        "goal_id",
        "milestone_id",
        "due_is_hard",
        "recurrence",
    }
)


async def update_task(
    session: AsyncSession,
    ctx: AppContext,
    task_id: int,
    *,
    actor: Actor,
    reason: str | None = None,
    **changes: Any,
) -> Task:
    """Change descriptive fields. Dates/status have their own functions."""
    unknown = set(changes) - EDITABLE_FIELDS
    if unknown:
        raise ValueError(f"not editable: {sorted(unknown)}")
    task = await require_task(session, task_id)
    before: dict[str, Any] = {}
    after: dict[str, Any] = {}

    def put(attr: str, value: Any) -> None:
        old = getattr(task, attr)
        if old != value:
            before[attr] = _jsonable(old)
            after[attr] = _jsonable(value)
            setattr(task, attr, value)

    for key, value in changes.items():
        if key == "title":
            put("title", _clean_title(str(value)))
        elif key == "category":
            put(
                "category_id",
                (await get_or_create_category(session, ctx, value)).id if value else None,
            )
        elif key == "context":
            put(
                "context_id",
                (await get_or_create_context(session, ctx, value)).id if value else None,
            )
        elif key == "verification":
            if value not in VERIFICATION_MODES:
                raise TaskError("Неизвестный способ проверки.")
            put("verification", value)
        elif key == "done_criteria":
            text = (str(value).strip() or None) if value is not None else None
            had = task.done_criteria
            put("done_criteria", text)
            if text and not had and "verification" not in changes:
                put("verification", "criteria")
        elif key == "recurrence":
            if task.parent_id is not None and value:
                raise TaskError("Повтор задаётся только у основной задачи.")
            put("recurrence", _checked_rule(value))
        elif key == "estimate_minutes":
            put("estimate_minutes", value)
            put("estimate_source", _estimate_source(actor) if value else None)
        else:
            put(key, value)
    if not after:
        return task
    await _event(
        session,
        ctx,
        task,
        TaskEventType.UPDATED,
        actor,
        reason=reason,
        payload={"before": before, "after": after},
    )
    await _touch(session, ctx, task, sync=False)
    return task


async def set_reminder_overrides(
    session: AsyncSession, ctx: AppContext, task_id: int, *, actor: Actor, **overrides: int | None
) -> Task:
    """Per-task reminder timing ({"pre_task_lead_min", "start_delay_min"});
    a None value clears that override."""
    allowed = {"pre_task_lead_min", "start_delay_min"}
    if set(overrides) - allowed:
        raise ValueError(f"unknown overrides: {sorted(set(overrides) - allowed)}")
    task = await require_task(session, task_id)
    if task.status in CLOSED_STATUSES:
        raise TaskError("Задача закрыта — напоминания не нужны.")
    before = dict(task.reminder_overrides or {})
    merged = dict(before)
    for k, v in overrides.items():
        if v is None:
            merged.pop(k, None)
        else:
            merged[k] = int(v)
    if merged == before:
        return task
    task.reminder_overrides = merged
    await _event(
        session,
        ctx,
        task,
        TaskEventType.UPDATED,
        actor,
        payload={"reminder_overrides": {"before": before, "after": merged}},
    )
    await _touch(session, ctx, task, sync=True)
    return task


# --------------------------------------------------------------------------- schedule


def _plan_from(
    tz: str, planned_date: date | None, planned_start_at: datetime | None
) -> tuple[date | None, datetime | None]:
    if planned_start_at is not None:
        return local_date(planned_start_at, tz), planned_start_at
    return planned_date, None


async def _apply_schedule(
    session: AsyncSession,
    ctx: AppContext,
    task: Task,
    *,
    planned_date: date | None,
    planned_start_at: datetime | None,
    to_backlog: bool,
    to_someday: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if task.status in CLOSED_STATUSES:
        raise TaskError("Задача уже закрыта — сначала верни её в работу.")
    tz = await owner_tz(session)
    old = _schedule_state(task)
    if to_someday:
        task.planned_date = None
        task.planned_start_at = None
        task.is_backlog = False
        task.status = TaskStatus.SOMEDAY
    elif to_backlog:
        task.planned_date = None
        task.planned_start_at = None
        task.is_backlog = True
        if task.status == TaskStatus.SOMEDAY:
            task.status = TaskStatus.TODO
    else:
        if planned_date is None and planned_start_at is None:
            raise TaskError("Не указано, на когда перенести.")
        task.planned_date, task.planned_start_at = _plan_from(tz, planned_date, planned_start_at)
        task.is_backlog = False
        if task.status == TaskStatus.SOMEDAY:
            task.status = TaskStatus.TODO
    return old, _schedule_state(task)


async def plan_task(
    session: AsyncSession,
    ctx: AppContext,
    task_id: int,
    *,
    actor: Actor,
    planned_date: date | None = None,
    planned_start_at: datetime | None = None,
    reason: str | None = None,
) -> Task:
    """Set (or move) the planned day / start time. A first plan is UPDATED, a
    move of an existing plan is RESCHEDULED (both keep old/new values)."""
    return await reschedule_task(
        session,
        ctx,
        task_id,
        actor=actor,
        planned_date=planned_date,
        planned_start_at=planned_start_at,
        reason=reason,
    )


async def reschedule_task(
    session: AsyncSession,
    ctx: AppContext,
    task_id: int,
    *,
    actor: Actor,
    planned_date: date | None = None,
    planned_start_at: datetime | None = None,
    to_backlog: bool = False,
    reason: str | None = None,
) -> Task:
    task = await require_task(session, task_id)
    had_plan = task.planned_date is not None or task.planned_start_at is not None
    old, new = await _apply_schedule(
        session,
        ctx,
        task,
        planned_date=planned_date,
        planned_start_at=planned_start_at,
        to_backlog=to_backlog,
    )
    if old == new:
        return task
    etype = TaskEventType.RESCHEDULED if had_plan else TaskEventType.UPDATED
    await _event(
        session,
        ctx,
        task,
        etype,
        actor,
        reason=reason,
        payload={"field": "plan", "old": old, "new": new},
    )
    await _touch(session, ctx, task, sync=True)
    return task


async def move_to_backlog(
    session: AsyncSession, ctx: AppContext, task_id: int, *, actor: Actor, reason: str | None = None
) -> Task:
    return await reschedule_task(session, ctx, task_id, actor=actor, to_backlog=True, reason=reason)


async def postpone_task(
    session: AsyncSession,
    ctx: AppContext,
    task_id: int,
    *,
    actor: Actor,
    planned_date: date | None = None,
    planned_start_at: datetime | None = None,
    to_backlog: bool = False,
    to_someday: bool = False,
    reason: str | None = None,
) -> tuple[Task, TaskEvent]:
    """The user consciously puts the task off (FR-04: a valid outcome, not a
    failure). Increments postpone_count; POSTPONED event keeps old/new values."""
    task = await require_task(session, task_id)
    old, new = await _apply_schedule(
        session,
        ctx,
        task,
        planned_date=planned_date,
        planned_start_at=planned_start_at,
        to_backlog=to_backlog,
        to_someday=to_someday,
    )
    task.postpone_count = (task.postpone_count or 0) + 1
    ev = await _event(
        session,
        ctx,
        task,
        TaskEventType.POSTPONED,
        actor,
        reason=reason,
        payload={"old": old, "new": new, "count": task.postpone_count},
    )
    await _touch(session, ctx, task, sync=True)
    return task, ev


async def report_obstacle(
    session: AsyncSession,
    ctx: AppContext,
    task_id: int,
    *,
    reason: ObstacleReason | str,
    actor: Actor,
    related_event_id: int | None = None,
    note: str | None = None,
) -> TaskEvent:
    """Record why the task was put off. The categories are hypotheses, not
    diagnoses (FR-04); the user can pick "other" or skip entirely."""
    task = await require_task(session, task_id)
    value = ObstacleReason(reason).value if isinstance(reason, ObstacleReason) else str(reason)
    payload: dict[str, Any] = {"category": value}
    if related_event_id is not None:
        payload["event_id"] = related_event_id
    if note:
        payload["note"] = note
    return await _event(
        session, ctx, task, TaskEventType.OBSTACLE_REPORTED, actor, reason=value, payload=payload
    )


async def set_deadline(
    session: AsyncSession,
    ctx: AppContext,
    task_id: int,
    *,
    actor: Actor,
    due_at: datetime | None = None,
    due_date: date | None = None,
    due_is_hard: bool | None = None,
    reason: str | None = None,
) -> Task:
    """Set, move or clear (both None) the deadline."""
    task = await require_task(session, task_id)
    if task.status in CLOSED_STATUSES:
        raise TaskError("Задача уже закрыта.")
    had = task.due_at is not None or task.due_date is not None
    old = _due_state(task)
    task.due_at = due_at
    task.due_date = None if due_at is not None else due_date
    if due_is_hard is not None:
        task.due_is_hard = due_is_hard
    new = _due_state(task)
    if old == new:
        return task
    etype = TaskEventType.RESCHEDULED if had else TaskEventType.UPDATED
    await _event(
        session,
        ctx,
        task,
        etype,
        actor,
        reason=reason,
        payload={"field": "due", "old": old, "new": new},
    )
    await _touch(session, ctx, task, sync=True)
    return task


async def reorder_tasks(
    session: AsyncSession,
    ctx: AppContext,
    task_ids: Sequence[int],
    *,
    actor: Actor,
    reason: str | None = None,
) -> list[int]:
    """Put the listed open tasks (same parent) in the given order. Changes ONLY
    sort_order: deadlines and planned times stay untouched (FR-15). Other
    tasks keep their relative order. Returns ids whose position changed."""
    ids = list(task_ids)
    if len(set(ids)) != len(ids):
        raise TaskError("Повторяющиеся задачи в новом порядке.")
    listed: list[Task] = []
    for tid in ids:
        t = await require_task(session, tid)
        if not is_open(t):
            raise TaskError(f"Задача #{tid} не открыта.")
        listed.append(t)
    parents = {t.parent_id for t in listed}
    if len(parents) != 1:
        raise TaskError("Переставлять можно только задачи одного уровня.")
    parent_id = parents.pop()
    cond = Task.parent_id.is_(None) if parent_id is None else Task.parent_id == parent_id
    scope = sorted(
        (await session.execute(select(Task).where(cond, Task.status.in_(OPEN_STATUSES))))
        .scalars()
        .all(),
        key=sort_key,
    )
    wanted = set(ids)
    slots = [i for i, t in enumerate(scope) if t.id in wanted]
    new_scope = list(scope)
    for slot, task in zip(slots, listed, strict=True):
        new_scope[slot] = task
    changed: list[int] = []
    for pos, task in enumerate(new_scope):
        order = (pos + 1) * 10
        if task.sort_order != order:
            old = task.sort_order
            task.sort_order = order
            task.updated_at = ctx.clock.now()
            if task.id in wanted:
                changed.append(task.id)
                await _event(
                    session,
                    ctx,
                    task,
                    TaskEventType.REORDERED,
                    actor,
                    reason=reason,
                    payload={"old": old, "new": order},
                )
    await session.flush()
    return changed


# --------------------------------------------------------------------------- status


async def complete_task(
    session: AsyncSession,
    ctx: AppContext,
    task_id: int,
    *,
    actor: Actor,
    note: str | None = None,
    session_id: int | None = None,
    criteria_confirmed: bool | None = None,
) -> Task:
    """Mark done. No-op when already done. `criteria_confirmed` records that the
    user compared the result with the definition of done."""
    task = await require_task(session, task_id)
    if task.status == TaskStatus.DONE:
        return task
    old = task.status
    task.status = TaskStatus.DONE
    task.completed_at = ctx.clock.now()
    payload: dict[str, Any] = {"from": old}
    if session_id is not None:
        payload["session_id"] = session_id
    if criteria_confirmed is not None:
        payload["criteria_confirmed"] = criteria_confirmed
    if note:
        payload["note"] = note
    await _event(session, ctx, task, TaskEventType.COMPLETED, actor, payload=payload)
    await _touch(session, ctx, task, sync=True)
    await spawn_next_occurrence(session, ctx, task)
    return task


async def undo_complete(
    session: AsyncSession, ctx: AppContext, task_id: int, *, actor: Actor
) -> Task:
    """ "Отменить" right after "Готово": reopen the task and drop the next copy of
    a recurring task if nobody touched it yet. No-op when the task is not done."""
    task = await require_task(session, task_id)
    if task.status != TaskStatus.DONE:
        return task
    task = await reopen_task(session, ctx, task_id, actor=actor, reason="отмена «Готово»")
    nxt = await next_occurrence_of(session, task_id)
    if nxt is not None and nxt.status == TaskStatus.TODO:
        events = await task_events(session, nxt.id)
        if all(e.event_type == str(TaskEventType.CREATED) for e in events):
            await delete_task(session, ctx, nxt.id, actor=actor, reason="отмена «Готово»")
    return task


async def next_occurrence_of(session: AsyncSession, task_id: int) -> Task | None:
    """The copy created when this recurring task was done (if any)."""
    row = (
        await session.execute(
            select(TaskEvent.task_id)
            .where(
                TaskEvent.event_type == str(TaskEventType.CREATED),
                TaskEvent.payload["recurred_from"].as_integer() == task_id,
            )
            .limit(1)
        )
    ).scalar()
    return await session.get(Task, row) if row is not None else None


async def spawn_next_occurrence(session: AsyncSession, ctx: AppContext, task: Task) -> Task | None:
    """Done recurring task -> its next copy on the next matching day. Code only
    (the owner set the rule). Once per task: reopening and finishing it again,
    or a repeated press, never makes a second copy."""
    if task.parent_id is not None or task.status != TaskStatus.DONE:
        return None
    try:
        rule = parse_rule(task.recurrence)
    except RecurrenceError:
        return None
    if rule is None or await next_occurrence_of(session, task.id) is not None:
        return None
    tz = await owner_tz(session)
    today = local_date(ctx.clock.now(), tz)
    plan_day = _plan_day(task, tz)
    due_day = _due_day(task, tz)
    base = plan_day or due_day or today
    nxt = next_date(rule, base=base, after=max(base, today))
    shift = nxt - base
    planned_start = None
    planned_date = None
    if task.planned_start_at is not None:
        planned_start = local_time_on(nxt, to_local(task.planned_start_at, tz).time(), tz)
    elif plan_day is not None or due_day is None:
        planned_date = nxt
    due_at = None
    if task.due_at is not None and due_day is not None:
        due_at = local_time_on(due_day + shift, to_local(task.due_at, tz).time(), tz)
    category = await session.get(Category, task.category_id) if task.category_id else None
    context = await session.get(Context, task.context_id) if task.context_id else None
    draft = TaskDraft(
        title=task.title,
        description=task.description,
        due_at=due_at,
        due_date=task.due_date + shift if task.due_date is not None else None,
        due_is_hard=task.due_is_hard,
        planned_start_at=planned_start,
        planned_date=planned_date,
        estimate_minutes=task.estimate_minutes,
        priority=task.priority,
        importance=task.importance,
        done_criteria=task.done_criteria,
        verification="criteria" if task.verification == "criteria" else "button",
        goal_id=task.goal_id,
        project_id=task.project_id,
        milestone_id=task.milestone_id,
        category=category.name if category is not None else None,
        context=context.name if context is not None else None,
        recurrence=rule,
    )
    return await create_task(
        session,
        ctx,
        draft,
        actor=Actor.SYSTEM,
        reason="повторяющаяся задача",
        event_payload={"recurred_from": task.id},
    )


async def record_progress(
    session: AsyncSession,
    ctx: AppContext,
    task_id: int,
    *,
    actor: Actor,
    note: str | None = None,
    session_id: int | None = None,
) -> Task:
    """Partial result: in_progress + note. NEVER done (acceptance 17).
    Repeating it with nothing new (double press) is a no-op."""
    task = await require_task(session, task_id)
    if task.status == TaskStatus.CANCELLED:
        raise TaskError("Задача отменена — сначала верни её.")
    nothing_new = note is None or note == task.progress_note
    if task.status == TaskStatus.IN_PROGRESS and nothing_new and session_id is None:
        return task
    old = task.status
    task.status = TaskStatus.IN_PROGRESS
    task.completed_at = None
    if note:
        task.progress_note = note
    payload: dict[str, Any] = {"from": old, "note": note}
    if session_id is not None:
        payload["session_id"] = session_id
    await _event(session, ctx, task, TaskEventType.PROGRESSED, actor, payload=payload)
    await _touch(session, ctx, task, sync=True)
    return task


async def reopen_task(
    session: AsyncSession, ctx: AppContext, task_id: int, *, actor: Actor, reason: str | None = None
) -> Task:
    task = await require_task(session, task_id)
    if task.status in OPEN_STATUSES:
        return task
    old = task.status
    task.status = TaskStatus.IN_PROGRESS if task.progress_note else TaskStatus.TODO
    task.completed_at = None
    await _event(
        session,
        ctx,
        task,
        TaskEventType.REOPENED,
        actor,
        reason=reason,
        payload={"from": old, "to": task.status},
    )
    await _touch(session, ctx, task, sync=True)
    return task


async def cancel_task(
    session: AsyncSession, ctx: AppContext, task_id: int, *, actor: Actor, reason: str | None = None
) -> Task:
    """Drop the task by decision (a valid outcome, FR-04). Reversible: reopen."""
    task = await require_task(session, task_id)
    if task.status == TaskStatus.CANCELLED:
        return task
    old = task.status
    task.status = TaskStatus.CANCELLED
    await _event(
        session, ctx, task, TaskEventType.CANCELLED, actor, reason=reason, payload={"from": old}
    )
    await _touch(session, ctx, task, sync=True)
    return task


async def defer_to_someday(
    session: AsyncSession, ctx: AppContext, task_id: int, *, actor: Actor, reason: str | None = None
) -> Task:
    task = await require_task(session, task_id)
    if task.status == TaskStatus.SOMEDAY:
        return task
    old, new = await _apply_schedule(
        session,
        ctx,
        task,
        planned_date=None,
        planned_start_at=None,
        to_backlog=False,
        to_someday=True,
    )
    await _event(
        session,
        ctx,
        task,
        TaskEventType.DEFERRED,
        actor,
        reason=reason,
        payload={"old": old, "new": new},
    )
    await _touch(session, ctx, task, sync=True)
    return task


async def delete_task(
    session: AsyncSession, ctx: AppContext, task_id: int, *, actor: Actor, reason: str | None = None
) -> None:
    """Hard delete (user request, with confirmation in the UI). Its journal rows
    go with it (FK cascade), so a snapshot is kept in the audit log."""
    task = await require_task(session, task_id)
    subs = await list_subtasks(session, task.id, include_cancelled=True)
    await audit(
        session,
        now=ctx.clock.now(),
        actor=actor,
        action="task.deleted",
        entity="task",
        entity_id=task.id,
        details={"task": snapshot(task), "subtasks": [snapshot(s) for s in subs], "reason": reason},
    )
    task.status = TaskStatus.CANCELLED
    await sync_task_reminders(session, ctx, task)
    for sub in subs:
        await session.delete(sub)
    await session.delete(task)
    await session.flush()


async def record_work_outcome(
    session: AsyncSession,
    ctx: AppContext,
    *,
    task_id: int,
    outcome: SessionOutcome,
    actor: Actor,
    session_id: int | None = None,
    note: str | None = None,
) -> Task:
    """Apply a work result to a task: STARTED/PROGRESSED -> in_progress (never done),
    COMPLETED -> done. Writes task_events and re-syncs reminders.

    Idempotent per (task, outcome, session_id). A STARTED/PROGRESSED result on a
    task that is already done or cancelled is journaled but does not reopen it
    (reopening is an explicit user decision)."""
    task = await require_task(session, task_id)
    outcome = SessionOutcome(outcome)
    etype = {
        SessionOutcome.STARTED: TaskEventType.STARTED,
        SessionOutcome.PROGRESSED: TaskEventType.PROGRESSED,
        SessionOutcome.COMPLETED: TaskEventType.COMPLETED,
    }[outcome]
    if session_id is not None:
        dup = (
            await session.execute(
                select(TaskEvent.id).where(
                    TaskEvent.task_id == task.id,
                    TaskEvent.event_type == str(etype),
                    func.json_extract(TaskEvent.payload, "$.session_id") == session_id,
                )
            )
        ).first()
        if dup is not None:
            return task
    old = task.status
    payload: dict[str, Any] = {"from": old, "session_id": session_id}
    if note:
        payload["note"] = note
    if outcome is SessionOutcome.COMPLETED:
        if task.status != TaskStatus.DONE:
            task.status = TaskStatus.DONE
            task.completed_at = ctx.clock.now()
    elif task.status not in CLOSED_STATUSES:
        task.status = TaskStatus.IN_PROGRESS
        if outcome is SessionOutcome.PROGRESSED and note:
            task.progress_note = note
    payload["to"] = task.status
    await _event(session, ctx, task, etype, actor, payload=payload)
    await _touch(session, ctx, task, sync=True)
    return task


# --------------------------------------------------------------------------- queries


@dataclass
class TodayBuckets:
    day: date
    tz: str
    overdue: list[Task] = field(default_factory=list)  # deadline passed
    missed: list[Task] = field(default_factory=list)  # planned for an earlier day
    planned: list[Task] = field(default_factory=list)  # planned today: timed first
    due_today: list[Task] = field(default_factory=list)  # deadline today, not planned
    backlog_count: int = 0

    def all(self) -> list[Task]:
        return [*self.overdue, *self.missed, *self.planned, *self.due_today]


def _due_moment(task: Task, tz: str) -> datetime | None:
    if task.due_at is not None:
        return task.due_at
    if task.due_date is not None:
        return local_day_bounds(task.due_date, tz)[1]
    return None


def _plan_day(task: Task, tz: str) -> date | None:
    if task.planned_start_at is not None:
        return local_date(task.planned_start_at, tz)
    return task.planned_date


def _due_day(task: Task, tz: str) -> date | None:
    if task.due_at is not None:
        return local_date(task.due_at, tz)
    return task.due_date


def is_overdue(task: Task, now: datetime, tz: str) -> bool:
    due = _due_moment(task, tz)
    return is_open(task) and due is not None and due <= now


async def _open_top_level(session: AsyncSession) -> list[Task]:
    stmt = select(Task).where(Task.parent_id.is_(None), Task.status.in_(OPEN_STATUSES))
    return sorted((await session.execute(stmt)).scalars().all(), key=sort_key)


def _classify(task: Task, today: date, now: datetime, tz: str) -> str:
    plan_day = _plan_day(task, tz)
    due_day = _due_day(task, tz)
    if is_overdue(task, now, tz):
        return "overdue"
    if plan_day == today:
        return "planned"
    if due_day == today:
        return "due_today"
    if plan_day is not None and plan_day < today:
        return "missed"
    if (plan_day is not None and plan_day > today) or (due_day is not None and due_day > today):
        return "upcoming"
    return "backlog"


async def today_buckets(
    session: AsyncSession, ctx: AppContext, *, day: date | None = None
) -> TodayBuckets:
    tz = await owner_tz(session)
    now = ctx.clock.now()
    today = day or local_date(now, tz)
    out = TodayBuckets(day=today, tz=tz)
    for t in await _open_top_level(session):
        kind = _classify(t, today, now, tz)
        if kind == "overdue":
            out.overdue.append(t)
        elif kind == "planned":
            out.planned.append(t)
        elif kind == "due_today":
            out.due_today.append(t)
        elif kind == "missed":
            out.missed.append(t)
        elif kind == "backlog":
            out.backlog_count += 1
    far = datetime.max.replace(tzinfo=now.tzinfo)
    out.overdue.sort(key=lambda t: (_due_moment(t, tz) or far, *sort_key(t)))
    out.planned.sort(
        key=lambda t: (t.planned_start_at is None, t.planned_start_at or far, *sort_key(t))
    )
    return out


async def list_tasks(
    session: AsyncSession, ctx: AppContext, view: TaskView, *, limit: int = 50
) -> list[Task]:
    """Top-level tasks for one list filter (FR-16 "Задачи")."""
    view = TaskView(view)
    tz = await owner_tz(session)
    now = ctx.clock.now()
    today = local_date(now, tz)
    if view is TaskView.TODAY:
        return (await today_buckets(session, ctx)).all()[:limit]
    if view in (TaskView.UPCOMING, TaskView.BACKLOG):
        want = "upcoming" if view is TaskView.UPCOMING else "backlog"
        rows = [t for t in await _open_top_level(session) if _classify(t, today, now, tz) == want]
        if view is TaskView.UPCOMING:
            far = date.max
            rows.sort(
                key=lambda t: (min(_plan_day(t, tz) or far, _due_day(t, tz) or far), *sort_key(t))
            )
        return rows[:limit]
    if view is TaskView.SOMEDAY:
        stmt = select(Task).where(Task.parent_id.is_(None), Task.status == TaskStatus.SOMEDAY)
        return sorted((await session.execute(stmt)).scalars().all(), key=sort_key)[:limit]
    since = now - timedelta(days=RECENT_DONE_DAYS)
    stmt = (
        select(Task)
        .where(Task.parent_id.is_(None), Task.status == TaskStatus.DONE, Task.completed_at >= since)
        .order_by(Task.completed_at.desc())
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars())


# --------------------------------------------------------------------------- formatting


def format_due(task: Task, tz: str, today: date | None = None) -> str:
    """ "срок завтра 18:00 (обязательный)" or "" when there is no deadline."""
    if task.due_at is not None:
        text = f"срок {format_moment(task.due_at, tz, today)}"
    elif task.due_date is not None:
        text = f"срок {day_label(task.due_date, today)}"
    else:
        return ""
    return text + (" (обязательный)" if task.due_is_hard else "")


def format_moment(moment: datetime, tz: str, today: date | None = None) -> str:
    return f"{day_label(local_date(moment, tz), today)} {fmt_local(moment, tz, '%H:%M')}"


def format_plan(task: Task, tz: str, today: date | None = None) -> str:
    if task.planned_start_at is not None:
        return format_moment(task.planned_start_at, tz, today)
    if task.planned_date is not None:
        return day_label(task.planned_date, today)
    if task.is_backlog:
        return "бэклог"
    return ""


def format_when(task: Task, tz: str, today: date | None = None) -> str:
    """Short Russian description of plan + deadline, e.g. "сегодня 14:00 · срок 27.10"."""
    parts = [format_plan(task, tz, today), format_due(task, tz, today)]
    return " · ".join(p for p in parts if p)


def day_label(d: date, today: date | None) -> str:
    if today is not None:
        delta = (d - today).days
        if delta == 0:
            return "сегодня"
        if delta == 1:
            return "завтра"
        if delta == -1:
            return "вчера"
    return d.strftime("%d.%m")


def local_time_on(day: date, at: time, tz: str) -> datetime:
    return resolve_local(day, at, tz)


def local_now(ctx: AppContext, tz: str) -> datetime:
    return to_local(ctx.clock.now(), tz)


# --------------------------------------------------------------------------- candidates


_BULLET_RE = re.compile(r"^\s*(?:[-*•–—]+|\d{1,2}[.)])\s*")


def split_candidates(text: str) -> list[str]:
    """Stage A candidate extraction (no AI): one candidate per line or ';'.
    Bullets/numbering are stripped. Nothing is created from this by itself."""
    out: list[str] = []
    for line in re.split(r"[\n;]+", text):
        clean = _BULLET_RE.sub("", line).strip(" \t.,")
        if clean:
            out.append(clean[:500])
    return out


async def propose_task_candidates(
    session: AsyncSession,
    ctx: AppContext,
    drafts: Sequence[TaskDraft],
    *,
    op_prefix: str,
    origin: Actor,
    ai_call_id: int | None = None,
) -> list[PendingApproval]:
    """Turn candidate tasks found in conversation into ONE batch of pending
    approvals (group f"{op_prefix}:cand"). Never creates a task by itself:
    creation happens only when the user accepts (rule 3, open question 36).
    Idempotent by op_prefix."""
    if origin == Actor.USER:
        raise ValueError("candidates are proposals; an explicit user command creates directly")
    group = f"{op_prefix}:cand"
    out: list[PendingApproval] = []
    for i, draft in enumerate(drafts[:MAX_CANDIDATES]):
        res = await actions.execute(
            session,
            ctx,
            CreateTask(task=draft),
            origin=origin,
            op_key=f"{op_prefix}:cand:{i}",
            group_key=group,
            ai_call_id=ai_call_id,
        )
        if res.approval_id is not None:
            row = await approvals.get_approval(session, res.approval_id)
            if row is not None:
                out.append(row)
    return out


# --------------------------------------------------------------------------- appliers


async def _title_of(session: AsyncSession, task_id: int) -> str:
    t = await require_task(session, task_id)
    return t.title


async def _apply_create_task(
    session: AsyncSession, ctx: AppContext, action: CreateTask, actor: Actor
) -> ApplyOutcome:
    task = await create_task(session, ctx, action.task, actor=actor)
    return ApplyOutcome({"task_id": task.id, "title": task.title})


async def _describe_create_task(
    session: AsyncSession, ctx: AppContext, action: CreateTask
) -> dict[str, Any]:
    d = action.task
    tz = await owner_tz(session)
    today = local_date(ctx.clock.now(), tz)
    lines: list[str] = []
    if d.planned_start_at is not None:
        lines.append(
            "Начало: "
            + day_label(local_date(d.planned_start_at, tz), today)
            + " "
            + fmt_local(d.planned_start_at, tz, "%H:%M")
        )
    elif d.planned_date is not None:
        lines.append("План: " + day_label(d.planned_date, today))
    if d.due_at is not None:
        lines.append("Срок: " + fmt_local(d.due_at, tz))
    elif d.due_date is not None:
        lines.append("Срок: " + day_label(d.due_date, today))
    if d.estimate_minutes:
        lines.append(f"Оценка: ~{d.estimate_minutes} мин")
    if d.done_criteria:
        lines.append(f"Критерий готовности: {d.done_criteria}")
    if d.parent_id is not None:
        lines.append(f"Подзадача для: {await _title_of(session, d.parent_id)}")
    return {"summary": f"Создать задачу «{d.title}»", "lines": lines, "title": d.title}


async def _apply_create_subtasks(
    session: AsyncSession, ctx: AppContext, action: CreateSubtasks, actor: Actor
) -> ApplyOutcome:
    created = await create_subtasks(session, ctx, action.parent_id, action.subtasks, actor=actor)
    return ApplyOutcome({"parent_id": action.parent_id, "created": [t.id for t in created]})


async def _describe_subtasks(
    session: AsyncSession, ctx: AppContext, action: CreateSubtasks | ReplaceSubtasks
) -> dict[str, Any]:
    parent = await require_task(session, action.parent_id)
    _check_parent_open(parent)
    verb = "Добавить шаги" if action.type == ActionType.CREATE_SUBTASKS else "Перестроить шаги"
    return {
        "summary": f"{verb} для «{parent.title}»",
        "lines": [f"• {s.title}" for s in action.subtasks],
    }


async def _apply_replace_subtasks(
    session: AsyncSession, ctx: AppContext, action: ReplaceSubtasks, actor: Actor
) -> ApplyOutcome:
    created, removed = await replace_open_subtasks(
        session, ctx, action.parent_id, action.subtasks, actor=actor
    )
    return ApplyOutcome(
        {
            "parent_id": action.parent_id,
            "created": [t.id for t in created],
            "replaced": [t.id for t in removed],
        }
    )


async def _apply_reorder(
    session: AsyncSession, ctx: AppContext, action: ReorderTasks, actor: Actor
) -> ApplyOutcome:
    changed = await reorder_tasks(session, ctx, action.task_ids, actor=actor)
    return ApplyOutcome({"reordered": changed})


async def _describe_reorder(
    session: AsyncSession, ctx: AppContext, action: ReorderTasks
) -> dict[str, Any]:
    titles = [await _title_of(session, tid) for tid in action.task_ids]
    return {
        "summary": "Изменить порядок задач",
        "lines": [f"{i}. {t}" for i, t in enumerate(titles, 1)],
    }


async def _apply_adjust_reminders(
    session: AsyncSession, ctx: AppContext, action: AdjustReminders, actor: Actor
) -> ApplyOutcome:
    fields = action.model_fields_set & {"pre_task_lead_min", "start_delay_min"}
    if not fields:
        raise TaskError("Не указано, что изменить в напоминаниях.")
    overrides = {k: getattr(action, k) for k in fields}
    task = await set_reminder_overrides(session, ctx, action.task_id, actor=actor, **overrides)
    return ApplyOutcome({"task_id": task.id, "reminder_overrides": dict(task.reminder_overrides)})


async def _describe_adjust_reminders(
    session: AsyncSession, ctx: AppContext, action: AdjustReminders
) -> dict[str, Any]:
    lines = []
    if action.pre_task_lead_min is not None:
        lines.append(f"Предупреждать за {action.pre_task_lead_min} мин до начала")
    if action.start_delay_min is not None:
        lines.append(f"Спрашивать через {action.start_delay_min} мин, если старт не подтверждён")
    title = await _title_of(session, action.task_id)
    return {"summary": f"Напоминания для «{title}»", "lines": lines}


async def _apply_reschedule(
    session: AsyncSession, ctx: AppContext, action: RescheduleTask, actor: Actor
) -> ApplyOutcome:
    task = await reschedule_task(
        session,
        ctx,
        action.task_id,
        actor=actor,
        planned_date=action.planned_date,
        planned_start_at=action.planned_start_at,
        to_backlog=action.to_backlog,
        reason=action.reason,
    )
    return ApplyOutcome({"task_id": task.id, **_schedule_state(task)})


async def _describe_reschedule(
    session: AsyncSession, ctx: AppContext, action: RescheduleTask
) -> dict[str, Any]:
    task = await require_task(session, action.task_id)
    if task.status in CLOSED_STATUSES:
        raise TaskError("Задача уже закрыта.")
    tz = await owner_tz(session)
    today = local_date(ctx.clock.now(), tz)
    if action.to_backlog:
        new = "в бэклог"
    elif action.planned_start_at is not None:
        new = (
            day_label(local_date(action.planned_start_at, tz), today)
            + " "
            + fmt_local(action.planned_start_at, tz, "%H:%M")
        )
    elif action.planned_date is not None:
        new = day_label(action.planned_date, today)
    else:
        raise TaskError("Не указано, на когда перенести.")
    lines = [f"Было: {format_when(task, tz, today) or 'без даты'}", f"Станет: {new}"]
    if action.reason:
        lines.append(f"Причина: {action.reason}")
    return {"summary": f"Перенести «{task.title}»", "lines": lines}


async def _apply_change_deadline(
    session: AsyncSession, ctx: AppContext, action: ChangeDeadline, actor: Actor
) -> ApplyOutcome:
    task = await set_deadline(
        session,
        ctx,
        action.task_id,
        actor=actor,
        due_at=action.due_at,
        due_date=action.due_date,
        reason=action.reason,
    )
    return ApplyOutcome({"task_id": task.id, **_due_state(task)})


async def _describe_change_deadline(
    session: AsyncSession, ctx: AppContext, action: ChangeDeadline
) -> dict[str, Any]:
    task = await require_task(session, action.task_id)
    tz = await owner_tz(session)
    old = (
        fmt_local(task.due_at, tz)
        if task.due_at
        else task.due_date.strftime("%d.%m")
        if task.due_date
        else "нет"
    )
    new = (
        fmt_local(action.due_at, tz)
        if action.due_at
        else action.due_date.strftime("%d.%m")
        if action.due_date
        else "убрать срок"
    )
    lines = [f"Было: {old}", f"Станет: {new}"]
    if task.due_is_hard:
        lines.append("Внимание: срок отмечен как обязательный.")
    if action.reason:
        lines.append(f"Причина: {action.reason}")
    return {"summary": f"Изменить срок «{task.title}»", "lines": lines}


async def _apply_change_status(
    session: AsyncSession, ctx: AppContext, action: ChangeTaskStatus, actor: Actor
) -> ApplyOutcome:
    if action.status == "someday":
        task = await defer_to_someday(
            session, ctx, action.task_id, actor=actor, reason=action.reason
        )
    elif action.status == "cancelled":
        task = await cancel_task(session, ctx, action.task_id, actor=actor, reason=action.reason)
    else:
        task = await reopen_task(session, ctx, action.task_id, actor=actor, reason=action.reason)
        if task.status == TaskStatus.SOMEDAY:
            task = await move_to_backlog(session, ctx, task.id, actor=actor, reason=action.reason)
    return ApplyOutcome({"task_id": task.id, "status": task.status})


async def _describe_change_status(
    session: AsyncSession, ctx: AppContext, action: ChangeTaskStatus
) -> dict[str, Any]:
    task = await require_task(session, action.task_id)
    lines = [
        f"Статус: {STATUS_LABELS.get(task.status, task.status)} → "
        f"{STATUS_LABELS.get(action.status, action.status)}"
    ]
    if action.reason:
        lines.append(f"Причина: {action.reason}")
    return {"summary": f"Изменить статус «{task.title}»", "lines": lines}


_UPDATE_FIELDS = (
    "title",
    "description",
    "done_criteria",
    "estimate_minutes",
    "importance",
    "priority",
)
_FIELD_LABELS = {
    "title": "Название",
    "description": "Описание",
    "done_criteria": "Критерий готовности",
    "estimate_minutes": "Оценка, мин",
    "importance": "Важность",
    "priority": "Приоритет",
}


async def _apply_update_task(
    session: AsyncSession, ctx: AppContext, action: UpdateTask, actor: Actor
) -> ApplyOutcome:
    changes = {k: getattr(action, k) for k in _UPDATE_FIELDS if k in action.model_fields_set}
    if not changes:
        raise TaskError("Нет изменений.")
    task = await update_task(
        session, ctx, action.task_id, actor=actor, reason=action.reason, **changes
    )
    return ApplyOutcome({"task_id": task.id, "changed": sorted(changes)})


async def _describe_update_task(
    session: AsyncSession, ctx: AppContext, action: UpdateTask
) -> dict[str, Any]:
    task = await require_task(session, action.task_id)
    lines = []
    for k in _UPDATE_FIELDS:
        if k in action.model_fields_set:
            lines.append(
                f"{_FIELD_LABELS[k]}: {getattr(task, k) or '—'} → {getattr(action, k) or '—'}"
            )
    if action.reason:
        lines.append(f"Причина: {action.reason}")
    return {"summary": f"Изменить задачу «{task.title}»", "lines": lines}


def register(ctx: AppContext) -> None:
    """Register action appliers for task actions."""
    reg = actions.register_applier
    reg(ctx, ActionType.CREATE_TASK, _apply_create_task, describe=_describe_create_task)
    reg(ctx, ActionType.CREATE_SUBTASKS, _apply_create_subtasks, describe=_describe_subtasks)
    reg(ctx, ActionType.REPLACE_SUBTASKS, _apply_replace_subtasks, describe=_describe_subtasks)
    reg(ctx, ActionType.REORDER_TASKS, _apply_reorder, describe=_describe_reorder)
    reg(
        ctx,
        ActionType.ADJUST_REMINDERS,
        _apply_adjust_reminders,
        describe=_describe_adjust_reminders,
    )
    reg(ctx, ActionType.RESCHEDULE_TASK, _apply_reschedule, describe=_describe_reschedule)
    reg(ctx, ActionType.CHANGE_DEADLINE, _apply_change_deadline, describe=_describe_change_deadline)
    reg(ctx, ActionType.CHANGE_TASK_STATUS, _apply_change_status, describe=_describe_change_status)
    reg(ctx, ActionType.UPDATE_TASK, _apply_update_task, describe=_describe_update_task)
