"""GoalService: long-term goals, roadmaps, honest progress, alignment (FR-10).

Owner of the `goals` table; co-owner (with services.structure, which keeps the
stage A projects/milestones CRUD) of `projects` and `milestones`. Task links are
written through TaskService (`services.tasks.update_task`).

Rules:
- a goal has a measurable outcome, success criteria, a horizon date, a roadmap
  of milestones (key milestones flagged), projects and linked tasks;
- progress is shown honestly and split: what is *measured* (milestones done,
  linked tasks closed, tracked work minutes) vs what the owner *reported*
  (`goals.progress_note`). There is no blended percentage;
- `goal_alignment` compares the daily load with the goals: share of closed
  work linked to each active goal, and active goals with no activity at all;
- the critical review (`goal_review` AI task, strong tier) only *proposes*:
  every proposed change becomes a CHANGE_GOAL / CHANGE_MILESTONE pending
  approval (FR-15), never applied silently. Without AI the review is a
  code-only summary of observations and proposes nothing;
- changes the owner makes in the goals menu are explicit commands (origin USER).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.ai import AITaskSpec, AIUnavailable, DataBlock, Prompt, Tier, register_task
from procbot.ai import prompting as P
from procbot.ai.audit import record_actions
from procbot.context import AppContext
from procbot.db.models import Goal, Milestone, Project, Task, WorkSession
from procbot.domain.actions import ActionType, ChangeGoal, ChangeMilestone, CreateGoal
from procbot.domain.enums import Actor, EntityStatus, SessionStatus, TaskStatus, TimeSource
from procbot.services import actions, approvals
from procbot.services import structure as st
from procbot.services import tasks as task_service
from procbot.services.actions import ActionError, ApplyOutcome
from procbot.services.audit import audit
from procbot.services.profile import get_profile
from procbot.timeutil import local_date

log = logging.getLogger(__name__)

GOAL_STATUSES: frozenset[str] = frozenset(s.value for s in EntityStatus)
MILESTONE_STATUSES: frozenset[str] = frozenset(
    {EntityStatus.ACTIVE.value, EntityStatus.DONE.value, EntityStatus.ARCHIVED.value}
)
GOAL_STATUS_LABELS: dict[str, str] = {
    "active": "активна",
    "paused": "на паузе",
    "done": "достигнута",
    "archived": "в архиве",
}
MILESTONE_ICON: dict[str, str] = {"active": "○", "done": "✅", "archived": "✖️"}
MILESTONE_STATUS_LABELS: dict[str, str] = {
    "active": "в работе",
    "done": "выполнен",
    "archived": "снят",
}
KEY_PREFIX = "!"  # "!Title" in a milestone list = key milestone
REVIEW_WINDOW_DAYS = 14
# Presentation only: "the horizon is near" for review observations.
HORIZON_NEAR_DAYS = 30
MAX_REVIEW_GOALS = 10


class GoalError(ActionError):
    """Goal operation not possible; str() is a short Russian message."""


# --------------------------------------------------------------------------- basics


async def owner_tz(session: AsyncSession) -> str:
    profile = await get_profile(session)
    return (profile.timezone if profile is not None else None) or "UTC"


async def get_goal(session: AsyncSession, goal_id: int) -> Goal | None:
    return await session.get(Goal, goal_id)


async def require_goal(session: AsyncSession, goal_id: int) -> Goal:
    goal = await session.get(Goal, goal_id)
    if goal is None:
        raise GoalError(f"Цель #{goal_id} не найдена.")
    return goal


async def list_goals(
    session: AsyncSession, *, statuses: tuple[str, ...] | None = ("active", "paused")
) -> list[Goal]:
    stmt = select(Goal).order_by(Goal.horizon_date.is_(None), Goal.horizon_date, Goal.id)
    if statuses is not None:
        stmt = stmt.where(Goal.status.in_(statuses))
    return list((await session.execute(stmt)).scalars())


def _clean(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    clean = " ".join(str(text).split()).strip()
    return clean[:limit] or None


def _snap(goal: Goal) -> dict[str, Any]:
    return {
        "title": goal.title,
        "measurable_outcome": goal.measurable_outcome,
        "success_criteria": goal.success_criteria,
        "horizon_date": goal.horizon_date.isoformat() if goal.horizon_date else None,
        "status": goal.status,
        "progress_note": goal.progress_note,
    }


def split_milestone_title(raw: str) -> tuple[str, bool]:
    """ "!Запуск" -> ("Запуск", True)."""
    text = raw.strip()
    key = text.startswith(KEY_PREFIX) or text.startswith("🔑")
    text = text.lstrip(KEY_PREFIX).removeprefix("🔑").strip()
    return text, key


async def create_goal(
    session: AsyncSession,
    ctx: AppContext,
    *,
    title: str,
    actor: Actor,
    measurable_outcome: str | None = None,
    success_criteria: str | None = None,
    horizon_date: date | None = None,
    description: str | None = None,
    milestones: list[str] | None = None,
) -> Goal:
    clean = _clean(title, 300)
    if not clean:
        raise GoalError("Пустое название цели.")
    now = ctx.clock.now()
    goal = Goal(
        title=clean,
        description=description,
        measurable_outcome=_clean(measurable_outcome, 2000),
        success_criteria=_clean(success_criteria, 2000),
        horizon_date=horizon_date,
        status=EntityStatus.ACTIVE.value,
        created_at=now,
        updated_at=now,
    )
    session.add(goal)
    await session.flush()
    await audit(
        session,
        now=now,
        actor=actor,
        action="goal.created",
        entity="goal",
        entity_id=goal.id,
        details=_snap(goal),
    )
    for raw in milestones or []:
        text, key = split_milestone_title(raw)
        if text:
            await add_milestone(session, ctx, goal.id, title=text, is_key=key, actor=actor)
    return goal


_UNSET: Any = object()


async def update_goal(
    session: AsyncSession,
    ctx: AppContext,
    goal_id: int,
    *,
    actor: Actor,
    reason: str | None = None,
    title: str | None = None,
    measurable_outcome: str | None = _UNSET,
    success_criteria: str | None = _UNSET,
    horizon_date: date | None = _UNSET,
    status: str | None = None,
    progress_note: str | None = _UNSET,
) -> tuple[Goal, dict[str, Any]]:
    """Change a goal. Callers: the owner's explicit edit (USER) or an approved
    CHANGE_GOAL card. Returns (goal, {"before", "after"}) of what changed."""
    goal = await require_goal(session, goal_id)
    before = _snap(goal)
    if title is not None:
        clean = _clean(title, 300)
        if not clean:
            raise GoalError("Пустое название цели.")
        goal.title = clean
    if measurable_outcome is not _UNSET:
        goal.measurable_outcome = _clean(measurable_outcome, 2000)
    if success_criteria is not _UNSET:
        goal.success_criteria = _clean(success_criteria, 2000)
    if horizon_date is not _UNSET:
        goal.horizon_date = horizon_date
    if progress_note is not _UNSET:
        goal.progress_note = _clean(progress_note, 2000)
    if status is not None:
        if status not in GOAL_STATUSES:
            raise GoalError("Неизвестный статус цели.")
        goal.status = status
    after = _snap(goal)
    changed = [k for k in after if after[k] != before[k]]
    if not changed:
        return goal, {}
    goal.updated_at = ctx.clock.now()
    diff = {"before": {k: before[k] for k in changed}, "after": {k: after[k] for k in changed}}
    await session.flush()
    await audit(
        session,
        now=goal.updated_at,
        actor=actor,
        action="goal.updated",
        entity="goal",
        entity_id=goal.id,
        details={**diff, "reason": reason},
    )
    return goal, diff


# --------------------------------------------------------------------------- roadmap


async def goal_projects(session: AsyncSession, goal_id: int) -> list[Project]:
    stmt = select(Project).where(Project.goal_id == goal_id).order_by(Project.id)
    return list((await session.execute(stmt)).scalars())


async def goal_milestones(session: AsyncSession, goal_id: int) -> list[Milestone]:
    """Milestones of the goal itself and of its projects (roadmap order)."""
    project_ids = [p.id for p in await goal_projects(session, goal_id)]
    cond = Milestone.goal_id == goal_id
    if project_ids:
        cond = or_(cond, Milestone.project_id.in_(project_ids))
    stmt = select(Milestone).where(cond).order_by(Milestone.sort_order, Milestone.id)
    return list((await session.execute(stmt)).scalars())


async def add_milestone(
    session: AsyncSession,
    ctx: AppContext,
    goal_id: int,
    *,
    title: str,
    actor: Actor,
    is_key: bool = False,
    due_date: date | None = None,
    project_id: int | None = None,
) -> Milestone:
    await require_goal(session, goal_id)
    try:
        row = await st.create_milestone(
            session,
            ctx,
            title=title,
            actor=actor,
            project_id=project_id,
            goal_id=goal_id,
            due_date=due_date,
        )
    except ValueError:
        raise GoalError("Пустое название этапа.") from None
    if project_id is None:
        # structure orders per project; keep goal-level milestones in roadmap order too
        last = [m.sort_order for m in await goal_milestones(session, goal_id) if m.id != row.id]
        row.sort_order = max(last, default=0) + 1
    row.is_key = bool(is_key)
    await session.flush()
    return row


async def require_milestone(session: AsyncSession, milestone_id: int) -> Milestone:
    row = await session.get(Milestone, milestone_id)
    if row is None:
        raise GoalError(f"Этап #{milestone_id} не найден.")
    return row


async def milestone_goal_id(session: AsyncSession, ms: Milestone) -> int | None:
    if ms.goal_id is not None:
        return ms.goal_id
    if ms.project_id is not None:
        project = await session.get(Project, ms.project_id)
        return project.goal_id if project is not None else None
    return None


async def update_milestone(
    session: AsyncSession,
    ctx: AppContext,
    milestone_id: int,
    *,
    actor: Actor,
    reason: str | None = None,
    title: str | None = None,
    due_date: date | None = _UNSET,
    status: str | None = None,
    is_key: bool | None = None,
) -> tuple[Milestone, dict[str, Any]]:
    ms = await require_milestone(session, milestone_id)

    def snap() -> dict[str, Any]:
        return {
            "title": ms.title,
            "due_date": ms.due_date.isoformat() if ms.due_date else None,
            "status": ms.status,
            "is_key": ms.is_key,
        }

    before = snap()
    if title is not None:
        clean = _clean(title, 300)
        if not clean:
            raise GoalError("Пустое название этапа.")
        ms.title = clean
    if due_date is not _UNSET:
        ms.due_date = due_date
    if status is not None:
        if status not in MILESTONE_STATUSES:
            raise GoalError("Неизвестный статус этапа.")
        ms.status = status
    if is_key is not None:
        ms.is_key = is_key
    after = snap()
    changed = [k for k in after if after[k] != before[k]]
    if not changed:
        return ms, {}
    ms.updated_at = ctx.clock.now()
    diff = {"before": {k: before[k] for k in changed}, "after": {k: after[k] for k in changed}}
    await session.flush()
    await audit(
        session,
        now=ms.updated_at,
        actor=actor,
        action="milestone.updated",
        entity="milestone",
        entity_id=ms.id,
        details={**diff, "reason": reason},
    )
    return ms, diff


async def set_project_goal(
    session: AsyncSession,
    ctx: AppContext,
    project_id: int,
    goal_id: int | None,
    *,
    actor: Actor,
) -> Project:
    project = await session.get(Project, project_id)
    if project is None:
        raise GoalError(f"Проект #{project_id} не найден.")
    if goal_id is not None:
        await require_goal(session, goal_id)
    if project.goal_id == goal_id:
        return project
    before = project.goal_id
    project.goal_id = goal_id
    project.updated_at = ctx.clock.now()
    await session.flush()
    await audit(
        session,
        now=project.updated_at,
        actor=actor,
        action="project.goal",
        entity="project",
        entity_id=project.id,
        details={"before": before, "after": goal_id},
    )
    return project


async def link_task(
    session: AsyncSession,
    ctx: AppContext,
    task_id: int,
    *,
    actor: Actor,
    goal_id: int | None = None,
    project_id: int | None = None,
    milestone_id: int | None = None,
) -> Task:
    """Link a task to a goal / project / milestone (through TaskService)."""
    changes: dict[str, Any] = {}
    if goal_id is not None:
        await require_goal(session, goal_id)
        changes["goal_id"] = goal_id
    if project_id is not None:
        project = await session.get(Project, project_id)
        if project is None:
            raise GoalError(f"Проект #{project_id} не найден.")
        changes["project_id"] = project_id
    if milestone_id is not None:
        await require_milestone(session, milestone_id)
        changes["milestone_id"] = milestone_id
    if not changes:
        raise GoalError("Не указано, к чему привязать задачу.")
    return await task_service.update_task(session, ctx, task_id, actor=actor, **changes)


# --------------------------------------------------------------------------- task -> goal


@dataclass
class _GoalIndex:
    project_goal: dict[int, int | None]
    milestone_goal: dict[int, int | None]
    parent_of: dict[int, int | None]
    task_rows: dict[int, Task]

    def goal_of(self, task: Task) -> int | None:
        seen = 0
        cur: Task | None = task
        while cur is not None and seen < 5:
            if cur.goal_id is not None:
                return cur.goal_id
            if cur.milestone_id is not None and self.milestone_goal.get(cur.milestone_id):
                return self.milestone_goal[cur.milestone_id]
            if cur.project_id is not None and self.project_goal.get(cur.project_id):
                return self.project_goal[cur.project_id]
            parent_id = cur.parent_id
            cur = self.task_rows.get(parent_id) if parent_id is not None else None
            seen += 1
        return None


async def _index(session: AsyncSession) -> _GoalIndex:
    projects = {p.id: p.goal_id for p in (await session.execute(select(Project))).scalars()}
    milestones: dict[int, int | None] = {}
    for m in (await session.execute(select(Milestone))).scalars():
        milestones[m.id] = m.goal_id if m.goal_id is not None else projects.get(m.project_id or 0)
    rows = {t.id: t for t in (await session.execute(select(Task))).scalars()}
    return _GoalIndex(projects, milestones, {t.id: t.parent_id for t in rows.values()}, rows)


async def linked_tasks(session: AsyncSession, goal_id: int) -> list[Task]:
    """Top-level tasks linked to the goal directly or via its projects/milestones."""
    idx = await _index(session)
    return [t for t in idx.task_rows.values() if t.parent_id is None and idx.goal_of(t) == goal_id]


async def open_linked_tasks(session: AsyncSession, goal_id: int) -> list[Task]:
    return sorted(
        (t for t in await linked_tasks(session, goal_id) if task_service.is_open(t)),
        key=task_service.sort_key,
    )


async def project_open_tasks(session: AsyncSession, project_id: int) -> list[Task]:
    stmt = select(Task).where(
        Task.project_id == project_id,
        Task.parent_id.is_(None),
        Task.status.in_(sorted(task_service.OPEN_STATUSES)),
    )
    return sorted((await session.execute(stmt)).scalars(), key=task_service.sort_key)


async def unlinked_open_tasks(session: AsyncSession, *, limit: int = 20) -> list[Task]:
    idx = await _index(session)
    rows = [
        t
        for t in idx.task_rows.values()
        if t.parent_id is None and task_service.is_open(t) and idx.goal_of(t) is None
    ]
    return sorted(rows, key=task_service.sort_key)[:limit]


# --------------------------------------------------------------------------- progress


@dataclass
class GoalProgress:
    """Measured vs reported progress of one goal (no blended percentage)."""

    goal_id: int
    title: str
    status: str
    horizon_date: date | None
    days_left: int | None
    milestones_total: int = 0
    milestones_done: int = 0
    key_total: int = 0
    key_done: int = 0
    tasks_total: int = 0  # linked top-level tasks, cancelled ones excluded
    tasks_done: int = 0
    tasks_open: int = 0
    reported: str | None = None  # the owner's own words (goals.progress_note)
    last_activity_at: datetime | None = None  # last closed linked task

    @property
    def milestone_share(self) -> float | None:
        return self.milestones_done / self.milestones_total if self.milestones_total else None

    @property
    def task_share(self) -> float | None:
        return self.tasks_done / self.tasks_total if self.tasks_total else None

    @property
    def has_measure(self) -> bool:
        return bool(self.milestones_total or self.tasks_total)

    def as_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "title": self.title,
            "status": self.status,
            "horizon_date": self.horizon_date.isoformat() if self.horizon_date else None,
            "days_left": self.days_left,
            "measured": {
                "milestones_done": self.milestones_done,
                "milestones_total": self.milestones_total,
                "key_milestones_done": self.key_done,
                "key_milestones_total": self.key_total,
                "tasks_done": self.tasks_done,
                "tasks_total": self.tasks_total,
                "tasks_open": self.tasks_open,
                "last_activity_at": self.last_activity_at.isoformat()
                if self.last_activity_at
                else None,
            },
            "reported": self.reported,
        }


async def goal_progress(session: AsyncSession, ctx: AppContext, goal_id: int) -> GoalProgress:
    """Progress of one goal. Called by the goal card and the weekly analytics."""
    goal = await require_goal(session, goal_id)
    tz = await owner_tz(session)
    today = local_date(ctx.clock.now(), tz)
    p = GoalProgress(
        goal_id=goal.id,
        title=goal.title,
        status=goal.status,
        horizon_date=goal.horizon_date,
        days_left=(goal.horizon_date - today).days if goal.horizon_date else None,
        reported=goal.progress_note,
    )
    for m in await goal_milestones(session, goal_id):
        if m.status == EntityStatus.ARCHIVED:
            continue
        p.milestones_total += 1
        p.milestones_done += m.status == EntityStatus.DONE
        if m.is_key:
            p.key_total += 1
            p.key_done += m.status == EntityStatus.DONE
    for t in await linked_tasks(session, goal_id):
        if t.status == TaskStatus.CANCELLED:
            continue
        p.tasks_total += 1
        if t.status == TaskStatus.DONE:
            p.tasks_done += 1
            if t.completed_at and (
                p.last_activity_at is None or t.completed_at > p.last_activity_at
            ):
                p.last_activity_at = t.completed_at
        elif task_service.is_open(t):
            p.tasks_open += 1
    return p


def progress_lines(p: GoalProgress) -> list[str]:
    """Russian lines that keep measured and reported progress apart."""
    lines: list[str] = []
    measured: list[str] = []
    if p.milestones_total:
        text = f"этапы {p.milestones_done}/{p.milestones_total}"
        if p.key_total:
            text += f" (ключевые {p.key_done}/{p.key_total})"
        measured.append(text)
    if p.tasks_total:
        measured.append(f"связанные задачи {p.tasks_done}/{p.tasks_total} закрыто")
    if measured:
        lines.append("📏 Измерено: " + " · ".join(measured))
    else:
        lines.append("📏 Измерено: пока нечего — нет этапов и связанных задач")
    lines.append(f"🗣 Со слов: {p.reported}" if p.reported else "🗣 Со слов: не указано")
    return lines


def progress_short(p: GoalProgress) -> str:
    parts: list[str] = []
    if p.milestones_total:
        parts.append(f"этапы {p.milestones_done}/{p.milestones_total}")
    if p.tasks_total:
        parts.append(f"задачи {p.tasks_done}/{p.tasks_total}")
    if p.days_left is not None:
        parts.append(f"осталось {p.days_left} дн." if p.days_left >= 0 else "горизонт прошёл")
    return " · ".join(parts) if parts else "без измеримого прогресса"


# --------------------------------------------------------------------------- alignment


@dataclass
class GoalActivity:
    goal_id: int
    title: str
    tasks_done: int = 0
    minutes_exact: int = 0  # live-tracked work sessions
    minutes_other: int = 0  # retro / approximate entries (less precise)
    share_of_done: float | None = None  # of all closed top-level tasks in the period


@dataclass
class AlignmentReport:
    """Daily load vs strategic goals over [since, until)."""

    since: datetime
    until: datetime
    total_done: int = 0
    unlinked_done: int = 0
    total_minutes: int = 0
    unlinked_minutes: int = 0
    goals: list[GoalActivity] = field(default_factory=list)
    idle_goals: list[GoalActivity] = field(default_factory=list)  # active, zero activity

    @property
    def linked_share(self) -> float | None:
        if not self.total_done:
            return None
        return (self.total_done - self.unlinked_done) / self.total_done

    @property
    def contradiction(self) -> bool:
        """Work got done, but some active goals got none of it."""
        return (self.total_done > 0 or self.total_minutes > 0) and bool(self.idle_goals)

    def as_dict(self) -> dict[str, Any]:
        return {
            "since": self.since.isoformat(),
            "until": self.until.isoformat(),
            "total_done": self.total_done,
            "unlinked_done": self.unlinked_done,
            "linked_share": self.linked_share,
            "total_minutes": self.total_minutes,
            "unlinked_minutes": self.unlinked_minutes,
            "goals": [g.__dict__ for g in self.goals],
            "idle_goals": [{"goal_id": g.goal_id, "title": g.title} for g in self.idle_goals],
            "contradiction": self.contradiction,
        }


async def goal_alignment(
    session: AsyncSession, ctx: AppContext, since: datetime, until: datetime | None = None
) -> AlignmentReport:
    """Share of closed work linked to each active goal since `since`.
    Called by the weekly analytics and the critical review."""
    from procbot.services.sessions import load_pauses, net_seconds

    until = until or ctx.clock.now()
    idx = await _index(session)
    active = await list_goals(session, statuses=("active",))
    acts = {g.id: GoalActivity(goal_id=g.id, title=g.title) for g in active}
    rep = AlignmentReport(since=since, until=until)
    for t in idx.task_rows.values():
        if t.parent_id is not None or t.status != TaskStatus.DONE or t.completed_at is None:
            continue
        if not (since <= t.completed_at < until):
            continue
        rep.total_done += 1
        gid = idx.goal_of(t)
        if gid is None:
            rep.unlinked_done += 1
        elif gid in acts:
            acts[gid].tasks_done += 1
    stmt = select(WorkSession).where(
        WorkSession.status == SessionStatus.FINISHED,
        WorkSession.ended_at.is_not(None),
        WorkSession.ended_at >= since,
        WorkSession.ended_at < until,
    )
    for ws in (await session.execute(stmt)).scalars():
        minutes = round(net_seconds(ws, await load_pauses(session, ws.id), until) / 60)
        if minutes <= 0:
            continue
        rep.total_minutes += minutes
        task = idx.task_rows.get(ws.task_id) if ws.task_id is not None else None
        gid = idx.goal_of(task) if task is not None else None
        if gid is None:
            rep.unlinked_minutes += minutes
        elif gid in acts:
            if ws.time_source == TimeSource.EXACT:
                acts[gid].minutes_exact += minutes
            else:
                acts[gid].minutes_other += minutes
    for a in acts.values():
        if rep.total_done:
            a.share_of_done = round(a.tasks_done / rep.total_done, 4)
        if a.tasks_done == 0 and a.minutes_exact == 0 and a.minutes_other == 0:
            rep.idle_goals.append(a)
    rep.goals = sorted(acts.values(), key=lambda a: (-a.tasks_done, a.goal_id))
    return rep


def alignment_lines(rep: AlignmentReport) -> list[str]:
    lines: list[str] = []
    if rep.total_done:
        linked = rep.total_done - rep.unlinked_done
        lines.append(f"Закрыто задач: {rep.total_done}, из них связаны с целями: {linked}.")
    else:
        lines.append("Закрытых задач за период нет.")
    for a in rep.goals:
        if a.tasks_done or a.minutes_exact or a.minutes_other:
            extra = ""
            if a.minutes_exact:
                extra += f", {a.minutes_exact} мин по таймеру"
            if a.minutes_other:
                extra += f", ~{a.minutes_other} мин со слов"
            lines.append(f"• «{a.title}»: задач {a.tasks_done}{extra}")
    if rep.contradiction:
        names = ", ".join(f"«{a.title}»" for a in rep.idle_goals)
        lines.append(f"⚠️ Работа шла, но по целям {names} за период не было ничего.")
    return lines


# --------------------------------------------------------------------------- review (AI)


class ReviewChange(BaseModel):
    """One proposed change. Becomes CHANGE_GOAL / CHANGE_MILESTONE (approval)."""

    target: Literal["goal", "milestone"]
    goal_id: int | None = None
    milestone_id: int | None = None
    title: str | None = Field(default=None, max_length=300)
    measurable_outcome: str | None = Field(default=None, max_length=2000)
    success_criteria: str | None = Field(default=None, max_length=2000)
    horizon_date: date | None = None
    due_date: date | None = None
    status: Literal["active", "paused", "done", "archived"] | None = None
    reason: str = Field(min_length=1, max_length=600)


class GoalReviewOutput(BaseModel):
    summary: str = Field(max_length=1500)
    observations: list[str] = Field(default_factory=list, max_length=10)
    changes: list[ReviewChange] = Field(default_factory=list, max_length=8)
    questions: list[str] = Field(default_factory=list, max_length=5)


GOAL_REVIEW = register_task(
    AITaskSpec(
        name="goal_review",
        tier=Tier.STRONG,
        output=GoalReviewOutput,
        max_output_tokens=2000,
        data_categories=frozenset({P.TASKS, P.ANALYTICS}),
        system=(
            "Ты делаешь критический пересмотр долгосрочных целей владельца (FR-10). "
            "Опирайся только на переданные данные и посчитанные кодом цифры, не выдумывай "
            "факты. Пиши по-русски, коротко и честно, без морализаторства. "
            "observations — наблюдения (что видно из данных); changes — предлагаемые "
            "изменения целей (target=goal, goal_id) или этапов (target=milestone, "
            "milestone_id) с обязательным объяснением reason; указывай только меняемые поля. "
            "Ничего не применяется без подтверждения владельца. Используй только id из данных. "
            "questions — вопросы владельцу, если без них вывод сделать нельзя."
        ),
        description="Критический разбор целей, сроков и приоритетов (FR-10)",
    )
)


@dataclass
class ReviewOutcome:
    text: str
    approval_ids: list[int] = field(default_factory=list)
    used_ai: bool = False
    dropped: list[str] = field(default_factory=list)


async def code_observations(session: AsyncSession, ctx: AppContext, goal: Goal) -> list[str]:
    """Deterministic observations (used without AI and as data for the AI)."""
    tz = await owner_tz(session)
    today = local_date(ctx.clock.now(), tz)
    p = await goal_progress(session, ctx, goal.id)
    name = f"«{goal.title}»"
    out: list[str] = []
    if not goal.measurable_outcome:
        out.append(
            f"{name}: не записан измеримый результат — трудно понять, когда цель достигнута."
        )
    if not goal.success_criteria:
        out.append(f"{name}: нет критериев успеха.")
    if goal.horizon_date is None:
        out.append(f"{name}: не задан горизонт (срок).")
    elif p.days_left is not None and p.days_left < 0 and goal.status == EntityStatus.ACTIVE:
        out.append(f"{name}: горизонт {goal.horizon_date:%d.%m.%Y} уже прошёл, а цель активна.")
    elif (
        p.days_left is not None
        and p.days_left <= HORIZON_NEAR_DAYS
        and p.milestones_total
        and (p.milestone_share or 0) < 0.5
    ):
        out.append(
            f"{name}: до горизонта {p.days_left} дн., а этапов закрыто "
            f"{p.milestones_done}/{p.milestones_total}."
        )
    if not p.milestones_total:
        out.append(f"{name}: нет дорожной карты — этапы не заданы.")
    for m in await goal_milestones(session, goal.id):
        if m.status == EntityStatus.ACTIVE and m.due_date is not None and m.due_date < today:
            key = " (ключевой)" if m.is_key else ""
            out.append(f"{name}: этап «{m.title}»{key} просрочен ({m.due_date:%d.%m}).")
    since = ctx.clock.now() - timedelta(days=REVIEW_WINDOW_DAYS)
    if p.last_activity_at is None or p.last_activity_at < since:
        out.append(f"{name}: за {REVIEW_WINDOW_DAYS} дн. не закрыто ни одной связанной задачи.")
    if p.tasks_open == 0 and goal.status == EntityStatus.ACTIVE:
        out.append(f"{name}: нет открытых задач — непонятен следующий шаг.")
    return out


async def _review_data(
    session: AsyncSession, ctx: AppContext, goal_id: int | None
) -> tuple[list[Goal], dict[str, Any], list[str], set[int], set[int]]:
    goals = (
        [await require_goal(session, goal_id)]
        if goal_id is not None
        else (await list_goals(session, statuses=("active", "paused")))[:MAX_REVIEW_GOALS]
    )
    since = ctx.clock.now() - timedelta(days=REVIEW_WINDOW_DAYS)
    alignment = await goal_alignment(session, ctx, since)
    goal_ids: set[int] = set()
    milestone_ids: set[int] = set()
    observations: list[str] = []
    data: list[dict[str, Any]] = []
    for g in goals:
        goal_ids.add(g.id)
        ms = await goal_milestones(session, g.id)
        milestone_ids.update(m.id for m in ms)
        prog = await goal_progress(session, ctx, g.id)
        obs = await code_observations(session, ctx, g)
        observations.extend(obs)
        data.append(
            {
                "goal_id": g.id,
                **_snap(g),
                "milestones": [
                    {
                        "milestone_id": m.id,
                        "title": m.title,
                        "status": m.status,
                        "is_key": m.is_key,
                        "due_date": m.due_date.isoformat() if m.due_date else None,
                    }
                    for m in ms
                ],
                "projects": [p.title for p in await goal_projects(session, g.id)],
                "open_tasks": [t.title for t in (await open_linked_tasks(session, g.id))[:10]],
                "progress": prog.as_dict(),
            }
        )
    payload = {
        "today": local_date(ctx.clock.now(), await owner_tz(session)).isoformat(),
        "goals": data,
        "alignment_last_14_days": alignment.as_dict(),
        "code_observations": observations,
    }
    return goals, payload, observations, goal_ids, milestone_ids


async def _to_actions(
    session: AsyncSession,
    out: GoalReviewOutput,
    goal_ids: set[int],
    milestone_ids: set[int],
) -> tuple[list[Any], list[str]]:
    """Typed actions for the proposed changes; invalid / no-op ones are dropped."""
    result: list[Any] = []
    dropped: list[str] = []
    for ch in out.changes:
        if ch.target == "goal":
            if ch.goal_id is None or ch.goal_id not in goal_ids:
                dropped.append(f"неизвестная цель #{ch.goal_id}")
                continue
            goal = await require_goal(session, ch.goal_id)
            fields: dict[str, Any] = {}
            for name in ("title", "measurable_outcome", "success_criteria"):
                value = getattr(ch, name)
                if value is not None and _clean(value, 2000) != getattr(goal, name):
                    fields[name] = value
            if ch.horizon_date is not None and ch.horizon_date != goal.horizon_date:
                fields["horizon_date"] = ch.horizon_date
            if ch.status is not None and ch.status != goal.status:
                fields["status"] = ch.status
            if not fields:
                dropped.append(f"без изменений: цель #{ch.goal_id}")
                continue
            result.append(ChangeGoal(goal_id=goal.id, reason=ch.reason, **fields))
        else:
            if ch.milestone_id is None or ch.milestone_id not in milestone_ids:
                dropped.append(f"неизвестный этап #{ch.milestone_id}")
                continue
            ms = await require_milestone(session, ch.milestone_id)
            fields = {}
            if ch.title is not None and _clean(ch.title, 300) != ms.title:
                fields["title"] = ch.title
            if ch.due_date is not None and ch.due_date != ms.due_date:
                fields["due_date"] = ch.due_date
            if ch.status == "paused":
                dropped.append(f"этап #{ch.milestone_id}: пауза для этапа не поддерживается")
                continue
            if ch.status is not None and ch.status != ms.status:
                fields["status"] = ch.status
            if not fields:
                dropped.append(f"без изменений: этап #{ch.milestone_id}")
                continue
            result.append(ChangeMilestone(milestone_id=ms.id, reason=ch.reason, **fields))
    return result, dropped


def _bullets(items: list[str]) -> list[str]:
    return [f"• {x}" for x in items]


async def run_goal_review(
    ctx: AppContext, *, op_prefix: str, goal_id: int | None = None
) -> ReviewOutcome:
    """Critical review of one goal (or all active ones). Must be called outside
    a transaction (AI network call). Proposals become pending approvals."""
    async with ctx.db.transaction() as s:
        goals, payload, observations, goal_ids, milestone_ids = await _review_data(s, ctx, goal_id)
    if not goals:
        return ReviewOutcome(text="Активных целей нет — разбирать пока нечего.")
    fallback_head = "ИИ сейчас недоступен — разбор только по цифрам, без предложений изменений."
    if ctx.ai is None:
        return ReviewOutcome(text=_fallback_text(fallback_head, observations))
    from procbot.ai.persona import load_persona

    persona = await load_persona(ctx)
    prompt = Prompt(
        persona=persona,
        instruction=(
            "Сделай критический разбор целей: реалистичны ли сроки, соответствует ли "
            "ежедневная занятость целям, что стоит пересмотреть (формулировку, результат, "
            "горизонт, статус, этапы). Каждое предложение — с объяснением."
        ),
        data=[
            DataBlock(P.TASKS, "goals", payload["goals"]),
            DataBlock(
                P.ANALYTICS,
                "metrics",
                {
                    "today": payload["today"],
                    "alignment_last_14_days": payload["alignment_last_14_days"],
                    "code_observations": payload["code_observations"],
                },
            ),
        ],
    )
    try:
        result = await ctx.ai.run("goal_review", prompt, initiator="services.goals.review")
    except AIUnavailable:
        return ReviewOutcome(text=_fallback_text(fallback_head, observations))
    out = result.as_(GoalReviewOutput)
    approval_ids: list[int] = []
    async with ctx.db.transaction() as s:
        proposed, dropped = await _to_actions(s, out, goal_ids, milestone_ids)
        results = await actions.execute_many(
            s,
            ctx,
            proposed,
            origin=Actor.AI,
            op_prefix=op_prefix,
            group_key=op_prefix,
            ai_call_id=result.call_id,
        )
        for r in results:
            if r.approval_id is not None and r.status is actions.ExecStatus.PENDING_APPROVAL:
                approval_ids.append(r.approval_id)
            elif not r.ok:
                dropped.append(r.error or "не прошло проверку")
    try:
        await record_actions(
            ctx,
            result.call_id,
            [actions.dump_action(a) for a in proposed],
            [{"status": r.status.value, "approval_id": r.approval_id} for r in results],
        )
    except Exception:  # audit detail must not break the review
        log.exception("record_actions failed")
    lines = ["🧐 Критический разбор", "", out.summary]
    if out.observations:
        lines += ["", "Наблюдения:", *_bullets(out.observations)]
    if out.questions:
        lines += ["", "Вопросы к тебе:", *_bullets(out.questions)]
    if approval_ids:
        lines += ["", "Предложения ниже — без твоего подтверждения ничего не изменится."]
    else:
        lines += ["", "Изменений не предлагаю."]
    return ReviewOutcome(
        text="\n".join(lines), approval_ids=approval_ids, used_ai=True, dropped=dropped
    )


def _fallback_text(head: str, observations: list[str]) -> str:
    lines = ["🧐 Критический разбор", "", head]
    if observations:
        lines += ["", *_bullets(observations)]
    else:
        lines += ["", "По цифрам явных проблем не видно."]
    return "\n".join(lines)


async def pending_review_cards(session: AsyncSession, approval_ids: list[int]) -> list[Any]:
    rows = [await approvals.get_approval(session, i) for i in approval_ids]
    return [r for r in rows if r is not None]


# --------------------------------------------------------------------------- appliers


async def _apply_create(
    session: AsyncSession, ctx: AppContext, action: CreateGoal, actor: Actor
) -> ApplyOutcome:
    goal = await create_goal(
        session,
        ctx,
        title=action.title,
        actor=actor,
        measurable_outcome=action.measurable_outcome,
        success_criteria=action.success_criteria,
        horizon_date=action.horizon_date,
        milestones=list(action.milestones),
    )
    return ApplyOutcome(result={"goal_id": goal.id, "title": goal.title})


async def _describe_create(
    session: AsyncSession, ctx: AppContext, action: CreateGoal
) -> dict[str, Any]:
    if not _clean(action.title, 300):
        raise GoalError("Пустое название цели.")
    lines: list[str] = []
    if action.measurable_outcome:
        lines.append(f"Результат: {action.measurable_outcome}")
    if action.success_criteria:
        lines.append(f"Критерии: {action.success_criteria}")
    if action.horizon_date:
        lines.append(f"Горизонт: {action.horizon_date:%d.%m.%Y}")
    for raw in action.milestones:
        text, key = split_milestone_title(raw)
        if text:
            lines.append(f"Этап: {'🔑 ' if key else ''}{text}")
    return {"summary": f"Новая цель «{_clean(action.title, 300)}»", "lines": lines}


async def _apply_change_goal(
    session: AsyncSession, ctx: AppContext, action: ChangeGoal, actor: Actor
) -> ApplyOutcome:
    kwargs: dict[str, Any] = {}
    for name in ("measurable_outcome", "success_criteria", "horizon_date"):
        value = getattr(action, name)
        if value is not None:
            kwargs[name] = value
    goal, diff = await update_goal(
        session,
        ctx,
        action.goal_id,
        actor=actor,
        reason=action.reason,
        title=action.title,
        status=action.status,
        **kwargs,
    )
    return ApplyOutcome(result={"goal_id": goal.id, "changed": sorted(diff.get("after", {}))})


def _fmt(v: Any) -> str:
    if v is None or v == "":
        return "не задано"
    if isinstance(v, date):
        return v.strftime("%d.%m.%Y")
    return str(v)


async def _describe_change_goal(
    session: AsyncSession, ctx: AppContext, action: ChangeGoal
) -> dict[str, Any]:
    goal = await require_goal(session, action.goal_id)
    labels = {
        "title": "Название",
        "measurable_outcome": "Результат",
        "success_criteria": "Критерии",
        "horizon_date": "Горизонт",
    }
    lines: list[str] = []
    for name, label in labels.items():
        value = getattr(action, name)
        if value is not None:
            lines.append(f"{label}: {_fmt(getattr(goal, name))} → {_fmt(value)}")
    if action.status is not None:
        if action.status not in GOAL_STATUSES:
            raise GoalError("Неизвестный статус цели.")
        lines.append(
            f"Статус: {GOAL_STATUS_LABELS.get(goal.status, goal.status)} → "
            f"{GOAL_STATUS_LABELS[action.status]}"
        )
    if not lines:
        raise GoalError("Предложение ничего не меняет.")
    if action.reason:
        lines.append(f"Почему: {action.reason}")
    return {"summary": f"Изменить цель «{goal.title}»", "lines": lines}


async def _apply_change_milestone(
    session: AsyncSession, ctx: AppContext, action: ChangeMilestone, actor: Actor
) -> ApplyOutcome:
    kwargs: dict[str, Any] = {}
    if action.due_date is not None:
        kwargs["due_date"] = action.due_date
    ms, diff = await update_milestone(
        session,
        ctx,
        action.milestone_id,
        actor=actor,
        reason=action.reason,
        title=action.title,
        status=action.status,
        **kwargs,
    )
    return ApplyOutcome(result={"milestone_id": ms.id, "changed": sorted(diff.get("after", {}))})


async def _describe_change_milestone(
    session: AsyncSession, ctx: AppContext, action: ChangeMilestone
) -> dict[str, Any]:
    ms = await require_milestone(session, action.milestone_id)
    lines: list[str] = []
    gid = await milestone_goal_id(session, ms)
    goal = await session.get(Goal, gid) if gid is not None else None
    if goal is not None:
        lines.append(f"Цель: {goal.title}")
    if action.title is not None:
        lines.append(f"Название: {ms.title} → {action.title}")
    if action.due_date is not None:
        lines.append(f"Срок: {_fmt(ms.due_date)} → {_fmt(action.due_date)}")
    if action.status is not None:
        if action.status not in MILESTONE_STATUSES:
            raise GoalError("Неизвестный статус этапа.")
        lines.append(
            f"Статус: {MILESTONE_STATUS_LABELS.get(ms.status, ms.status)} → "
            f"{MILESTONE_STATUS_LABELS[action.status]}"
        )
    if len(lines) <= (1 if goal is not None else 0):
        raise GoalError("Предложение ничего не меняет.")
    if action.reason:
        lines.append(f"Почему: {action.reason}")
    key = "ключевой " if ms.is_key else ""
    return {"summary": f"Изменить {key}этап «{ms.title}»", "lines": lines}


def register(ctx: AppContext) -> None:
    """Register appliers for goal actions (all NEEDS_APPROVAL in FR-15)."""
    reg = actions.register_applier
    reg(ctx, ActionType.CREATE_GOAL, _apply_create, describe=_describe_create)
    reg(ctx, ActionType.CHANGE_GOAL, _apply_change_goal, describe=_describe_change_goal)
    reg(
        ctx,
        ActionType.CHANGE_MILESTONE,
        _apply_change_milestone,
        describe=_describe_change_milestone,
    )
