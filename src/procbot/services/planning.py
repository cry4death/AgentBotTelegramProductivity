"""PlanningService: the daily adaptive plan (FR-03, scenario B, acceptance 6/7).

Code first, AI second:

1. ``build_day_context`` (code, one short transaction) computes the local day
   window (sleep routine, otherwise "now .. end of day" — said in the
   assumptions), busy calendar intervals (cache rows; a staleness flag when the
   connection is missing / failed / not synced recently), already fixed items
   (tasks the user gave an exact time), free windows, the free reserve
   (``config.planner_free_buffer_ratio``: the plan never fills all free time),
   optional breaks (only when the owner set a break length — open question 5),
   and the candidates: overdue / due / planned / missed tasks, near hard
   deadlines, habits due today, a few backlog tasks (goal-linked first), each
   with a calibrated estimate (services.estimates).
2. ``plan_day`` (strong tier) receives that as DATA and returns a layout with
   explanations. A business validator (passed to the gateway, so an invalid
   answer goes through the repair path) checks: known task ids only, inside the
   day and the free windows, never over busy calendar time or fixed items, no
   overlaps, hard deadlines respected, the reserve kept.
3. ``AIUnavailable`` (incl. still-invalid answers) -> a deterministic greedy plan
   (deadline / priority order, reserve kept) marked "составлен без ИИ".
   Postponements never raise a task's rank (FR-03).
4. The proposal is saved as a new ``PlanRevision`` (proposed) with items. The
   owner's "Утвердить" executes ``ApplyDayPlan`` (origin USER = consent, FR-15
   "занимать свободные окна — только предложение"): the planned times are set
   through services.tasks (task_events + reminders follow). A later version
   supersedes the earlier one (revised); at the end of the day the plan is
   closed with item outcomes.

Replanning (category REPLAN): a periodic job notices that the approved plan
became unrealistic (remaining work > remaining free time, or several items
missed) and schedules ONE reminder per approved version (the gate enforces
quiet hours / limits). Its renderer proposes an updated version (`replan` task
or a deterministic fallback): reordering is applied automatically (AUTO), moves
of tasks to other days become pending approvals that are part of the version the
owner approves. Morning/evening times are never changed here.

Calendar blocks are NOT created here (services.calendar offers that separately,
after approval).
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import math
import re
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Literal

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.ai import prompting as P
from procbot.ai.errors import AIUnavailable
from procbot.ai.gateway import run_task
from procbot.ai.prompting import DataBlock, Prompt
from procbot.ai.registry import AITaskSpec, register_task
from procbot.ai.types import Tier
from procbot.context import AppContext
from procbot.db.models import (
    CalendarConnection,
    CalendarEventCache,
    DailyPlan,
    Goal,
    Habit,
    HabitOccurrence,
    Intervention,
    JobLease,
    PendingApproval,
    PlanItem,
    PlanRevision,
    Project,
    ReminderSchedule,
    Task,
    TaskEvent,
    WorkSession,
)
from procbot.domain.actions import ActionType, ApplyDayPlan, ReorderTasks, RescheduleTask
from procbot.domain.enums import (
    Actor,
    ApprovalStatus,
    MessageCategory,
    ObstacleReason,
    PlanItemKind,
    PlanStatus,
    SessionStatus,
    TaskEventType,
    TaskStatus,
)
from procbot.services import actions, approvals, estimates, leases
from procbot.services import tasks as ts
from procbot.services.actions import ActionError, ApplyOutcome
from procbot.services.audit import audit
from procbot.services.jobs import PeriodicJob, register_job
from procbot.services.outbound import Outgoing, register_renderer
from procbot.services.preferences import flag
from procbot.services.profile import get_profile
from procbot.services.reminders import schedule_reminder
from procbot.telegram.callbacks import DayCb, SessionCb
from procbot.timeutil import fmt_local, local_date, local_day_bounds, resolve_local

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- constants

MIN_WINDOW_MIN = 10  # free gaps shorter than this are not planned into
MIN_ITEM_MIN = 5
MAX_BACKLOG_CANDIDATES = 5
SOON_DEADLINE_DAYS = 3
# Technical defaults (UNCONFIRMED, docs/open-questions.md):
CALENDAR_FRESH = timedelta(minutes=30)  # older sync -> "календарь мог устареть"
DEFAULT_HABIT_MIN = 15  # a habit without a minutes target
LOW_ENERGY_CAPACITY = 0.75  # stated low energy today -> lighter deterministic plan
REPLAN_JOB = "planning.replan_check"
REPLAN_INTERVAL_SECONDS = 600.0
MISSED_ITEMS_FOR_REPLAN = 2
MAX_REPLANS_PER_DAY = 3
CLOSE_JOB = "planning.close"
CLOSE_INTERVAL_SECONDS = 1800.0
REUSE_PROPOSAL_WITHIN = timedelta(minutes=3)  # double "Составить план" press
AI_RENDER_TIMEOUT_SECONDS = 150.0
CAPACITY_TOLERANCE_MIN = 5

NO_AI_MARK = "составлен без ИИ"

GROUP_ORDER: dict[str, int] = {
    "hard": 0,
    "due_today": 1,
    "planned": 2,
    "overdue": 3,
    "soon": 4,
    "missed": 5,
    "habit": 6,
    "goal": 7,
    "backlog": 8,
}
GROUP_LABELS: dict[str, str] = {
    "hard": "обязательный срок",
    "due_today": "срок сегодня",
    "planned": "запланировано на сегодня",
    "overdue": "просрочено",
    "soon": "срок в ближайшие дни",
    "missed": "не сделано в запланированный день",
    "habit": "привычка",
    "goal": "связано с целью",
    "backlog": "из бэклога",
    "fixed": "время уже назначено",
}
WEEKDAYS = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")

# --------------------------------------------------------------------------- AI schemas


class PlanItemOut(BaseModel):
    """One timed entry of the AI layout (local wall-clock times of the plan day)."""

    kind: Literal["task", "habit", "break"] = "task"
    task_id: int | None = None
    habit_id: int | None = None
    title: str | None = None
    start: str = Field(description="Local start time HH:MM")
    end: str = Field(description="Local end time HH:MM")
    rationale: str = ""


class UnplannedOut(BaseModel):
    task_id: int | None = None
    habit_id: int | None = None
    reason: str = ""


class PlanDayOutput(BaseModel):
    items: list[PlanItemOut] = Field(default_factory=list)
    priorities: str = Field(
        description=(
            "Объяснение приоритетов: сроки, последствия, цели, время, трудоёмкость, "
            "самочувствие, история переносов"
        )
    )
    assumptions: list[str] = Field(default_factory=list)
    approximate_task_ids: list[int] = Field(default_factory=list)
    unplanned: list[UnplannedOut] = Field(default_factory=list)


class MoveOut(BaseModel):
    task_id: int
    to: Literal["tomorrow", "backlog", "date"]
    date: str | None = Field(default=None, description="YYYY-MM-DD when to == 'date'")
    reason: str = ""


class SimplifyOut(BaseModel):
    task_id: int
    suggestion: str


class ReplanOutput(BaseModel):
    items: list[PlanItemOut] = Field(default_factory=list)
    move: list[MoveOut] = Field(default_factory=list)
    simplify: list[SimplifyOut] = Field(default_factory=list)
    explanation: str
    assumptions: list[str] = Field(default_factory=list)


_PLAN_SYSTEM = (
    "Ты планируешь день одного человека. Ты только предлагаешь раскладку: "
    "код приложения её проверит, а применит только после согласия пользователя. "
    "Не придумывай новых задач — используй только task_id / habit_id из данных или "
    "перерывы (kind=break). Время — локальное HH:MM дня плана. Никогда не ставь работу "
    "поверх занятых интервалов календаря и закреплённых дел. Не заполняй всё свободное "
    "время: оставь резерв. Количество переносов само по себе не делает второстепенную "
    "задачу главной. Самочувствие учитывай бережно: усталость — повод облегчить план, "
    "а не упрёк."
)

PLAN_DAY = register_task(
    AITaskSpec(
        name="plan_day",
        tier=Tier.STRONG,
        output=PlanDayOutput,
        max_output_tokens=2500,
        context_budget_tokens=9000,
        data_categories=frozenset(
            {P.TASKS, P.CALENDAR, P.PROFILE, P.MEMORY, P.ANALYTICS, P.MESSAGE}
        ),
        system=_PLAN_SYSTEM,
        description="Утренний план дня по свободным окнам, срокам и истории оценок",
    )
)

REPLAN = register_task(
    AITaskSpec(
        name="replan",
        tier=Tier.STRONG,
        output=ReplanOutput,
        max_output_tokens=2500,
        context_budget_tokens=9000,
        data_categories=frozenset(
            {P.TASKS, P.CALENDAR, P.PROFILE, P.MEMORY, P.ANALYTICS, P.MESSAGE}
        ),
        system=_PLAN_SYSTEM
        + " Если план перестал быть реалистичным, предложи, что оставить, что упростить "
        "и куда перенести. Перенос — только предложение: решение за пользователем.",
        description="Пересмотр плана дня (что оставить / упростить / перенести)",
    )
)

# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Interval:
    start: datetime
    end: datetime
    label: str = ""

    @property
    def minutes(self) -> int:
        return int((self.end - self.start).total_seconds() // 60)


def overlaps(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    return a_start < b_end and b_start < a_end


def subtract(windows: Sequence[Interval], blocks: Sequence[Interval]) -> list[Interval]:
    """Windows minus blocks (all half-open), sorted."""
    out = sorted(windows, key=lambda w: w.start)
    for b in sorted(blocks, key=lambda x: x.start):
        nxt: list[Interval] = []
        for w in out:
            if not overlaps(w.start, w.end, b.start, b.end):
                nxt.append(w)
                continue
            if w.start < b.start:
                nxt.append(Interval(w.start, b.start))
            if b.end < w.end:
                nxt.append(Interval(b.end, w.end))
        out = nxt
    return out


@dataclass
class Candidate:
    kind: PlanItemKind
    title: str
    minutes: int
    approximate: bool
    estimate_note: str
    group: str
    task_id: int | None = None
    habit_id: int | None = None
    due_at: datetime | None = None
    due_hard: bool = False
    fixed_start: datetime | None = None
    preferred_start: datetime | None = None
    priority: int | None = None
    importance: int | None = None
    sort_order: int = 0
    goal: str | None = None
    postpone_count: int = 0
    status: str = TaskStatus.TODO
    next_step: str | None = None
    in_base_plan: bool = False

    @property
    def is_fixed(self) -> bool:
        return self.fixed_start is not None

    @property
    def fixed_interval(self) -> Interval | None:
        if self.fixed_start is None:
            return None
        return Interval(
            self.fixed_start, self.fixed_start + timedelta(minutes=self.minutes), self.title
        )

    def rank(self) -> tuple[int, int, int, float, int, int, int]:
        """Deadline / commitment first, then priority. postpone_count is NOT used."""
        far = float("inf")
        return (
            0 if self.group == "hard" else 1,
            0 if self.in_base_plan else 1,
            GROUP_ORDER.get(self.group, 99),
            self.due_at.timestamp() if self.due_at else far,
            self.priority if self.priority is not None else 1 << 30,
            self.sort_order,
            self.task_id or self.habit_id or 0,
        )


@dataclass
class DayContext:
    day: date
    tz: str
    now: datetime
    start: datetime
    end: datetime
    busy: list[Interval]
    fixed: list[Candidate]
    windows: list[Interval]
    free_minutes: int
    buffer_minutes: int
    capacity_minutes: int
    break_minutes: int | None
    calendar_stale: bool
    calendar_note: str | None
    candidates: list[Candidate]
    assumptions: list[str] = field(default_factory=list)
    energy_note: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    user_request: str | None = None
    base_revision_id: int | None = None
    base_plan_id: int | None = None
    replan_reason: str | None = None

    def task_candidate(self, task_id: int) -> Candidate | None:
        for c in [*self.candidates, *self.fixed]:
            if c.task_id == task_id:
                return c
        return None

    def habit_candidate(self, habit_id: int) -> Candidate | None:
        for c in self.candidates:
            if c.habit_id == habit_id:
                return c
        return None

    @property
    def hard_minutes(self) -> int:
        return sum(c.minutes for c in self.candidates if c.group == "hard")


@dataclass
class DraftItem:
    kind: PlanItemKind
    start: datetime
    end: datetime
    title: str
    task_id: int | None = None
    habit_id: int | None = None
    approximate: bool = True
    rationale: str = ""
    fixed: bool = False

    @property
    def minutes(self) -> int:
        return int((self.end - self.start).total_seconds() // 60)


@dataclass
class MoveDraft:
    task_id: int
    title: str
    planned_date: date | None
    to_backlog: bool
    reason: str


@dataclass
class PlanDraft:
    items: list[DraftItem]
    explanation: str
    assumptions: list[str] = field(default_factory=list)
    unplanned: list[tuple[str, str]] = field(default_factory=list)
    used_ai: bool = False
    ai_call_id: int | None = None
    moves: list[MoveDraft] = field(default_factory=list)
    simplify: list[tuple[int, str]] = field(default_factory=list)

    @property
    def work_minutes(self) -> int:
        return sum(
            i.minutes
            for i in self.items
            if i.kind in (PlanItemKind.TASK, PlanItemKind.HABIT) and not i.fixed
        )


# --------------------------------------------------------------------------- helpers


def _ceil5(dt: datetime) -> datetime:
    dt = dt.replace(second=0, microsecond=0)
    extra = (-dt.minute) % 5
    return dt + timedelta(minutes=extra)


def _parse_hhmm(raw: Any) -> time | None:
    try:
        return time.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None


def _hm(dt: datetime, tz: str) -> str:
    return fmt_local(dt, tz, "%H:%M")


async def _owner(session: AsyncSession) -> tuple[str, Any]:
    profile = await get_profile(session)
    tz = (profile.timezone if profile is not None else None) or "UTC"
    return tz, profile


def _day_bounds(
    day: date, tz: str, routine: dict[str, Any] | None
) -> tuple[datetime, datetime, str | None]:
    """Waking window of the local day from routine.sleep, else the whole day."""
    sleep = (routine or {}).get("sleep") or {}
    bed = _parse_hhmm(sleep.get("start")) if isinstance(sleep, dict) else None
    wake = _parse_hhmm(sleep.get("end")) if isinstance(sleep, dict) else None
    day_start, day_end = local_day_bounds(day, tz)
    if bed is None or wake is None:
        return (
            day_start,
            day_end,
            "Режим сна не задан — планирую с текущего момента до конца дня "
            "(часы сна можно указать в настройках).",
        )
    start = resolve_local(day, wake, tz)
    end = resolve_local(day, bed, tz) if bed > wake else resolve_local(day + timedelta(1), bed, tz)
    return start, end, None


async def _calendar_state(
    session: AsyncSession, now: datetime, start: datetime, end: datetime, tz: str
) -> tuple[list[Interval], bool, str | None]:
    """Busy intervals from the calendar cache + staleness. Read-only: the
    calendar module owns these tables."""
    conns = list((await session.execute(select(CalendarConnection))).scalars())
    if not conns:
        return [], True, "Календарь не подключён — встречи и занятость не учтены."
    rows = (
        (
            await session.execute(
                select(CalendarEventCache).where(
                    CalendarEventCache.busy.is_(True),
                    CalendarEventCache.start_at < end,
                    CalendarEventCache.end_at > start,
                )
            )
        )
        .scalars()
        .all()
    )
    busy = [
        Interval(max(r.start_at, start), min(r.end_at, end), (r.summary or "Занято")[:80])
        for r in rows
        if r.end_at > r.start_at
    ]
    active = [c for c in conns if c.status == "active"]
    note: str | None = None
    stale = False
    if not active:
        stale, note = True, "Календарь отключён — занятость могла измениться, проверь встречи."
    else:
        syncs = [c.last_sync_at for c in active]
        if any(s is None for s in syncs):
            stale, note = True, "Календарь ещё не синхронизирован — встречи могут быть не учтены."
        else:
            oldest = min(s for s in syncs if s is not None)
            errors = [c for c in active if c.last_error]
            if errors:
                stale = True
                note = (
                    "Календарь мог устареть: последняя синхронизация с ошибкой "
                    f"(успешно — {fmt_local(oldest, tz)})."
                )
            elif now - oldest > CALENDAR_FRESH:
                stale = True
                note = f"Календарь мог устареть: последняя синхронизация {fmt_local(oldest, tz)}."
    return sorted(busy, key=lambda b: b.start), stale, note


async def _goal_title(session: AsyncSession, task: Task) -> str | None:
    goal_id = task.goal_id
    if goal_id is None and task.project_id is not None:
        proj = await session.get(Project, task.project_id)
        goal_id = proj.goal_id if proj else None
    if goal_id is None:
        return None
    g = await session.get(Goal, goal_id)
    return g.title if g is not None and g.status == "active" else None


def _due_moment(task: Task, tz: str) -> datetime | None:
    if task.due_at is not None:
        return task.due_at
    if task.due_date is not None:
        return local_day_bounds(task.due_date, tz)[1]
    return None


async def _habits_due(
    session: AsyncSession, day: date, tz: str, start: datetime, end: datetime
) -> list[Candidate]:
    """Active habits scheduled for `day` without an occurrence yet (read-only)."""
    habits = list((await session.execute(select(Habit).where(Habit.status == "active"))).scalars())
    out: list[Candidate] = []
    week_start = day - timedelta(days=day.weekday())
    for h in habits:
        sched = h.schedule or {}
        kind = sched.get("type")
        due = False
        if kind == "daily":
            due = True
        elif kind == "weekly":
            due = day.weekday() in {int(x) for x in sched.get("days", []) or []}
        elif kind == "times_per_week":
            done = (
                await session.execute(
                    select(func.count(HabitOccurrence.id)).where(
                        HabitOccurrence.habit_id == h.id,
                        HabitOccurrence.local_date >= week_start,
                        HabitOccurrence.local_date <= day,
                        HabitOccurrence.status.in_(["done", "partial"]),
                    )
                )
            ).scalar_one()
            due = int(done) < int(sched.get("n", 0) or 0)
        if not due:
            continue
        today_occ = (
            await session.execute(
                select(HabitOccurrence.id).where(
                    HabitOccurrence.habit_id == h.id, HabitOccurrence.local_date == day
                )
            )
        ).first()
        if today_occ is not None:
            continue
        target = h.target or {}
        minutes: int | None = None
        if str(target.get("unit", "")).startswith("мин") or target.get("unit") == "min":
            try:
                minutes = int(float(target.get("amount") or 0)) or None
            except (TypeError, ValueError):
                minutes = None
        pref = resolve_local(day, h.preferred_time, tz) if h.preferred_time else None
        if pref is not None and not start <= pref < end:
            pref = None
        out.append(
            Candidate(
                kind=PlanItemKind.HABIT,
                title=h.title,
                minutes=minutes or DEFAULT_HABIT_MIN,
                approximate=minutes is None,
                estimate_note=(
                    "по цели привычки"
                    if minutes
                    else f"длительность не указана — заложено ≈{DEFAULT_HABIT_MIN} мин"
                ),
                group="habit",
                habit_id=h.id,
                preferred_start=pref,
            )
        )
    return out


async def _low_energy_today(session: AsyncSession, day: date, tz: str) -> bool:
    start, end = local_day_bounds(day, tz)
    row = (
        await session.execute(
            select(Intervention.id).where(
                Intervention.reason_category == ObstacleReason.LOW_ENERGY,
                Intervention.created_at >= start,
                Intervention.created_at < end,
            )
        )
    ).first()
    return row is not None


async def _base_items(session: AsyncSession, revision_id: int | None) -> list[PlanItem]:
    if revision_id is None:
        return []
    return list(
        (
            await session.execute(
                select(PlanItem)
                .where(PlanItem.revision_id == revision_id)
                .order_by(PlanItem.sort_order, PlanItem.id)
            )
        ).scalars()
    )


# --------------------------------------------------------------------------- context


async def build_day_context(
    session: AsyncSession,
    ctx: AppContext,
    *,
    day: date | None = None,
    base_revision_id: int | None = None,
    energy: str | None = None,
    user_request: str | None = None,
) -> DayContext:
    """Everything the planner needs, computed by code (no AI, no network)."""
    now = ctx.clock.now()
    tz, profile = await _owner(session)
    today = local_date(now, tz)
    day = day or today
    assumptions: list[str] = []
    if profile is None or not profile.timezone:
        assumptions.append("Часовой пояс не задан — время считаю по UTC.")
    day_start, day_end, routine_note = _day_bounds(day, tz, profile.routine if profile else None)
    if routine_note:
        assumptions.append(routine_note)
    start = max(day_start, _ceil5(now)) if day <= today else day_start
    end = day_end
    busy, cal_stale, cal_note = await _calendar_state(session, now, start, max(start, end), tz)
    if cal_note:
        assumptions.append(cal_note)
    # An active work session occupies its remaining planned time.
    active = (
        await session.execute(
            select(WorkSession).where(
                WorkSession.status.in_([SessionStatus.ACTIVE, SessionStatus.PAUSED])
            )
        )
    ).scalar_one_or_none()
    if active is not None and active.started_at is not None:
        until = active.started_at + timedelta(minutes=active.planned_minutes or 0)
        busy.append(Interval(start, max(until, start + timedelta(minutes=15)), "рабочая сессия"))

    base_items = await _base_items(session, base_revision_id)
    base_task_ids = {i.task_id for i in base_items if i.task_id is not None}
    samples = await estimates.collect_samples(session, ctx, now=now)
    buckets = await ts.today_buckets(session, ctx, day=day)
    picked: dict[int, tuple[Task, str]] = {}

    def pick(t: Task, group: str) -> None:
        if t.id not in picked:
            picked[t.id] = (t, group)

    for t in buckets.overdue:
        pick(t, "hard" if t.due_is_hard else "overdue")
    for t in buckets.due_today:
        pick(t, "hard" if t.due_is_hard else "due_today")
    for t in buckets.planned:
        pick(t, "planned")
    for t in buckets.missed:
        pick(t, "missed")
    for t in await ts.list_tasks(session, ctx, ts.TaskView.UPCOMING, limit=50):
        if t.planned_date is not None or t.planned_start_at is not None:
            continue  # the owner already chose a later day for it
        due = _due_moment(t, tz)
        if due is not None and due <= local_day_bounds(day + timedelta(SOON_DEADLINE_DAYS), tz)[0]:
            pick(t, "soon")
    backlog = await ts.list_tasks(session, ctx, ts.TaskView.BACKLOG, limit=100)
    with_goal = [t for t in backlog if t.goal_id is not None or t.project_id is not None]
    for t in [*with_goal, *[t for t in backlog if t not in with_goal]][:MAX_BACKLOG_CANDIDATES]:
        pick(t, "goal" if t in with_goal else "backlog")
    for tid in base_task_ids:  # replan: approved items stay candidates whatever their bucket
        if tid not in picked:
            t = await session.get(Task, tid)
            if t is not None and ts.is_open(t):
                pick(t, "planned")

    candidates: list[Candidate] = []
    fixed: list[Candidate] = []
    for t, group in picked.values():
        est = await estimates.estimate_task(session, ctx, t, samples=samples)
        due = _due_moment(t, tz)
        step = await ts.next_open_subtask(session, t.id)
        c = Candidate(
            kind=PlanItemKind.TASK,
            title=t.title,
            minutes=max(MIN_ITEM_MIN, est.minutes),
            approximate=est.approximate,
            estimate_note=est.note,
            group=group,
            task_id=t.id,
            due_at=due,
            due_hard=t.due_is_hard,
            priority=t.priority,
            importance=t.importance,
            sort_order=t.sort_order,
            goal=await _goal_title(session, t),
            postpone_count=t.postpone_count or 0,
            status=t.status,
            next_step=step.title if step is not None else None,
            in_base_plan=t.id in base_task_ids,
        )
        timed_today = t.planned_start_at is not None and local_date(t.planned_start_at, tz) == day
        # A time the owner set himself (not one from the plan being revised) is fixed.
        if (
            timed_today
            and t.id not in base_task_ids
            and t.planned_start_at is not None
            and t.planned_start_at >= start
        ):
            c.fixed_start = t.planned_start_at
            c.group = "fixed"
            fixed.append(c)
            continue
        candidates.append(c)
    candidates += await _habits_due(session, day, tz, start, end)

    fixed_blocks = [f.fixed_interval for f in fixed if f.fixed_interval is not None]
    for f in fixed:
        iv = f.fixed_interval
        assert iv is not None
        for b in busy:
            if overlaps(iv.start, iv.end, b.start, b.end):
                assumptions.append(
                    f"«{f.title}» в {_hm(iv.start, tz)} пересекается с занятым временем "
                    f"({b.label}, {_hm(b.start, tz)}–{_hm(b.end, tz)}) — время не меняю без "
                    "твоего согласия."
                )
                break
    windows = [
        w
        for w in subtract([Interval(start, end)] if end > start else [], [*busy, *fixed_blocks])
        if w.minutes >= MIN_WINDOW_MIN
    ]
    free = sum(w.minutes for w in windows)
    ratio = ctx.config.planner_free_buffer_ratio
    buffer = math.ceil(free * ratio)
    capacity = max(0, free - buffer)
    energy_note = energy
    if energy_note is None and await _low_energy_today(session, day, tz):
        energy_note = "сегодня отмечалась нехватка сил"
    if energy_note:
        capacity = int(capacity * LOW_ENERGY_CAPACITY)
    break_min = profile.pomodoro_break_min if profile is not None else None
    if break_min is None:
        assumptions.append("Длительность перерывов не задана — перерывы входят в свободный резерв.")
    history = await estimates.history_summary(session, ctx, samples=samples)
    return DayContext(
        day=day,
        tz=tz,
        now=now,
        start=start,
        end=max(start, end),
        busy=sorted(busy, key=lambda b: b.start),
        fixed=sorted(fixed, key=lambda f: f.fixed_start or start),
        windows=windows,
        free_minutes=free,
        buffer_minutes=free - capacity,
        capacity_minutes=capacity,
        break_minutes=break_min,
        calendar_stale=cal_stale,
        calendar_note=cal_note,
        candidates=sorted(candidates, key=Candidate.rank),
        assumptions=assumptions,
        energy_note=energy_note,
        history=history,
        user_request=user_request,
        base_revision_id=base_revision_id,
    )


# --------------------------------------------------------------------------- deterministic


def _fixed_items(dc: DayContext) -> list[DraftItem]:
    out: list[DraftItem] = []
    for f in dc.fixed:
        iv = f.fixed_interval
        assert iv is not None
        out.append(
            DraftItem(
                kind=PlanItemKind.TASK,
                start=iv.start,
                end=iv.end,
                title=f.title,
                task_id=f.task_id,
                approximate=f.approximate,
                rationale=GROUP_LABELS["fixed"],
                fixed=True,
            )
        )
    return out


def deterministic_plan(dc: DayContext) -> PlanDraft:
    """Greedy layout: rank order (deadline / commitment / priority, never the
    number of postponements), earliest window that fits, deadlines respected,
    the reserve kept (hard deadlines may use it, with a warning)."""
    slots = [Interval(w.start, w.end) for w in dc.windows]
    items = _fixed_items(dc)
    unplanned: list[tuple[str, str]] = []
    assumptions: list[str] = []
    used = 0
    gap = timedelta(minutes=dc.break_minutes or 0)

    def place(c: Candidate) -> list[DraftItem]:
        """Put `c` into the earliest fitting slot (its preferred time first);
        a break of the owner's length follows when it fits in the same slot."""
        need = timedelta(minutes=c.minutes)
        order = list(range(len(slots)))
        pref = c.preferred_start
        if pref is not None:
            order.sort(key=lambda i: abs((slots[i].start - pref).total_seconds()))
        for i in order:
            s = slots[i]
            begin = s.start
            if pref is not None and s.start <= pref < s.end:
                begin = pref
            if begin + need > s.end:
                begin = s.start
            finish = begin + need
            if finish > s.end:
                continue
            if c.due_hard and c.due_at is not None and finish > c.due_at:
                continue
            placed = [
                DraftItem(
                    kind=c.kind,
                    start=begin,
                    end=finish,
                    title=c.title,
                    task_id=c.task_id,
                    habit_id=c.habit_id,
                    approximate=c.approximate,
                    rationale=GROUP_LABELS.get(c.group, ""),
                )
            ]
            right_start = finish
            if gap and finish + gap <= s.end:
                placed.append(
                    DraftItem(
                        PlanItemKind.BREAK, finish, finish + gap, "Перерыв", approximate=False
                    )
                )
                right_start = finish + gap
            left = Interval(s.start, begin) if begin > s.start else None
            right = Interval(right_start, s.end) if right_start < s.end else None
            slots[i : i + 1] = [
                x for x in (left, right) if x is not None and x.minutes >= MIN_WINDOW_MIN
            ]
            slots.sort(key=lambda x: x.start)
            return placed
        return []

    for c in dc.candidates:
        if used + c.minutes > dc.capacity_minutes and c.group != "hard":
            unplanned.append((c.title, "не хватает времени с учётом резерва"))
            continue
        # fillers only while the higher groups did not use most of the day
        if c.group in ("goal", "backlog") and used + c.minutes > dc.capacity_minutes * 0.8:
            unplanned.append((c.title, "оставил место под резерв"))
            continue
        placed = place(c)
        if not placed:
            reason = (
                "не помещается до обязательного срока"
                if c.due_hard
                else "нет подходящего свободного окна"
            )
            unplanned.append((c.title, reason))
            if c.group == "hard":
                assumptions.append(
                    f"⚠️ «{c.title}» с обязательным сроком не помещается — нужно решить, "
                    "что сдвинуть или упростить."
                )
            continue
        if used + c.minutes > dc.capacity_minutes:
            assumptions.append(f"Резерв уменьшен ради обязательного срока «{c.title}».")
        used += c.minutes
        items.extend(placed)
    items.sort(key=lambda i: i.start)
    return PlanDraft(
        items=items,
        explanation=_deterministic_explanation(dc, used),
        assumptions=assumptions,
        unplanned=unplanned,
        used_ai=False,
    )


def _deterministic_explanation(dc: DayContext, used: int) -> str:
    groups = {c.group for c in dc.candidates}
    parts: list[str] = []
    if "hard" in groups:
        parts.append("сначала — дела с обязательным сроком")
    if groups & {"due_today", "planned"}:
        parts.append("затем то, что нужно сделать сегодня по сроку или по плану")
    if groups & {"overdue", "missed", "soon"}:
        parts.append("потом просроченное, перенесённое с прошлых дней и близкие сроки")
    if groups & {"habit", "goal", "backlog"}:
        parts.append("привычки и задачи, связанные с целями, — если остаётся время")
    head = "Порядок: " + "; ".join(parts) + "." if parts else "Дел на сегодня немного."
    tail = (
        f" Внутри групп — по сроку и приоритету; количество переносов само по себе "
        f"приоритет не повышает. Свободно ≈{dc.free_minutes} мин, в план взято ≈{used} мин, "
        f"резерв ≈{max(0, dc.free_minutes - used)} мин на перерывы и непредвиденное."
    )
    if dc.energy_note:
        tail += f" Учёл самочувствие ({dc.energy_note}) — объём меньше обычного."
    return head + tail


def deterministic_replan(dc: DayContext) -> PlanDraft:
    """Fallback for REPLAN: keep what fits from now on (approved items first),
    propose moving the rest of the approved items to tomorrow (needs consent)."""
    draft = deterministic_plan(dc)
    placed = {i.task_id for i in draft.items if i.task_id is not None}
    for c in dc.candidates:
        if not c.in_base_plan or c.task_id is None or c.task_id in placed:
            continue
        if c.due_hard and c.due_at is not None and c.due_at <= local_day_bounds(dc.day, dc.tz)[1]:
            continue  # never propose moving past a hard deadline
        draft.moves.append(
            MoveDraft(
                task_id=c.task_id,
                title=c.title,
                planned_date=dc.day + timedelta(days=1),
                to_backlog=False,
                reason="не помещается в оставшееся время",
            )
        )
    draft.unplanned = [
        (t, r) for t, r in draft.unplanned if t not in {m.title for m in draft.moves}
    ]
    draft.explanation = (
        "План перестал помещаться в оставшееся время. Оставил то, что успевается "
        "(сначала обязательные сроки и уже утверждённое), остальное предлагаю перенести "
        "на завтра — перенос только с твоего согласия. " + draft.explanation
    )
    return draft


# --------------------------------------------------------------------------- AI


_HHMM = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")


def parse_local_time(dc: DayContext, raw: str) -> datetime | None:
    """'HH:MM' on the plan day (next day if the waking window crosses midnight),
    or an ISO datetime (naive = local)."""
    m = _HHMM.match(raw or "")
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if h == 24 and mi == 0:
            return local_day_bounds(dc.day, dc.tz)[1]
        if not (0 <= h < 24 and 0 <= mi < 60):
            return None
        moment = resolve_local(dc.day, time(h, mi), dc.tz)
        if moment < dc.start and dc.end > local_day_bounds(dc.day, dc.tz)[1]:
            moment = resolve_local(dc.day + timedelta(days=1), time(h, mi), dc.tz)
        return moment
    try:
        parsed = datetime.fromisoformat(str(raw).strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return resolve_local(parsed.date(), parsed.time(), dc.tz)
    return parsed.astimezone(dc.now.tzinfo)


def _convert_items(dc: DayContext, out: Sequence[PlanItemOut]) -> tuple[list[DraftItem], list[str]]:
    """AI items -> draft items + validation errors (business rules)."""
    errors: list[str] = []
    items: list[DraftItem] = []
    tz = dc.tz
    for n, it in enumerate(out, 1):
        start = parse_local_time(dc, it.start)
        end = parse_local_time(dc, it.end)
        label = f"пункт {n}"
        if start is None or end is None:
            errors.append(f"{label}: время должно быть в формате HH:MM")
            continue
        if end <= start:
            errors.append(f"{label}: конец должен быть позже начала")
            continue
        if end - start < timedelta(minutes=MIN_ITEM_MIN):
            errors.append(f"{label}: слишком короткий интервал (меньше {MIN_ITEM_MIN} мин)")
            continue
        if start < dc.start or end > dc.end:
            errors.append(f"{label}: вне доступного окна дня {_hm(dc.start, tz)}–{_hm(dc.end, tz)}")
            continue
        if it.kind == "task":
            c = dc.task_candidate(it.task_id) if it.task_id is not None else None
            if c is None:
                errors.append(
                    f"{label}: task_id {it.task_id} нет среди кандидатов — новые задачи "
                    "создавать нельзя"
                )
                continue
            if c.is_fixed:
                iv = c.fixed_interval
                assert iv is not None
                if start != iv.start:
                    errors.append(
                        f"{label}: «{c.title}» закреплена на {_hm(iv.start, tz)} — переносить "
                        "её нельзя, просто не включай её в items"
                    )
                continue  # fixed items are added by the code
            if c.due_hard and c.due_at is not None and end > c.due_at:
                errors.append(
                    f"{label}: «{c.title}» должна закончиться до срока {_hm(c.due_at, tz)}"
                )
                continue
            items.append(
                DraftItem(
                    PlanItemKind.TASK,
                    start,
                    end,
                    c.title,
                    task_id=c.task_id,
                    approximate=c.approximate,
                    rationale=it.rationale[:400],
                )
            )
        elif it.kind == "habit":
            c = dc.habit_candidate(it.habit_id) if it.habit_id is not None else None
            if c is None:
                errors.append(f"{label}: habit_id {it.habit_id} нет среди привычек на сегодня")
                continue
            items.append(
                DraftItem(
                    PlanItemKind.HABIT,
                    start,
                    end,
                    c.title,
                    habit_id=c.habit_id,
                    approximate=c.approximate,
                    rationale=it.rationale[:400],
                )
            )
        else:
            items.append(
                DraftItem(
                    PlanItemKind.BREAK,
                    start,
                    end,
                    (it.title or "Перерыв")[:100],
                    approximate=False,
                    rationale=it.rationale[:400],
                )
            )
    for it in items:
        for b in dc.busy:
            if overlaps(it.start, it.end, b.start, b.end):
                errors.append(
                    f"«{it.title}» {_hm(it.start, tz)}–{_hm(it.end, tz)} пересекается с "
                    f"занятым временем {_hm(b.start, tz)}–{_hm(b.end, tz)}"
                )
        for f in dc.fixed:
            iv = f.fixed_interval
            assert iv is not None
            if overlaps(it.start, it.end, iv.start, iv.end):
                errors.append(
                    f"«{it.title}» {_hm(it.start, tz)}–{_hm(it.end, tz)} пересекается с "
                    f"закреплённым делом «{f.title}» {_hm(iv.start, tz)}–{_hm(iv.end, tz)}"
                )
    ordered = sorted(items, key=lambda i: i.start)
    for a, b in itertools.pairwise(ordered):
        if overlaps(a.start, a.end, b.start, b.end):
            errors.append(f"«{a.title}» и «{b.title}» пересекаются по времени")
    work = sum(i.minutes for i in items if i.kind != PlanItemKind.BREAK)
    hard = sum(
        i.minutes
        for i in items
        if i.task_id is not None
        and (c := dc.task_candidate(i.task_id)) is not None
        and c.group == "hard"
    )
    allowed = max(dc.capacity_minutes, hard) + CAPACITY_TOLERANCE_MIN
    if work > allowed:
        errors.append(
            f"слишком плотно: запланировано {work} мин при допустимых {dc.capacity_minutes} "
            f"(свободно {dc.free_minutes}, резерв не меньше {dc.buffer_minutes} мин)"
        )
    return ordered, errors


def plan_validator(dc: DayContext) -> Callable[[BaseModel], list[str]]:
    def validate(output: BaseModel) -> list[str]:
        if isinstance(output, PlanDayOutput):
            _, errors = _convert_items(dc, output.items)
            return errors
        if isinstance(output, ReplanOutput):
            _, errors = _convert_items(dc, output.items)
            kept = {i.task_id for i in output.items if i.task_id is not None}
            for mv in output.move:
                c = dc.task_candidate(mv.task_id)
                if c is None:
                    errors.append(f"move: task_id {mv.task_id} нет среди задач плана")
                    continue
                if mv.task_id in kept:
                    errors.append(f"move: «{c.title}» одновременно оставлена и перенесена")
                target = _move_target(dc, mv)
                if target is None and mv.to == "date":
                    errors.append(f"move: для «{c.title}» нужна дата YYYY-MM-DD после сегодня")
                if c.due_hard and c.due_at is not None:
                    deadline_day = local_date(c.due_at, dc.tz)
                    if mv.to == "backlog" or (target is not None and target > deadline_day):
                        errors.append(
                            f"move: «{c.title}» нельзя переносить позже обязательного срока"
                        )
            for sp in output.simplify:
                if dc.task_candidate(sp.task_id) is None:
                    errors.append(f"simplify: task_id {sp.task_id} нет среди задач плана")
            return errors
        return ["неожиданный формат ответа"]

    return validate


def _move_target(dc: DayContext, mv: MoveOut) -> date | None:
    if mv.to == "tomorrow":
        return dc.day + timedelta(days=1)
    if mv.to == "date" and mv.date:
        try:
            d = date.fromisoformat(mv.date)
        except ValueError:
            return None
        return d if d > dc.day else None
    return None


def _cand_json(c: Candidate, tz: str) -> dict[str, Any]:
    data: dict[str, Any] = {
        "minutes": c.minutes,
        "approximate": c.approximate,
        "estimate_note": c.estimate_note,
        "group": GROUP_LABELS.get(c.group, c.group),
    }
    if c.task_id is not None:
        data["task_id"] = c.task_id
    if c.habit_id is not None:
        data["habit_id"] = c.habit_id
    data["title"] = c.title
    if c.due_at is not None:
        data["due"] = fmt_local(c.due_at, tz)
        data["due_is_hard"] = c.due_hard
    if c.priority is not None:
        data["priority"] = c.priority
    if c.importance is not None:
        data["importance"] = c.importance
    if c.goal:
        data["goal"] = c.goal
    if c.postpone_count:
        data["postpone_count"] = c.postpone_count
    if c.status == TaskStatus.IN_PROGRESS:
        data["in_progress"] = True
    if c.next_step:
        data["next_step"] = c.next_step
    if c.preferred_start is not None:
        data["preferred_time"] = _hm(c.preferred_start, tz)
    if c.in_base_plan:
        data["in_approved_plan"] = True
    if c.fixed_start is not None:
        data["fixed_at"] = _hm(c.fixed_start, tz)
    return data


def _plan_data(dc: DayContext) -> dict[str, Any]:
    tz = dc.tz
    return {
        "date": f"{dc.day.isoformat()} ({WEEKDAYS[dc.day.weekday()]})",
        "timezone": tz,
        "now": _hm(dc.now, tz),
        "day_window": {"start": _hm(dc.start, tz), "end": _hm(dc.end, tz)},
        "free_windows": [
            {"start": _hm(w.start, tz), "end": _hm(w.end, tz), "minutes": w.minutes}
            for w in dc.windows
        ],
        "fixed_items": [_cand_json(f, tz) for f in dc.fixed],
        "free_minutes": dc.free_minutes,
        "capacity_minutes": dc.capacity_minutes,
        "reserve_minutes": dc.buffer_minutes,
        "break_minutes": dc.break_minutes,
        "candidates": [_cand_json(c, tz) for c in dc.candidates],
        "wellbeing": dc.energy_note,
        "estimate_history": dc.history,
        "code_assumptions": dc.assumptions,
        "replan_reason": dc.replan_reason,
    }


def _busy_data(dc: DayContext) -> dict[str, Any]:
    return {
        "busy": [
            {"start": _hm(b.start, dc.tz), "end": _hm(b.end, dc.tz), "title": b.label}
            for b in dc.busy
        ],
        "calendar_may_be_stale": dc.calendar_stale,
    }


_PLAN_INSTRUCTION = (
    "Составь реалистичный план дня из candidates.\n"
    "Правила:\n"
    "1. Размещай дела только внутри free_windows (локальное время HH:MM). Никогда не "
    "ставь ничего поверх busy и fixed_items; fixed_items в items не включай.\n"
    "2. Дела с due_is_hard=true должны закончиться до срока.\n"
    "3. Суммарная длительность задач и привычек — не больше capacity_minutes: "
    "reserve_minutes остаётся свободным на перерывы и непредвиденное.\n"
    "4. Длительность бери из minutes (уже с поправкой на историю оценок). Если оценка "
    "приблизительная — добавь task_id в approximate_task_ids.\n"
    "5. postpone_count сам по себе не повышает приоритет второстепенной задачи.\n"
    "6. В priorities коротко объясни порядок: сроки, последствия невыполнения, связь с "
    "целями, доступное время, трудоёмкость, самочувствие (wellbeing), история переносов.\n"
    "7. Что не поместилось — в unplanned с причиной. Допущения — в assumptions.\n"
    "8. Новых задач не придумывай: только task_id / habit_id из данных или kind=break."
)

_REPLAN_INSTRUCTION = (
    "Утверждённый план на сегодня перестал быть реалистичным (replan_reason). Предложи "
    "обновлённую версию на оставшееся время.\n"
    "- items: что оставить, с новым временем внутри free_windows (правила как для плана: "
    "не поверх busy и fixed_items, резерв reserve_minutes не занимать, сроки due_is_hard "
    "соблюдать, новых задач не придумывать);\n"
    "- simplify: что можно упростить (конкретная подсказка);\n"
    "- move: что перенести (tomorrow / backlog / date) — это только предложение, решает "
    "пользователь; задачи с обязательным сроком не переносить позже срока;\n"
    "- explanation: коротко, почему так; assumptions — допущения.\n"
    "Если есть сообщение пользователя с просьбой изменить план — учти его."
)


async def _ai_draft(
    ctx: AppContext, dc: DayContext, *, mode: Literal["plan", "replan"]
) -> PlanDraft:
    from procbot.ai.context import ContextBuilder, Extra
    from procbot.ai.persona import load_persona

    task = "plan_day" if mode == "plan" else "replan"
    extra = [
        Extra(DataBlock(P.TASKS, "План дня: окна и кандидаты", _plan_data(dc)), 98, None),
        Extra(DataBlock(P.CALENDAR, "Занятость по календарю", _busy_data(dc)), 97, None),
    ]
    blocks = await ContextBuilder(ctx, task=task).build(
        message=dc.user_request,
        message_label="Просьба пользователя изменить план",
        profile=True,
        history=0,
        tasks=0,
        memory=True,
        extra=extra,
    )
    prompt = Prompt(
        instruction=_PLAN_INSTRUCTION if mode == "plan" else _REPLAN_INSTRUCTION,
        data=blocks,
        persona=await load_persona(ctx),
    )
    result = await run_task(
        ctx, task, prompt, initiator="services.planning", validator=plan_validator(dc)
    )
    if mode == "plan":
        out = result.as_(PlanDayOutput)
        items, _ = _convert_items(dc, out.items)
        approx = set(out.approximate_task_ids)
        for i in items:
            if i.task_id in approx:
                i.approximate = True
        unplanned = []
        for u in out.unplanned:
            c = (
                dc.task_candidate(u.task_id)
                if u.task_id is not None
                else dc.habit_candidate(u.habit_id)
                if u.habit_id is not None
                else None
            )
            if c is not None:
                unplanned.append((c.title, u.reason[:200]))
        return PlanDraft(
            items=sorted([*_fixed_items(dc), *items], key=lambda i: i.start),
            explanation=out.priorities.strip()[:2000],
            assumptions=[a.strip()[:300] for a in out.assumptions if a.strip()][:10],
            unplanned=unplanned,
            used_ai=True,
            ai_call_id=result.call_id,
        )
    rout = result.as_(ReplanOutput)
    items, _ = _convert_items(dc, rout.items)
    moves: list[MoveDraft] = []
    for mv in rout.move:
        c = dc.task_candidate(mv.task_id)
        if c is None or c.task_id is None:
            continue
        moves.append(
            MoveDraft(
                task_id=c.task_id,
                title=c.title,
                planned_date=_move_target(dc, mv),
                to_backlog=mv.to == "backlog",
                reason=mv.reason[:200],
            )
        )
    return PlanDraft(
        items=sorted([*_fixed_items(dc), *items], key=lambda i: i.start),
        explanation=rout.explanation.strip()[:2000],
        assumptions=[a.strip()[:300] for a in rout.assumptions if a.strip()][:10],
        used_ai=True,
        ai_call_id=result.call_id,
        moves=moves,
        simplify=[(s.task_id, s.suggestion.strip()[:300]) for s in rout.simplify],
    )


@contextlib.asynccontextmanager
async def keep_lease(ctx: AppContext, name: str | None) -> AsyncIterator[None]:
    """Renew a lease held by the current processing run while a long render (AI)
    is in progress (the scheduler lease TTL is short and not renewed by itself).
    No-op when the lease is not held (e.g. a renderer called directly)."""
    if not name:
        yield
        return
    async with ctx.db.transaction() as s:
        row = await s.get(JobLease, name)
        owner = row.owner if row is not None and row.expires_at > ctx.clock.now() else None
    if owner is None:
        yield
        return
    interval = max(5.0, ctx.config.scheduler.lease_seconds / 3)

    async def renew_loop() -> None:
        while True:
            await asyncio.sleep(interval)
            if not await leases.renew(ctx, name, owner=owner):
                return

    task = asyncio.create_task(renew_loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


async def make_draft(
    ctx: AppContext,
    dc: DayContext,
    *,
    mode: Literal["plan", "replan"] = "plan",
    lease_name: str | None = None,
    require_ai: bool = False,
) -> PlanDraft:
    """AI layout with the deterministic fallback (hard rule 8). `require_ai`
    re-raises AIUnavailable (a free-text change request cannot be guessed)."""
    try:
        async with keep_lease(ctx, lease_name):
            return await asyncio.wait_for(
                _ai_draft(ctx, dc, mode=mode), timeout=AI_RENDER_TIMEOUT_SECONDS
            )
    except (AIUnavailable, TimeoutError) as exc:
        if require_ai:
            if isinstance(exc, AIUnavailable):
                raise
            raise AIUnavailable("timeout", "plan render took too long") from exc
        log.info("planning without AI", extra={"mode": mode, "reason": str(exc)[:200]})
    draft = deterministic_plan(dc) if mode == "plan" else deterministic_replan(dc)
    draft.assumptions.insert(0, f"План {NO_AI_MARK}: ИИ сейчас недоступен.")
    return draft


# --------------------------------------------------------------------------- persistence


async def get_plan(session: AsyncSession, day: date) -> DailyPlan | None:
    return (
        await session.execute(select(DailyPlan).where(DailyPlan.local_date == day))
    ).scalar_one_or_none()


async def revisions(session: AsyncSession, plan_id: int) -> list[PlanRevision]:
    return list(
        (
            await session.execute(
                select(PlanRevision)
                .where(PlanRevision.plan_id == plan_id)
                .order_by(PlanRevision.version)
            )
        ).scalars()
    )


async def approved_revision(session: AsyncSession, plan_id: int) -> PlanRevision | None:
    for r in reversed(await revisions(session, plan_id)):
        if r.status == PlanStatus.APPROVED:
            return r
    return None


async def revision_items(session: AsyncSession, revision_id: int) -> list[PlanItem]:
    return await _base_items(session, revision_id)


def plan_group_key(revision_id: int) -> str:
    return f"plan:{revision_id}"


async def save_draft(
    session: AsyncSession,
    ctx: AppContext,
    dc: DayContext,
    draft: PlanDraft,
    *,
    created_by: Actor,
    change_reason: str | None = None,
    reorder: bool = False,
    ai_records: list[tuple[int, list[Any], list[Any]]] | None = None,
) -> PlanRevision:
    """Store the draft as a new proposed revision (earlier proposals -> revised).
    Moves become pending approvals of this revision; reordering (AUTO, FR-15)
    is applied through the executor when `reorder` is set."""
    now = ctx.clock.now()
    plan = await get_plan(session, dc.day)
    if plan is None:
        plan = DailyPlan(
            local_date=dc.day, status=PlanStatus.PROPOSED, created_at=now, updated_at=now
        )
        session.add(plan)
        await session.flush()
    revs = await revisions(session, plan.id)
    for r in revs:
        if r.status == PlanStatus.PROPOSED:
            r.status = PlanStatus.REVISED
            await _reject_group(session, ctx, r.id)
    assumptions = [*dc.assumptions, *draft.assumptions]
    for title, reason in draft.unplanned:
        assumptions.append(f"Не вошло: «{title}» — {reason}.")
    for tid, text in draft.simplify:
        c = dc.task_candidate(tid)
        if c is not None:
            assumptions.append(f"Упростить «{c.title}»: {text}")
    rev = PlanRevision(
        plan_id=plan.id,
        version=(max((r.version for r in revs), default=0) + 1),
        status=PlanStatus.PROPOSED,
        created_by=str(created_by),
        explanation=draft.explanation,
        assumptions=_dedup(assumptions),
        calendar_stale=dc.calendar_stale,
        used_ai=draft.used_ai,
        change_reason=change_reason,
        created_at=now,
    )
    session.add(rev)
    await session.flush()
    order = 0
    for it in sorted(draft.items, key=lambda i: i.start):
        order += 10
        session.add(
            PlanItem(
                revision_id=rev.id,
                kind=str(it.kind),
                task_id=it.task_id,
                habit_id=it.habit_id,
                title=it.title[:500],
                start_at=it.start,
                end_at=it.end,
                estimate_is_approximate=it.approximate,
                rationale=(it.rationale or None) and it.rationale[:1000],
                sort_order=order,
            )
        )
    for b in dc.busy:
        session.add(
            PlanItem(
                revision_id=rev.id,
                kind=str(PlanItemKind.EVENT),
                title=b.label[:500],
                start_at=b.start,
                end_at=b.end,
                estimate_is_approximate=False,
                sort_order=0,
            )
        )
    reserve = max(0, dc.free_minutes - draft.work_minutes)
    session.add(
        PlanItem(
            revision_id=rev.id,
            kind=str(PlanItemKind.BUFFER),
            title=f"Резерв ≈{reserve} мин на перерывы и непредвиденное",
            estimate_is_approximate=True,
            rationale=str(reserve),
            sort_order=1_000_000,
        )
    )
    plan.current_revision_id = rev.id
    plan.status = (
        PlanStatus.APPROVED
        if any(r.status == PlanStatus.APPROVED for r in revs)
        else PlanStatus.PROPOSED
    )
    plan.updated_at = now
    await session.flush()
    results: list[dict[str, Any]] = []
    proposed: list[Any] = []
    for mv in draft.moves:
        action = RescheduleTask(
            task_id=mv.task_id,
            planned_date=mv.planned_date,
            to_backlog=mv.to_backlog,
            reason=mv.reason or "пересмотр плана дня",
        )
        res = await actions.execute(
            session,
            ctx,
            action,
            origin=Actor.AI if draft.used_ai else Actor.SYSTEM,
            op_key=f"plan:{rev.id}:move:{mv.task_id}",
            group_key=plan_group_key(rev.id),
            ai_call_id=draft.ai_call_id,
        )
        proposed.append(action)
        results.append({"status": res.status.value, "approval_id": res.approval_id})
    if reorder:
        order_ids = [
            i.task_id
            for i in sorted(draft.items, key=lambda i: i.start)
            if i.task_id is not None and not i.fixed
        ]
        top = [tid for tid in dict.fromkeys(order_ids) if await _is_open_top(session, tid)]
        profile = await get_profile(session)
        allowed = flag(profile.autonomy_settings if profile else None, "reorder_tasks")
        if len(top) >= 2 and allowed:
            action = ReorderTasks(task_ids=top)
            res = await actions.execute(
                session,
                ctx,
                action,
                origin=Actor.AI if draft.used_ai else Actor.SYSTEM,
                op_key=f"plan:{rev.id}:reorder",
                ai_call_id=draft.ai_call_id,
            )
            proposed.append(action)
            results.append({"status": res.status.value, "result": res.result})
    await audit(
        session,
        now=now,
        actor=created_by,
        action="plan.proposed",
        entity="plan_revision",
        entity_id=rev.id,
        details={
            "plan_id": plan.id,
            "version": rev.version,
            "used_ai": draft.used_ai,
            "items": len(draft.items),
            "moves": len(draft.moves),
            "calendar_stale": dc.calendar_stale,
        },
    )
    if draft.ai_call_id is not None and proposed and ai_records is not None:
        ai_records.append((draft.ai_call_id, proposed, results))
    return rev


async def _is_open_top(session: AsyncSession, task_id: int) -> bool:
    t = await session.get(Task, task_id)
    return t is not None and t.parent_id is None and ts.is_open(t)


def _dedup(lines: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(x for x in lines if x))


async def _pending_group(session: AsyncSession, revision_id: int) -> list[PendingApproval]:
    return [
        a
        for a in await approvals.list_group(session, plan_group_key(revision_id))
        if a.status == ApprovalStatus.PENDING
    ]


async def _reject_group(session: AsyncSession, ctx: AppContext, revision_id: int) -> None:
    for a in await _pending_group(session, revision_id):
        await approvals.reject(session, ctx, a.id)


async def propose(
    ctx: AppContext,
    *,
    day: date | None = None,
    mode: Literal["plan", "replan"] = "plan",
    energy: str | None = None,
    user_request: str | None = None,
    change_reason: str | None = None,
    replan_reason: str | None = None,
    lease_name: str | None = None,
    op_key: str | None = None,
) -> int:
    """Build (AI or fallback) and store a new proposed revision; returns its id.

    Idempotent by `op_key` (a repeated render/press returns the same revision).
    Raises AIUnavailable only when a free-text `user_request` needs the AI.
    Network work happens between two short transactions."""
    async with ctx.db.transaction() as s:
        if op_key is not None:
            done = await actions.get_operation(s, op_key)
            if done is not None and done.result and done.result.get("revision_id"):
                return int(done.result["revision_id"])
        tz, _ = await _owner(s)
        day = day or local_date(ctx.clock.now(), tz)
        base_id: int | None = None
        plan = await get_plan(s, day)
        if plan is not None and mode == "replan":
            base = await approved_revision(s, plan.id)
            if base is None and plan.current_revision_id is not None:
                base = await s.get(PlanRevision, plan.current_revision_id)
            base_id = base.id if base is not None else None
        dc = await build_day_context(
            s, ctx, day=day, base_revision_id=base_id, energy=energy, user_request=user_request
        )
        dc.replan_reason = replan_reason
    draft = await make_draft(
        ctx, dc, mode=mode, lease_name=lease_name, require_ai=bool(user_request)
    )
    async with ctx.db.transaction() as s:
        if op_key is not None:
            done = await actions.get_operation(s, op_key)
            if done is not None and done.result and done.result.get("revision_id"):
                return int(done.result["revision_id"])
        pending: list[tuple[int, list[Any], list[Any]]] = []
        rev = await save_draft(
            s,
            ctx,
            dc,
            draft,
            created_by=Actor.AI if draft.used_ai else Actor.SYSTEM,
            change_reason=change_reason,
            reorder=mode == "replan",
            ai_records=pending,
        )
        if op_key is not None:
            await actions.claim_operation(s, ctx, op_key, {"revision_id": rev.id})
        rev_id = rev.id
    for call_id, proposed_actions, results in pending:
        with contextlib.suppress(Exception):
            await _record_ai_actions(ctx, call_id, proposed_actions, results)
    return rev_id


async def _record_ai_actions(
    ctx: AppContext, call_id: int, proposed: list[Any], results: list[Any]
) -> None:
    from procbot.ai.audit import record_actions

    await record_actions(ctx, call_id, proposed, results)


async def recent_proposal(session: AsyncSession, ctx: AppContext, day: date) -> PlanRevision | None:
    """A proposal created moments ago (double press of "Составить план")."""
    plan = await get_plan(session, day)
    if plan is None or plan.current_revision_id is None:
        return None
    rev = await session.get(PlanRevision, plan.current_revision_id)
    if (
        rev is not None
        and rev.status == PlanStatus.PROPOSED
        and ctx.clock.now() - rev.created_at <= REUSE_PROPOSAL_WITHIN
    ):
        return rev
    return None


# --------------------------------------------------------------------------- approve / reject


async def _apply_day_plan(
    session: AsyncSession, ctx: AppContext, action: ApplyDayPlan, actor: Actor
) -> ApplyOutcome:
    rev = await session.get(PlanRevision, action.revision_id)
    plan = await session.get(DailyPlan, action.plan_id)
    if rev is None or plan is None or rev.plan_id != plan.id:
        raise ActionError("Эта версия плана не найдена.")
    if rev.status == PlanStatus.APPROVED:
        return ApplyOutcome({"revision_id": rev.id, "already": True})
    if rev.status != PlanStatus.PROPOSED:
        raise ActionError("Эта версия плана уже не актуальна — открой «📅 Мой день».")
    now = ctx.clock.now()
    tz, _ = await _owner(session)
    if plan.local_date < local_date(now, tz):
        raise ActionError("Этот план относится к прошедшему дню.")
    changed: list[int] = []
    skipped: list[int] = []
    for it in await revision_items(session, rev.id):
        if it.kind != PlanItemKind.TASK or it.task_id is None or it.start_at is None:
            continue
        task = await session.get(Task, it.task_id)
        if task is None or not ts.is_open(task):
            continue
        if task.planned_start_at == it.start_at:
            continue
        if it.end_at is not None and it.end_at <= now:
            skipped.append(task.id)  # that slot is already over: the time is not set
            continue
        await ts.plan_task(
            session,
            ctx,
            task.id,
            actor=actor,
            planned_start_at=it.start_at,
            reason=f"план дня, версия {rev.version}",
        )
        changed.append(task.id)
    moved: list[int] = []
    for a in await _pending_group(session, rev.id):
        res = await approvals.approve(session, ctx, a.id)
        if res.status == ApprovalStatus.APPLIED:
            moved.append(int((res.approval.result or {}).get("task_id") or 0))
    for r in await revisions(session, plan.id):
        if r.id != rev.id and r.status == PlanStatus.APPROVED:
            r.status = PlanStatus.REVISED
    rev.status = PlanStatus.APPROVED
    rev.approved_at = now
    plan.status = PlanStatus.APPROVED
    plan.current_revision_id = rev.id
    plan.updated_at = now
    await audit(
        session,
        now=now,
        actor=actor,
        action="plan.approved",
        entity="plan_revision",
        entity_id=rev.id,
        details={"planned": changed, "moved": moved, "skipped_past": skipped},
    )
    return ApplyOutcome(
        {"revision_id": rev.id, "planned": changed, "moved": moved, "skipped_past": skipped}
    )


async def _describe_day_plan(
    session: AsyncSession, ctx: AppContext, action: ApplyDayPlan
) -> dict[str, Any]:
    rev = await session.get(PlanRevision, action.revision_id)
    if rev is None:
        raise ActionError("Эта версия плана не найдена.")
    tz, _ = await _owner(session)
    lines = [
        f"{_hm(i.start_at, tz)}–{_hm(i.end_at, tz)} {i.title}"
        for i in await revision_items(session, rev.id)
        if i.kind in (PlanItemKind.TASK, PlanItemKind.HABIT)
        and i.start_at is not None
        and i.end_at is not None
    ]
    return {"summary": f"Утвердить план дня (версия {rev.version})", "lines": lines}


async def approve_revision(
    session: AsyncSession, ctx: AppContext, *, plan_id: int, revision_id: int
) -> actions.ExecResult:
    """The owner pressed "Утвердить" = explicit consent (origin USER)."""
    return await actions.execute(
        session,
        ctx,
        ApplyDayPlan(plan_id=plan_id, revision_id=revision_id),
        origin=Actor.USER,
        op_key=f"dayplan:approve:{revision_id}",
    )


async def reject_revision(
    session: AsyncSession, ctx: AppContext, *, revision_id: int
) -> PlanRevision | None:
    rev = await session.get(PlanRevision, revision_id)
    if rev is None or rev.status != PlanStatus.PROPOSED:
        return rev
    plan = await session.get(DailyPlan, rev.plan_id)
    rev.status = PlanStatus.REJECTED
    await _reject_group(session, ctx, rev.id)
    if plan is not None:
        approved = await approved_revision(session, plan.id)
        plan.status = PlanStatus.APPROVED if approved is not None else PlanStatus.REJECTED
        if approved is not None:
            plan.current_revision_id = approved.id
        plan.updated_at = ctx.clock.now()
    await audit(
        session,
        now=ctx.clock.now(),
        actor=Actor.USER,
        action="plan.rejected",
        entity="plan_revision",
        entity_id=rev.id,
        details={},
    )
    return rev


# --------------------------------------------------------------------------- close


async def _item_outcome(session: AsyncSession, item: PlanItem, day: date, tz: str) -> str:
    start, end = local_day_bounds(day, tz)
    if item.kind == PlanItemKind.HABIT and item.habit_id is not None:
        occ = (
            await session.execute(
                select(HabitOccurrence.status).where(
                    HabitOccurrence.habit_id == item.habit_id, HabitOccurrence.local_date == day
                )
            )
        ).scalar_one_or_none()
        return {"done": "done", "partial": "partial"}.get(str(occ), "skipped")
    task = await session.get(Task, item.task_id) if item.task_id is not None else None
    if task is None:
        return "skipped"
    if task.status == TaskStatus.DONE and task.completed_at is not None and task.completed_at < end:
        return "done"
    worked = (
        await session.execute(
            select(TaskEvent.id).where(
                TaskEvent.task_id == task.id,
                TaskEvent.event_type.in_([TaskEventType.STARTED, TaskEventType.PROGRESSED]),
                TaskEvent.created_at >= start,
                TaskEvent.created_at < end,
            )
        )
    ).first()
    if worked is not None:
        return "partial"
    planned_day = (
        local_date(task.planned_start_at, tz)
        if task.planned_start_at is not None
        else task.planned_date
    )
    if planned_day != day:
        return "moved"
    return "skipped"


async def close_past_plans(ctx: AppContext) -> int:
    """End of day: close plans of past local days with item outcomes."""
    now = ctx.clock.now()
    closed = 0
    async with ctx.db.transaction() as s:
        tz, _ = await _owner(s)
        today = local_date(now, tz)
        plans = list(
            (
                await s.execute(
                    select(DailyPlan).where(
                        DailyPlan.local_date < today, DailyPlan.status != PlanStatus.CLOSED
                    )
                )
            ).scalars()
        )
        for plan in plans:
            approved = await approved_revision(s, plan.id)
            outcomes: dict[str, int] = {}
            if approved is not None:
                for it in await revision_items(s, approved.id):
                    if it.kind not in (PlanItemKind.TASK, PlanItemKind.HABIT):
                        continue
                    outcome = await _item_outcome(s, it, plan.local_date, tz)
                    it.outcome = outcome
                    outcomes[outcome] = outcomes.get(outcome, 0) + 1
            for r in await revisions(s, plan.id):
                if r.status in (PlanStatus.APPROVED, PlanStatus.PROPOSED):
                    r.status = PlanStatus.CLOSED
                    await _reject_group(s, ctx, r.id)
            plan.status = PlanStatus.CLOSED
            plan.updated_at = now
            await audit(
                s,
                now=now,
                actor=Actor.SYSTEM,
                action="plan.closed",
                entity="daily_plan",
                entity_id=plan.id,
                details={"outcomes": outcomes, "was_approved": approved is not None},
            )
            closed += 1
    return closed


# --------------------------------------------------------------------------- replan detection


@dataclass
class RealityCheck:
    unrealistic: bool
    missed: list[int]
    remaining_work: int
    remaining_free: int
    reason: str


async def check_reality(
    session: AsyncSession, ctx: AppContext, plan: DailyPlan
) -> RealityCheck | None:
    """Is the approved plan still doable today? None = nothing to check."""
    rev = await approved_revision(session, plan.id)
    if rev is None:
        return None
    now = ctx.clock.now()
    tz, profile = await _owner(session)
    items = [
        i
        for i in await revision_items(session, rev.id)
        if i.kind == PlanItemKind.TASK and i.task_id is not None and i.start_at and i.end_at
    ]
    missed: list[int] = []
    work = 0
    for it in items:
        task = await session.get(Task, it.task_id)
        if task is None or not ts.is_open(task):
            continue
        assert it.start_at is not None and it.end_at is not None
        minutes = int((it.end_at - it.start_at).total_seconds() // 60)
        if it.end_at <= now:
            started = (
                await session.execute(
                    select(WorkSession.id).where(
                        WorkSession.task_id == task.id,
                        WorkSession.started_at.is_not(None),
                        WorkSession.started_at >= it.start_at - timedelta(minutes=30),
                    )
                )
            ).first()
            if started is None and task.status == TaskStatus.TODO:
                missed.append(task.id)
                work += minutes
            continue
        if it.start_at < now:
            minutes = int((it.end_at - now).total_seconds() // 60)
        work += minutes
    _, day_end, _ = _day_bounds(plan.local_date, tz, profile.routine if profile else None)
    start = _ceil5(now)
    busy, _, _ = await _calendar_state(session, now, start, max(start, day_end), tz)
    free = sum(
        w.minutes
        for w in subtract([Interval(start, day_end)] if day_end > start else [], busy)
        if w.minutes >= MIN_WINDOW_MIN
    )
    reasons: list[str] = []
    if len(missed) >= MISSED_ITEMS_FOR_REPLAN:
        reasons.append(f"пропущено пунктов плана: {len(missed)}")
    if work > free:
        reasons.append(f"осталось работы ≈{work} мин, а свободного времени ≈{free} мин")
    return RealityCheck(bool(reasons), missed, work, free, "; ".join(reasons))


def replan_key(plan_id: int, revision_id: int) -> str:
    return f"replan:{plan_id}:{revision_id}"


async def detect_replan(ctx: AppContext) -> int | None:
    """Periodic job: schedule ONE REPLAN reminder for an approved plan that
    became unrealistic. Returns the reminder id (None if nothing to do)."""
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        tz, _ = await _owner(s)
        plan = await get_plan(s, local_date(now, tz))
        if plan is None or plan.status == PlanStatus.CLOSED:
            return None
        rev = await approved_revision(s, plan.id)
        if rev is None:
            return None
        if plan.current_revision_id is not None and plan.current_revision_id != rev.id:
            cur = await s.get(PlanRevision, plan.current_revision_id)
            if cur is not None and cur.status == PlanStatus.PROPOSED:
                return None  # a newer proposal waits for the owner
        active = (
            await s.execute(
                select(WorkSession.id).where(
                    WorkSession.status.in_([SessionStatus.ACTIVE, SessionStatus.PAUSED])
                )
            )
        ).first()
        if active is not None:
            return None  # no checks during work (acceptance 14); later ticks re-check
        existing = (
            await s.execute(
                select(ReminderSchedule.id).where(
                    ReminderSchedule.dedup_key == replan_key(plan.id, rev.id)
                )
            )
        ).first()
        if existing is not None:
            return None
        count = (
            await s.execute(
                select(func.count(ReminderSchedule.id)).where(
                    ReminderSchedule.dedup_key.like(f"replan:{plan.id}:%")
                )
            )
        ).scalar_one()
        if int(count) >= MAX_REPLANS_PER_DAY:
            return None
        check = await check_reality(s, ctx, plan)
        if check is None or not check.unrealistic:
            return None
        r = await schedule_reminder(
            s,
            now=now,
            category=MessageCategory.REPLAN,
            due_at=now,
            dedup_key=replan_key(plan.id, rev.id),
            plan_id=plan.id,
            payload={
                "revision_id": rev.id,
                "local_date": plan.local_date.isoformat(),
                "reason": check.reason,
                "missed": check.missed,
            },
            stale_at=local_day_bounds(plan.local_date, tz)[1],
        )
        return r.id


async def _replan_job(ctx: AppContext) -> None:
    await detect_replan(ctx)


async def _close_job(ctx: AppContext) -> None:
    await close_past_plans(ctx)


# --------------------------------------------------------------------------- rendering


@dataclass
class PlanView:
    plan: DailyPlan
    revision: PlanRevision
    items: list[PlanItem]
    tz: str
    today: date
    approved: PlanRevision | None
    pending_moves: list[PendingApproval] = field(default_factory=list)

    @property
    def is_proposal(self) -> bool:
        return self.revision.status == PlanStatus.PROPOSED


async def load_plan_view(
    session: AsyncSession,
    ctx: AppContext,
    *,
    day: date | None = None,
    revision_id: int | None = None,
) -> PlanView | None:
    tz, _ = await _owner(session)
    today = local_date(ctx.clock.now(), tz)
    if revision_id is not None:
        rev = await session.get(PlanRevision, revision_id)
        plan = await session.get(DailyPlan, rev.plan_id) if rev is not None else None
    else:
        plan = await get_plan(session, day or today)
        rev = (
            await session.get(PlanRevision, plan.current_revision_id)
            if plan is not None and plan.current_revision_id is not None
            else None
        )
    if plan is None or rev is None:
        return None
    return PlanView(
        plan=plan,
        revision=rev,
        items=await revision_items(session, rev.id),
        tz=tz,
        today=today,
        approved=await approved_revision(session, plan.id),
        pending_moves=await _pending_group(session, rev.id),
    )


STATUS_LABELS: dict[str, str] = {
    PlanStatus.PROPOSED: "предложение",
    PlanStatus.APPROVED: "утверждён",
    PlanStatus.REVISED: "заменён новой версией",
    PlanStatus.REJECTED: "отклонён",
    PlanStatus.CLOSED: "закрыт",
}
OUTCOME_LABELS = {"done": "✅", "partial": "🔸", "skipped": "⏭", "moved": "↪️"}


def render_plan_text(view: PlanView, *, header: str | None = None) -> str:
    rev, tz = view.revision, view.tz
    d = view.plan.local_date
    lines: list[str] = []
    if header:
        lines += [header, ""]
    lines.append(
        f"📅 План на {WEEKDAYS[d.weekday()]}, {d.strftime('%d.%m')} — версия {rev.version} "
        f"({STATUS_LABELS.get(rev.status, rev.status)})"
    )
    if not rev.used_ai:
        lines.append(f"🛠 {NO_AI_MARK.capitalize()} — по срокам и приоритетам.")
    if rev.calendar_stale:
        lines.append("⚠️ Календарь мог устареть или не подключён — проверь встречи.")
    if view.is_proposal and view.approved is not None:
        lines.append(f"Сейчас утверждена версия {view.approved.version}.")
    lines.append("")
    timed = sorted(
        (i for i in view.items if i.start_at is not None and i.end_at is not None),
        key=lambda i: (i.start_at, i.sort_order),  # type: ignore[arg-type,return-value]
    )
    if not any(i.kind != PlanItemKind.EVENT for i in timed):
        lines.append("В плане пока нет дел — свободное время остаётся свободным.")
    has_approx = False
    for i in timed:
        assert i.start_at is not None and i.end_at is not None
        span = f"{_hm(i.start_at, tz)}–{_hm(i.end_at, tz)}"
        if i.kind == PlanItemKind.EVENT:
            lines.append(f"{span} 📆 {i.title} (занято)")
            continue
        if i.kind == PlanItemKind.BREAK:
            lines.append(f"{span} ☕ {i.title}")
            continue
        mark = " ≈" if i.estimate_is_approximate else ""
        has_approx = has_approx or i.estimate_is_approximate
        icon = "🔁 " if i.kind == PlanItemKind.HABIT else ""
        done = f" {OUTCOME_LABELS[i.outcome]}" if i.outcome in OUTCOME_LABELS else ""
        lines.append(f"{span} {icon}{i.title}{mark}{done}")
    for i in view.items:
        if i.kind == PlanItemKind.BUFFER:
            lines.append(f"🫧 {i.title}")
    if view.pending_moves:
        lines.append("")
        lines.append("Предлагаю перенести (применится, только если утвердишь эту версию):")
        for a in view.pending_moves:
            diff = a.diff or {}
            title = str(diff.get("summary", "")).removeprefix("Перенести ").strip()
            target = next(
                (
                    str(x).removeprefix("Станет: ")
                    for x in diff.get("lines", [])
                    if str(x).startswith("Станет:")
                ),
                "",
            )
            lines.append(f"↪️ {title} → {target}" if target else f"↪️ {title}")
    if rev.explanation:
        lines += ["", "Почему так: " + rev.explanation]
    if rev.assumptions:
        lines += ["", "Допущения:"]
        lines += [f"• {a}" for a in rev.assumptions[:12]]
    if has_approx:
        lines += ["", "≈ — оценка длительности приблизительная."]
    return "\n".join(lines)


def _btn(text: str, cb: DayCb) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=cb.pack())


def render_plan_keyboard(view: PlanView) -> InlineKeyboardMarkup:
    p, r = view.plan.id, view.revision.id
    rows: list[list[InlineKeyboardButton]] = []
    if view.is_proposal and view.plan.local_date >= view.today:
        rows.append(
            [
                _btn("✅ Утвердить", DayCb(action="approve", plan_id=p, revision_id=r)),
                _btn("✏️ Изменить", DayCb(action="revise", plan_id=p, revision_id=r)),
                _btn("✖️ Отклонить", DayCb(action="reject", plan_id=p, revision_id=r)),
            ]
        )
    elif view.plan.local_date == view.today:
        rows.append(
            [
                _btn("✏️ Изменить", DayCb(action="revise", plan_id=p, revision_id=r)),
                _btn("🔄 Пересобрать", DayCb(action="replan", plan_id=p, revision_id=r)),
            ]
        )
    starts = [
        i
        for i in view.items
        if i.kind == PlanItemKind.TASK and i.task_id is not None and i.outcome is None
    ]
    if view.revision.status == PlanStatus.APPROVED:
        for i in starts[:3]:
            assert i.task_id is not None
            title = i.title if len(i.title) <= 28 else i.title[:27] + "…"
            rows.append(
                [
                    InlineKeyboardButton(
                        text=f"▶️ {title}",
                        callback_data=SessionCb(action="pick", task_id=i.task_id).pack(),
                    )
                ]
            )
    rows.append(
        [
            _btn("📋 Все дела дня", DayCb(action="tasks")),
            _btn("🗂 Версии", DayCb(action="history", plan_id=p)),
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def render_plan(view: PlanView, *, header: str | None = None) -> tuple[str, InlineKeyboardMarkup]:
    return render_plan_text(view, header=header), render_plan_keyboard(view)


async def render_history(session: AsyncSession, ctx: AppContext, plan_id: int) -> str:
    """Plan versions: proposed / approved / changed / done (FR-03)."""
    plan = await session.get(DailyPlan, plan_id)
    if plan is None:
        return "План не найден."
    tz, _ = await _owner(session)
    lines = [f"🗂 Версии плана на {plan.local_date.strftime('%d.%m')}:"]
    for r in await revisions(session, plan.id):
        items = await revision_items(session, r.id)
        work = [i for i in items if i.kind in (PlanItemKind.TASK, PlanItemKind.HABIT)]
        line = (
            f"v{r.version} — {STATUS_LABELS.get(r.status, r.status)}, "
            f"создана {fmt_local(r.created_at, tz, '%H:%M')}"
        )
        if r.approved_at is not None:
            line += f", утверждена {fmt_local(r.approved_at, tz, '%H:%M')}"
        line += f", дел: {len(work)}"
        if not r.used_ai:
            line += f" ({NO_AI_MARK})"
        done = sum(1 for i in work if i.outcome == "done")
        if any(i.outcome for i in work):
            line += f", выполнено: {done}"
        if r.change_reason:
            line += f"\n   причина: {r.change_reason}"
        lines.append(line)
    return "\n".join(lines)


# --------------------------------------------------------------------------- renderers


def _payload_day(r: ReminderSchedule, now: datetime, tz: str) -> date:
    raw = (r.payload or {}).get("local_date")
    if raw:
        try:
            return date.fromisoformat(str(raw))
        except ValueError:
            pass
    return local_date(now, tz)


async def render_morning_plan(ctx: AppContext, r: ReminderSchedule) -> Outgoing | None:
    """MORNING_PLAN: a plan proposal built outside transactions, sent through the
    normal reminder path. Falls back to the stage-A template on any failure."""
    from procbot.services import initiative
    from procbot.services.scheduler import reminder_lease

    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        tz, _ = await _owner(s)
        day = _payload_day(r, now, tz)
        today = local_date(now, tz)
        existing: int | None = None
        plan = await get_plan(s, day)
        if plan is not None and plan.current_revision_id is not None:
            # a proposal / approved plan of today already exists: show it, no new version
            cur = await s.get(PlanRevision, plan.current_revision_id)
            if cur is not None and cur.status in (PlanStatus.PROPOSED, PlanStatus.APPROVED):
                existing = cur.id
    if day != today:
        return await initiative.render_morning(ctx, r)
    try:
        rev_id = existing or await propose(
            ctx,
            day=day,
            change_reason="утреннее предложение",
            lease_name=reminder_lease(r.id),
            op_key=f"plan:morning:{r.id}",
        )
        async with ctx.db.transaction() as s:
            view = await load_plan_view(s, ctx, revision_id=rev_id)
    except Exception:
        log.exception("morning plan failed; stage-A template is used")
        return await initiative.render_morning(ctx, r)
    if view is None:
        return await initiative.render_morning(ctx, r)
    header = (
        "Доброе утро! План на сегодня уже утверждён:"
        if view.revision.status == PlanStatus.APPROVED
        else "Доброе утро! Вот предложение плана на сегодня — утверди или поправь."
    )
    text, kb = render_plan(view, header=header)
    return Outgoing(text=text, reply_markup=kb)


async def render_replan(ctx: AppContext, r: ReminderSchedule) -> Outgoing | None:
    """REPLAN: propose an updated version (keep / simplify / move). Obsolete when
    the approved version changed or the plan is no longer unrealistic."""
    from procbot.services.scheduler import reminder_lease

    payload = r.payload or {}
    op_key = f"plan:replan:{r.id}"
    async with ctx.db.transaction() as s:
        done = await actions.get_operation(s, op_key)
        existing = int((done.result or {}).get("revision_id") or 0) if done else 0
        if not existing:
            if r.plan_id is None:
                return None
            plan = await s.get(DailyPlan, r.plan_id)
            if plan is None or plan.status == PlanStatus.CLOSED:
                return None
            approved = await approved_revision(s, plan.id)
            if approved is None or approved.id != payload.get("revision_id"):
                return None
            check = await check_reality(s, ctx, plan)
            if check is None or not check.unrealistic:
                return None
            reason = check.reason
            day = plan.local_date
        else:
            reason = str(payload.get("reason") or "")
            day = None
    rev_id = existing or await propose(
        ctx,
        day=day,
        mode="replan",
        change_reason=f"план перестал быть реалистичным: {reason}",
        replan_reason=reason,
        lease_name=reminder_lease(r.id),
        op_key=op_key,
    )
    async with ctx.db.transaction() as s:
        view = await load_plan_view(s, ctx, revision_id=rev_id)
    if view is None or view.revision.status != PlanStatus.PROPOSED:
        return None
    text, kb = render_plan(
        view,
        header=(
            f"Похоже, сегодняшний план перестал быть реалистичным ({reason}). "
            "Предлагаю обновлённую версию — решать тебе:"
        ),
    )
    return Outgoing(text=text, reply_markup=kb)


# --------------------------------------------------------------------------- wiring


def register(ctx: AppContext) -> None:
    """Renderers (MORNING_PLAN, REPLAN), the ApplyDayPlan applier and jobs.

    Must run after services.initiative in wiring order so these renderers
    replace the stage-A ones for the same categories."""
    register_renderer(ctx, MessageCategory.MORNING_PLAN, render_morning_plan)
    register_renderer(ctx, MessageCategory.REPLAN, render_replan)
    actions.register_applier(
        ctx, ActionType.APPLY_DAY_PLAN, _apply_day_plan, describe=_describe_day_plan
    )
    register_job(
        ctx,
        PeriodicJob(name=REPLAN_JOB, interval_seconds=REPLAN_INTERVAL_SECONDS, run=_replan_job),
    )
    register_job(
        ctx,
        PeriodicJob(
            name=CLOSE_JOB,
            interval_seconds=CLOSE_INTERVAL_SECONDS,
            run=_close_job,
            run_on_start=True,
        ),
    )


__all__ = [
    "Candidate",
    "DayContext",
    "DraftItem",
    "Interval",
    "MoveDraft",
    "PlanDayOutput",
    "PlanDraft",
    "PlanItemOut",
    "PlanView",
    "ReplanOutput",
    "approve_revision",
    "approved_revision",
    "build_day_context",
    "check_reality",
    "close_past_plans",
    "detect_replan",
    "deterministic_plan",
    "deterministic_replan",
    "get_plan",
    "load_plan_view",
    "make_draft",
    "plan_validator",
    "propose",
    "recent_proposal",
    "register",
    "reject_revision",
    "render_history",
    "render_morning_plan",
    "render_plan",
    "render_replan",
    "revisions",
    "save_draft",
]
