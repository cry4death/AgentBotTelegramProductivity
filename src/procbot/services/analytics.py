"""AnalyticsService (FR-11): the six agreed directions, computed by CODE.

1. tasks_plans        - task completion and match to daily plans, reasons for changes;
2. start_delay        - delay between planned and confirmed start;
3. procrastination    - recurring reasons and circumstances (time of day, category);
4. help_effectiveness - outcomes of suggested help (never claims causality);
5. goals              - goal / milestone progress (+ habits);
6. hypotheses         - AI hypotheses with concrete ways to test them.

Every metric carries its source quality (точная отметка / ретроспективно /
неизвестно, acceptance 16) and an empty direction is still returned and marked
"недостаточно данных" (acceptance 22). The AI never computes numbers: the
weekly_analysis task (services.reports) receives ``ai_payload(...)`` - only the
numbers computed here - and writes text and hypotheses.

Templates (how the numbers are worded in Telegram) live in services.reports, so
report forms (open question 9) can change without touching the computation.
"""

from __future__ import annotations

import importlib
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.models import (
    Category,
    DailyPlan,
    Goal,
    Habit,
    HabitOccurrence,
    Intervention,
    InterventionOutcome,
    MemoryItem,
    Milestone,
    PlanItem,
    PlanRevision,
    Task,
    TaskEvent,
    WorkSession,
)
from procbot.domain.enums import (
    EntityStatus,
    InterventionResult,
    MemoryKind,
    MemoryStatus,
    PlanItemKind,
    PlanStatus,
    SessionStatus,
    TaskEventType,
    TaskStatus,
    TimeSource,
)
from procbot.services.profile import get_profile
from procbot.services.sessions import session_stats
from procbot.timeutil import local_date, local_day_bounds, to_local

INSUFFICIENT = "недостаточно данных"
NO_CAUSALITY = (
    "Это не доказывает причинно-следственную связь: совпадение по времени не значит, "
    "что помог именно этот способ."
)
REASONS_ARE_HYPOTHESES = "Причины — то, что ты отмечал сам (или предположения), а не диагноз."
# UNCONFIRMED technical default (docs/open-questions.md): how many marks of the
# same reason within a period make it "recurring" for a measured memory pattern.
RECURRING_REASON_MIN = 3
# How long after a planned start a confirmed start still counts as "that" start.
START_MATCH_BEFORE = timedelta(hours=3)


class Quality(StrEnum):
    EXACT = "exact"
    RETRO = "retro"
    UNKNOWN = "unknown"
    MIXED = "mixed"


QUALITY_LABELS: dict[str, str] = {
    Quality.EXACT: "точная отметка",
    Quality.RETRO: "ретроспективно",
    Quality.UNKNOWN: "неизвестно",
    Quality.MIXED: "смешанные источники",
}


def quality_of(time_source: str | None) -> Quality:
    if time_source == TimeSource.EXACT:
        return Quality.EXACT
    if time_source in (TimeSource.RETRO, TimeSource.APPROXIMATE):
        return Quality.RETRO
    return Quality.UNKNOWN


def combine(qualities: Sequence[Quality]) -> Quality:
    found = set(qualities)
    if not found:
        return Quality.UNKNOWN
    return found.pop() if len(found) == 1 else Quality.MIXED


REASON_LABELS: dict[str, str] = {
    "unclear": "непонятно, с чего начать",
    "too_big": "слишком большая задача",
    "unpleasant": "неприятно",
    "distractions": "отвлечения",
    "low_energy": "нет сил",
    "priority_conflict": "другое важнее",
    "external": "внешние дела",
    "other": "другое",
}
INTERVENTION_LABELS: dict[str, str] = {
    "tiny_step": "очень маленький шаг",
    "short_start": "короткий старт",
    "split": "разбить на шаги",
    "simplify": "упростить",
    "reschedule": "перенести",
    "defer_long": "отложить надолго",
    "change_approach": "другой подход",
    "drop": "отказаться",
    "rest": "отдых",
    "deep_dialog": "углублённый разбор",
    "remove_distractions": "убрать отвлечения",
}
DAYPARTS: tuple[tuple[str, int, int], ...] = (
    ("ночь", 0, 5),
    ("утро", 5, 12),
    ("день", 12, 17),
    ("вечер", 17, 23),
    ("ночь", 23, 24),
)
POSITIVE_RESULTS = frozenset(
    {InterventionResult.STARTED, InterventionResult.PROGRESSED, InterventionResult.COMPLETED}
)
NEGATIVE_RESULTS = frozenset({InterventionResult.NO_EFFECT, InterventionResult.DECLINED})
APPROVED_PLAN_STATUSES = frozenset({PlanStatus.APPROVED, PlanStatus.REVISED, PlanStatus.CLOSED})


def daypart(dt: datetime, tz: str | None) -> str:
    h = to_local(dt, tz).hour
    return next(name for name, a, b in DAYPARTS if a <= h < b)


# --------------------------------------------------------------------------- result types


@dataclass
class Metric:
    key: str
    label: str
    value: float | int | str | None
    unit: str = ""
    quality: Quality = Quality.EXACT
    n: int | None = None  # observations behind the value
    note: str | None = None


@dataclass
class Direction:
    key: str
    title: str
    metrics: list[Metric] = field(default_factory=list)
    details: list[str] = field(default_factory=list)  # code-generated lines
    notes: list[str] = field(default_factory=list)  # always shown (disclaimers)
    has_data: bool = False
    data: dict[str, Any] = field(default_factory=dict)  # numbers for the AI payload


DIRECTION_KEYS: tuple[str, ...] = (
    "tasks_plans",
    "start_delay",
    "procrastination",
    "help_effectiveness",
    "goals",
    "hypotheses",
)
DIRECTION_TITLES: dict[str, str] = {
    "tasks_plans": "1. Выполнение задач и дневных планов",
    "start_delay": "2. Задержка начала работы",
    "procrastination": "3. Причины и обстоятельства прокрастинации",
    "help_effectiveness": "4. Эффективность предложенной помощи",
    "goals": "5. Продвижение по целям",
    "hypotheses": "6. Гипотезы о закономерностях",
}


@dataclass(frozen=True)
class Period:
    first_day: date
    last_day: date
    tz: str | None
    start: datetime
    end: datetime

    @classmethod
    def of(cls, first_day: date, last_day: date, tz: str | None) -> Period:
        if last_day < first_day:
            first_day, last_day = last_day, first_day
        return cls(
            first_day=first_day,
            last_day=last_day,
            tz=tz,
            start=local_day_bounds(first_day, tz)[0],
            end=local_day_bounds(last_day, tz)[1],
        )

    @property
    def days(self) -> int:
        return (self.last_day - self.first_day).days + 1

    @property
    def label(self) -> str:
        if self.first_day == self.last_day:
            return self.first_day.strftime("%d.%m")
        return f"{self.first_day.strftime('%d.%m')}–{self.last_day.strftime('%d.%m')}"


@dataclass
class Analytics:
    period: Period
    directions: dict[str, Direction]
    # ids useful for linking memory sources (e.g. interventions behind a reason)
    evidence: dict[str, list[int]] = field(default_factory=dict)

    def ordered(self) -> list[Direction]:
        return [self.directions[k] for k in DIRECTION_KEYS]


def _pct(part: int, whole: int) -> int | None:
    return None if whole <= 0 else round(100 * part / whole)


def _median(values: Sequence[float]) -> float | None:
    return None if not values else float(statistics.median(values))


# --------------------------------------------------------------------------- 1. tasks / plans


async def _tasks_plans(s: AsyncSession, ctx: AppContext, p: Period) -> Direction:
    d = Direction("tasks_plans", DIRECTION_TITLES["tasks_plans"])
    now = ctx.clock.now()
    completed = list(
        (
            await s.execute(
                select(Task).where(
                    Task.status == TaskStatus.DONE,
                    Task.completed_at >= p.start,
                    Task.completed_at < p.end,
                )
            )
        )
        .scalars()
        .all()
    )
    events = list(
        (
            await s.execute(
                select(TaskEvent).where(
                    TaskEvent.created_at >= p.start, TaskEvent.created_at < p.end
                )
            )
        )
        .scalars()
        .all()
    )
    created = sum(1 for e in events if e.event_type == TaskEventType.CREATED)
    moved = [
        e
        for e in events
        if e.event_type
        in (TaskEventType.POSTPONED, TaskEventType.RESCHEDULED, TaskEventType.DEFERRED)
    ]
    progressed_tasks = {
        e.task_id
        for e in events
        if e.event_type == TaskEventType.PROGRESSED and e.task_id not in {t.id for t in completed}
    }
    cancelled = sum(1 for e in events if e.event_type == TaskEventType.CANCELLED)
    moved_reasons = Counter(
        (e.reason or "").strip()[:80] for e in moved if (e.reason or "").strip()
    )
    by_actor = Counter(e.actor for e in moved)

    # daily plans
    plans = list(
        (
            await s.execute(
                select(DailyPlan).where(
                    DailyPlan.local_date >= p.first_day, DailyPlan.local_date <= p.last_day
                )
            )
        )
        .scalars()
        .all()
    )
    planned_items = 0
    outcome_counts: Counter[str] = Counter()
    approved_plans = 0
    revision_reasons: list[str] = []
    revisions_total = 0
    for plan in plans:
        revs = list(
            (
                await s.execute(
                    select(PlanRevision)
                    .where(PlanRevision.plan_id == plan.id)
                    .order_by(PlanRevision.version)
                )
            )
            .scalars()
            .all()
        )
        revisions_total += max(0, len(revs) - 1)
        revision_reasons += [r.change_reason.strip()[:80] for r in revs if r.change_reason]
        if plan.status not in APPROVED_PLAN_STATUSES:
            continue
        approved_plans += 1
        current = next(
            (r for r in revs if r.id == plan.current_revision_id), revs[-1] if revs else None
        )
        if current is None:
            continue
        items = (
            (
                await s.execute(
                    select(PlanItem).where(
                        PlanItem.revision_id == current.id, PlanItem.kind == PlanItemKind.TASK
                    )
                )
            )
            .scalars()
            .all()
        )
        day_end = local_day_bounds(plan.local_date, p.tz)[1]
        for it in items:
            planned_items += 1
            outcome = it.outcome
            if outcome is None and it.task_id is not None:
                task = await s.get(Task, it.task_id)
                if (
                    task is not None
                    and task.status == TaskStatus.DONE
                    and task.completed_at is not None
                    and task.completed_at < day_end
                ):
                    outcome = "done"
            outcome_counts[outcome or "unknown"] += 1
    stats = await session_stats(s, start=p.start, end=p.end, now=now)
    by_source = stats.minutes_by_source()
    exact_min = round(by_source.get(TimeSource.EXACT, 0.0))
    retro_min = round(
        by_source.get(TimeSource.RETRO, 0.0) + by_source.get(TimeSource.APPROXIMATE, 0.0)
    )
    unknown_min = round(by_source.get(TimeSource.UNKNOWN, 0.0))

    d.metrics.append(Metric("completed", "Завершено задач", len(completed), n=len(completed)))
    if progressed_tasks:
        d.metrics.append(
            Metric(
                "partial",
                "С частичным прогрессом (не считается выполнением)",
                len(progressed_tasks),
            )
        )
    d.metrics.append(Metric("created", "Добавлено задач", created))
    d.metrics.append(Metric("moved", "Переносов и отложенных", len(moved)))
    if cancelled:
        d.metrics.append(Metric("cancelled", "Отменено", cancelled))
    if stats.items:
        d.metrics.append(Metric("work_exact", "Работа по таймеру", exact_min, "мин", Quality.EXACT))
        if retro_min:
            d.metrics.append(
                Metric(
                    "work_retro",
                    "Работа, отмеченная задним числом",
                    retro_min,
                    "мин",
                    Quality.RETRO,
                    note="приблизительно, не точный таймер",
                )
            )
        if unknown_min:
            d.metrics.append(
                Metric(
                    "work_unknown",
                    "Работа, источник неизвестен",
                    unknown_min,
                    "мин",
                    Quality.UNKNOWN,
                )
            )
    if stats.unconfirmed_count:
        d.details.append(
            f"Запланированных сессий без подтверждённого начала: {stats.unconfirmed_count} "
            "(не считаются отработанным временем)."
        )
    done_in_plan = outcome_counts.get("done", 0)
    if approved_plans:
        unknown = outcome_counts.get("unknown", 0)
        d.metrics.append(
            Metric(
                "plan_match",
                "Выполнено из утверждённых планов",
                f"{done_in_plan} из {planned_items}"
                + (f" ({_pct(done_in_plan, planned_items)}%)" if planned_items else ""),
                quality=Quality.MIXED if unknown else Quality.EXACT,
                n=planned_items,
                note=f"у {unknown} пунктов нет отметки — исход неизвестен" if unknown else None,
            )
        )
        parts = [
            f"{label}: {outcome_counts[k]}"
            for k, label in (
                ("partial", "частично"),
                ("moved", "перенесено"),
                ("skipped", "пропущено"),
            )
            if outcome_counts.get(k)
        ]
        if parts:
            d.details.append("Пункты плана: " + ", ".join(parts) + ".")
    else:
        d.details.append("Утверждённых дневных планов за период нет — сравнивать не с чем.")
    if revisions_total:
        line = f"План пересматривался {revisions_total} раз"
        if revision_reasons:
            line += ": " + "; ".join(list(dict.fromkeys(revision_reasons))[:3])
        d.details.append(line + ".")
    if moved_reasons:
        d.details.append(
            "Причины переносов: "
            + "; ".join(f"«{r}» — {n}" for r, n in moved_reasons.most_common(4))
            + "."
        )
    if moved and by_actor:
        actors = {"user": "тобой", "system": "системой", "ai": "по предложению ИИ"}
        d.details.append(
            "Кем перенесено: "
            + ", ".join(f"{actors.get(a, a)} — {n}" for a, n in by_actor.most_common())
            + "."
        )
    d.has_data = bool(completed or created or moved or plans or stats.items or progressed_tasks)
    d.data = {
        "completed": len(completed),
        "partial_progress_tasks": len(progressed_tasks),
        "created": created,
        "moved": len(moved),
        "cancelled": cancelled,
        "approved_plans": approved_plans,
        "planned_task_items": planned_items,
        "plan_items_done": done_in_plan,
        "plan_items_unknown": outcome_counts.get("unknown", 0),
        "plan_match_pct": _pct(done_in_plan, planned_items),
        "plan_revisions": revisions_total,
        "work_minutes_exact": exact_min,
        "work_minutes_retro_approximate": retro_min,
        "unconfirmed_sessions": stats.unconfirmed_count,
    }
    return d


# --------------------------------------------------------------------------- 2. start delay


@dataclass(frozen=True)
class DelayObs:
    task_id: int | None
    planned: datetime
    started: datetime | None
    quality: Quality

    @property
    def minutes(self) -> float | None:
        if self.started is None:
            return None
        return (self.started - self.planned).total_seconds() / 60


async def start_delays(s: AsyncSession, p: Period) -> list[DelayObs]:
    """Planned starts in the period and the confirmed start that answered them.

    Planned moment: the session's own planned start, else the approved plan
    item's start, else the task's planned start. A planned start without a
    confirmed session has ``started=None`` (delay unknown, acceptance 15)."""
    obs: list[DelayObs] = []
    used: set[int] = set()
    started_statuses = [SessionStatus.ACTIVE, SessionStatus.PAUSED, SessionStatus.FINISHED]
    # (a) sessions that carry their own planned start
    for ws in (
        (
            await s.execute(
                select(WorkSession).where(
                    WorkSession.planned_start_at >= p.start,
                    WorkSession.planned_start_at < p.end,
                    WorkSession.status.in_([*started_statuses, SessionStatus.UNCONFIRMED]),
                )
            )
        )
        .scalars()
        .all()
    ):
        assert ws.planned_start_at is not None
        if ws.status == SessionStatus.UNCONFIRMED or ws.started_at is None:
            obs.append(DelayObs(ws.task_id, ws.planned_start_at, None, Quality.UNKNOWN))
        else:
            obs.append(
                DelayObs(ws.task_id, ws.planned_start_at, ws.started_at, quality_of(ws.time_source))
            )
            used.add(ws.id)
    planned_tasks = {o.task_id for o in obs if o.task_id is not None}
    # (b) plan items with a start time, (c) tasks with an exact planned start
    planned: dict[int, datetime] = {}
    for it, plan_status in (
        await s.execute(
            select(PlanItem, DailyPlan.status)
            .join(PlanRevision, PlanRevision.id == PlanItem.revision_id)
            .join(DailyPlan, DailyPlan.id == PlanRevision.plan_id)
            .where(
                DailyPlan.current_revision_id == PlanRevision.id,
                PlanItem.kind == PlanItemKind.TASK,
                PlanItem.task_id.is_not(None),
                PlanItem.start_at >= p.start,
                PlanItem.start_at < p.end,
            )
        )
    ).all():
        if plan_status in APPROVED_PLAN_STATUSES and it.task_id not in planned_tasks:
            assert it.task_id is not None and it.start_at is not None
            planned.setdefault(it.task_id, it.start_at)
    for task in (
        (
            await s.execute(
                select(Task).where(
                    Task.planned_start_at >= p.start,
                    Task.planned_start_at < p.end,
                    Task.status != TaskStatus.CANCELLED,
                )
            )
        )
        .scalars()
        .all()
    ):
        if task.id not in planned_tasks and task.planned_start_at is not None:
            planned.setdefault(task.id, task.planned_start_at)
    for task_id, moment in sorted(planned.items(), key=lambda kv: kv[1]):
        day_end = local_day_bounds(local_date(moment, p.tz), p.tz)[1]
        ws = (
            (
                await s.execute(
                    select(WorkSession)
                    .where(
                        WorkSession.task_id == task_id,
                        WorkSession.status.in_(started_statuses),
                        WorkSession.started_at >= moment - START_MATCH_BEFORE,
                        WorkSession.started_at < day_end,
                    )
                    .order_by(WorkSession.started_at)
                )
            )
            .scalars()
            .all()
        )
        match = next((w for w in ws if w.id not in used), None)
        if match is None:
            obs.append(DelayObs(task_id, moment, None, Quality.UNKNOWN))
        else:
            used.add(match.id)
            obs.append(DelayObs(task_id, moment, match.started_at, quality_of(match.time_source)))
    return obs


def fmt_delay(m: float) -> str:
    v = round(m)
    if v == 0:
        return "вовремя"
    return f"+{v} мин" if v > 0 else f"на {-v} мин раньше"


async def _start_delay(s: AsyncSession, ctx: AppContext, p: Period) -> Direction:
    d = Direction("start_delay", DIRECTION_TITLES["start_delay"])
    obs = await start_delays(s, p)
    exact = [o.minutes for o in obs if o.quality == Quality.EXACT and o.minutes is not None]
    retro = [o.minutes for o in obs if o.quality == Quality.RETRO and o.minutes is not None]
    unknown = [o for o in obs if o.started is None or o.quality == Quality.UNKNOWN]
    med_exact, med_retro = _median(exact), _median(retro)
    if exact:
        assert med_exact is not None
        d.metrics.append(
            Metric(
                "delay_exact",
                "Задержка старта (медиана)",
                fmt_delay(med_exact),
                quality=Quality.EXACT,
                n=len(exact),
            )
        )
        late = sum(1 for m in exact if m > 15)
        if late:
            d.details.append(
                f"Позже плана больше чем на 15 мин: {late} из {len(exact)} (по таймеру)."
            )
    if retro:
        assert med_retro is not None
        d.metrics.append(
            Metric(
                "delay_retro",
                "Задержка старта по отметкам задним числом (медиана)",
                fmt_delay(med_retro),
                quality=Quality.RETRO,
                n=len(retro),
                note="время приблизительное, это не точный таймер",
            )
        )
    if unknown:
        d.metrics.append(
            Metric(
                "delay_unknown",
                "Запланированных стартов без подтверждённого начала",
                len(unknown),
                quality=Quality.UNKNOWN,
                note="задержка неизвестна",
            )
        )
    d.has_data = bool(obs)
    d.data = {
        "planned_starts": len(obs),
        "exact_n": len(exact),
        "exact_median_min": None if med_exact is None else round(med_exact),
        "retro_n": len(retro),
        "retro_median_min": None if med_retro is None else round(med_retro),
        "unconfirmed_n": len(unknown),
    }
    return d


# --------------------------------------------------------------------------- 3. procrastination


async def _procrastination(
    s: AsyncSession, ctx: AppContext, p: Period, evidence: dict[str, list[int]]
) -> Direction:
    d = Direction("procrastination", DIRECTION_TITLES["procrastination"])
    ivs = list(
        (
            await s.execute(
                select(Intervention).where(
                    Intervention.created_at >= p.start,
                    Intervention.created_at < p.end,
                    Intervention.reason_category.is_not(None),
                )
            )
        )
        .scalars()
        .all()
    )
    obstacles = list(
        (
            await s.execute(
                select(TaskEvent).where(
                    TaskEvent.created_at >= p.start,
                    TaskEvent.created_at < p.end,
                    TaskEvent.event_type == TaskEventType.OBSTACLE_REPORTED,
                )
            )
        )
        .scalars()
        .all()
    )
    # (when, task_id, reason, origin, source) - one incident may be recorded both
    # as an intervention and an obstacle event: merge marks within 10 minutes.
    marks: list[tuple[datetime, int | None, str, str, str, int]] = []
    for iv in ivs:
        assert iv.reason_category is not None
        marks.append(
            (
                iv.created_at,
                iv.task_id,
                iv.reason_category,
                iv.reason_origin or "user_said",
                "iv",
                iv.id,
            )
        )
    for ev in obstacles:
        reason = str((ev.payload or {}).get("category") or ev.reason or "other")
        dup = any(
            m[1] == ev.task_id
            and m[2] == reason
            and abs((m[0] - ev.created_at).total_seconds()) < 600
            for m in marks
        )
        if not dup:
            marks.append((ev.created_at, ev.task_id, reason, "user_said", "ev", ev.id))
    reasons = Counter(m[2] for m in marks)
    by_part: Counter[str] = Counter(daypart(m[0], p.tz) for m in marks)
    task_ids = {m[1] for m in marks if m[1] is not None}
    cat_names: dict[int, str] = {}
    if task_ids:
        for t_id, c_name in (
            await s.execute(
                select(Task.id, Category.name)
                .join(Category, Category.id == Task.category_id)
                .where(Task.id.in_(task_ids))
            )
        ).all():
            cat_names[int(t_id)] = str(c_name)
    by_cat: Counter[str] = Counter(
        cat_names.get(m[1], "без категории") for m in marks if m[1] is not None
    )
    ai_marks = sum(1 for m in marks if m[3] == "ai_inferred")
    # repeated postponements of the same task
    moved = list(
        (
            await s.execute(
                select(TaskEvent.task_id).where(
                    TaskEvent.created_at >= p.start,
                    TaskEvent.created_at < p.end,
                    TaskEvent.event_type.in_([TaskEventType.POSTPONED, TaskEventType.RESCHEDULED]),
                )
            )
        )
        .scalars()
        .all()
    )
    repeat = {tid: n for tid, n in Counter(moved).items() if n >= 2}
    for reason, n in reasons.most_common(5):
        d.metrics.append(
            Metric(f"reason:{reason}", f"«{REASON_LABELS.get(reason, reason)}»", n, "раз", n=n)
        )
    if marks:
        d.details.append(
            "Время суток: " + ", ".join(f"{k} — {n}" for k, n in by_part.most_common()) + "."
        )
        if by_cat:
            d.details.append(
                "Категории задач: "
                + ", ".join(f"{k} — {n}" for k, n in by_cat.most_common(4))
                + "."
            )
    if repeat:
        titles = []
        for tid, n in sorted(repeat.items(), key=lambda kv: -kv[1])[:3]:
            t = await s.get(Task, tid)
            titles.append(f"«{t.title if t else '#' + str(tid)}» — {n} раза")
        d.details.append("Повторно переносились: " + ", ".join(titles) + ".")
    recurring = {r: n for r, n in reasons.items() if n >= 2}
    if marks and not recurring and not repeat:
        d.details.append("Повторяющихся причин пока не видно — отметок слишком мало.")
    d.notes.append(REASONS_ARE_HYPOTHESES)
    if ai_marks:
        d.notes.append(f"{ai_marks} из {len(marks)} отметок — предположения ИИ, не твои слова.")
    for reason in recurring:
        evidence[f"reason:{reason}"] = [m[5] for m in marks if m[2] == reason and m[4] == "iv"]
        evidence[f"reason_ev:{reason}"] = [m[5] for m in marks if m[2] == reason and m[4] == "ev"]
    d.has_data = bool(marks or repeat)
    d.data = {
        "marks": len(marks),
        "reasons": {REASON_LABELS.get(k, k): n for k, n in reasons.most_common()},
        "reason_codes": dict(reasons.most_common()),
        "by_daypart": dict(by_part.most_common()),
        "by_category": dict(by_cat.most_common()),
        "ai_inferred_marks": ai_marks,
        "tasks_postponed_repeatedly": len(repeat),
    }
    return d


# --------------------------------------------------------------------------- 4. effectiveness


async def _effectiveness(s: AsyncSession, ctx: AppContext, p: Period) -> Direction:
    d = Direction("help_effectiveness", DIRECTION_TITLES["help_effectiveness"])
    ivs = list(
        (
            await s.execute(
                select(Intervention).where(
                    Intervention.created_at >= p.start, Intervention.created_at < p.end
                )
            )
        )
        .scalars()
        .all()
    )
    per_kind: dict[str, dict[str, int]] = defaultdict(
        lambda: {"offered": 0, "positive": 0, "negative": 0, "unknown": 0}
    )
    qualities: list[Quality] = []
    for iv in ivs:
        outcomes = list(
            (
                await s.execute(
                    select(InterventionOutcome)
                    .where(InterventionOutcome.intervention_id == iv.id)
                    .order_by(InterventionOutcome.recorded_at, InterventionOutcome.id)
                )
            )
            .scalars()
            .all()
        )
        row = per_kind[iv.kind]
        row["offered"] += 1
        results = {o.result for o in outcomes}
        if results & POSITIVE_RESULTS:
            row["positive"] += 1
            qualities += [
                quality_of(o.time_source) for o in outcomes if o.result in POSITIVE_RESULTS
            ]
        elif results & NEGATIVE_RESULTS:
            row["negative"] += 1
            qualities += [
                quality_of(o.time_source) for o in outcomes if o.result in NEGATIVE_RESULTS
            ]
        else:
            row["unknown"] += 1
    q = combine(qualities) if qualities else Quality.UNKNOWN
    for kind, row in sorted(per_kind.items(), key=lambda kv: -kv[1]["offered"]):
        known = row["positive"] + row["negative"]
        label = INTERVENTION_LABELS.get(kind, kind)
        value = (
            f"после {row['positive']} из {known} начал(а) или продвинулся(ась)"
            if known
            else "исход не отмечен"
        )
        note = None
        if known and known < 3:
            note = "очень мало наблюдений"
        if row["unknown"]:
            note = (note + "; " if note else "") + f"без отметки исхода: {row['unknown']}"
        d.metrics.append(
            Metric(
                f"help:{kind}",
                f"«{label}» (предложено {row['offered']})",
                value,
                quality=q,
                n=known,
                note=note,
            )
        )
    d.notes.append(NO_CAUSALITY)
    d.has_data = bool(ivs)
    d.data = {
        INTERVENTION_LABELS.get(k, k): {
            "offered": v["offered"],
            "positive": v["positive"],
            "negative": v["negative"],
            "unknown": v["unknown"],
            "positive_pct": _pct(v["positive"], v["positive"] + v["negative"]),
        }
        for k, v in per_kind.items()
    }
    return d


# --------------------------------------------------------------------------- 5. goals


def _optional(module: str) -> Any | None:
    """The goals/habits services (stage C) are used when present."""
    try:
        return importlib.import_module(module)
    except ImportError:  # pragma: no cover - module not built
        return None


async def _goal_rows_fallback(
    s: AsyncSession, ctx: AppContext, p: Period, d: Direction, goals: Sequence[Goal]
) -> list[dict[str, Any]]:
    stats = await session_stats(s, start=p.start, end=p.end, now=ctx.clock.now())
    rows: list[dict[str, Any]] = []
    for g in goals:
        ms = list(
            (await s.execute(select(Milestone).where(Milestone.goal_id == g.id))).scalars().all()
        )
        ms_done = sum(1 for m in ms if m.status == EntityStatus.DONE)
        tasks = list((await s.execute(select(Task).where(Task.goal_id == g.id))).scalars().all())
        ids = {t.id for t in tasks}
        done_now = sum(
            1
            for t in tasks
            if t.status == TaskStatus.DONE
            and t.completed_at is not None
            and p.start <= t.completed_at < p.end
        )
        exact = round(sum(i.net_minutes for i in stats.items if i.task_id in ids and i.is_exact))
        retro = round(
            sum(i.net_minutes for i in stats.items if i.task_id in ids and not i.is_exact)
        )
        parts = [f"этапы {ms_done}/{len(ms)}"] if ms else []
        parts.append(f"задач завершено за период: {done_now}")
        if exact:
            parts.append(f"{exact} мин по таймеру")
        if retro:
            parts.append(f"~{retro} мин задним числом (приблизительно)")
        if not ms and not tasks:
            parts = ["к цели пока не привязаны задачи и этапы"]
        d.details.append(f"«{g.title}»: " + ", ".join(parts) + ".")
        rows.append(
            {
                "goal": g.title,
                "milestones_done": ms_done,
                "milestones_total": len(ms),
                "tasks_done_in_period": done_now,
                "minutes_exact": exact,
                "minutes_retro_approximate": retro,
            }
        )
    return rows


async def _goals(s: AsyncSession, ctx: AppContext, p: Period) -> Direction:
    d = Direction("goals", DIRECTION_TITLES["goals"])
    goals = list(
        (await s.execute(select(Goal).where(Goal.status == EntityStatus.ACTIVE).order_by(Goal.id)))
        .scalars()
        .all()
    )
    shown = goals[:10]
    gmod = _optional("procbot.services.goals")
    goal_data: list[dict[str, Any]] = []
    alignment: dict[str, Any] | None = None
    moved = 0
    has_retro = False
    if gmod is not None and all(
        hasattr(gmod, n) for n in ("goal_progress", "progress_short", "goal_alignment")
    ):
        for g in shown:
            prog = await gmod.goal_progress(s, ctx, g.id)
            d.details.append(f"«{g.title}»: {gmod.progress_short(prog)}.")
            goal_data.append(prog.as_dict())
        if goals:
            rep = await gmod.goal_alignment(s, ctx, p.start, p.end)
            alignment = rep.as_dict()
            moved = sum(1 for a in rep.goals if a.tasks_done or a.minutes_exact or a.minutes_other)
            has_retro = any(a.minutes_other for a in rep.goals)
            if hasattr(gmod, "alignment_lines"):
                d.details += list(gmod.alignment_lines(rep))
    else:
        goal_data = await _goal_rows_fallback(s, ctx, p, d, shown)
        moved = sum(
            1
            for r in goal_data
            if r["tasks_done_in_period"] or r["minutes_exact"] or r["minutes_retro_approximate"]
        )
        has_retro = any(r["minutes_retro_approximate"] for r in goal_data)
    if len(goals) > len(shown):
        d.details.append(f"…и ещё целей: {len(goals) - len(shown)}.")
    if goals:
        d.metrics.append(Metric("goals_active", "Активных целей", len(goals)))
        d.metrics.append(
            Metric(
                "goals_moved",
                "Целей с продвижением за период",
                moved,
                quality=Quality.MIXED if has_retro else Quality.EXACT,
                note="часть времени отмечена задним числом" if has_retro else None,
            )
        )
    # habits: supporting data for the goals direction (not a separate section)
    habit_data: dict[str, Any] = {}
    hmod = _optional("procbot.services.habits")
    habits = list(
        (
            await s.execute(
                select(Habit).where(Habit.status == EntityStatus.ACTIVE).order_by(Habit.id)
            )
        )
        .scalars()
        .all()
    )[:5]
    if habits and hmod is not None and hasattr(hmod, "habit_stability"):
        for h in habits:
            st = await hmod.habit_stability(s, ctx, h.id, days=p.days, end=p.last_day)
            text = hmod.stability_text(st) if hasattr(hmod, "stability_text") else ""
            d.details.append(f"Привычка «{h.title}»: {text}.")
            habit_data[h.title] = st.as_dict()
    elif habits:
        occ = (
            (
                await s.execute(
                    select(HabitOccurrence).where(
                        HabitOccurrence.local_date >= p.first_day,
                        HabitOccurrence.local_date <= p.last_day,
                    )
                )
            )
            .scalars()
            .all()
        )
        for h in habits:
            c = Counter(o.status for o in occ if o.habit_id == h.id)
            habit_data[h.title] = dict(c)
            if c:
                d.details.append(
                    f"Привычка «{h.title}»: выполнено {c.get('done', 0)}, частично "
                    f"{c.get('partial', 0)}, пропущено {c.get('skipped', 0) + c.get('missed', 0)}."
                )
    d.has_data = bool(goals or habit_data)
    d.data = {"goals": goal_data, "alignment": alignment, "habits": habit_data}
    return d


# --------------------------------------------------------------------------- 6. hypotheses


def hypotheses_direction(items: Sequence[MemoryItem], *, new_ids: Sequence[int] = ()) -> Direction:
    d = Direction("hypotheses", DIRECTION_TITLES["hypotheses"])
    d.has_data = bool(items)
    d.data = {
        "hypotheses": [
            {
                "id": it.id,
                "content": it.content,
                "origin": it.origin,
                "how_to_check": it.verification_hint,
            }
            for it in items
        ],
        "new_ids": list(new_ids),
    }
    d.notes.append("Это предположения, а не факты: их можно оспорить кнопкой «Не согласен».")
    return d


async def _hypotheses(s: AsyncSession) -> Direction:
    items = list(
        (
            await s.execute(
                select(MemoryItem)
                .where(
                    MemoryItem.kind == MemoryKind.HYPOTHESIS,
                    MemoryItem.status == MemoryStatus.ACTIVE,
                )
                .order_by(MemoryItem.updated_at.desc(), MemoryItem.id.desc())
                .limit(10)
            )
        )
        .scalars()
        .all()
    )
    return hypotheses_direction(items)


# --------------------------------------------------------------------------- public API


async def owner_tz(s: AsyncSession) -> str | None:
    p = await get_profile(s)
    return p.timezone if p is not None else None


async def compute(s: AsyncSession, ctx: AppContext, period: Period) -> Analytics:
    """All six directions for the period (each one present even when empty)."""
    evidence: dict[str, list[int]] = {}
    directions = {
        "tasks_plans": await _tasks_plans(s, ctx, period),
        "start_delay": await _start_delay(s, ctx, period),
        "procrastination": await _procrastination(s, ctx, period, evidence),
        "help_effectiveness": await _effectiveness(s, ctx, period),
        "goals": await _goals(s, ctx, period),
        "hypotheses": await _hypotheses(s),
    }
    return Analytics(period=period, directions=directions, evidence=evidence)


def ai_payload(a: Analytics) -> dict[str, Any]:
    """Only numbers computed above (+ labels). The model gets nothing to count."""
    return {
        "period": {
            "from": a.period.first_day.strftime("%d.%m"),
            "to": a.period.last_day.strftime("%d.%m"),
            "days": a.period.days,
        },
        "directions": {
            d.key: {"title": d.title, "has_data": d.has_data, "numbers": d.data}
            for d in a.ordered()
        },
        "time_source_rule": "exact = точный таймер; retro/approximate = задним числом, приблизительно",
    }


def recurring_reasons(a: Analytics, minimum: int = RECURRING_REASON_MIN) -> dict[str, int]:
    codes: dict[str, int] = a.directions["procrastination"].data.get("reason_codes", {})
    return {k: n for k, n in codes.items() if n >= minimum}


# --------------------------------------------------------------------------- one day (evening)


@dataclass
class DaySummary:
    day: date
    tz: str | None
    done: list[Task]
    in_progress: list[Task]  # started / partial progress, not done
    not_started: list[Task]  # planned for the day, no confirmed start
    postponed_today: list[Task]
    work_exact_min: int
    work_retro_min: int
    unconfirmed_sessions: int
    delays: list[DelayObs]
    plan_revised: bool

    @property
    def deviations(self) -> list[str]:
        """Substantial deviations (UNCONFIRMED rule, see docs/open-questions.md):
        a planned task not started, a planned start not confirmed, a postponed
        task, a plan revised during the day. A small delay alone is not one."""
        out: list[str] = []
        if self.not_started:
            out.append(f"не начато из запланированного: {len(self.not_started)}")
        unconf = [o for o in self.delays if o.started is None]
        if unconf:
            out.append(f"старт не подтверждён: {len(unconf)}")
        if self.postponed_today:
            out.append(f"перенесено: {len(self.postponed_today)}")
        if self.plan_revised:
            out.append("план пересматривался")
        return out


async def day_summary(s: AsyncSession, ctx: AppContext, day: date) -> DaySummary:
    tz = await owner_tz(s)
    p = Period.of(day, day, tz)
    now = ctx.clock.now()
    done = list(
        (
            await s.execute(
                select(Task).where(
                    Task.status == TaskStatus.DONE,
                    Task.completed_at >= p.start,
                    Task.completed_at < p.end,
                )
            )
        )
        .scalars()
        .all()
    )
    done_ids = {t.id for t in done}
    stats = await session_stats(s, start=p.start, end=p.end, now=now)
    worked_ids = {i.task_id for i in stats.items if i.task_id is not None}
    progressed_ids = set(
        (
            await s.execute(
                select(TaskEvent.task_id).where(
                    TaskEvent.created_at >= p.start,
                    TaskEvent.created_at < p.end,
                    TaskEvent.event_type.in_([TaskEventType.PROGRESSED, TaskEventType.STARTED]),
                )
            )
        )
        .scalars()
        .all()
    )
    in_progress = [
        t
        for tid in sorted((worked_ids | progressed_ids) - done_ids)
        if (t := await s.get(Task, tid)) is not None and t.status != TaskStatus.CANCELLED
    ]
    postponed_ids = set(
        (
            await s.execute(
                select(TaskEvent.task_id).where(
                    TaskEvent.created_at >= p.start,
                    TaskEvent.created_at < p.end,
                    TaskEvent.event_type.in_(
                        [TaskEventType.POSTPONED, TaskEventType.RESCHEDULED, TaskEventType.DEFERRED]
                    ),
                )
            )
        )
        .scalars()
        .all()
    )
    postponed = [t for tid in sorted(postponed_ids) if (t := await s.get(Task, tid)) is not None]
    planned: list[Task] = []
    try:
        from procbot.services.tasks import today_buckets

        b = await today_buckets(s, ctx, day=day)
        planned = [*b.planned, *b.due_today]
    except (ImportError, AttributeError, TypeError):  # pragma: no cover - defensive
        planned = []
    seen: set[int] = set()
    not_started = []
    for t in planned:
        if t.id in seen or t.id in done_ids or t.id in worked_ids or t.id in progressed_ids:
            continue
        seen.add(t.id)
        if t.status == TaskStatus.TODO and t.parent_id is None:
            not_started.append(t)
    delays = await start_delays(s, p)
    plan = (
        await s.execute(select(DailyPlan).where(DailyPlan.local_date == day))
    ).scalar_one_or_none()
    revised = False
    if plan is not None:
        revs = (
            await s.execute(select(PlanRevision.id).where(PlanRevision.plan_id == plan.id))
        ).all()
        revised = len(revs) > 1
    by_source = stats.minutes_by_source()
    return DaySummary(
        day=day,
        tz=tz,
        done=done,
        in_progress=in_progress,
        not_started=not_started,
        postponed_today=postponed,
        work_exact_min=round(by_source.get(TimeSource.EXACT, 0.0)),
        work_retro_min=round(
            by_source.get(TimeSource.RETRO, 0.0) + by_source.get(TimeSource.APPROXIMATE, 0.0)
        ),
        unconfirmed_sessions=stats.unconfirmed_count,
        delays=delays,
        plan_revised=revised,
    )
