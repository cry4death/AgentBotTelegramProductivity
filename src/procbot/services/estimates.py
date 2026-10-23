"""Estimate calibration from history (FR-03: compare predicted durations with
the actual time of similar finished work and gradually improve estimates).

How it works:
- a *sample* is a finished task that had an estimate (user / AI / history) and
  linked work sessions (its own and, for a top-level task, its subtasks');
- the actual duration is the NET time of those sessions (pauses excluded) from
  `services.sessions.session_stats`;
- only live timer sessions are exact. Retro / approximate entries still count,
  but with a lower weight, and a calibration built mostly from them is flagged
  (acceptance 15/16: retro time is never presented as a precise measurement);
- the factor is a weighted geometric mean of actual/estimate, shrunk toward 1.0
  by a prior (few samples -> barely any correction) and clamped;
- scope: the task's category if it has enough data, else its context, else all
  tasks, else no correction.

Every adjusted estimate is shown with "≈" and a short explanation. Read-only:
this module writes nothing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.models import Category, Context, Task
from procbot.domain.enums import EstimateSource, TaskStatus, TimeSource
from procbot.services.sessions import session_stats

# Weight of one minute of work by time source (exact timer = 1).
SOURCE_WEIGHTS: dict[str, float] = {
    TimeSource.EXACT: 1.0,
    TimeSource.RETRO: 0.5,
    TimeSource.APPROXIMATE: 0.25,
    TimeSource.UNKNOWN: 0.0,
}
LOOKBACK_DAYS = 90
# Pseudo-weight of "estimates are right" (factor 1.0) mixed into every average.
PRIOR_WEIGHT = 2.0
# Minimal sample weight for a narrower scope (category / context) to be used.
MIN_SCOPE_WEIGHT = 2.0
MIN_FACTOR, MAX_FACTOR = 0.5, 3.0
# Ratios outside this range are treated as data errors (forgotten timers...).
MIN_RATIO, MAX_RATIO = 0.1, 10.0
# Technical planning assumption when a task has no estimate at all. UNCONFIRMED:
# listed in docs/open-questions.md and always shown as an approximate value.
DEFAULT_ESTIMATE_MIN = 30
# A factor this close to 1.0 is not applied (noise).
NEUTRAL_BAND = 0.1


@dataclass(frozen=True)
class Sample:
    task_id: int
    estimate_minutes: int
    actual_minutes: float
    weight: float  # minutes-weighted source weight, 0..1
    exact: bool  # all time came from live timers
    category_id: int | None
    context_id: int | None
    estimate_source: str | None

    @property
    def ratio(self) -> float:
        return self.actual_minutes / self.estimate_minutes


@dataclass(frozen=True)
class Calibration:
    factor: float = 1.0
    scope: str = "none"  # category / context / all / none
    scope_label: str = ""
    weight: float = 0.0
    samples: int = 0
    exact_samples: int = 0

    @property
    def applies(self) -> bool:
        return self.samples > 0 and abs(self.factor - 1.0) >= NEUTRAL_BAND

    @property
    def mostly_approximate(self) -> bool:
        return self.samples > 0 and self.exact_samples * 2 < self.samples

    def explanation(self) -> str:
        if self.samples == 0:
            return "истории выполнения похожих задач пока нет"
        where = {
            "category": f"в категории «{self.scope_label}»",
            "context": f"в контексте «{self.scope_label}»",
            "all": "по всем задачам",
        }.get(self.scope, "")
        exact = f", {self.exact_samples} с точным таймером" if self.exact_samples else ""
        base = f"по {self.samples} завершённым задачам {where}{exact}".replace("  ", " ")
        if not self.applies:
            text = f"{base} оценки в среднем совпадают с фактом"
        elif self.factor > 1:
            text = f"{base} на деле уходит примерно в {self.factor:.1f} раза больше оценки"
        else:
            text = f"{base} на деле уходит примерно {self.factor:.1f} от оценки"
        if self.mostly_approximate:
            text += " (в основном по отметкам задним числом — неточно)"
        return text


@dataclass(frozen=True)
class AdjustedEstimate:
    minutes: int
    base_minutes: int | None
    approximate: bool
    note: str
    source: str  # user / ai / history / subtasks / default
    calibration: Calibration = field(default_factory=Calibration)
    spent_minutes: float = 0.0

    @property
    def label(self) -> str:
        return f"≈{self.minutes} мин" if self.approximate else f"{self.minutes} мин"


# --------------------------------------------------------------------------- samples


async def collect_samples(
    session: AsyncSession, ctx: AppContext, *, now: datetime | None = None
) -> list[Sample]:
    """Finished tasks with an estimate and recorded work, last LOOKBACK_DAYS."""
    now = now or ctx.clock.now()
    since = now - timedelta(days=LOOKBACK_DAYS)
    done = list(
        (
            await session.execute(
                select(Task).where(
                    Task.status == TaskStatus.DONE,
                    Task.estimate_minutes.is_not(None),
                    Task.completed_at >= since,
                )
            )
        ).scalars()
    )
    if not done:
        return []
    # Sessions may start well before completion: look a further month back.
    stats = await session_stats(session, start=since - timedelta(days=30), end=now, now=now)
    by_task: dict[int, list[tuple[float, str]]] = {}
    for it in stats.items:
        if it.task_id is None or it.live or it.net_minutes <= 0:
            continue
        by_task.setdefault(it.task_id, []).append((it.net_minutes, it.time_source))
    children: dict[int, list[int]] = {}
    for tid, parent in (await session.execute(select(Task.id, Task.parent_id))).all():
        if parent is not None:
            children.setdefault(parent, []).append(tid)
    out: list[Sample] = []
    for t in done:
        assert t.estimate_minutes is not None
        parts = list(by_task.get(t.id, []))
        if t.parent_id is None:
            for child in children.get(t.id, []):
                parts += by_task.get(child, [])
        actual = sum(m for m, _ in parts)
        if actual < 1 or t.estimate_minutes < 1:
            continue
        ratio = actual / t.estimate_minutes
        if not MIN_RATIO <= ratio <= MAX_RATIO:
            continue
        weight = sum(m * SOURCE_WEIGHTS.get(src, 0.0) for m, src in parts) / actual
        if weight <= 0:
            continue
        out.append(
            Sample(
                task_id=t.id,
                estimate_minutes=t.estimate_minutes,
                actual_minutes=actual,
                weight=weight,
                exact=all(src == TimeSource.EXACT for _, src in parts),
                category_id=t.category_id,
                context_id=t.context_id,
                estimate_source=t.estimate_source,
            )
        )
    return out


def factor_from(samples: Sequence[Sample]) -> tuple[float, float]:
    """(factor, total weight): weighted geometric mean of ratios with a prior."""
    total = sum(s.weight for s in samples)
    if total <= 0:
        return 1.0, 0.0
    log_mean = sum(s.weight * math.log(s.ratio) for s in samples) / (total + PRIOR_WEIGHT)
    return max(MIN_FACTOR, min(MAX_FACTOR, math.exp(log_mean))), total


def _calibration(samples: Sequence[Sample], scope: str, label: str) -> Calibration:
    factor, weight = factor_from(samples)
    return Calibration(
        factor=round(factor, 2),
        scope=scope,
        scope_label=label,
        weight=weight,
        samples=len(samples),
        exact_samples=sum(1 for s in samples if s.exact),
    )


async def calibration_for(
    session: AsyncSession,
    ctx: AppContext,
    *,
    category_id: int | None,
    context_id: int | None,
    samples: Sequence[Sample] | None = None,
) -> Calibration:
    """Most specific calibration with enough data (category -> context -> all)."""
    if samples is None:
        samples = await collect_samples(session, ctx)
    if not samples:
        return Calibration()
    if category_id is not None:
        same = [s for s in samples if s.category_id == category_id]
        if sum(s.weight for s in same) >= MIN_SCOPE_WEIGHT:
            cat = await session.get(Category, category_id)
            return _calibration(same, "category", cat.name if cat else "")
    if context_id is not None:
        same = [s for s in samples if s.context_id == context_id]
        if sum(s.weight for s in same) >= MIN_SCOPE_WEIGHT:
            c = await session.get(Context, context_id)
            return _calibration(same, "context", c.name if c else "")
    return _calibration(samples, "all", "")


# --------------------------------------------------------------------------- estimates


def adjust(
    base_minutes: int | None, source: str | None, calibration: Calibration
) -> AdjustedEstimate:
    """Apply a calibration to one estimate (pure function)."""
    if base_minutes is None or base_minutes <= 0:
        return AdjustedEstimate(
            minutes=DEFAULT_ESTIMATE_MIN,
            base_minutes=None,
            approximate=True,
            note=f"оценки нет — заложено ≈{DEFAULT_ESTIMATE_MIN} мин",
            source="default",
            calibration=calibration,
        )
    src = source or EstimateSource.USER
    if calibration.applies:
        minutes = max(5, round(base_minutes * calibration.factor / 5.0) * 5)
        note = f"оценка {base_minutes} мин × {calibration.factor:.1f}: {calibration.explanation()}"
        return AdjustedEstimate(minutes, base_minutes, True, note, src, calibration)
    # The user's own estimate without a measured correction is shown as given;
    # AI / history estimates are always approximate.
    approximate = src != EstimateSource.USER or calibration.mostly_approximate
    who = {"user": "твоя оценка", "ai": "оценка ИИ", "history": "оценка по истории"}.get(
        src, "оценка"
    )
    return AdjustedEstimate(base_minutes, base_minutes, approximate, who, src, calibration)


async def spent_minutes(
    session: AsyncSession, ctx: AppContext, task_id: int, *, now: datetime | None = None
) -> float:
    now = now or ctx.clock.now()
    stats = await session_stats(
        session, start=now - timedelta(days=LOOKBACK_DAYS), end=now, now=now, task_id=task_id
    )
    return stats.total_minutes


async def estimate_task(
    session: AsyncSession,
    ctx: AppContext,
    task: Task,
    *,
    samples: Sequence[Sample] | None = None,
    remaining: bool = True,
) -> AdjustedEstimate:
    """Calibrated estimate of a task (remaining work for in-progress tasks).

    Without an own estimate the open subtasks' estimates are summed; without any
    estimate DEFAULT_ESTIMATE_MIN is assumed (approximate, explained)."""
    if samples is None:
        samples = await collect_samples(session, ctx)
    cal = await calibration_for(
        session, ctx, category_id=task.category_id, context_id=task.context_id, samples=samples
    )
    base, source = task.estimate_minutes, task.estimate_source
    sub_note = ""
    if base is None and task.parent_id is None:
        subs = list(
            (
                await session.execute(
                    select(Task).where(
                        Task.parent_id == task.id,
                        Task.status.in_([TaskStatus.TODO, TaskStatus.IN_PROGRESS]),
                    )
                )
            ).scalars()
        )
        known = [s.estimate_minutes for s in subs if s.estimate_minutes]
        if known:
            base = sum(known)
            source = "subtasks"
            sub_note = f"сумма оценок шагов ({len(known)} из {len(subs)})"
    est = adjust(base, source, cal)
    if source == "subtasks":
        est = AdjustedEstimate(
            est.minutes,
            est.base_minutes,
            True,
            f"{sub_note}; {est.note}" if cal.applies else sub_note,
            "subtasks",
            cal,
        )
    if remaining and task.status == TaskStatus.IN_PROGRESS:
        spent = await spent_minutes(session, ctx, task.id)
        if spent >= 1:
            left = max(round(est.minutes - spent), max(5, int(est.minutes * 0.25)))
            return AdjustedEstimate(
                left,
                est.base_minutes,
                True,
                f"{est.note}; уже потрачено ≈{round(spent)} мин",
                est.source,
                cal,
                spent,
            )
    return est


async def history_summary(
    session: AsyncSession, ctx: AppContext, *, samples: Sequence[Sample] | None = None
) -> list[dict[str, object]]:
    """Compact accuracy data for AI prompts and reports (no free text)."""
    if samples is None:
        samples = await collect_samples(session, ctx)
    if not samples:
        return []
    out: list[dict[str, object]] = []
    overall = _calibration(samples, "all", "")
    out.append(
        {
            "scope": "all",
            "factor": overall.factor,
            "samples": overall.samples,
            "exact_samples": overall.exact_samples,
        }
    )
    by_cat: dict[int, list[Sample]] = {}
    for s in samples:
        if s.category_id is not None:
            by_cat.setdefault(s.category_id, []).append(s)
    for cat_id, items in by_cat.items():
        if sum(s.weight for s in items) < MIN_SCOPE_WEIGHT:
            continue
        cat = await session.get(Category, cat_id)
        c = _calibration(items, "category", cat.name if cat else str(cat_id))
        out.append(
            {
                "scope": f"category:{c.scope_label}",
                "factor": c.factor,
                "samples": c.samples,
                "exact_samples": c.exact_samples,
            }
        )
    return out
