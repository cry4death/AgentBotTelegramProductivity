"""Estimate calibration from history (FR-03): actual net time of finished work vs
the estimate; exact timer data weighs more, retro/approximate data is flagged."""

from __future__ import annotations

from datetime import timedelta

import pytest

from procbot.context import AppContext
from procbot.db.models import Category, Task, WorkSession
from procbot.domain.enums import EstimateSource, SessionStatus, TaskStatus, TimeSource
from procbot.services import estimates as est


async def _done_task(
    ctx: AppContext,
    *,
    estimate: int,
    actual: list[tuple[int, str]],
    category_id: int | None = None,
    source: str = EstimateSource.USER,
    status: str = TaskStatus.DONE,
) -> int:
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        t = Task(
            title=f"t{estimate}",
            estimate_minutes=estimate,
            estimate_source=source,
            status=status,
            completed_at=now - timedelta(days=1) if status == TaskStatus.DONE else None,
            category_id=category_id,
            created_at=now - timedelta(days=3),
            updated_at=now,
        )
        s.add(t)
        await s.flush()
        start = now - timedelta(days=2)
        for minutes, src in actual:
            s.add(
                WorkSession(
                    task_id=t.id,
                    mode="free",
                    status=SessionStatus.FINISHED,
                    started_at=start,
                    ended_at=start + timedelta(minutes=minutes),
                    time_source=src,
                    created_at=start,
                    updated_at=start,
                )
            )
            start += timedelta(hours=3)
        return t.id


async def _cal(ctx: AppContext, **kw):
    async with ctx.db.transaction() as s:
        samples = await est.collect_samples(s, ctx)
        return samples, await est.calibration_for(
            s, ctx, category_id=kw.get("category_id"), context_id=None, samples=samples
        )


async def test_no_history_means_no_correction(ctx, profile) -> None:
    samples, cal = await _cal(ctx)
    assert samples == [] and cal.factor == 1.0 and not cal.applies
    user = est.adjust(40, EstimateSource.USER, cal)
    assert (user.minutes, user.approximate) == (40, False)
    ai = est.adjust(40, EstimateSource.AI, cal)
    assert ai.approximate and ai.minutes == 40  # AI estimates are always "≈"
    none = est.adjust(None, None, cal)
    assert none.approximate and none.minutes == est.DEFAULT_ESTIMATE_MIN
    assert "оценки нет" in none.note


async def test_exact_history_calibrates_with_prior(ctx, profile) -> None:
    await _done_task(ctx, estimate=30, actual=[(60, TimeSource.EXACT)])
    await _done_task(ctx, estimate=30, actual=[(45, TimeSource.EXACT)])
    samples, cal = await _cal(ctx)
    assert len(samples) == 2 and cal.exact_samples == 2
    # weighted log-mean shrunk by the prior: exp((ln 2 + ln 1.5) / (2 + 2)) ≈ 1.32
    assert cal.factor == pytest.approx(1.32, abs=0.01)
    adj = est.adjust(30, EstimateSource.USER, cal)
    assert adj.approximate and adj.minutes == 40 and adj.label == "≈40 мин"
    assert "× 1.3" in adj.note and "точным таймером" in adj.note


async def test_retro_time_weighs_less_and_is_flagged(ctx, profile) -> None:
    await _done_task(ctx, estimate=30, actual=[(90, TimeSource.RETRO)])
    await _done_task(ctx, estimate=30, actual=[(90, TimeSource.APPROXIMATE)])
    samples, cal = await _cal(ctx)
    assert all(not s.exact for s in samples)
    assert [s.weight for s in samples] == [0.5, 0.25]
    assert cal.mostly_approximate
    assert "неточно" in cal.explanation()
    # same ratios with exact time would correct much more strongly
    exact = est.factor_from([est.Sample(1, 30, 90, 1.0, True, None, None, "user")] * 2)[0]
    assert cal.factor < exact


async def test_unknown_source_and_live_or_unfinished_ignored(ctx, profile) -> None:
    await _done_task(ctx, estimate=30, actual=[(60, TimeSource.UNKNOWN)])
    await _done_task(ctx, estimate=30, actual=[(60, TimeSource.EXACT)], status=TaskStatus.TODO)
    samples, _ = await _cal(ctx)
    assert samples == []


async def test_category_scope_preferred_when_enough_data(ctx, profile) -> None:
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        cat = Category(name="код", created_at=now)
        s.add(cat)
        await s.flush()
        cat_id = cat.id
    for _ in range(2):
        await _done_task(ctx, estimate=30, actual=[(90, TimeSource.EXACT)], category_id=cat_id)
    await _done_task(ctx, estimate=60, actual=[(30, TimeSource.EXACT)])
    _, cal = await _cal(ctx, category_id=cat_id)
    assert cal.scope == "category" and cal.scope_label == "код" and cal.factor > 1.5
    _, other = await _cal(ctx, category_id=None)
    assert other.scope == "all" and other.factor < cal.factor
    async with ctx.db.transaction() as s:
        summary = await est.history_summary(s, ctx)
    assert {row["scope"] for row in summary} == {"all", "category:код"}


async def test_in_progress_task_gets_remaining_time(ctx, profile) -> None:
    tid = await _done_task(
        ctx, estimate=60, actual=[(20, TimeSource.EXACT)], status=TaskStatus.IN_PROGRESS
    )
    async with ctx.db.transaction() as s:
        task = await s.get(Task, tid)
        assert task is not None
        e = await est.estimate_task(s, ctx, task)
    assert e.minutes == 40 and e.approximate and "уже потрачено ≈20 мин" in e.note
