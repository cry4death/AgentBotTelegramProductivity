"""Analytics helpers and the evening day summary: data quality is never upgraded,
partial progress is not "done", unstarted planned tasks count as deviations."""

from __future__ import annotations

from datetime import UTC, datetime

from procbot.domain.actions import TaskDraft
from procbot.domain.enums import Actor
from procbot.services import analytics as an
from procbot.services import tasks as task_service


def test_quality_is_never_upgraded():
    assert an.quality_of("exact") == an.Quality.EXACT
    assert an.quality_of("approximate") == an.Quality.RETRO
    assert an.quality_of(None) == an.Quality.UNKNOWN
    assert an.combine([]) == an.Quality.UNKNOWN
    assert an.combine([an.Quality.EXACT, an.Quality.EXACT]) == an.Quality.EXACT
    assert an.combine([an.Quality.EXACT, an.Quality.RETRO]) == an.Quality.MIXED


def test_daypart_uses_owner_timezone():
    dt = datetime(2026, 10, 26, 6, 30, tzinfo=UTC)
    assert an.daypart(dt, "UTC") == "утро"
    assert an.daypart(dt, "America/New_York") == "ночь"  # 02:30 local
    assert an.daypart(dt, "Asia/Tokyo") == "день"  # 15:30 local


async def test_day_summary_separates_done_partial_and_not_started(ctx, profile):
    today = await _today(ctx)
    async with ctx.db.transaction() as s:
        ids = []
        for title in ("Готовая", "Частичная", "Не начатая"):
            t = await task_service.create_task(
                s, ctx, TaskDraft(title=title, planned_date=today), actor=Actor.USER
            )
            ids.append(t.id)
        await task_service.complete_task(s, ctx, ids[0], actor=Actor.USER)
        await task_service.record_progress(s, ctx, ids[1], actor=Actor.USER, note="половина")
    async with ctx.db.transaction() as s:
        d = await an.day_summary(s, ctx, today)
    assert [t.title for t in d.done] == ["Готовая"]
    assert [t.title for t in d.in_progress] == ["Частичная"]
    assert [t.title for t in d.not_started] == ["Не начатая"]
    assert any("не начато" in x for x in d.deviations)


async def _today(ctx):
    async with ctx.db.transaction() as s:
        return await task_service.local_today(s, ctx)
