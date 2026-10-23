"""AnalyticsService (FR-11): six directions by code, source quality, honest gaps
(acceptance 16, 22), effectiveness disclaimer."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import select

from procbot.db.models import (
    Category,
    DailyPlan,
    Goal,
    Intervention,
    InterventionOutcome,
    MemoryItem,
    MemorySource,
    Milestone,
    PlanItem,
    PlanRevision,
    Task,
    TaskEvent,
    WorkSession,
)
from procbot.services import analytics, memory, reports
from procbot.services.analytics import DIRECTION_KEYS, INSUFFICIENT, NO_CAUSALITY, Period, Quality

TZ = "Europe/Berlin"
FIRST, LAST = date(2026, 10, 20), date(2026, 10, 26)


def at(day: int, hh: int, mm: int = 0) -> datetime:
    """Berlin local time on 2026-10-<day> as UTC (CEST until the 25th, then CET)."""
    offset = 2 if day < 25 else 1
    return datetime(2026, 10, day, hh - offset, mm, tzinfo=UTC)


async def task(ctx, title="Отчёт", **kw) -> int:
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        t = Task(title=title, created_at=kw.pop("created_at", now), updated_at=now, **kw)
        s.add(t)
        await s.flush()
        return t.id


async def session(ctx, task_id, start, minutes, *, source="exact", planned=None, status="finished"):
    async with ctx.db.transaction() as s:
        ws = WorkSession(
            task_id=task_id,
            mode="free",
            status=status,
            planned_start_at=planned,
            started_at=start if status != "unconfirmed" else None,
            ended_at=start + timedelta(minutes=minutes) if status == "finished" else None,
            time_source=source,
            confidence=None if source == "exact" else "medium",
            created_at=start,
            updated_at=start,
        )
        s.add(ws)
        await s.flush()
        return ws.id


async def intervention(
    ctx, task_id, when, *, kind="tiny_step", reason=None, result=None, origin="user_said"
):
    async with ctx.db.transaction() as s:
        iv = Intervention(
            task_id=task_id,
            kind=kind,
            reason_category=reason,
            reason_origin=origin if reason else None,
            created_at=when,
        )
        s.add(iv)
        await s.flush()
        if result:
            s.add(InterventionOutcome(intervention_id=iv.id, result=result, recorded_at=when))
        return iv.id


async def compute(ctx, first=FIRST, last=LAST):
    async with ctx.db.transaction() as s:
        return await analytics.compute(s, ctx, Period.of(first, last, TZ))


def metric(a, direction, key):
    found = [m for m in a.directions[direction].metrics if m.key == key]
    assert found, [m.key for m in a.directions[direction].metrics]
    return found[0]


# --------------------------------------------------------------------------- acceptance 22


async def test_all_six_directions_present_and_marked_when_empty(ctx, profile):
    a = await compute(ctx)
    assert [d.key for d in a.ordered()] == list(DIRECTION_KEYS)
    assert all(not d.has_data for d in a.ordered())
    text = reports.render_weekly_text(
        a, reports.WeeklyAI(status="unavailable"), measured=[], pending_proposals=0
    )
    for title in analytics.DIRECTION_TITLES.values():
        assert title in text
    assert text.count(INSUFFICIENT) >= 6
    assert NO_CAUSALITY in text
    assert "ИИ сейчас недоступен" in text


# --------------------------------------------------------------------------- acceptance 16


async def test_retro_time_is_never_presented_as_exact(ctx, profile):
    t1 = await task(ctx, "Сайт", planned_start_at=at(22, 10))
    t2 = await task(ctx, "Статья", planned_start_at=at(23, 14))
    await session(ctx, t1, at(22, 10, 30), 60, source="retro")
    await session(ctx, t2, at(23, 14, 10), 30, source="exact")
    await session(ctx, None, at(24, 9), 45, source="approximate")
    a = await compute(ctx)
    retro = metric(a, "tasks_plans", "work_retro")
    exact = metric(a, "tasks_plans", "work_exact")
    assert (retro.value, retro.quality) == (105, Quality.RETRO)
    assert (exact.value, exact.quality) == (30, Quality.EXACT)
    d_exact = metric(a, "start_delay", "delay_exact")
    d_retro = metric(a, "start_delay", "delay_retro")
    assert (d_exact.value, d_exact.n, d_exact.quality) == ("+10 мин", 1, Quality.EXACT)
    assert (d_retro.value, d_retro.n, d_retro.quality) == ("+30 мин", 1, Quality.RETRO)
    text = reports.render_weekly_text(
        a, reports.WeeklyAI(status="disabled"), measured=[], pending_proposals=0
    )
    for line in text.splitlines():
        if "задним числом" in line and "мин" in line:
            assert "ретроспективно" in line and "точная отметка" not in line, line
    retro_line = next(x for x in text.splitlines() if "Работа, отмеченная задним числом" in x)
    assert "105 мин" in retro_line and "приблизительно" in retro_line


async def test_start_delay_unconfirmed_is_unknown(ctx, profile):
    t1 = await task(ctx, "Звонок", planned_start_at=at(21, 9))
    await task(ctx, "Почта", planned_start_at=at(21, 11))  # never started
    await session(ctx, t1, at(21, 9, 40), 20)
    planned = await task(ctx, "Пробежка")
    await session(ctx, planned, at(22, 8), 0, status="unconfirmed", planned=at(22, 8))
    a = await compute(ctx)
    d = a.directions["start_delay"]
    assert metric(a, "start_delay", "delay_exact").value == "+40 мин"
    unk = metric(a, "start_delay", "delay_unknown")
    assert (unk.value, unk.quality) == (2, Quality.UNKNOWN)
    assert d.data["unconfirmed_n"] == 2 and d.data["exact_median_min"] == 40


# --------------------------------------------------------------------------- 1. plans


async def test_plan_match_with_reasons_for_changes(ctx, profile):
    t_done = await task(ctx, "A", status="done", completed_at=at(21, 12))
    t_open = await task(ctx, "B")
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        plan = DailyPlan(
            local_date=date(2026, 10, 21), status="approved", created_at=now, updated_at=now
        )
        s.add(plan)
        await s.flush()
        r1 = PlanRevision(
            plan_id=plan.id, version=1, status="revised", created_by="ai", created_at=now
        )
        r2 = PlanRevision(
            plan_id=plan.id,
            version=2,
            status="approved",
            created_by="user",
            change_reason="встреча сдвинулась",
            created_at=now,
        )
        s.add_all([r1, r2])
        await s.flush()
        plan.current_revision_id = r2.id
        s.add_all(
            [
                PlanItem(revision_id=r2.id, kind="task", task_id=t_done, title="A"),
                PlanItem(revision_id=r2.id, kind="task", task_id=t_open, title="B"),
                PlanItem(revision_id=r2.id, kind="break", title="Перерыв"),
            ]
        )
        s.add(
            TaskEvent(
                task_id=t_open,
                event_type="postponed",
                actor="user",
                reason="не успел",
                created_at=at(21, 18),
            )
        )
    a = await compute(ctx)
    m = metric(a, "tasks_plans", "plan_match")
    assert m.value == "1 из 2 (50%)" and m.quality == Quality.MIXED
    assert "нет отметки" in (m.note or "")
    d = a.directions["tasks_plans"]
    assert any("встреча сдвинулась" in x for x in d.details)
    assert any("«не успел» — 1" in x for x in d.details)
    assert metric(a, "tasks_plans", "completed").value == 1


# --------------------------------------------------------------------------- 3, 4


async def test_reasons_circumstances_and_measured_pattern(ctx, profile):
    async with ctx.db.transaction() as s:
        cat = Category(name="Работа", created_at=ctx.clock.now())
        s.add(cat)
        await s.flush()
        cat_id = cat.id
    t = await task(ctx, "Отчёт", category_id=cat_id)
    ivs = [
        await intervention(ctx, t, at(20, 9), reason="unclear", result="started"),
        await intervention(ctx, t, at(22, 10), reason="unclear", result="no_effect"),
        await intervention(ctx, t, at(24, 19), reason="unclear"),
        await intervention(
            ctx, t, at(24, 20), reason="low_energy", kind="rest", origin="ai_inferred"
        ),
    ]
    a = await compute(ctx)
    d = a.directions["procrastination"]
    assert metric(a, "procrastination", "reason:unclear").value == 3
    assert any("утро — 2" in x for x in d.details)
    assert any("Работа — 4" in x for x in d.details)
    assert any("предположения ИИ" in n for n in d.notes)
    assert analytics.recurring_reasons(a) == {"unclear": 3}

    eff = a.directions["help_effectiveness"]
    tiny = metric(a, "help_effectiveness", "help:tiny_step")
    assert tiny.value == "после 1 из 2 начал(а) или продвинулся(ась)"
    assert "без отметки исхода: 1" in (tiny.note or "")
    assert NO_CAUSALITY in eff.notes

    async with ctx.db.transaction() as s:
        created = await reports.record_measured_patterns(s, ctx, a)
        again = await reports.record_measured_patterns(s, ctx, a)
        item = (await s.execute(select(MemoryItem))).scalar_one()
        srcs = (await s.execute(select(MemorySource.source_id))).scalars().all()
    assert created == ["Часто отмечаемая причина задержек: «непонятно, с чего начать»"]
    assert again == []
    assert (item.kind, item.origin) == ("pattern", "system_measured")
    assert sorted(srcs) == sorted(ivs[:3])
    # a refuted measured pattern is not re-added next week
    async with ctx.db.transaction() as s:
        await memory.refute(s, ctx, item.id)
        assert await reports.record_measured_patterns(s, ctx, a) == []


async def test_effectiveness_disclaimer_even_without_data(ctx, profile):
    a = await compute(ctx)
    assert NO_CAUSALITY in a.directions["help_effectiveness"].notes


# --------------------------------------------------------------------------- 5, 6


async def test_goal_progress_direction(ctx, profile):
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        g = Goal(title="Выучить испанский", status="active", created_at=now, updated_at=now)
        s.add(g)
        await s.flush()
        s.add(Milestone(goal_id=g.id, title="A1", status="done", created_at=now, updated_at=now))
        s.add(Milestone(goal_id=g.id, title="A2", status="active", created_at=now, updated_at=now))
        gid = g.id
    t = await task(ctx, "Урок 1", goal_id=gid, status="done", completed_at=at(22, 19))
    await session(ctx, t, at(22, 18), 40)
    a = await compute(ctx)
    d = a.directions["goals"]
    assert d.has_data
    assert any("«Выучить испанский»" in x and "1/2" in x for x in d.details)
    assert metric(a, "goals", "goals_moved").value == 1


async def test_hypotheses_direction_lists_active_only(ctx, profile):
    async with ctx.db.transaction() as s:
        h1 = await memory.add_hypothesis(
            s, ctx, "Возможно, утром легче", verification_hint="5 дней"
        )
        h2 = await memory.add_hypothesis(s, ctx, "Возможно, мешает телефон")
        assert h1 and h2
        await memory.refute(s, ctx, h2.id)
    a = await compute(ctx)
    d = a.directions["hypotheses"]
    assert [h["content"] for h in d.data["hypotheses"]] == ["Возможно, утром легче"]
    lines = reports.render_hypotheses(d)
    assert any("Как проверить: 5 дней" in x and "предположение ИИ" in x for x in lines)


async def test_ai_payload_contains_only_numbers_and_labels(ctx, profile):
    t = await task(ctx, "Сайт", planned_start_at=at(22, 10))
    await session(ctx, t, at(22, 10, 5), 30)
    a = await compute(ctx)
    payload = analytics.ai_payload(a)
    assert set(payload["directions"]) == set(DIRECTION_KEYS)
    assert payload["directions"]["start_delay"]["numbers"]["exact_median_min"] == 5


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("01.10-15.10", (date(2026, 10, 1), date(2026, 10, 15))),
        ("1.10.2026 – 15.10.2026", (date(2026, 10, 1), date(2026, 10, 15))),
        ("20.12 по 05.01", (date(2025, 12, 20), date(2026, 1, 5))),
    ],
)
def test_parse_period(text, expected):
    assert reports.parse_period(text, date(2026, 10, 26)) == expected


@pytest.mark.parametrize("text", ["вчера", "01.10-30.12.2026", "01.01.2026-30.06.2026"])
def test_parse_period_rejects(text):
    with pytest.raises(ValueError):
        reports.parse_period(text, date(2026, 10, 26))
