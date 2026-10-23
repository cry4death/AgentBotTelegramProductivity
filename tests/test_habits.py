"""HabitService (FR-09): schedules and due-ness (weeks, DST), flexible N/week,
stability without zeroing, skipped vs missed, miss analysis, idempotent marks,
FR-15: bot/AI habit changes need approval while the owner's edit applies."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

import pytest
from sqlalchemy import func, select

from procbot.db.models import Habit, HabitOccurrence, PendingApproval
from procbot.domain.actions import ChangeHabit, CreateHabit
from procbot.domain.enums import Actor, ApprovalStatus, EntityStatus
from procbot.services import actions, approvals
from procbot.services import habits as hs
from procbot.timeutil import resolve_local

TZ = "Europe/Berlin"
TODAY = date(2026, 10, 26)  # Monday; conftest clock = 2026-10-26 09:00 UTC (10:00 CET)


@pytest.fixture
def wired(ctx, profile):
    hs.register(ctx)
    approvals.register(ctx)
    return ctx


async def make_habit(ctx, clock, *, created: date, **kw) -> Habit:
    saved = clock.now()
    clock.set(resolve_local(created, time(8, 0), TZ))
    try:
        async with ctx.db.transaction() as s:
            return await hs.create_habit(s, ctx, actor=Actor.USER, **kw)
    finally:
        clock.set(saved)


async def mark(ctx, habit_id: int, status: str, day: date, **kw) -> hs.MarkResult:
    async with ctx.db.transaction() as s:
        return await hs.mark(s, ctx, habit_id, status, day=day, **kw)


async def stability(ctx, habit_id: int, days: int) -> hs.HabitStability:
    async with ctx.db.transaction() as s:
        return await hs.habit_stability(s, ctx, habit_id, days=days)


async def due(ctx, day: date | None = None) -> list[hs.DueHabit]:
    async with ctx.db.transaction() as s:
        return await hs.due_habits(s, ctx, day)


def d(day: int, month: int = 10) -> date:
    return date(2026, month, day)


# --------------------------------------------------------------------------- parsing


def test_parse_schedule_weekdays_and_target() -> None:
    assert hs.parse_weekdays("пн ср пт") == frozenset({0, 2, 4})
    assert hs.parse_weekdays("пн-пт") == frozenset(range(5))
    assert hs.parse_weekdays("каждый") is None
    assert hs.parse_schedule({"type": "weekly", "days": list(range(7))}).type == hs.DAILY
    with pytest.raises(hs.HabitError):
        hs.parse_schedule({"type": "times_per_week", "n": 0})
    assert hs.parse_target("10 мин, минимум 2") == {"amount": 10, "min_amount": 2, "unit": "мин"}
    assert hs.parse_target("20 отжиманий минимум 5") == {
        "amount": 20,
        "min_amount": 5,
        "unit": "отжиманий",
    }
    assert hs.parse_target("просто так") is None
    assert hs.target_text({"amount": 10, "min_amount": 2, "unit": "мин"}) == "10 мин (минимум 2 мин)"


# --------------------------------------------------------------------------- due-ness


async def test_weekly_due_across_weeks_and_dst(wired, clock) -> None:
    ctx = wired
    h = await make_habit(
        ctx,
        clock,
        created=d(19),
        title="Зарядка",
        schedule={"type": "weekly", "days": [0, 2, 4]},
        target={"amount": 10, "unit": "мин"},
        preferred_time=time(7, 30),
    )
    # Friday before the DST switch (CEST, UTC+2) and Monday after it (CET, UTC+1).
    fri = await due(ctx, d(23))
    assert [x.habit_id for x in fri] == [h.id]
    assert fri[0].start_at == datetime(2026, 10, 23, 5, 30, tzinfo=UTC)
    assert fri[0].estimate_minutes == 10 and fri[0].pending
    mon = await due(ctx, d(26))
    assert mon[0].start_at == datetime(2026, 10, 26, 6, 30, tzinfo=UTC)
    assert await due(ctx, d(27)) == []  # Tuesday: not a habit day
    assert [x.habit_id for x in await due(ctx, date(2026, 11, 2))] == [h.id]
    assert await due(ctx, d(16)) == []  # before the habit existed

    # Local "today" right after midnight CET is already Monday.
    clock.set(datetime(2026, 10, 25, 23, 30, tzinfo=UTC))
    assert [x.local_date for x in await due(ctx)] == [d(26)]


async def test_preferred_time_in_spring_gap_resolves_after_gap(wired, clock) -> None:
    ctx = wired
    await make_habit(
        ctx, clock, created=d(19), title="Чтение", schedule={"type": "daily"}, preferred_time=time(2, 30)
    )
    (item,) = await due(ctx, date(2027, 3, 28))  # Berlin springs forward 02:00 -> 03:00
    assert item.start_at == datetime(2027, 3, 28, 1, 0, tzinfo=UTC)  # 03:00 CEST


async def test_flexible_times_per_week(wired, clock) -> None:
    ctx = wired
    h = await make_habit(
        ctx, clock, created=d(19), title="Бег", schedule={"type": "times_per_week", "n": 3}
    )
    for day in (d(19), d(21), d(22)):
        await mark(ctx, h.id, hs.DONE, day)
    # The weekly target is reached: no longer due later that week, no pressure.
    assert await due(ctx, d(24)) == []
    # A marked day stays listed (with its status) for the planner/day view.
    (marked,) = await due(ctx, d(22))
    assert marked.status == hs.DONE and not marked.pending
    # New week: due again, 0/3 so far.
    (item,) = await due(ctx, d(26))
    assert item.flexible and item.week_kept == 0 and item.week_target == 3

    st = await stability(ctx, h.id, 14)
    # Last week 3/3; the current week (Monday, nothing certain yet) adds nothing.
    assert (st.counted, st.kept, st.missed) == (3, 3, 0)
    assert st.regularity == 1.0

    # By Sunday with nothing done, the deficit that can't be made up counts.
    clock.set(resolve_local(date(2026, 11, 1), time(12, 0), TZ))
    st = await stability(ctx, h.id, 14)
    assert (st.counted, st.kept, st.missed) == (5, 3, 2)


# --------------------------------------------------------------------------- stability


async def test_single_miss_does_not_zero_progress(wired, clock) -> None:
    ctx = wired
    h = await make_habit(ctx, clock, created=d(12), title="Испанский", schedule={"type": "daily"})
    for i in range(12, 26):
        if i != 20:
            await mark(ctx, h.id, hs.DONE, d(i))
    st = await stability(ctx, h.id, 14)
    # Window 13..26; today (26) is not over and not marked -> left out.
    assert (st.counted, st.kept, st.missed) == (13, 12, 1)
    assert st.regularity is not None and st.regularity > 0.9
    assert st.returns == [1] and st.open_gap_since is None
    text = hs.stability_text(st)
    assert "12 из 13" in text
    for word in ("серия", "стрик", "обнул", "с нуля"):
        assert word not in text.lower()
    # Today's mark brings it in without any reset logic.
    await mark(ctx, h.id, hs.DONE, TODAY)
    st = await stability(ctx, h.id, 14)
    assert (st.counted, st.kept) == (14, 13)


async def test_skipped_is_not_missed_and_miss_analysis(wired, clock) -> None:
    ctx = wired
    h = await make_habit(ctx, clock, created=d(19), title="Медитация", schedule={"type": "daily"})
    for i in (19, 20, 21, 22, 25):
        await mark(ctx, h.id, hs.DONE, d(i))
    await mark(ctx, h.id, hs.SKIPPED, d(23))  # conscious decision
    # 24: nothing -> missed (derived, no row)
    st = await stability(ctx, h.id, 14)
    assert (st.counted, st.kept, st.skipped, st.missed) == (6, 5, 1, 1)
    assert st.returns == [1]

    async with ctx.db.transaction() as s:
        a = await hs.missed_analysis(s, ctx, h.id, days=14)
    assert a.unexplained == [d(24)] and a.skipped == 1 and a.missed == 1

    async with ctx.db.transaction() as s:
        await hs.record_miss(s, ctx, h.id, d(24), "low_energy", text="поздно вернулся")
        a = await hs.missed_analysis(s, ctx, h.id, days=14)
    assert a.unexplained == [] and a.by_reason["low_energy"] == 1
    assert a.notes == [(d(24), "поздно вернулся")]
    assert "Мало сил" in hs.analysis_text(a)
    st = await stability(ctx, h.id, 14)
    assert st.missed == 1 and st.skipped == 1  # a reason doesn't turn a miss into a skip

    async with ctx.db.transaction() as s:
        with pytest.raises(hs.HabitError):
            await hs.record_miss(s, ctx, h.id, d(25), "forgot")  # was done that day
        with pytest.raises(hs.HabitError):
            await hs.record_miss(s, ctx, h.id, TODAY, "forgot")  # today isn't over


async def test_pause_days_are_not_misses(wired, clock) -> None:
    ctx = wired
    h = await make_habit(ctx, clock, created=d(12), title="Растяжка", schedule={"type": "daily"})
    for i in range(13, 18):
        await mark(ctx, h.id, hs.DONE, d(i))
    clock.set(resolve_local(d(18), time(9, 0), TZ))
    async with ctx.db.transaction() as s:
        await hs.set_status(s, ctx, h.id, EntityStatus.PAUSED, actor=Actor.USER)
    assert await due(ctx, d(20)) == []
    clock.set(resolve_local(d(24), time(9, 0), TZ))
    async with ctx.db.transaction() as s:
        await hs.set_status(s, ctx, h.id, EntityStatus.ACTIVE, actor=Actor.USER)
    clock.set(datetime(2026, 10, 26, 9, 0, tzinfo=UTC))
    for i in (24, 25):
        await mark(ctx, h.id, hs.DONE, d(i))
    st = await stability(ctx, h.id, 14)
    assert (st.counted, st.kept, st.missed) == (7, 7, 0)


# --------------------------------------------------------------------------- marks


async def test_marking_is_idempotent_per_day(wired, clock) -> None:
    ctx = wired
    h = await make_habit(
        ctx,
        clock,
        created=d(19),
        title="Вода",
        target={"amount": 10, "min_amount": 2, "unit": "мин"},
    )
    first = await mark(ctx, h.id, hs.DONE, TODAY)
    again = await mark(ctx, h.id, hs.DONE, TODAY)
    assert first.changed and not again.changed
    assert first.occurrence.time_source == "exact" and first.occurrence.amount == 10
    async with ctx.db.transaction() as s:
        n = (
            await s.execute(select(func.count()).select_from(HabitOccurrence))
        ).scalar_one()
    assert n == 1
    corrected = await mark(ctx, h.id, hs.PARTIAL, TODAY)
    assert corrected.changed and corrected.previous == hs.DONE
    assert corrected.occurrence.amount == 2  # the minimum version
    retro = await mark(ctx, h.id, hs.DONE, d(24))
    assert retro.occurrence.time_source == "retro"
    async with ctx.db.transaction() as s:
        with pytest.raises(hs.HabitError):
            await hs.mark(s, ctx, h.id, hs.DONE, day=TODAY + timedelta(days=1))
        with pytest.raises(hs.HabitError):
            await hs.mark(s, ctx, h.id, hs.DONE, day=TODAY - timedelta(days=30))
        res = await hs.mark_amount(s, ctx, h.id, 3, day=d(23))
        assert res.occurrence.status == hs.PARTIAL
        res = await hs.mark_amount(s, ctx, h.id, 12, day=d(23))
        assert res.occurrence.status == hs.DONE


# --------------------------------------------------------------------------- FR-15


async def test_ai_or_bot_habit_change_needs_approval_user_edit_applies(wired, clock) -> None:
    ctx = wired
    h = await make_habit(ctx, clock, created=d(19), title="Отжимания", target={"amount": 10})
    for origin in (Actor.AI, Actor.SYSTEM):
        async with ctx.db.transaction() as s:
            res = await actions.execute(
                s,
                ctx,
                ChangeHabit(habit_id=h.id, target={"amount": 30}, preferred_time="06:00"),
                origin=origin,
                op_key=f"t:{origin}",
            )
        assert res.status is actions.ExecStatus.PENDING_APPROVAL
    async with ctx.db.transaction() as s:
        row = await s.get(Habit, h.id)
        assert row is not None and row.target == {"amount": 10} and row.preferred_time is None
        cards = await approvals.list_pending(s)
        assert {c.origin for c in cards} == {"ai", "system"}
        assert "Объём: 10 → 30" in cards[0].diff["lines"]
        assert await hs.has_pending_change(s, h.id)
        # AI-created habits also wait for consent.
        res = await actions.execute(
            s,
            ctx,
            CreateHabit(title="Йога", schedule={"type": "daily"}),
            origin=Actor.AI,
            op_key="t:create",
        )
        assert res.status is actions.ExecStatus.PENDING_APPROVAL
        assert (await s.execute(select(func.count()).select_from(Habit))).scalar_one() == 1

    # Consent applies it.
    async with ctx.db.transaction() as s:
        r = await approvals.approve(s, ctx, cards[0].id)
        assert r.status == ApprovalStatus.APPLIED
        row = await s.get(Habit, h.id)
        assert row is not None and row.target == {"amount": 30}
        assert row.preferred_time == time(6, 0)

    # The owner's explicit edit applies directly.
    async with ctx.db.transaction() as s:
        res = await actions.execute(
            s,
            ctx,
            ChangeHabit(habit_id=h.id, preferred_time=""),
            origin=Actor.USER,
            op_key="t:user",
        )
        assert res.status is actions.ExecStatus.APPLIED
        row = await s.get(Habit, h.id)
        assert row is not None and row.preferred_time is None


async def test_increase_is_only_suggested(wired, clock, monkeypatch) -> None:
    ctx = wired
    h = await make_habit(ctx, clock, created=d(12), title="Чтение", target={"amount": 10, "unit": "мин"})
    flex = await make_habit(
        ctx, clock, created=d(12), title="Бег", schedule={"type": "times_per_week", "n": 2}
    )
    for i in range(12, 26):
        await mark(ctx, h.id, hs.DONE, d(i))
    async with ctx.db.transaction() as s:
        sug = await hs.suggest_increase(s, ctx, h.id)
        assert sug is not None and sug.action.target == {"amount": 12, "unit": "мин"}
        flex_sug = await hs.suggest_increase(s, ctx, flex.id)
        assert flex_sug is not None and flex_sug.action.schedule == {"type": "times_per_week", "n": 3}
        # Threshold X is not decided -> the weekly review suggests nothing.
        assert hs.increase_threshold(ctx) is None
        assert await hs.increase_candidates(s, ctx) == []
    monkeypatch.setattr(hs, "INCREASE_MIN_REGULARITY", 0.8)
    async with ctx.db.transaction() as s:
        cands = await hs.increase_candidates(s, ctx)
        assert [c.habit_id for c in cands] == [h.id]  # the flexible one has no data
        res = await hs.propose_change(s, ctx, cands[0].action, origin=Actor.SYSTEM, op_key="t:inc")
        assert res.status is actions.ExecStatus.PENDING_APPROVAL
        row = await s.get(Habit, h.id)
        assert row is not None and row.target == {"amount": 10, "unit": "мин"}
        assert await hs.increase_candidates(s, ctx) == []  # already waiting for a decision
        n = (await s.execute(select(func.count()).select_from(PendingApproval))).scalar_one()
    assert n == 1


async def test_day_section_quick_mark_buttons(wired, clock) -> None:
    ctx = wired
    h = await make_habit(ctx, clock, created=d(19), title="Зарядка", preferred_time=time(7, 0))
    async with ctx.db.transaction() as s:
        lines, rows = await hs.day_section(s, ctx)
    assert lines[0] == "🔁 Привычки" and "07:00 Зарядка" in lines[1]
    assert rows[0][0].callback_data == f"hb:m:{h.id}:d.20261026.d"
    await mark(ctx, h.id, hs.DONE, TODAY)
    async with ctx.db.transaction() as s:
        lines, rows = await hs.day_section(s, ctx)
    assert lines[1].startswith("✅") and rows == []
