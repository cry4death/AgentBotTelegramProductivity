"""services.capture: date resolution by code, validators/repair, proposals only."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

import pytest
from sqlalchemy import func, select

from procbot.ai.errors import AIUnavailable
from procbot.ai.testing import FakeError, make_fake_gateway
from procbot.db.models import AICall, PendingApproval, Task
from procbot.domain.actions import TaskDraft
from procbot.domain.enums import Actor, AIErrorKind, ApprovalStatus
from procbot.services import approvals, capture
from procbot.services import tasks as ts
from procbot.wiring import register_all

MONDAY = date(2026, 10, 26)


@pytest.fixture
def wired(ctx):
    return register_all(ctx)


async def count(ctx, model, *where) -> int:
    async with ctx.db.transaction() as s:
        stmt = select(func.count()).select_from(model)
        for w in where:
            stmt = stmt.where(w)
        return (await s.execute(stmt)).scalar_one()


# --------------------------------------------------------------------------- dates


@pytest.mark.parametrize(
    ("when", "at", "day", "tm", "unc_date", "unc_time", "backlog"),
    [
        ("завтра", None, date(2026, 10, 27), None, False, False, False),
        ("в пятницу", "14:00", date(2026, 10, 30), time(14), False, False, False),
        ("на выходных", None, date(2026, 10, 31), None, True, False, False),
        ("сегодня", "вечером", MONDAY, None, False, True, False),
        ("через 3 дня", None, date(2026, 10, 29), None, False, False, False),
        ("через неделю", None, date(2026, 11, 2), None, False, False, False),
        ("2026-11-05", None, date(2026, 11, 5), None, False, False, False),
        ("25.12", None, date(2026, 12, 25), None, False, False, False),
        ("на этой неделе", None, None, None, True, False, False),
        ("когда-нибудь", None, None, None, False, False, True),
        (None, None, None, None, False, False, False),
    ],
)
def test_resolve_when(when, at, day, tm, unc_date, unc_time, backlog) -> None:
    r = capture.resolve_when(when, at, today=MONDAY)
    assert r is not None
    assert (r.day, r.at, r.uncertain_date, r.uncertain_time, r.backlog) == (
        day,
        tm,
        unc_date,
        unc_time,
        backlog,
    )


def test_resolve_when_garbage_is_not_understood() -> None:
    assert capture.resolve_when("в четверг после дождичка", None, today=MONDAY) is None
    assert capture.resolve_when("31.02", None, today=MONDAY) is None


def test_extract_validator_rejects_past_and_garbage_dates() -> None:
    clock = capture._Now(datetime(2026, 10, 26, 9, tzinfo=UTC), "Europe/Berlin", MONDAY)
    check = capture.extract_validator(clock)
    out = capture.ExtractedTasks.model_validate(
        {
            "candidates": [
                {"title": "ok", "when": "завтра"},
                {"title": "past", "when": "2026-10-01"},
                {"title": "junk", "when": "когда рак на горе свистнет"},
            ]
        }
    )
    errors = check(out)
    assert len(errors) == 2
    assert "candidates.1.when" in errors[0] and "прошла" in errors[0]
    assert "candidates.2.when/at" in errors[1]


def test_route_validator() -> None:
    check = capture.route_validator(frozenset({5}))
    bad = capture.RouteDecision.model_validate(
        {"intent": "session_control", "confidence": 0.9, "task_id": 7, "retro_start": "25:99"}
    )
    errors = check(bad)
    assert any("task_id" in e for e in errors)
    assert any("session_action" in e for e in errors)
    assert any("retro_start" in e for e in errors)
    good = capture.RouteDecision.model_validate(
        {"intent": "postpone", "confidence": 0.9, "task_id": 5}
    )
    assert check(good) == []


def test_resolve_retro_prefers_model_clock_times_placed_by_code() -> None:
    now = datetime(2026, 10, 26, 16, 30, tzinfo=UTC)  # 17:30 Berlin
    d = capture.RouteDecision.model_validate(
        {
            "intent": "retro_report",
            "confidence": 0.9,
            "retro_start": "14:00",
            "retro_end": "16:00",
            "retro_approximate": True,
        }
    )
    iv = capture.resolve_retro(d, "я работал с 2 до 4", now=now, tz="Europe/Berlin")
    assert iv is not None and iv.start is not None and iv.end is not None
    assert iv.start == datetime(2026, 10, 26, 13, 0, tzinfo=UTC)
    assert iv.end == datetime(2026, 10, 26, 15, 0, tzinfo=UTC)
    assert iv.approximate and iv.time_source == "approximate"
    # no model times -> code parses the text
    iv2 = capture.resolve_retro(None, "вчера 10-12", now=now, tz="Europe/Berlin")
    assert iv2 is not None and iv2.start == datetime(2026, 10, 25, 9, 0, tzinfo=UTC)
    # duration only
    d3 = capture.RouteDecision.model_validate(
        {"intent": "retro_report", "confidence": 0.9, "retro_duration_min": 90}
    )
    iv3 = capture.resolve_retro(d3, "работал полтора", now=now, tz="Europe/Berlin")
    assert iv3 is not None and iv3.duration == timedelta(minutes=90) and iv3.start is None


# --------------------------------------------------------------------------- extraction


async def test_capture_tasks_proposes_only_and_records_actions(wired, profile) -> None:
    ctx = wired
    gw = await make_fake_gateway(
        ctx,
        {
            "extract_tasks": {
                "candidates": [
                    {"title": "Доделать сайт", "when": "завтра"},
                    {
                        "title": "Позаниматься проектом",
                        "when": "на выходных",
                        "uncertainties": ["priority"],
                        "uncertainty_note": "не сказано, сколько времени",
                    },
                    {"title": "Отчёт", "when": "к пятнице", "date_kind": "due", "at": "18:00"},
                ]
            }
        },
    )
    res = await capture.capture_tasks(ctx, "Завтра доделать сайт, на выходных проект", update_id=11)
    assert res.kind == "proposed" and len(res.approvals) == 3
    assert await count(ctx, Task) == 0
    assert await count(ctx, PendingApproval, PendingApproval.status == "pending") == 3
    async with ctx.db.transaction() as s:
        rows = await approvals.list_group(s, "u11:cand")
        drafts = [r.action_payload["task"] for r in rows]
        call = await s.get(AICall, res.ai_call_id)
    assert drafts[0]["planned_date"] == "2026-10-27"
    assert drafts[1]["planned_date"] == "2026-10-31"
    assert drafts[2]["due_at"].startswith("2026-10-30T17:00")  # 18:00 Berlin = 17:00 UTC
    assert all(r.origin == "ai" and r.ai_call_id == res.ai_call_id for r in rows)
    lines = rows[1].diff["lines"]
    assert any(line.startswith("❓ Неясно: срок / дата, приоритет") for line in lines)
    assert any("не сказано, сколько времени" in line for line in lines)
    assert any("31.10" in line for line in lines)
    assert call is not None and len(call.proposed_actions) == 3
    assert {r["status"] for r in call.actions_result} == {"pending_approval"}
    # the prompt carries the message as data
    user = gw.fake.calls_for("extract_tasks")[0].user
    assert "category=message" in user and "Завтра доделать сайт" in user
    # same update again -> the same cards, no duplicates
    gw.fake.push("extract_tasks", {"candidates": [{"title": "Доделать сайт", "when": "завтра"}]})
    again = await capture.capture_tasks(ctx, "Завтра доделать сайт", update_id=11)
    assert [a.id for a in again.approvals] == [rows[0].id]
    assert await count(ctx, PendingApproval) == 3


async def test_invalid_dates_go_through_repair(wired, profile) -> None:
    ctx = wired
    gw = await make_fake_gateway(
        ctx,
        {
            "extract_tasks": [
                {"candidates": [{"title": "Купить билеты", "when": "2020-01-01"}]},
                {"candidates": [{"title": "Купить билеты", "when": "завтра"}]},
            ]
        },
    )
    res = await capture.capture_tasks(ctx, "купить билеты завтра", update_id=12)
    calls = gw.fake.calls_for("extract_tasks")
    assert len(calls) == 2 and calls[1].repair
    assert "уже прошла" in calls[1].messages[-1][1]
    async with ctx.db.transaction() as s:
        row = await s.get(PendingApproval, res.approvals[0].id)
    assert row is not None and row.action_payload["task"]["planned_date"] == "2026-10-27"


async def test_reflection_only_asks_instead_of_creating(wired, profile) -> None:
    ctx = wired
    await make_fake_gateway(
        ctx,
        {
            "extract_tasks": {
                "candidates": [],
                "is_reflection_only": True,
                "clarifying_question": "Хочешь, запишу «больше спорта» как цель или задачу?",
            }
        },
    )
    res = await capture.capture_tasks(ctx, "надо бы как-то больше спорта", update_id=13)
    assert res.kind == "clarify" and "больше спорта" in (res.text or "")
    assert await count(ctx, PendingApproval) == 0 and await count(ctx, Task) == 0


async def test_reflection_with_draft_is_offered_for_confirmation(wired, profile) -> None:
    ctx = wired
    await make_fake_gateway(
        ctx,
        {
            "extract_tasks": {
                "candidates": [{"title": "Записаться в зал", "uncertainties": ["intent"]}],
                "is_reflection_only": True,
            }
        },
    )
    res = await capture.capture_tasks(ctx, "может, записаться в зал", update_id=14)
    assert res.kind == "proposed" and res.header is not None
    assert "размышления" in res.header and "подтвердишь" in res.header
    assert await count(ctx, Task) == 0


async def test_goal_hint_matches_existing_goal_only(wired, profile) -> None:
    ctx = wired
    from procbot.db.models import Goal

    async with ctx.db.transaction() as s:
        g = Goal(title="Английский", status="active", created_at=ctx.clock.now(), updated_at=ctx.clock.now())
        s.add(g)
        await s.flush()
        gid = g.id
    await make_fake_gateway(
        ctx,
        {
            "extract_tasks": {
                "candidates": [
                    {"title": "Урок", "goal_hint": "английский"},
                    {"title": "Тренировка", "goal_hint": "марафон"},
                ]
            }
        },
    )
    res = await capture.capture_tasks(ctx, "урок английского, тренировка", update_id=15)
    async with ctx.db.transaction() as s:
        rows = [await s.get(PendingApproval, a.id) for a in res.approvals]
    assert rows[0] is not None and rows[0].action_payload["task"]["goal_id"] == gid
    assert rows[1] is not None and "goal_id" not in rows[1].action_payload["task"]
    assert any("не нашёл" in line for line in rows[1].diff["lines"])


async def test_unavailable_raises_and_fallback_splits_lines(wired, profile) -> None:
    ctx = wired
    await make_fake_gateway(ctx, {"extract_tasks": [FakeError(AIErrorKind.SERVER)] * 10})
    with pytest.raises(AIUnavailable):
        await capture.capture_tasks(ctx, "a\nb", update_id=16)
    assert await count(ctx, PendingApproval) == 0
    res = await capture.fallback_capture(ctx, "молоко\nхлеб", update_id=16)
    assert res.kind == "proposed" and len(res.approvals) == 2
    assert {a.origin for a in res.approvals} == {"system"}
    assert await count(ctx, Task) == 0


async def test_route_low_confidence_becomes_unclear(wired, profile) -> None:
    ctx = wired
    await make_fake_gateway(
        ctx,
        {
            "route_message": [
                {"intent": "session_control", "session_action": "stop", "confidence": 0.3},
                {"intent": "new_tasks", "confidence": 0.3},
            ]
        },
    )
    d = await capture.route_message(ctx, "ну такое", update_id=17)
    assert d.intent == "unclear"
    d2 = await capture.route_message(ctx, "может купить молока", update_id=18)
    assert d2.intent == "new_tasks"  # task intents stay: they only lead to a card


async def test_route_unknown_task_id_is_repaired(wired, profile) -> None:
    ctx = wired
    async with ctx.db.transaction() as s:
        t = await ts.create_task(s, ctx, TaskDraft(title="Отчёт"), actor=Actor.USER)
        tid = t.id
    gw = await make_fake_gateway(
        ctx,
        {
            "route_message": [
                {"intent": "postpone", "confidence": 0.9, "task_id": 999},
                {"intent": "postpone", "confidence": 0.9, "task_id": tid},
            ]
        },
    )
    d = await capture.route_message(ctx, "перенеси отчёт", update_id=19)
    assert d.task_id == tid
    assert gw.fake.calls_for("route_message")[1].repair
    async with ctx.db.transaction() as s:
        task = await s.get(Task, tid)
    assert task is not None and task.status == "todo" and task.planned_date is None


async def test_approving_ai_card_creates_task_with_resolved_date(wired, profile) -> None:
    ctx = wired
    await make_fake_gateway(
        ctx, {"extract_tasks": {"candidates": [{"title": "Сайт", "when": "завтра", "at": "15:00"}]}}
    )
    res = await capture.capture_tasks(ctx, "завтра в 15 сайт", update_id=20)
    async with ctx.db.transaction() as s:
        r = await approvals.approve(s, ctx, res.approvals[0].id)
    assert r.status == ApprovalStatus.APPLIED
    async with ctx.db.transaction() as s:
        task = (await s.execute(select(Task))).scalar_one()
    assert task.planned_start_at == datetime(2026, 10, 27, 14, 0, tzinfo=UTC)
    assert task.created_by == "user"  # attributed to the confirming user
