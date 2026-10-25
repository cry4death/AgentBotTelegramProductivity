"""Mini App management flows: everything the chat can do, without the chat.
AI proposals still need approval; calendar insert only after approval."""

from __future__ import annotations

import json

import pytest
from aiohttp.test_utils import TestClient, TestServer
from sqlalchemy import select

from procbot.db.models import Goal, Habit, PendingApproval, Task
from procbot.domain.actions import TaskDraft
from procbot.domain.enums import Actor
from procbot.http_server import build_app
from procbot.miniapp.auth import sign_init_data
from procbot.services import memory
from procbot.services import tasks as task_service
from procbot.testing import OWNER_ID
from procbot.wiring import register_all


@pytest.fixture
async def client(ctx, profile):
    register_all(ctx)
    c = TestClient(TestServer(build_app(ctx)))
    await c.start_server()
    fields = {
        "auth_date": str(int(ctx.clock.now().timestamp())),
        "user": json.dumps({"id": OWNER_ID}),
    }
    c.headers = {"Authorization": "tma " + sign_init_data(fields, ctx.env.telegram_bot_token)}
    yield c
    await c.close()


async def _get(c, path):
    r = await c.get("/miniapp/api/" + path, headers=c.headers)
    return r.status, await r.json()


async def _post(c, path, body=None):
    r = await c.post("/miniapp/api/" + path, json=body or {}, headers=c.headers)
    return r.status, await r.json()


async def _task(ctx, title="Отчёт", **kw) -> int:
    async with ctx.db.transaction() as s:
        t = await task_service.create_task(s, ctx, TaskDraft(title=title, **kw), actor=Actor.USER)
        return t.id


async def test_edit_task_fields_plan_and_deadline(ctx, client):
    task_id = await _task(ctx)
    st, data = await _post(
        client,
        f"tasks/{task_id}/edit",
        {
            "title": "Квартальный отчёт",
            "done_criteria": "Отправлен руководителю",
            "estimate_minutes": 90,
            "planned_day": "2026-10-27",
            "planned_time": "14:30",
            "due_date": "2026-10-30",
            "due_is_hard": True,
        },
    )
    assert st == 200, data
    assert data["title"] == "Квартальный отчёт" and data["needs_criteria"] is True
    assert data["start"] == "14:30" and data["estimate_min"] == 90
    _, detail = await _get(client, f"tasks/{task_id}")
    assert detail["planned_day"] == "2026-10-27" and detail["due_date"] == "2026-10-30"
    assert detail["due_is_hard"] is True
    assert {"updated", "rescheduled"} & {h["type"] for h in detail["history"]} or detail["history"]
    st, data = await _post(client, f"tasks/{task_id}/edit", {"title": "  "})
    assert st == 400


async def test_status_actions(ctx, client):
    task_id = await _task(ctx)
    assert (await _post(client, f"tasks/{task_id}/status", {"action": "tomorrow"}))[1][
        "status"
    ] == "todo"
    assert (await _post(client, f"tasks/{task_id}/status", {"action": "someday"}))[1][
        "status"
    ] == "someday"
    assert (await _post(client, f"tasks/{task_id}/status", {"action": "reopen"}))[1][
        "status"
    ] == "todo"
    assert (await _post(client, f"tasks/{task_id}/status", {"action": "cancel"}))[1][
        "status"
    ] == "cancelled"
    assert (await _post(client, f"tasks/{task_id}/status", {"action": "fly"}))[0] == 400


async def test_capture_creates_cards_and_only_approval_creates_tasks(ctx, client):
    st, data = await _post(
        client, "capture", {"text": "Купить молоко\nПозвонить маме", "op_key": "cap-12345678"}
    )
    assert st == 200 and len(data["approvals"]) == 2
    async with ctx.db.transaction() as s:
        assert (await s.execute(select(Task))).scalars().all() == []  # nothing yet
    _, listed = await _get(client, "approvals")
    ids = [a["id"] for a in listed["approvals"]]
    assert len(ids) == 2
    st, res = await _post(client, f"approvals/{ids[0]}/approve")
    assert st == 200 and res["status"] == "applied"
    st, res = await _post(client, f"approvals/{ids[1]}/reject")
    assert res["status"] == "rejected"
    async with ctx.db.transaction() as s:
        titles = [t.title for t in (await s.execute(select(Task))).scalars()]
    assert titles == ["Купить молоко"]


async def test_day_plan_propose_and_approve_without_ai(ctx, client):
    today = None
    async with ctx.db.transaction() as s:
        today = await task_service.local_today(s, ctx)
    await _task(ctx, planned_date=today, estimate_minutes=30)
    st, data = await _post(client, "plan", {"op_key": "plan-1234567"})
    assert st == 200 and data["plan"]["is_proposal"] is True and data["plan"]["text"]
    rev = data["plan"]["revision_id"]
    st, data = await _post(client, f"plan/{rev}/approve")
    assert st == 200 and data["plan"]["status"] == "approved"
    st, data = await _get(client, "plan")
    assert data["plan"]["revision_id"] == rev


async def test_goals_and_milestones_crud(ctx, client):
    st, data = await _post(
        client,
        "goals",
        {
            "title": "Марафон",
            "horizon_date": "2027-05-01",
            "milestones": ["10 км"],
            "op_key": "goal-1234567",
        },
    )
    assert st == 200, data
    async with ctx.db.transaction() as s:
        goal = (await s.execute(select(Goal))).scalar_one()
        goal_id = goal.id
    st, ms = await _post(
        client, f"goals/{goal_id}/milestones", {"title": "Полумарафон", "is_key": True}
    )
    again = await _post(client, f"goals/{goal_id}/milestones", {"title": "Полумарафон"})
    assert again[1]["id"] == ms["id"]  # repeated press: no twin
    assert (await _post(client, f"milestones/{ms['id']}/edit", {"status": "done"}))[0] == 200
    assert (await _post(client, f"goals/{goal_id}/edit", {"status": "paused"}))[0] == 200
    _, data = await _get(client, "goals")
    g = data["goals"][0]
    assert g["status"] == "paused" and g["measured"]["milestones_done"] == 1
    task_id = await _task(ctx)
    assert (await _post(client, f"tasks/{task_id}/edit", {"goal_id": goal_id}))[0] == 200
    _, data = await _get(client, "goals")
    assert data["goals"][0]["measured"]["tasks_total"] == 1


async def test_habit_create_edit_pause(ctx, client):
    body = {
        "title": "Отжимания",
        "schedule": {"type": "weekly", "days": [0, 2, 4]},
        "target": {"amount": 20, "unit": "раз", "min_amount": 5},
        "time": "08:00",
        "op_key": "hab-1234567",
    }
    assert (await _post(client, "habits", body))[0] == 200
    assert (await _post(client, "habits", body))[1]["repeated"] is True
    async with ctx.db.transaction() as s:
        habit_id = (await s.execute(select(Habit))).scalar_one().id
    bad = await _post(
        client,
        "habits",
        {**body, "schedule": {"type": "weekly", "days": []}, "op_key": "hab-7654321"},
    )
    assert bad[0] == 400
    assert (
        await _post(
            client, f"habits/{habit_id}/edit", {"title": "Отжимания утром", "status": "paused"}
        )
    )[0] == 200
    _, data = await _get(client, "habits")
    assert (
        data["habits"][0]["title"] == "Отжимания утром" and data["habits"][0]["status"] == "paused"
    )


async def test_memory_review_from_the_app(ctx, client):
    memory.register(ctx)
    st, data = await _post(client, "memory", {"content": "Лучше работаю утром"})
    assert st == 200
    item_id = data["id"]
    _, data = await _get(client, "memory")
    assert data["items"][0]["origin_label"] == "ты сказал"
    assert (await _post(client, f"memory/{item_id}/refute"))[0] == 200
    _, data = await _get(client, "memory")
    assert data["items"][0]["status"] == "refuted"
    assert (await _post(client, f"memory/{item_id}/correct", {"content": "Лучше работаю днём"}))[
        0
    ] == 200
    assert (await _post(client, f"memory/{item_id}/delete"))[0] == 200
    _, data = await _get(client, "memory")
    assert data["items"] == []


async def test_style_and_memory_settings(client):
    st, data = await _post(
        client,
        "settings",
        {"style_preset": "coach", "strictness": 3, "memory_settings": {"hypotheses": False}},
    )
    assert st == 200 and data["style_preset"] == "coach" and data["strictness"] == 3
    assert data["memory_settings"]["hypotheses"] is False
    assert (await _post(client, "settings", {"strictness": 9}))[0] == 400
    assert (await _post(client, "settings", {"memory_settings": {"x": True}}))[0] == 400


async def test_calendar_block_is_only_a_card(ctx, client):
    task_id = await _task(ctx)
    st, data = await _post(client, f"tasks/{task_id}/calendar", {"time": "15:00", "minutes": 60})
    # no calendar connected: refused with a Russian reason, nothing created
    assert st == 400 and "Calendar" in data["error"]
    async with ctx.db.transaction() as s:
        assert (await s.execute(select(PendingApproval))).scalars().all() == []
