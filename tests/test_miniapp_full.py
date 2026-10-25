"""Mini App sections (stage D, FR-16) over the same services as the chat:
task tree, week with calendar, goals roadmap, habits, work session, report,
settings. Every call needs the owner's signed initData."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from aiohttp.test_utils import TestClient, TestServer
from sqlalchemy import select

from procbot.db.models import CalendarConnection, CalendarEventCache, ReminderSchedule, Task
from procbot.domain.actions import TaskDraft
from procbot.domain.enums import Actor
from procbot.http_server import build_app
from procbot.miniapp.auth import sign_init_data
from procbot.services import goals, habits
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


async def _today(ctx):
    async with ctx.db.transaction() as s:
        return await task_service.local_today(s, ctx)


@pytest.mark.parametrize(
    "path", ["tasks", "week", "goals", "habits", "session", "report", "settings"]
)
async def test_every_section_requires_owner(client, path):
    r = await client.get("/miniapp/api/" + path)
    assert r.status == 401


async def test_task_tree_create_is_idempotent_and_subtasks_nest(ctx, client):
    today = (await _today(ctx)).isoformat()
    body = {"title": "Отчёт", "planned_date": today, "op_key": "op-12345678"}
    assert (await _post(client, "tasks", body))[0] == 200
    assert (await _post(client, "tasks", body))[0] == 200  # same press again
    st, data = await _get(client, "tasks?view=today")
    assert st == 200 and [t["title"] for t in data["tasks"]] == ["Отчёт"]
    parent = data["tasks"][0]["id"]
    await _post(
        client, "tasks", {"title": "Собрать цифры", "parent_id": parent, "op_key": "op-abcdefgh"}
    )
    _, data = await _get(client, "tasks?view=today")
    assert [c["title"] for c in data["tasks"][0]["children"]] == ["Собрать цифры"]
    assert (await _post(client, "tasks", {"title": "", "op_key": "op-zzzzzzzz"}))[0] == 400
    assert (await _get(client, "tasks?view=nope"))[0] == 400


async def test_done_needs_criteria_check_and_progress_is_not_done(ctx, client):
    async with ctx.db.transaction() as s:
        t = await task_service.create_task(
            s,
            ctx,
            TaskDraft(title="Статья", done_criteria="3 страницы", verification="criteria"),
            actor=Actor.USER,
        )
        task_id = t.id
    st, data = await _post(client, f"tasks/{task_id}/progress", {"note": "1 страница"})
    assert st == 200 and data["status"] == "in_progress"
    st, data = await _post(client, f"tasks/{task_id}/done")
    assert st == 409 and data["done_criteria"] == "3 страницы"
    st, data = await _post(client, f"tasks/{task_id}/done", {"criteria_confirmed": True})
    assert st == 200 and data["status"] == "done"


async def test_week_shows_tasks_and_calendar_busy_time(ctx, client):
    now = ctx.clock.now()
    today = await _today(ctx)
    async with ctx.db.transaction() as s:
        conn = CalendarConnection(status="active", last_sync_at=now, created_at=now, updated_at=now)
        s.add(conn)
        await s.flush()
        s.add(
            CalendarEventCache(
                connection_id=conn.id,
                calendar_id="primary",
                external_id="e1",
                start_at=now + timedelta(hours=2),
                end_at=now + timedelta(hours=3),
                busy=True,
                summary="Встреча",
                synced_at=now,
            )
        )
        await task_service.create_task(
            s, ctx, TaskDraft(title="План", planned_date=today), actor=Actor.USER
        )
    st, data = await _get(client, "week")
    assert st == 200 and len(data["days"]) == 7
    day = next(d for d in data["days"] if d["is_today"])
    assert [t["title"] for t in day["tasks"]] == ["План"]
    assert day["events"][0]["title"] == "Встреча"
    assert day["events"][0]["start"] == "12:00"  # Europe/Berlin local time
    assert (await _get(client, "week?start=bad"))[0] == 400


async def test_goals_roadmap_and_habits_marking(ctx, client):
    async with ctx.db.transaction() as s:
        g = await goals.create_goal(s, ctx, title="Выучить испанский", actor=Actor.USER)
        await goals.add_milestone(s, ctx, g.id, title="A1", actor=Actor.USER, is_key=True)
        h = await habits.create_habit(s, ctx, title="Чтение", actor=Actor.USER)
        habit_id = h.id
    _, data = await _get(client, "goals")
    goal = data["goals"][0]
    assert goal["title"] == "Выучить испанский"
    assert goal["milestones"][0] == {**goal["milestones"][0], "title": "A1", "is_key": True}

    _, data = await _get(client, "habits")
    assert data["habits"][0]["due_today"] is True
    assert (await _post(client, f"habits/{habit_id}/mark", {"status": "partial"}))[0] == 200
    _, data = await _get(client, "habits")
    assert data["habits"][0]["today_status"] == "partial"
    assert (await _post(client, f"habits/{habit_id}/mark", {"status": "missed"}))[0] == 400


async def test_work_session_flow(ctx, client):
    async with ctx.db.transaction() as s:
        t = await task_service.create_task(s, ctx, TaskDraft(title="Код"), actor=Actor.USER)
        task_id = t.id
    st, data = await _post(client, "session/start", {"task_id": task_id, "mode": "free"})
    assert st == 200 and data["session"]["phase"] == "work"
    sid = data["session"]["id"]
    assert (await _post(client, "session/start", {"mode": "free"}))[0] == 409  # one at a time
    ctx.clock.advance(minutes=10)
    _, data = await _post(client, "session/pause", {"session_id": sid})
    assert data["session"]["status"] == "paused"
    _, data = await _post(client, "session/resume", {"session_id": sid})
    _, data = await _post(client, "session/stop", {"session_id": sid})
    assert data["session"]["needs_outcome"] is True
    assert data["session"]["net_seconds"] == 600
    _, data = await _post(client, "session/outcome", {"session_id": sid, "outcome": "progressed"})
    assert data["session"] is None
    async with ctx.db.transaction() as s:
        task = await s.get(Task, task_id)
        assert task is not None and task.status == "in_progress"  # partial is never done


async def test_report_has_six_directions(client):
    st, data = await _get(client, "report?days=7")
    assert st == 200 and len(data["directions"]) == 6
    assert (await _get(client, "report?days=3"))[0] == 400


async def test_settings_roundtrip_validates_and_reseeds_reminders(ctx, client):
    st, data = await _get(client, "settings")
    assert st == 200 and data["timezone"] == "Europe/Berlin"
    st, data = await _post(
        client,
        "settings",
        {
            "morning_time": "08:30",
            "start_delay_min": 15,
            "quiet_hours": [{"start": "23:00", "end": "07:00"}],
            "pause_hours": 2,
        },
    )
    assert st == 200
    assert data["morning_time"] == "08:30" and data["start_delay_min"] == 15
    assert data["quiet_hours"][0]["start"] == "23:00" and data["pause_until"]
    async with ctx.db.transaction() as s:
        rows = (
            await s.execute(
                select(ReminderSchedule).where(ReminderSchedule.category == "morning_plan")
            )
        ).scalars()
        assert any(r.state == "pending" for r in rows)
    for bad in (
        {"timezone": "Mars/Base"},
        {"start_delay_min": 999},
        {"morning_time": "25:00"},
        {"style_preset": "rude"},  # not editable from the Mini App
    ):
        assert (await _post(client, "settings", bad))[0] == 400
