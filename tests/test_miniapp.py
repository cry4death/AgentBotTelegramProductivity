"""Mini App (stage D): initData check, "Мой день" JSON, the "done" button."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from aiohttp.test_utils import TestClient, TestServer

from procbot.domain.actions import TaskDraft
from procbot.domain.enums import Actor, TaskStatus
from procbot.http_server import build_app
from procbot.miniapp.auth import InitDataError, sign_init_data, verify_init_data
from procbot.services import tasks as task_service
from procbot.testing import OWNER_ID


def _init_data(ctx, *, user_id=OWNER_ID, token=None, age=timedelta(0)) -> str:
    fields = {
        "auth_date": str(int((ctx.clock.now() - age).timestamp())),
        "query_id": "AAH",
        "user": json.dumps({"id": user_id, "first_name": "U"}),
    }
    return sign_init_data(fields, token or ctx.env.telegram_bot_token)


@pytest.fixture
async def client(ctx, profile):
    c = TestClient(TestServer(build_app(ctx)))
    await c.start_server()
    yield c
    await c.close()


def test_init_data_rules(ctx):
    kw = {"bot_token": ctx.env.telegram_bot_token, "owner_id": OWNER_ID, "now": ctx.clock.now()}
    assert verify_init_data(_init_data(ctx), **kw) == OWNER_ID
    for bad in (
        _init_data(ctx, user_id=OWNER_ID + 1),  # someone else
        _init_data(ctx, token="1:OTHER"),  # signed with another bot token
        _init_data(ctx, age=timedelta(days=2)),  # too old
        _init_data(ctx).replace("first_name", "first_nam"),  # tampered
        "auth_date=1&user=%7B%7D",  # no hash
    ):
        with pytest.raises(InitDataError):
            verify_init_data(bad, **kw)


async def test_day_requires_owner_init_data(ctx, client):
    assert (await client.get("/miniapp/api/day")).status == 401
    r = await client.get(
        "/miniapp/api/day", headers={"Authorization": "tma " + _init_data(ctx, user_id=1)}
    )
    assert r.status == 403
    page = await client.get("/miniapp")
    assert page.status == 200 and "Мой день" in await page.text()


async def test_day_lists_tasks_and_done_marks_task(ctx, client):
    today = ctx.clock.now().date()
    async with ctx.db.transaction() as s:
        task = await task_service.create_task(
            s, ctx, TaskDraft(title="Написать отчёт", planned_date=today), actor=Actor.USER
        )
        task_id = task.id
    headers = {"Authorization": "tma " + _init_data(ctx)}

    r = await client.get("/miniapp/api/day", headers=headers)
    assert r.status == 200
    data = await r.json()
    titles = [t["title"] for sec in data["sections"] for t in sec["tasks"]]
    assert titles == ["Написать отчёт"]

    for _ in range(2):  # repeated press is a no-op
        r = await client.post(f"/miniapp/api/tasks/{task_id}/done", headers=headers)
        assert r.status == 200 and (await r.json())["status"] == "done"
    async with ctx.db.transaction() as s:
        t = await task_service.require_task(s, task_id)
        assert t.status == TaskStatus.DONE
        events = await task_service.task_events(s, task_id)
    assert [e.event_type for e in events].count("completed") == 1

    r = await client.post("/miniapp/api/tasks/999/done", headers=headers)
    assert r.status == 404
