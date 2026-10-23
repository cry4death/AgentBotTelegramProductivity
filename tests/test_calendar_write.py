"""Google Calendar write safety (FR-11, hard rule 7): an event is inserted only
after the owner approves the card, exactly once, and the code cannot update or
delete external events."""

from __future__ import annotations

import json
import re
from datetime import timedelta
from pathlib import Path

import httpx
import pytest

from procbot.domain.actions import ActionType, DeleteCalendarEvent, UpdateCalendarEvent
from procbot.domain.enums import Actor
from procbot.services import actions, approvals
from procbot.services import calendar as cal
from procbot.services.calendar_google import GoogleClient, TokenSet
from procbot.services.profile import set_setting

SRC = Path(__file__).resolve().parents[1] / "src" / "procbot"


class FakeGoogle:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "POST" and request.url.path.endswith("/events"):
            body = json.loads(request.content)
            return httpx.Response(200, json={"id": body["id"], "status": "confirmed"})
        return httpx.Response(200, json={"items": []})

    def inserts(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == "POST" and r.url.path.endswith("/events")]


@pytest.fixture
async def google(ctx, profile):
    ctx.env.google_client_id = "cid"
    ctx.env.google_client_secret = "secret"
    fake = FakeGoogle()
    ctx.calendar = cal.CalendarService(ctx, transport=httpx.MockTransport(fake))
    cal.register(ctx)
    await cal.save_tokens(
        ctx,
        TokenSet(
            access_token="at",
            refresh_token="rt",
            expires_at=ctx.clock.now() + timedelta(hours=1),
            scopes=["https://www.googleapis.com/auth/calendar.events.owned"],
        ),
    )
    async with ctx.db.transaction() as s:
        await set_setting(
            s,
            cal.SETTING_CALENDARS,
            [{"id": "primary@x", "name": "Мой", "primary": True, "access_role": "owner"}],
            now=ctx.clock.now(),
        )
        assert await cal.set_write_calendar(s, ctx, "primary@x")
    return fake


async def _propose(ctx) -> int:
    start = ctx.clock.now() + timedelta(hours=2)
    async with ctx.db.transaction() as s:
        res = await cal.propose_calendar_block(
            s,
            ctx,
            title="Отчёт",
            start_at=start,
            end_at=start + timedelta(hours=1),
            origin=Actor.USER,  # even a user-triggered offer must show a card
        )
    assert res.status == actions.ExecStatus.PENDING_APPROVAL
    assert res.approval_id is not None
    return res.approval_id


async def _approve(ctx, approval_id: int) -> approvals.Resolution:
    async with ctx.db.transaction() as s:
        res = await approvals.approve(s, ctx, approval_id)
    await actions.run_after_commit(res.after_commit)
    return res


async def test_insert_happens_only_after_approval_and_only_once(ctx, google):
    approval_id = await _propose(ctx)
    assert google.inserts() == []  # proposing writes nothing to Google

    await _approve(ctx, approval_id)
    assert len(google.inserts()) == 1
    body = json.loads(google.inserts()[0].content)
    assert body["summary"] == "Отчёт"

    await _approve(ctx, approval_id)  # repeated button press
    assert len(google.inserts()) == 1


async def test_rejected_card_writes_nothing(ctx, google):
    approval_id = await _propose(ctx)
    async with ctx.db.transaction() as s:
        await approvals.reject(s, ctx, approval_id)
    await _approve(ctx, approval_id)  # late approve on a rejected card is a no-op
    await cal.reconcile_pending_inserts(ctx)
    assert google.inserts() == []


async def test_update_and_delete_of_external_events_are_impossible(ctx, google):
    for action in (
        UpdateCalendarEvent(external_event_id="e1"),
        DeleteCalendarEvent(external_event_id="e1"),
    ):
        assert actions.permission_for(action.type) == actions.Permission.FORBIDDEN
        assert actions.get_handler(ctx, action.type) is None
        async with ctx.db.transaction() as s:
            res = await actions.execute(
                s, ctx, action, origin=Actor.USER, op_key=f"t:{action.type}"
            )
        assert res.status != actions.ExecStatus.APPLIED
    assert actions.get_handler(ctx, ActionType.CREATE_CALENDAR_EVENT) is not None

    client = GoogleClient(client_id="c", client_secret="s", transport=httpx.MockTransport(google))
    with pytest.raises(ValueError):  # POST into an existing event path is blocked
        await client._send("POST", f"{cal_api()}/calendars/x/events/e1/move", token="t")

    code = "\n".join(p.read_text("utf-8") for p in SRC.rglob("calendar*.py"))
    assert not re.search(r"\"(PUT|PATCH|DELETE)\"", code)
    assert google.requests == []


def cal_api() -> str:
    from procbot.services.calendar_google import CALENDAR_API

    return CALENDAR_API
