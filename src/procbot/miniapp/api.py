"""Thin Mini App HTTP API over the same services as the bot (FR-16).

GET  /miniapp                  - the page
GET  /miniapp/api/day          - "Мой день" as JSON
POST /miniapp/api/tasks/{id}/done - mark a task done (the button press is the consent)
"""

from __future__ import annotations

from importlib.resources import files
from typing import Any

from aiohttp import web

from procbot.context import AppContext
from procbot.db.models import Task
from procbot.domain.enums import Actor, TaskStatus
from procbot.miniapp.auth import InitDataError, verify_init_data
from procbot.services import tasks as task_service
from procbot.services.dayview import DayView, build_day_view
from procbot.timeutil import fmt_local

CtxKey = web.AppKey[AppContext]


def _ctx(request: web.Request, key: CtxKey) -> AppContext:
    return request.app[key]


def _authorize(request: web.Request, ctx: AppContext) -> None:
    header = request.headers.get("Authorization", "")
    scheme, _, raw = header.partition(" ")
    if scheme.lower() != "tma" or not raw:
        raise web.HTTPUnauthorized(text="initData required")
    try:
        verify_init_data(
            raw,
            bot_token=ctx.env.telegram_bot_token,
            owner_id=ctx.owner_id,
            now=ctx.clock.now(),
        )
    except InitDataError as exc:
        raise web.HTTPForbidden(text=str(exc)) from exc


def _task_json(t: Task, view: DayView) -> dict[str, Any]:
    step = view.next_steps.get(t.id)
    due = t.due_at or None
    return {
        "id": t.id,
        "title": t.title,
        "status": str(t.status),
        "start": fmt_local(t.planned_start_at, view.tz, "%H:%M") if t.planned_start_at else None,
        "due": (
            fmt_local(due, view.tz)
            if due
            else (t.due_date.strftime("%d.%m") if t.due_date else None)
        ),
        "estimate_min": t.estimate_minutes,
        "next_step": step.title if step is not None else None,
    }


def day_payload(view: DayView) -> dict[str, Any]:
    b = view.buckets
    return {
        "day": b.day.isoformat(),
        "tz": b.tz,
        "sections": [
            {"key": key, "title": title, "tasks": [_task_json(t, view) for t in items]}
            for key, title, items in (
                ("overdue", "Просрочено", b.overdue),
                ("missed", "Не сделано в прошлые дни", b.missed),
                ("planned", "Запланировано на сегодня", b.planned),
                ("due_today", "Срок сегодня", b.due_today),
            )
            if items
        ],
        "backlog_count": b.backlog_count,
    }


def add_routes(app: web.Application, key: CtxKey) -> None:
    async def page(request: web.Request) -> web.Response:
        html = files("procbot.miniapp").joinpath("static/index.html").read_text("utf-8")
        return web.Response(text=html, content_type="text/html")

    async def day(request: web.Request) -> web.Response:
        ctx = _ctx(request, key)
        _authorize(request, ctx)
        async with ctx.db.transaction() as s:
            view = await build_day_view(s, ctx)
            payload = day_payload(view)
        return web.json_response(payload)

    async def done(request: web.Request) -> web.Response:
        ctx = _ctx(request, key)
        _authorize(request, ctx)
        try:
            task_id = int(request.match_info["task_id"])
        except ValueError as exc:
            raise web.HTTPBadRequest(text="bad task id") from exc
        async with ctx.db.transaction() as s:
            task = await task_service.get_task(s, task_id)
            if task is None or task.status in (TaskStatus.CANCELLED,):
                raise web.HTTPNotFound(text="task not found")
            task = await task_service.complete_task(s, ctx, task_id, actor=Actor.USER)
            status = str(task.status)
        return web.json_response({"id": task_id, "status": status})

    app.router.add_get("/miniapp", page)
    app.router.add_get("/miniapp/api/day", day)
    app.router.add_post("/miniapp/api/tasks/{task_id}/done", done)
