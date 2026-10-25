"""Thin Mini App HTTP API over the same services as the bot (FR-16).

Every /miniapp/api/* call needs `Authorization: tma <initData>` signed by
Telegram for the owner. Each button press in the Mini App is the owner's
explicit action for that exact change (like a chat button), so it is applied
with actor=USER; nothing here is proposed by the AI. Writes that create rows
take a client `op_key` so a repeated press is a no-op.

Routes (all under /miniapp):
  GET  /                         page
  GET  /api/day                  "Мой день" + today's habits + work session
  GET  /api/tasks?view=...       task tree (today/upcoming/backlog/someday/done)
  POST /api/tasks                create a task or a subtask
  POST /api/tasks/{id}/done      mark done (criteria check when the task has one)
  POST /api/tasks/{id}/progress  partial result (never "done")
  GET  /api/week?start=...       week: planned tasks + calendar busy time
  GET  /api/goals                goals, milestones (roadmap) and progress
  GET  /api/habits               habits with regularity
  POST /api/habits/{id}/mark     done / partial / skipped for today
  GET  /api/session              current work session
  POST /api/session/{action}     start / pause / resume / stop / outcome
  GET  /api/report?days=7        the six analytics directions (no AI)
  GET  /api/settings, POST /api/settings
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import date, time, timedelta
from importlib.resources import files
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aiohttp import web

from procbot.context import AppContext
from procbot.domain.actions import CreateSubtasks, CreateTask, SubtaskDraft, TaskDraft
from procbot.domain.enums import Actor, HabitOccurrenceStatus, SessionMode, SessionOutcome
from procbot.miniapp import views
from procbot.miniapp.auth import InitDataError, verify_init_data
from procbot.services import actions, habits, sessions
from procbot.services import tasks as task_service
from procbot.services.profile import get_profile, update_profile
from procbot.services.profile_hooks import after_profile_change

CtxKey = web.AppKey[AppContext]
Handler = Callable[[web.Request, AppContext], Awaitable[web.StreamResponse]]

MAX_TITLE = 500
REPORT_DAYS = (1, 7, 14, 30)
MAX_PAUSE_HOURS = 24 * 14


def _error(status: int, message: str, **extra: Any) -> web.Response:
    return web.json_response({"error": message, **extra}, status=status)


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


async def _body(request: web.Request) -> dict[str, Any]:
    if not request.can_read_body:
        return {}
    try:
        data = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise web.HTTPBadRequest(text="bad json") from exc
    if not isinstance(data, dict):
        raise web.HTTPBadRequest(text="json object expected")
    return data


def _int_id(request: web.Request, name: str) -> int:
    try:
        return int(request.match_info[name])
    except ValueError as exc:
        raise web.HTTPBadRequest(text=f"bad {name}") from exc


def _date(value: Any, field: str) -> date | None:
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"Неверная дата: {field}") from exc


def _hhmm(value: Any, field: str) -> time | None:
    if value in (None, ""):
        return None
    try:
        t = time.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"Неверное время: {field}") from exc
    return t.replace(second=0, microsecond=0, tzinfo=None)


def _minutes(value: Any, field: str, *, high: int = 240) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise ValueError(f"Нужно число минут: {field}")
    try:
        n = int(value)
    except ValueError as exc:
        raise ValueError(f"Нужно число минут: {field}") from exc
    if not 0 <= n <= high:
        raise ValueError(f"Значение вне диапазона 0–{high}: {field}")
    return n


def _route(key: CtxKey, fn: Handler) -> Callable[[web.Request], Awaitable[web.StreamResponse]]:
    """Auth + mapping of service errors to JSON answers with Russian text."""

    async def wrapped(request: web.Request) -> web.StreamResponse:
        ctx = request.app[key]
        _authorize(request, ctx)
        try:
            return await fn(request, ctx)
        except sessions.ActiveSessionExists as exc:
            return _error(409, "Уже идёт другая рабочая сессия.", session_id=exc.session_id)
        except sessions.SettingsMissing as exc:
            return _error(409, "Для этого режима не заданы длительности.", missing=list(exc.fields))
        except sessions.CriteriaCheckRequired:
            return _error(409, "Сначала сверь результат с критерием готовности.", criteria=True)
        except sessions.SessionError as exc:
            return _error(409, str(exc))
        except task_service.TaskNotFound:
            return _error(404, "Задача не найдена.")
        except actions.ActionError as exc:
            return _error(400, str(exc))
        except LookupError:
            return _error(404, "Не найдено.")
        except ValueError as exc:
            return _error(400, str(exc))

    return wrapped


# --------------------------------------------------------------------------- reads


async def day(request: web.Request, ctx: AppContext) -> web.Response:
    async with ctx.db.transaction() as s:
        payload = await views.day_payload(s, ctx)
    return web.json_response(payload)


async def task_tree(request: web.Request, ctx: AppContext) -> web.Response:
    view = request.query.get("view", "today")
    if view not in {v.value for v in task_service.TaskView}:
        raise ValueError("Неизвестный список задач.")
    async with ctx.db.transaction() as s:
        payload = await views.tasks_payload(s, ctx, view)
    return web.json_response(payload)


async def week(request: web.Request, ctx: AppContext) -> web.Response:
    start = _date(request.query.get("start"), "start")
    async with ctx.db.transaction() as s:
        payload = await views.week_payload(s, ctx, start)
    return web.json_response(payload)


async def goal_list(request: web.Request, ctx: AppContext) -> web.Response:
    async with ctx.db.transaction() as s:
        payload = await views.goals_payload(s, ctx)
    return web.json_response(payload)


async def habit_list(request: web.Request, ctx: AppContext) -> web.Response:
    async with ctx.db.transaction() as s:
        payload = await views.habits_payload(s, ctx)
    return web.json_response(payload)


async def session_get(request: web.Request, ctx: AppContext) -> web.Response:
    async with ctx.db.transaction() as s:
        payload = await views.session_payload(s, ctx)
    return web.json_response({"session": payload})


async def report(request: web.Request, ctx: AppContext) -> web.Response:
    try:
        days = int(request.query.get("days", "7"))
    except ValueError:
        days = 0
    if days not in REPORT_DAYS:
        raise ValueError("Период отчёта: 1, 7, 14 или 30 дней.")
    async with ctx.db.transaction() as s:
        payload = await views.report_payload(s, ctx, days)
    return web.json_response(payload)


# --------------------------------------------------------------------------- task writes


async def task_create(request: web.Request, ctx: AppContext) -> web.Response:
    body = await _body(request)
    title = " ".join(str(body.get("title") or "").split())[:MAX_TITLE]
    op_key = str(body.get("op_key") or "")
    if not title:
        raise ValueError("Нужно название задачи.")
    if not 8 <= len(op_key) <= 64:
        raise ValueError("Нужен op_key запроса.")
    estimate = _minutes(body.get("estimate_minutes"), "estimate_minutes", high=24 * 60)
    parent_id = body.get("parent_id")
    action: CreateTask | CreateSubtasks
    if parent_id is not None:
        action = CreateSubtasks(
            parent_id=int(parent_id),
            subtasks=[SubtaskDraft(title=title, estimate_minutes=estimate or None)],
        )
    else:
        action = CreateTask(
            task=TaskDraft(
                title=title,
                planned_date=_date(body.get("planned_date"), "planned_date"),
                due_date=_date(body.get("due_date"), "due_date"),
                estimate_minutes=estimate or None,
            )
        )
    async with ctx.db.transaction() as s:
        res = await actions.execute(s, ctx, action, origin=Actor.USER, op_key=f"miniapp:{op_key}")
    await actions.run_after_commit(res.after_commit)
    if res.status != actions.ExecStatus.APPLIED:
        return _error(400, res.error or "Не удалось создать задачу.")
    return web.json_response({"status": "created", "result": res.result, "repeated": res.repeated})


async def task_done(request: web.Request, ctx: AppContext) -> web.Response:
    task_id = _int_id(request, "task_id")
    body = await _body(request)
    confirmed = body.get("criteria_confirmed") is True
    async with ctx.db.transaction() as s:
        task = await task_service.require_task(s, task_id)
        if task_service.requires_criteria_check(task) and not confirmed:
            return _error(
                409,
                "Сверь результат с критерием готовности.",
                criteria=True,
                done_criteria=task.done_criteria,
            )
        task = await task_service.complete_task(
            s, ctx, task_id, actor=Actor.USER, criteria_confirmed=confirmed or None
        )
        status = str(task.status)
    return web.json_response({"id": task_id, "status": status})


async def task_progress(request: web.Request, ctx: AppContext) -> web.Response:
    task_id = _int_id(request, "task_id")
    body = await _body(request)
    note = " ".join(str(body.get("note") or "").split())[:1000] or None
    async with ctx.db.transaction() as s:
        task = await task_service.record_progress(s, ctx, task_id, actor=Actor.USER, note=note)
        status = str(task.status)
    return web.json_response({"id": task_id, "status": status})


# --------------------------------------------------------------------------- habits


async def habit_mark(request: web.Request, ctx: AppContext) -> web.Response:
    habit_id = _int_id(request, "habit_id")
    body = await _body(request)
    status = str(body.get("status") or "")
    allowed = {
        HabitOccurrenceStatus.DONE,
        HabitOccurrenceStatus.PARTIAL,
        HabitOccurrenceStatus.SKIPPED,
    }
    if status not in allowed:
        raise ValueError("Отметка: done, partial или skipped.")
    async with ctx.db.transaction() as s:
        await habits.mark(s, ctx, habit_id, status, actor=Actor.USER)
    return web.json_response({"id": habit_id, "status": status})


# --------------------------------------------------------------------------- work session


async def session_action(request: web.Request, ctx: AppContext) -> web.Response:
    what = request.match_info["action"]
    body = await _body(request)
    async with ctx.db.transaction() as s:
        if what == "start":
            mode = SessionMode(str(body.get("mode") or SessionMode.FREE))
            task_id = body.get("task_id")
            if task_id is not None:
                await task_service.require_task(s, int(task_id))
            await sessions.start_session(
                s, ctx, task_id=int(task_id) if task_id is not None else None, mode=mode
            )
        else:
            session_id = int(body.get("session_id") or 0)
            if await sessions.get_work_session(s, session_id) is None:
                raise LookupError("session")
            if what == "pause":
                await sessions.pause_session(s, ctx, session_id=session_id)
            elif what == "resume":
                await sessions.resume_session(s, ctx, session_id=session_id)
            elif what == "stop":
                await sessions.stop_session(s, ctx, session_id=session_id)
            elif what == "outcome":
                await sessions.record_outcome(
                    s,
                    ctx,
                    session_id=session_id,
                    outcome=SessionOutcome(str(body.get("outcome") or "")),
                    criteria_confirmed=True if body.get("criteria_confirmed") is True else None,
                    note=(str(body.get("note"))[:1000] if body.get("note") else None),
                )
            else:
                raise LookupError(what)
        payload = await views.session_payload(s, ctx)
    return web.json_response({"session": payload})


# --------------------------------------------------------------------------- settings


def _settings_changes(body: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    for key, value in body.items():
        if key == "timezone":
            try:
                ZoneInfo(str(value))
            except (ZoneInfoNotFoundError, ValueError) as exc:
                raise ValueError("Неизвестный часовой пояс.") from exc
            changes[key] = str(value)
        elif key in views.SETTINGS_TIMES:
            changes[key] = _hhmm(value, key)
        elif key in views.SETTINGS_MINUTES:
            changes[key] = _minutes(value, key)
        elif key == "notifications_enabled":
            if not isinstance(value, bool):
                raise ValueError("notifications_enabled: да или нет.")
            changes[key] = value
        elif key == "weekly_review_dow":
            if value is not None and (not isinstance(value, int) or not 0 <= value <= 6):
                raise ValueError("День недели: 0–6.")
            changes[key] = value
        elif key == "quiet_hours":
            if not isinstance(value, list):
                raise ValueError("quiet_hours: список окон.")
            windows = []
            for w in value:
                if not isinstance(w, dict):
                    raise ValueError("quiet_hours: окно {start, end}.")
                start, end = _hhmm(w.get("start"), "start"), _hhmm(w.get("end"), "end")
                if start is None or end is None or start == end:
                    raise ValueError("Тихие часы: нужно начало и конец.")
                window: dict[str, Any] = {
                    "start": start.strftime("%H:%M"),
                    "end": end.strftime("%H:%M"),
                }
                days = w.get("days")
                if days:
                    window["days"] = sorted({int(d) for d in days if 0 <= int(d) <= 6})
                windows.append(window)
            changes[key] = windows
        elif key == "pause_hours":
            hours = _minutes(value, key, high=MAX_PAUSE_HOURS) or 0
            changes["pause_until"] = ctx.clock.now() + timedelta(hours=hours) if hours > 0 else None
        else:
            raise ValueError(f"Эту настройку нельзя менять из Mini App: {key}")
    return changes


async def settings_get(request: web.Request, ctx: AppContext) -> web.Response:
    async with ctx.db.transaction() as s:
        payload = views.settings_json(await get_profile(s), ctx)
    return web.json_response(payload)


async def settings_post(request: web.Request, ctx: AppContext) -> web.Response:
    changes = _settings_changes(await _body(request), ctx)
    async with ctx.db.transaction() as s:
        if await get_profile(s) is None:
            raise LookupError("profile")
        if changes:
            await update_profile(s, now=ctx.clock.now(), actor=Actor.USER, **changes)
            await after_profile_change(s, ctx, changes.keys())
        payload = views.settings_json(await get_profile(s), ctx)
    return web.json_response(payload)


# --------------------------------------------------------------------------- wiring


def add_routes(app: web.Application, key: CtxKey) -> None:
    async def page(request: web.Request) -> web.Response:
        html = files("procbot.miniapp").joinpath("static/index.html").read_text("utf-8")
        return web.Response(text=html, content_type="text/html")

    r = app.router
    r.add_get("/miniapp", page)
    r.add_get("/miniapp/", page)
    r.add_get("/miniapp/api/day", _route(key, day))
    r.add_get("/miniapp/api/tasks", _route(key, task_tree))
    r.add_post("/miniapp/api/tasks", _route(key, task_create))
    r.add_post("/miniapp/api/tasks/{task_id}/done", _route(key, task_done))
    r.add_post("/miniapp/api/tasks/{task_id}/progress", _route(key, task_progress))
    r.add_get("/miniapp/api/week", _route(key, week))
    r.add_get("/miniapp/api/goals", _route(key, goal_list))
    r.add_get("/miniapp/api/habits", _route(key, habit_list))
    r.add_post("/miniapp/api/habits/{habit_id}/mark", _route(key, habit_mark))
    r.add_get("/miniapp/api/session", _route(key, session_get))
    r.add_post("/miniapp/api/session/{action}", _route(key, session_action))
    r.add_get("/miniapp/api/report", _route(key, report))
    r.add_get("/miniapp/api/settings", _route(key, settings_get))
    r.add_post("/miniapp/api/settings", _route(key, settings_post))
