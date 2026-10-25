"""Mini App write flows beyond the basics, so the owner can manage everything
without the chat: edit/reschedule tasks, approve proposals, day plan, quick
capture, goals and milestones, habits, memory, calendar blocks.

Each handler is one explicit owner action (actor=USER) through the same
services the chat uses. AI output is never applied here: AI proposals arrive
as approval cards and are applied only by "approve".
"""

from __future__ import annotations

import zlib
from datetime import timedelta
from typing import Any

from aiohttp import web
from sqlalchemy import select

from procbot.ai import AIUnavailable
from procbot.context import AppContext
from procbot.db.models import Milestone, PendingApproval, PlanRevision, TaskEvent
from procbot.domain.actions import CreateGoal, CreateHabit
from procbot.domain.enums import Actor, ApprovalStatus, EntityStatus, MemoryStatus
from procbot.miniapp import http as H
from procbot.miniapp import views
from procbot.miniapp.http import CtxKey
from procbot.services import actions, approvals, capture, goals, habits, memory, planning
from procbot.services import calendar as cal
from procbot.services import tasks as task_service
from procbot.timeutil import fmt_local, resolve_local

MAX_TEXT = 2000
TASK_TEXT_FIELDS = ("title", "description", "done_criteria")
GOAL_STATUSES = frozenset({"active", "paused", "done", "archived"})
MILESTONE_STATUSES = frozenset({"active", "done", "archived"})
HABIT_STATUSES = frozenset({"active", "paused", "archived"})
MAX_BLOCK_MIN = 8 * 60


def _text(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).split())[:limit] or None


def _op_key(body: dict[str, Any]) -> str:
    key = str(body.get("op_key") or "")
    if not 8 <= len(key) <= 64:
        raise ValueError("Нужен op_key запроса.")
    return key


def _update_id(op_key: str) -> int:
    """Capture/dedup code keys on Telegram update ids; Mini App presses get a
    stable negative id from their op_key (never collides with real updates)."""
    return -(zlib.crc32(op_key.encode()) or 1)


async def _apply(ctx: AppContext, res: approvals.Resolution) -> approvals.Resolution:
    await actions.run_after_commit(res.after_commit)
    return res


def _exec_answer(res: actions.ExecResult) -> web.Response:
    if res.status != actions.ExecStatus.APPLIED:
        return H.error(400, res.error or "Не удалось применить.")
    return web.json_response({"status": "ok", "result": res.result, "repeated": res.repeated})


# --------------------------------------------------------------------------- tasks


async def task_detail(request: web.Request, ctx: AppContext) -> web.Response:
    task_id = H.int_id(request, "task_id")
    async with ctx.db.transaction() as s:
        t = await task_service.require_task(s, task_id)
        tz = await task_service.owner_tz(s)
        data = views.task_json(t, tz)
        data.update(
            {
                "description": t.description,
                "parent_id": t.parent_id,
                "goal_id": t.goal_id,
                "milestone_id": t.milestone_id,
                "planned_time": (
                    fmt_local(t.planned_start_at, tz, "%H:%M") if t.planned_start_at else None
                ),
                "planned_day": (
                    fmt_local(t.planned_start_at, tz, "%Y-%m-%d")
                    if t.planned_start_at
                    else (t.planned_date.isoformat() if t.planned_date else None)
                ),
                "due_date": (
                    fmt_local(t.due_at, tz, "%Y-%m-%d")
                    if t.due_at
                    else (t.due_date.isoformat() if t.due_date else None)
                ),
                "postpone_count": t.postpone_count,
                "children": [
                    views.task_json(c, tz) for c in await task_service.list_subtasks(s, t.id)
                ],
            }
        )
        events = (
            await s.execute(
                select(TaskEvent)
                .where(TaskEvent.task_id == task_id)
                .order_by(TaskEvent.created_at.desc(), TaskEvent.id.desc())
                .limit(10)
            )
        ).scalars()
        data["history"] = [
            {"type": e.event_type, "at": fmt_local(e.created_at, tz), "actor": e.actor}
            for e in events
        ]
        data["goals"] = [{"id": g.id, "title": g.title} for g in await goals.list_goals(s)]
    return web.json_response(data)


async def task_edit(request: web.Request, ctx: AppContext) -> web.Response:
    """Edit fields the owner typed. Plan/deadline changes use the scheduling
    functions so reminders and history stay right."""
    task_id = H.int_id(request, "task_id")
    body = await H.body(request)
    async with ctx.db.transaction() as s:
        task = await task_service.require_task(s, task_id)
        tz = await task_service.owner_tz(s)
        fields: dict[str, Any] = {}
        for key in TASK_TEXT_FIELDS:
            if key not in body:
                continue
            if key == "title":
                value = _text(body[key], 500)
                if not value:
                    raise ValueError("Название не может быть пустым.")
            else:  # multi-line text keeps its line breaks
                value = str(body[key] or "").strip()[:MAX_TEXT] or None
            fields[key] = value
        if "done_criteria" in fields:
            fields["verification"] = "criteria" if fields["done_criteria"] else "button"
        if "estimate_minutes" in body:
            est = H.parse_minutes(body["estimate_minutes"], "estimate_minutes", high=24 * 60)
            fields["estimate_minutes"] = est or None
        if fields:
            await task_service.update_task(s, ctx, task_id, actor=Actor.USER, **fields)
        if "planned_day" in body or "planned_time" in body:
            day = H.parse_date(body.get("planned_day"), "planned_day")
            at = H.parse_hhmm(body.get("planned_time"), "planned_time")
            had_plan = task.planned_date is not None or task.planned_start_at is not None
            if day is None and at is None:
                if had_plan:
                    await task_service.move_to_backlog(s, ctx, task_id, actor=Actor.USER)
            else:
                if day is None:
                    day = await task_service.local_today(s, ctx)
                start = resolve_local(day, at, tz) if at is not None else None
                fn = task_service.reschedule_task if had_plan else task_service.plan_task
                await fn(
                    s,
                    ctx,
                    task_id,
                    actor=Actor.USER,
                    planned_date=None if start is not None else day,
                    planned_start_at=start,
                )
        if "due_date" in body or "due_is_hard" in body:
            due = H.parse_date(body.get("due_date"), "due_date")
            await task_service.set_deadline(
                s,
                ctx,
                task_id,
                actor=Actor.USER,
                due_date=due,
                due_is_hard=bool(body.get("due_is_hard")) if due is not None else False,
            )
        if "goal_id" in body:
            goal_id = body.get("goal_id")
            await goals.link_task(
                s, ctx, task_id, actor=Actor.USER, goal_id=int(goal_id) if goal_id else None
            )
        task = await task_service.require_task(s, task_id)
        data = views.task_json(task, tz)
    return web.json_response(data)


async def task_status(request: web.Request, ctx: AppContext) -> web.Response:
    task_id = H.int_id(request, "task_id")
    what = str((await H.body(request)).get("action") or "")
    async with ctx.db.transaction() as s:
        await task_service.require_task(s, task_id)
        if what == "cancel":
            await task_service.cancel_task(s, ctx, task_id, actor=Actor.USER)
        elif what == "someday":
            await task_service.defer_to_someday(s, ctx, task_id, actor=Actor.USER)
        elif what == "reopen":
            await task_service.reopen_task(s, ctx, task_id, actor=Actor.USER)
        elif what == "backlog":
            await task_service.move_to_backlog(s, ctx, task_id, actor=Actor.USER)
        elif what == "tomorrow":
            tomorrow = await task_service.local_today(s, ctx) + timedelta(days=1)
            await task_service.postpone_task(
                s, ctx, task_id, actor=Actor.USER, planned_date=tomorrow
            )
        elif what == "delete":
            await task_service.delete_task(s, ctx, task_id, actor=Actor.USER)
            return web.json_response({"id": task_id, "status": "deleted"})
        else:
            raise ValueError("Неизвестное действие с задачей.")
        status = str((await task_service.require_task(s, task_id)).status)
    return web.json_response({"id": task_id, "status": status})


async def task_calendar(request: web.Request, ctx: AppContext) -> web.Response:
    """Offer a calendar block for the task. Creates an approval card only: the
    event is inserted after the owner approves that card (hard rule 7)."""
    task_id = H.int_id(request, "task_id")
    body = await H.body(request)
    minutes = H.parse_minutes(body.get("minutes"), "minutes", high=MAX_BLOCK_MIN) or 0
    if minutes < 5:
        raise ValueError("Блок — от 5 минут.")
    async with ctx.db.transaction() as s:
        task = await task_service.require_task(s, task_id)
        tz = await task_service.owner_tz(s)
        day = H.parse_date(body.get("day"), "day") or await task_service.local_today(s, ctx)
        at = H.parse_hhmm(body.get("time"), "time")
        if at is None:
            raise ValueError("Укажи время начала.")
        start = resolve_local(day, at, tz)
        res = await cal.propose_calendar_block(
            s,
            ctx,
            title=task.title,
            start_at=start,
            end_at=start + timedelta(minutes=minutes),
            task_id=task.id,
        )
    if res.status != actions.ExecStatus.PENDING_APPROVAL:
        return H.error(400, res.error or "Не удалось предложить событие.")
    return web.json_response({"approval_id": res.approval_id})


# --------------------------------------------------------------------------- approvals


def approval_json(a: PendingApproval, ctx: AppContext, tz: str) -> dict[str, Any]:
    diff = a.diff or {}
    return {
        "id": a.id,
        "type": a.action_type,
        "summary": diff.get("summary") or a.action_type,
        "lines": [str(x) for x in diff.get("lines") or []][:20],
        "origin": a.origin,
        "group": a.group_key,
        "created": fmt_local(a.created_at, tz),
        "expires": fmt_local(a.expires_at, tz) if a.expires_at else None,
    }


async def approval_list(request: web.Request, ctx: AppContext) -> web.Response:
    async with ctx.db.transaction() as s:
        tz = await task_service.owner_tz(s)
        rows = await approvals.list_pending(s)
        payload = [approval_json(a, ctx, tz) for a in rows]
    return web.json_response({"approvals": payload})


async def approval_resolve(request: web.Request, ctx: AppContext) -> web.Response:
    approval_id = H.int_id(request, "approval_id")
    what = request.match_info["action"]
    if what not in ("approve", "reject"):
        raise LookupError(what)
    async with ctx.db.transaction() as s:
        if await approvals.get_approval(s, approval_id) is None:
            raise LookupError("approval")
        fn = approvals.approve if what == "approve" else approvals.reject
        res = await fn(s, ctx, approval_id)
        status, message = str(res.status), res.message
    await _apply(ctx, res)
    ok = status in (ApprovalStatus.APPLIED, ApprovalStatus.REJECTED)
    return web.json_response({"id": approval_id, "status": status, "message": message, "ok": ok})


# --------------------------------------------------------------------------- quick capture


async def capture_text(request: web.Request, ctx: AppContext) -> web.Response:
    """Free text -> task proposals as approval cards (AI when configured,
    otherwise one candidate per line). Nothing is created until approved."""
    body = await H.body(request)
    text = str(body.get("text") or "").strip()[:MAX_TEXT]
    if not text:
        raise ValueError("Напиши, что добавить.")
    update_id = _update_id(_op_key(body))
    result = None
    if capture.ai_configured(ctx):
        try:
            result = await capture.capture_tasks(ctx, text, update_id=update_id)
        except AIUnavailable:
            result = None
    if result is None:
        result = await capture.fallback_capture(ctx, text, update_id=update_id)
    async with ctx.db.transaction() as s:
        tz = await task_service.owner_tz(s)
        cards = [
            approval_json(a, ctx, tz)
            for a in [await approvals.get_approval(s, x.id) for x in result.approvals]
            if a is not None and a.status == ApprovalStatus.PENDING
        ]
    return web.json_response({"approvals": cards, "text": result.text, "header": result.header})


# --------------------------------------------------------------------------- day plan


async def _plan_payload(s: Any, ctx: AppContext, revision_id: int | None = None) -> dict[str, Any]:
    view = await planning.load_plan_view(s, ctx, revision_id=revision_id)
    if view is None:
        return {"plan": None}
    return {
        "plan": {
            "plan_id": view.plan.id,
            "revision_id": view.revision.id,
            "day": view.plan.local_date.isoformat(),
            "status": view.revision.status,
            "is_proposal": view.is_proposal,
            "text": planning.render_plan_text(view),
        }
    }


async def plan_get(request: web.Request, ctx: AppContext) -> web.Response:
    async with ctx.db.transaction() as s:
        today = await task_service.local_today(s, ctx)
        recent = await planning.recent_proposal(s, ctx, today)
        payload = await _plan_payload(s, ctx, recent.id if recent is not None else None)
    return web.json_response(payload)


async def plan_propose(request: web.Request, ctx: AppContext) -> web.Response:
    body = await H.body(request)
    op_key = _op_key(body)
    request_text = _text(body.get("request"), 1000)
    try:
        rev_id = await planning.propose(
            ctx,
            mode="replan" if body.get("replan") or request_text else "plan",
            user_request=request_text,
            change_reason=(f"просьба: {request_text[:200]}" if request_text else "из Mini App"),
            op_key=f"miniapp:{op_key}:plan",
        )
    except AIUnavailable:
        return H.error(
            409, "ИИ сейчас недоступен: пожелание текстом не учесть. Собери план без него."
        )
    async with ctx.db.transaction() as s:
        payload = await _plan_payload(s, ctx, rev_id)
    return web.json_response(payload)


async def plan_resolve(request: web.Request, ctx: AppContext) -> web.Response:
    revision_id = H.int_id(request, "revision_id")
    what = request.match_info["action"]
    async with ctx.db.transaction() as s:
        rev = await s.get(PlanRevision, revision_id)
        if rev is None:
            raise LookupError("revision")
        if what == "approve":
            res = await planning.approve_revision(
                s, ctx, plan_id=rev.plan_id, revision_id=revision_id
            )
            hooks = res.after_commit
            if res.status != actions.ExecStatus.APPLIED:
                return H.error(400, res.error or "План не утверждён.")
        elif what == "reject":
            await planning.reject_revision(s, ctx, revision_id=revision_id)
            hooks = []
        else:
            raise LookupError(what)
    await actions.run_after_commit(hooks)
    async with ctx.db.transaction() as s:
        payload = await _plan_payload(s, ctx, revision_id)
    return web.json_response(payload)


# --------------------------------------------------------------------------- goals


async def goal_create(request: web.Request, ctx: AppContext) -> web.Response:
    body = await H.body(request)
    title = _text(body.get("title"), 300)
    if not title:
        raise ValueError("Нужно название цели.")
    milestones = [m for m in (_text(x, 300) for x in body.get("milestones") or []) if m]
    action = CreateGoal(
        title=title,
        measurable_outcome=_text(body.get("measurable_outcome"), MAX_TEXT),
        horizon_date=H.parse_date(body.get("horizon_date"), "horizon_date"),
        milestones=milestones[:30],
    )
    async with ctx.db.transaction() as s:
        res = await actions.execute(
            s, ctx, action, origin=Actor.USER, op_key=f"miniapp:{_op_key(body)}"
        )
    await actions.run_after_commit(res.after_commit)
    return _exec_answer(res)


async def goal_edit(request: web.Request, ctx: AppContext) -> web.Response:
    goal_id = H.int_id(request, "goal_id")
    body = await H.body(request)
    changes: dict[str, Any] = {}
    if "title" in body:
        title = _text(body["title"], 300)
        if not title:
            raise ValueError("Название не может быть пустым.")
        changes["title"] = title
    for key in ("measurable_outcome", "success_criteria", "progress_note"):
        if key in body:
            changes[key] = _text(body[key], MAX_TEXT)
    if "horizon_date" in body:
        changes["horizon_date"] = H.parse_date(body["horizon_date"], "horizon_date")
    if "status" in body:
        if body["status"] not in GOAL_STATUSES:
            raise ValueError("Статус цели: active, paused, done, archived.")
        changes["status"] = body["status"]
    async with ctx.db.transaction() as s:
        await goals.update_goal(s, ctx, goal_id, actor=Actor.USER, **changes)
    return web.json_response({"id": goal_id, "status": "ok"})


async def milestone_add(request: web.Request, ctx: AppContext) -> web.Response:
    goal_id = H.int_id(request, "goal_id")
    body = await H.body(request)
    title = _text(body.get("title"), 300)
    if not title:
        raise ValueError("Нужно название этапа.")
    async with ctx.db.transaction() as s:
        existing = await goals.goal_milestones(s, goal_id)
        same = next((m for m in existing if m.title == title and m.status != "archived"), None)
        if same is None:  # a repeated press does not add a twin
            same = await goals.add_milestone(
                s,
                ctx,
                goal_id,
                title=title,
                actor=Actor.USER,
                is_key=bool(body.get("is_key")),
                due_date=H.parse_date(body.get("due_date"), "due_date"),
            )
        ms_id = same.id
    return web.json_response({"id": ms_id})


async def milestone_edit(request: web.Request, ctx: AppContext) -> web.Response:
    ms_id = H.int_id(request, "milestone_id")
    body = await H.body(request)
    changes: dict[str, Any] = {}
    if "title" in body:
        title = _text(body["title"], 300)
        if not title:
            raise ValueError("Название не может быть пустым.")
        changes["title"] = title
    if "due_date" in body:
        changes["due_date"] = H.parse_date(body["due_date"], "due_date")
    if "is_key" in body:
        changes["is_key"] = bool(body["is_key"])
    if "status" in body:
        if body["status"] not in MILESTONE_STATUSES:
            raise ValueError("Статус этапа: active, done, archived.")
        changes["status"] = body["status"]
    async with ctx.db.transaction() as s:
        if await s.get(Milestone, ms_id) is None:
            raise LookupError("milestone")
        await goals.update_milestone(s, ctx, ms_id, actor=Actor.USER, **changes)
    return web.json_response({"id": ms_id, "status": "ok"})


# --------------------------------------------------------------------------- habits


def _schedule(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"type": habits.DAILY}
    return {k: raw[k] for k in ("type", "days", "n") if k in raw}


def _target(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {k: raw[k] for k in ("amount", "unit", "min_amount") if raw.get(k) not in (None, "")}


async def habit_create(request: web.Request, ctx: AppContext) -> web.Response:
    body = await H.body(request)
    title = _text(body.get("title"), 200)
    if not title:
        raise ValueError("Нужно название привычки.")
    at = H.parse_hhmm(body.get("time"), "time")
    habits.parse_schedule(_schedule(body.get("schedule")))  # Russian error before saving
    habits.normalize_target(_target(body.get("target")))
    action = CreateHabit(
        title=title,
        schedule=_schedule(body.get("schedule")),
        target=_target(body.get("target")),
        preferred_time=at.strftime("%H:%M") if at is not None else None,
    )
    async with ctx.db.transaction() as s:
        res = await actions.execute(
            s, ctx, action, origin=Actor.USER, op_key=f"miniapp:{_op_key(body)}"
        )
    await actions.run_after_commit(res.after_commit)
    return _exec_answer(res)


async def habit_edit(request: web.Request, ctx: AppContext) -> web.Response:
    habit_id = H.int_id(request, "habit_id")
    body = await H.body(request)
    changes: dict[str, Any] = {}
    if "title" in body:
        title = _text(body["title"], 200)
        if not title:
            raise ValueError("Название не может быть пустым.")
        changes["title"] = title
    if "schedule" in body:
        changes["schedule"] = _schedule(body["schedule"])
    if "target" in body:
        changes["target"] = _target(body["target"])
    if "time" in body:
        changes["preferred_time"] = H.parse_hhmm(body["time"], "time")
    async with ctx.db.transaction() as s:
        await habits.require_habit(s, habit_id)
        if changes:
            await habits.update_habit(s, ctx, habit_id, actor=Actor.USER, **changes)
        if "status" in body:
            if body["status"] not in HABIT_STATUSES:
                raise ValueError("Статус привычки: active, paused, archived.")
            await habits.set_status(
                s, ctx, habit_id, EntityStatus(body["status"]), actor=Actor.USER
            )
    return web.json_response({"id": habit_id, "status": "ok"})


# --------------------------------------------------------------------------- memory


MEMORY_KIND_LABELS = {
    "fact": "факт",
    "preference": "предпочтение",
    "pattern": "закономерность",
    "hypothesis": "гипотеза",
}
MEMORY_ORIGIN_LABELS = {
    "user_said": "ты сказал",
    "system_measured": "измерено ботом",
    "ai_inferred": "предположение ИИ",
    "user_confirmed": "подтверждено тобой",
    "user_refuted": "опровергнуто",
}


async def memory_list(request: web.Request, ctx: AppContext) -> web.Response:
    async with ctx.db.transaction() as s:
        flags = await memory.memory_flags(s)
        items = await memory.list_items(
            s, statuses=(MemoryStatus.ACTIVE, MemoryStatus.REFUTED), limit=200
        )
        payload = {
            "flags": {
                "enabled": flags.enabled,
                "hypotheses": flags.hypotheses,
                "use_history": flags.use_history,
            },
            "items": [
                {
                    "id": m.id,
                    "kind": m.kind,
                    "kind_label": MEMORY_KIND_LABELS.get(m.kind, m.kind),
                    "origin": m.origin,
                    "origin_label": MEMORY_ORIGIN_LABELS.get(m.origin, m.origin),
                    "status": m.status,
                    "content": m.content,
                    "hint": m.verification_hint,
                }
                for m in items
            ],
        }
    return web.json_response(payload)


async def memory_change(request: web.Request, ctx: AppContext) -> web.Response:
    item_id = H.int_id(request, "item_id")
    what = request.match_info["action"]
    body = await H.body(request)
    async with ctx.db.transaction() as s:
        if what == "confirm":
            await memory.confirm(s, ctx, item_id)
        elif what == "refute":
            await memory.refute(s, ctx, item_id)
        elif what == "delete":
            await memory.delete_item(s, ctx, item_id)
        elif what == "correct":
            await memory.correct(s, ctx, item_id, str(body.get("content") or ""))
        else:
            raise LookupError(what)
    return web.json_response({"id": item_id, "status": "ok"})


async def memory_add(request: web.Request, ctx: AppContext) -> web.Response:
    body = await H.body(request)
    content = str(body.get("content") or "").strip()
    async with ctx.db.transaction() as s:
        item = await memory.add_fact(s, ctx, content, kind=str(body.get("kind") or "fact"))
        if item is None:
            return H.error(409, "Память выключена в настройках или запись уже есть.")
        item_id = item.id
    return web.json_response({"id": item_id})


# --------------------------------------------------------------------------- wiring


def add_manage_routes(app: web.Application, key: CtxKey) -> None:
    r = app.router
    r.add_get("/miniapp/api/tasks/{task_id}", H.route(key, task_detail))
    r.add_post("/miniapp/api/tasks/{task_id}/edit", H.route(key, task_edit))
    r.add_post("/miniapp/api/tasks/{task_id}/status", H.route(key, task_status))
    r.add_post("/miniapp/api/tasks/{task_id}/calendar", H.route(key, task_calendar))
    r.add_get("/miniapp/api/approvals", H.route(key, approval_list))
    r.add_post("/miniapp/api/approvals/{approval_id}/{action}", H.route(key, approval_resolve))
    r.add_post("/miniapp/api/capture", H.route(key, capture_text))
    r.add_get("/miniapp/api/plan", H.route(key, plan_get))
    r.add_post("/miniapp/api/plan", H.route(key, plan_propose))
    r.add_post("/miniapp/api/plan/{revision_id}/{action}", H.route(key, plan_resolve))
    r.add_post("/miniapp/api/goals", H.route(key, goal_create))
    r.add_post("/miniapp/api/goals/{goal_id}/edit", H.route(key, goal_edit))
    r.add_post("/miniapp/api/goals/{goal_id}/milestones", H.route(key, milestone_add))
    r.add_post("/miniapp/api/milestones/{milestone_id}/edit", H.route(key, milestone_edit))
    r.add_post("/miniapp/api/habits", H.route(key, habit_create))
    r.add_post("/miniapp/api/habits/{habit_id}/edit", H.route(key, habit_edit))
    r.add_get("/miniapp/api/memory", H.route(key, memory_list))
    r.add_post("/miniapp/api/memory", H.route(key, memory_add))
    r.add_post("/miniapp/api/memory/{item_id}/{action}", H.route(key, memory_change))
