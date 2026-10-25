"""JSON payloads for the Mini App. Read-only: built inside the caller's
transaction from the same services the chat uses (FR-16, no second system)."""

from __future__ import annotations

from datetime import date, time, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.models import CalendarEventCache, OwnerProfile, Task, WorkSession
from procbot.domain.enums import SessionStatus, TaskStatus
from procbot.services import analytics, goals, habits, reports, sessions
from procbot.services import calendar as cal
from procbot.services import tasks as task_service
from procbot.services.dayview import DayView, build_day_view
from procbot.timeutil import fmt_local, local_date, local_day_bounds, to_local

# A finished session still waits for "what came out of it" this long.
OUTCOME_WINDOW = timedelta(hours=12)
WEEK_DAYS = 7
MAX_TREE_DEPTH = 4


def _hm(value: time | None) -> str | None:
    return value.strftime("%H:%M") if value is not None else None


def task_json(t: Task, tz: str, *, next_step: Task | None = None) -> dict[str, Any]:
    return {
        "id": t.id,
        "title": t.title,
        "status": str(t.status),
        "start": fmt_local(t.planned_start_at, tz, "%H:%M") if t.planned_start_at else None,
        "planned_date": t.planned_date.isoformat() if t.planned_date else None,
        "due": (
            fmt_local(t.due_at, tz)
            if t.due_at
            else (t.due_date.strftime("%d.%m") if t.due_date else None)
        ),
        "due_is_hard": t.due_is_hard,
        "estimate_min": t.estimate_minutes,
        "needs_criteria": task_service.requires_criteria_check(t),
        "done_criteria": t.done_criteria,
        "progress_note": t.progress_note,
        "next_step": next_step.title if next_step is not None else None,
    }


# --------------------------------------------------------------------------- day


def _day_sections(view: DayView) -> list[dict[str, Any]]:
    b = view.buckets
    return [
        {
            "key": key,
            "title": title,
            "tasks": [task_json(t, view.tz, next_step=view.next_steps.get(t.id)) for t in items],
        }
        for key, title, items in (
            ("overdue", "Просрочено", b.overdue),
            ("missed", "Не сделано в прошлые дни", b.missed),
            ("planned", "Запланировано на сегодня", b.planned),
            ("due_today", "Срок сегодня", b.due_today),
        )
        if items
    ]


def habit_json(h: habits.DueHabit) -> dict[str, Any]:
    return {
        "id": h.habit_id,
        "title": h.title,
        "time": _hm(h.preferred_time),
        "target": h.target_text,
        "status": h.status,
        "flexible": h.flexible,
        "week_kept": h.week_kept,
        "week_target": h.week_target,
    }


async def day_payload(s: AsyncSession, ctx: AppContext) -> dict[str, Any]:
    view = await build_day_view(s, ctx)
    return {
        "day": view.day.isoformat(),
        "tz": view.tz,
        "sections": _day_sections(view),
        "backlog_count": view.buckets.backlog_count,
        "habits": [habit_json(h) for h in await habits.due_habits(s, ctx, view.day)],
        "session": await session_payload(s, ctx),
    }


# --------------------------------------------------------------------------- tasks


async def _tree(s: AsyncSession, t: Task, tz: str, depth: int) -> dict[str, Any]:
    node = task_json(t, tz)
    children = await task_service.list_subtasks(s, t.id) if depth < MAX_TREE_DEPTH else []
    node["children"] = [await _tree(s, c, tz, depth + 1) for c in children]
    return node


async def tasks_payload(s: AsyncSession, ctx: AppContext, view: str) -> dict[str, Any]:
    tz = await task_service.owner_tz(s)
    rows = await task_service.list_tasks(s, ctx, task_service.TaskView(view), limit=100)
    return {"view": view, "tasks": [await _tree(s, t, tz, 0) for t in rows]}


# --------------------------------------------------------------------------- week


def _plan_day(t: Task, tz: str) -> date | None:
    if t.planned_start_at is not None:
        return local_date(t.planned_start_at, tz)
    return t.planned_date


async def week_payload(s: AsyncSession, ctx: AppContext, start: date | None) -> dict[str, Any]:
    tz = await task_service.owner_tz(s)
    today = local_date(ctx.clock.now(), tz)
    first = start or today - timedelta(days=today.weekday())
    last = first + timedelta(days=WEEK_DAYS - 1)
    begin, end = local_day_bounds(first, tz)[0], local_day_bounds(last, tz)[1]
    busy, stale = await cal.busy_intervals(s, ctx, begin, end)

    events = (
        await s.execute(
            select(CalendarEventCache)
            .where(
                CalendarEventCache.busy.is_(True),
                CalendarEventCache.end_at > begin,
                CalendarEventCache.start_at < end,
            )
            .order_by(CalendarEventCache.start_at)
        )
    ).scalars()
    rows = (
        await s.execute(
            select(Task).where(
                Task.parent_id.is_(None),
                Task.status.in_([TaskStatus.TODO, TaskStatus.IN_PROGRESS, TaskStatus.DONE]),
            )
        )
    ).scalars()
    days: dict[date, dict[str, Any]] = {
        first + timedelta(days=i): {
            "date": (first + timedelta(days=i)).isoformat(),
            "is_today": first + timedelta(days=i) == today,
            "tasks": [],
            "events": [],
            "habits": 0,
        }
        for i in range(WEEK_DAYS)
    }
    for t in rows:
        pd = _plan_day(t, tz)
        if pd is not None and pd in days:
            days[pd]["tasks"].append(task_json(t, tz))
    for entry in days.values():
        entry["tasks"].sort(key=lambda x: (x["start"] is None, x["start"] or "", x["id"]))
    for ev in events:
        ed = local_date(max(ev.start_at, begin), tz)
        if ed in days:
            days[ed]["events"].append(
                {
                    "start": None if ev.all_day else to_local(ev.start_at, tz).strftime("%H:%M"),
                    "end": None if ev.all_day else to_local(ev.end_at, tz).strftime("%H:%M"),
                    "title": ev.summary or "Занято",
                    "all_day": ev.all_day,
                }
            )
    for day in days:
        days[day]["habits"] = len(await habits.due_habits(s, ctx, day))
    status = await cal.connection_status(s, ctx)
    return {
        "start": first.isoformat(),
        "days": list(days.values()),
        "calendar": {
            "connected": status.connected,
            "stale": stale if status.connected else False,
            "busy_count": len(busy),
        },
    }


# --------------------------------------------------------------------------- goals


async def goals_payload(s: AsyncSession, ctx: AppContext) -> dict[str, Any]:
    tz = await task_service.owner_tz(s)
    out = []
    for g in await goals.list_goals(s):
        prog = await goals.goal_progress(s, ctx, g.id)
        out.append(
            {
                **prog.as_dict(),
                "measurable_outcome": g.measurable_outcome,
                "success_criteria": g.success_criteria,
                "milestones": [
                    {
                        "id": m.id,
                        "title": m.title,
                        "status": m.status,
                        "is_key": m.is_key,
                        "due": m.due_date.isoformat() if m.due_date else None,
                    }
                    for m in await goals.goal_milestones(s, g.id)
                    if m.status != "archived"
                ],
                "open_tasks": [
                    task_json(t, tz) for t in (await goals.open_linked_tasks(s, g.id))[:20]
                ],
            }
        )
    return {"goals": out}


# --------------------------------------------------------------------------- habits


async def habits_payload(s: AsyncSession, ctx: AppContext) -> dict[str, Any]:
    today = await habits.local_today(s, ctx)
    due = {h.habit_id: h for h in await habits.due_habits(s, ctx, today)}
    out = []
    for h in await habits.list_habits(s):
        st = await habits.habit_stability(s, ctx, h.id)
        item = {
            "id": h.id,
            "title": h.title,
            "status": h.status,
            "schedule": habits.schedule_text(habits.habit_schedule(h)),
            "target": habits.target_text(h.target),
            "time": _hm(h.preferred_time),
            "stability": habits.stability_text(st),
            "due_today": h.id in due,
            "today_status": due[h.id].status if h.id in due else None,
        }
        out.append(item)
    return {"day": today.isoformat(), "habits": out}


# --------------------------------------------------------------------------- work session


async def session_payload(s: AsyncSession, ctx: AppContext) -> dict[str, Any] | None:
    now = ctx.clock.now()
    ws = await sessions.get_active_session(s)
    if ws is None:
        ws = (
            (
                await s.execute(
                    select(WorkSession)
                    .where(
                        WorkSession.status == SessionStatus.FINISHED,
                        WorkSession.outcome.is_(None),
                        WorkSession.ended_at >= now - OUTCOME_WINDOW,
                    )
                    .order_by(WorkSession.ended_at.desc())
                    .limit(1)
                )
            )
            .scalars()
            .first()
        )
        if ws is None:
            return None
    pauses = await sessions.load_pauses(s, ws.id)
    task = await s.get(Task, ws.task_id) if ws.task_id is not None else None
    return {
        "id": ws.id,
        "status": ws.status,
        "mode": ws.mode,
        "phase": sessions.current_phase(ws, pauses, now),
        "task_id": ws.task_id,
        "task_title": task.title if task is not None else None,
        "needs_criteria": task is not None and task_service.requires_criteria_check(task),
        "done_criteria": task.done_criteria if task is not None else None,
        "net_seconds": int(sessions.net_seconds(ws, pauses, now)),
        "planned_minutes": ws.planned_minutes or ws.work_minutes,
        "needs_outcome": ws.status == SessionStatus.FINISHED and ws.outcome is None,
    }


# --------------------------------------------------------------------------- report


async def report_payload(s: AsyncSession, ctx: AppContext, days: int) -> dict[str, Any]:
    tz = await analytics.owner_tz(s)
    today = local_date(ctx.clock.now(), tz)
    period = analytics.Period.of(today - timedelta(days=days - 1), today, tz)
    a = await analytics.compute(s, ctx, period)
    return {
        "period": period.label,
        "directions": [
            {"key": d.key, "title": d.title, "lines": reports.render_direction(d)[1:]}
            for d in a.ordered()
        ],
    }


# --------------------------------------------------------------------------- settings


SETTINGS_MINUTES = (
    "pre_task_lead_min",
    "start_delay_min",
    "pomodoro_work_min",
    "pomodoro_break_min",
    "short_start_min",
)
SETTINGS_TIMES = ("morning_time", "evening_time", "weekly_review_time")


def settings_json(p: OwnerProfile | None, ctx: AppContext) -> dict[str, Any]:
    if p is None:
        return {"ready": False}
    out: dict[str, Any] = {
        "ready": True,
        "timezone": p.timezone,
        "notifications_enabled": p.notifications_enabled,
        "weekly_review_dow": p.weekly_review_dow,
        "quiet_hours": [
            {"start": w.get("start"), "end": w.get("end"), "days": w.get("days")}
            for w in (p.quiet_hours or [])
            if isinstance(w, dict)
        ],
        "pause_until": (fmt_local(p.pause_until, p.timezone) if p.pause_until else None),
    }
    for key in SETTINGS_TIMES:
        out[key] = _hm(getattr(p, key))
    for key in SETTINGS_MINUTES:
        out[key] = getattr(p, key)
    return out
