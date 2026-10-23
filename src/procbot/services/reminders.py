"""ReminderService: plans reminders in reminder_schedule (the source of truth).

What may create a reminder (acceptance 13 - silence alone never does):
- a task's planned start (PRE_TASK before it, START_DELAY after it) - here;
- the owner's morning / evening / weekly times - services.initiative;
- timer signals of a running work session - services.sessions;
- a user's "later" button (snooze) - here;
- folded stale reminders (one DIGEST) - services.scheduler.

Reminder timing (open question 3): lead / delay come from
``task.reminder_overrides`` -> ``profile.pre_task_lead_min`` / ``start_delay_min``
-> ``config.reminders.*``. If none is set the value is None and that reminder type
is OFF - no interval is invented.

Delivery (gate checks, re-evaluation, folding) lives in services.scheduler.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.models import OwnerProfile, ReminderSchedule, Task, WorkSession
from procbot.domain.enums import MessageCategory, ReminderState, SessionStatus, TaskStatus
from procbot.services.outbound import Outgoing, register_renderer
from procbot.services.profile import get_profile
from procbot.telegram.callbacks import DayCb, SessionCb
from procbot.timeutil import fmt_local

LIVE_STATES = (ReminderState.PENDING, ReminderState.RESCHEDULED)
TASK_REMINDER_CATEGORIES: frozenset[MessageCategory] = frozenset(
    {MessageCategory.PRE_TASK, MessageCategory.START_DELAY}
)
ACTIVE_TASK_STATUSES = (TaskStatus.TODO, TaskStatus.IN_PROGRESS)


async def schedule_reminder(
    session: AsyncSession,
    *,
    now: datetime,
    category: MessageCategory,
    due_at: datetime,
    dedup_key: str,
    task_id: int | None = None,
    session_id: int | None = None,
    plan_id: int | None = None,
    payload: dict[str, Any] | None = None,
    stale_at: datetime | None = None,
) -> ReminderSchedule:
    """Idempotent on dedup_key: an existing pending row is updated, not duplicated.

    A row that already reached a final state (done/folded/dropped/cancelled) is
    left untouched and returned as is, so a repeated call never re-sends.
    """
    existing = (
        await session.execute(
            select(ReminderSchedule).where(ReminderSchedule.dedup_key == dedup_key)
        )
    ).scalar_one_or_none()
    if existing is not None:
        if existing.state in (ReminderState.PENDING, ReminderState.RESCHEDULED):
            existing.due_at = due_at
            existing.stale_at = stale_at
            existing.payload = payload or {}
            existing.state = ReminderState.PENDING
            existing.updated_at = now
        return existing
    row = ReminderSchedule(
        category=str(category),
        due_at=due_at,
        stale_at=stale_at,
        dedup_key=dedup_key,
        state=ReminderState.PENDING,
        task_id=task_id,
        session_id=session_id,
        plan_id=plan_id,
        payload=payload or {},
        created_at=now,
        updated_at=now,
    )
    session.add(row)
    await session.flush()
    return row


async def cancel_reminders(
    session: AsyncSession,
    *,
    now: datetime,
    reason: str,
    task_id: int | None = None,
    session_id: int | None = None,
    plan_id: int | None = None,
    categories: set[MessageCategory] | None = None,
) -> int:
    """Cancel pending reminders matching the filters; returns how many."""
    stmt = select(ReminderSchedule).where(
        ReminderSchedule.state.in_([ReminderState.PENDING, ReminderState.RESCHEDULED])
    )
    if task_id is not None:
        stmt = stmt.where(ReminderSchedule.task_id == task_id)
    if session_id is not None:
        stmt = stmt.where(ReminderSchedule.session_id == session_id)
    if plan_id is not None:
        stmt = stmt.where(ReminderSchedule.plan_id == plan_id)
    if categories:
        stmt = stmt.where(ReminderSchedule.category.in_([str(c) for c in categories]))
    rows = (await session.execute(stmt)).scalars().all()
    for r in rows:
        r.state = ReminderState.CANCELLED
        r.last_reason = reason
        r.updated_at = now
    return len(rows)


# --------------------------------------------------------------------------- task reminders


def _override(task: Task, key: str) -> int | None:
    raw = (task.reminder_overrides or {}).get(key)
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def resolve_pre_task_lead(ctx: AppContext, task: Task, profile: OwnerProfile | None) -> int | None:
    """Minutes before the planned start; None = no pre-task warning (open question 3)."""
    for value in (
        _override(task, "pre_task_lead_min"),
        profile.pre_task_lead_min if profile else None,
        ctx.config.reminders.pre_task_lead_minutes,
    ):
        if value is not None:
            return value
    return None


def resolve_start_delay(ctx: AppContext, task: Task, profile: OwnerProfile | None) -> int | None:
    """Minutes after the planned start; None = no start-delay check (open question 3)."""
    for value in (
        _override(task, "start_delay_min"),
        profile.start_delay_min if profile else None,
        ctx.config.reminders.start_delay_after_minutes,
    ):
        if value is not None:
            return value
    return None


def pre_task_key(task_id: int, planned_start: datetime) -> str:
    return f"pre:{task_id}:{planned_start.isoformat()}"


def start_delay_key(task_id: int, planned_start: datetime) -> str:
    return f"delay:{task_id}:{planned_start.isoformat()}"


async def upsert_reminder(
    session: AsyncSession, *, now: datetime, dedup_key: str, **kwargs: Any
) -> ReminderSchedule:
    """schedule_reminder, but a CANCELLED row with the same key is revived (it was
    never sent, so this cannot cause a re-send)."""
    existing = (
        await session.execute(
            select(ReminderSchedule).where(ReminderSchedule.dedup_key == dedup_key)
        )
    ).scalar_one_or_none()
    if existing is not None and existing.state == ReminderState.CANCELLED:
        existing.state = ReminderState.PENDING
        existing.reschedule_count = 0
        existing.attempts = 0
        existing.last_reason = None
    return await schedule_reminder(session, now=now, dedup_key=dedup_key, **kwargs)


async def sync_task_reminders(session: AsyncSession, ctx: AppContext, task: Task) -> None:
    """(Re)plan PRE_TASK / START_DELAY reminders after a task's planned start,
    status or reminder settings changed. Call it inside the transaction that
    changed the task. Cancels them for done/cancelled/someday/backlog tasks.

    - PRE_TASK at planned_start - lead, if that moment is still ahead;
    - START_DELAY at planned_start + delay, if that moment is still ahead;
    - lead/delay None (not configured anywhere) -> that type is not scheduled;
    - dedup keys contain the planned start, so re-planning cancels the old pair
      and creates a fresh one (a sent reminder is never re-sent).
    """
    now = ctx.clock.now()
    cats = set(TASK_REMINDER_CATEGORIES)
    if (
        task.id is None
        or task.status not in ACTIVE_TASK_STATUSES
        or task.is_backlog
        or task.planned_start_at is None
    ):
        if task.id is not None:
            await cancel_reminders(
                session, now=now, reason="task_not_planned", task_id=task.id, categories=cats
            )
        return

    start = task.planned_start_at
    profile = await get_profile(session)
    lead = resolve_pre_task_lead(ctx, task, profile)
    delay = resolve_start_delay(ctx, task, profile)
    # Keys valid for the current plan. A live row with such a key whose moment has
    # already passed (e.g. deferred by quiet hours) is kept as is.
    current: set[str] = set()
    if lead is not None:
        current.add(pre_task_key(task.id, start))
    if delay is not None:
        current.add(start_delay_key(task.id, start))
    wanted: dict[str, tuple[MessageCategory, datetime, int]] = {}
    if lead is not None and start - timedelta(minutes=lead) > now:
        wanted[pre_task_key(task.id, start)] = (
            MessageCategory.PRE_TASK,
            start - timedelta(minutes=lead),
            lead,
        )
    if delay is not None and start + timedelta(minutes=delay) > now:
        wanted[start_delay_key(task.id, start)] = (
            MessageCategory.START_DELAY,
            start + timedelta(minutes=delay),
            delay,
        )

    live = (
        (
            await session.execute(
                select(ReminderSchedule).where(
                    ReminderSchedule.task_id == task.id,
                    ReminderSchedule.category.in_([str(c) for c in cats]),
                    ReminderSchedule.state.in_([str(s) for s in LIVE_STATES]),
                    # snoozed copies follow the user's own "later", keep them
                    ~ReminderSchedule.dedup_key.like("snz:%"),
                )
            )
        )
        .scalars()
        .all()
    )
    for r in live:
        if r.dedup_key not in current:
            r.state = ReminderState.CANCELLED
            r.last_reason = "replanned"
            r.updated_at = now

    for key, (category, due, minutes) in wanted.items():
        await upsert_reminder(
            session,
            now=now,
            dedup_key=key,
            category=category,
            due_at=due,
            task_id=task.id,
            payload={
                "task_id": task.id,
                "planned_start_at": start.isoformat(),
                "minutes": minutes,
            },
        )


async def task_reminder_relevant(
    session: AsyncSession, task: Task | None, planned_start: datetime | None, now: datetime
) -> bool:
    """A task-bound reminder still makes sense: the task is open, still planned for
    the same start, and no confirmed work session overlaps [planned_start, now]."""
    if task is None or task.status not in ACTIVE_TASK_STATUSES or task.is_backlog:
        return False
    if planned_start is not None and task.planned_start_at != planned_start:
        return False
    since = planned_start or now
    started = (
        await session.execute(
            select(WorkSession.id)
            .where(
                WorkSession.task_id == task.id,
                WorkSession.status.in_(
                    [SessionStatus.ACTIVE, SessionStatus.PAUSED, SessionStatus.FINISHED]
                ),
                WorkSession.started_at.is_not(None),
                WorkSession.started_at <= now,
                or_(WorkSession.ended_at.is_(None), WorkSession.ended_at >= since),
            )
            .limit(1)
        )
    ).first()
    return started is None


def payload_dt(payload: dict[str, Any], key: str) -> datetime | None:
    raw = payload.get(key)
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except ValueError:
        return None


# --------------------------------------------------------------------------- snooze


def snooze_key(reminder_id: int) -> str:
    return f"snz:{reminder_id}"


async def snooze_reminder(
    session: AsyncSession, ctx: AppContext, *, reminder_id: int, minutes: int
) -> tuple[ReminderSchedule | None, bool]:
    """User pressed "later": plan a NEW reminder of the same kind `minutes` from now.

    It goes through the gate like any other reminder and counts toward the limit.
    One snooze per original reminder (a repeated press is a no-op): returns
    (row, created).
    """
    now = ctx.clock.now()
    original = await session.get(ReminderSchedule, reminder_id)
    if original is None:
        return None, False
    key = snooze_key(reminder_id)
    existing = (
        await session.execute(select(ReminderSchedule).where(ReminderSchedule.dedup_key == key))
    ).scalar_one_or_none()
    if existing is not None:
        return existing, False
    payload = dict(original.payload or {})
    payload["snoozed_from"] = reminder_id
    row = await schedule_reminder(
        session,
        now=now,
        category=MessageCategory(original.category),
        due_at=now + timedelta(minutes=minutes),
        dedup_key=key,
        task_id=original.task_id,
        session_id=original.session_id,
        plan_id=original.plan_id,
        payload=payload,
    )
    return row, True


# --------------------------------------------------------------------------- digest


_CATEGORY_LINE = {
    MessageCategory.MORNING_PLAN: "утренний план",
    MessageCategory.EVENING_REVIEW: "вечерний разбор",
    MessageCategory.WEEKLY_REVIEW: "недельный обзор",
    MessageCategory.REPLAN: "предложение пересмотреть план",
    MessageCategory.TIMER_SIGNAL: "сигнал таймера рабочей сессии",
    MessageCategory.SYSTEM_NOTICE: "служебное уведомление",
}


async def render_digest(ctx: AppContext, reminder: ReminderSchedule) -> Outgoing | None:
    """One short summary of reminders that were folded (quiet hours, pause, downtime).
    Items that became irrelevant in the meantime are left out; nothing left -> None."""
    ids = [int(x) for x in (reminder.payload or {}).get("items", [])]
    if not ids:
        return None
    now = ctx.clock.now()
    lines: list[str] = []
    task_buttons: list[tuple[int, str]] = []
    async with ctx.db.transaction() as s:
        profile = await get_profile(s)
        tz = profile.timezone if profile else None
        rows = (
            (
                await s.execute(
                    select(ReminderSchedule)
                    .where(ReminderSchedule.id.in_(ids))
                    .order_by(ReminderSchedule.due_at)
                )
            )
            .scalars()
            .all()
        )
        seen_tasks: set[int] = set()
        seen_kinds: set[str] = set()
        for r in rows:
            cat = MessageCategory(r.category)
            if r.task_id is not None and cat in (
                MessageCategory.PRE_TASK,
                MessageCategory.START_DELAY,
            ):
                if r.task_id in seen_tasks:
                    continue
                task = await s.get(Task, r.task_id)
                planned = payload_dt(r.payload or {}, "planned_start_at")
                if not await task_reminder_relevant(s, task, planned, now):
                    continue
                assert task is not None
                seen_tasks.add(task.id)
                when = fmt_local(planned, tz, "%H:%M") if planned else ""
                lines.append(
                    f"• «{task.title}» — начало по плану {when}, отметки о старте нет"
                    if when
                    else f"• «{task.title}»"
                )
                task_buttons.append((task.id, task.title))
                continue
            label = _CATEGORY_LINE.get(cat, str(cat))
            key = f"{cat}:{fmt_local(r.due_at, tz, '%d.%m')}"
            if key in seen_kinds:
                continue
            seen_kinds.add(key)
            lines.append(f"• {label} ({fmt_local(r.due_at, tz, '%d.%m %H:%M')})")
    if not lines:
        return None
    text = (
        "Пока сообщения были отложены, накопилось несколько напоминаний. "
        "Собрал их в одну сводку:\n"
        + "\n".join(lines)
        + "\n\nЕсли что-то ещё актуально — можно начать сейчас или пересмотреть день."
    )
    kb: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text=f"▶️ {title[:28]}",
                callback_data=SessionCb(action="pick", task_id=task_id).pack(),
            )
        ]
        for task_id, title in task_buttons[:3]
    ]
    kb.append([InlineKeyboardButton(text="📅 Мой день", callback_data=DayCb(action="show").pack())])
    return Outgoing(text=text, reply_markup=InlineKeyboardMarkup(inline_keyboard=kb))


async def render_payload_text(ctx: AppContext, reminder: ReminderSchedule) -> Outgoing | None:
    """Generic renderer: the text was prepared when the reminder was planned
    (system notices; the REPLAN hook until stage B registers its own renderer)."""
    text = (reminder.payload or {}).get("text")
    return Outgoing(text=str(text)) if text else None


def register(ctx: AppContext) -> None:
    """Register the digest renderer and generic payload-text renderers."""
    register_renderer(ctx, MessageCategory.DIGEST, render_digest)
    register_renderer(ctx, MessageCategory.SYSTEM_NOTICE, render_payload_text)
