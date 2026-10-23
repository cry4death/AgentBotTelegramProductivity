"""ReminderService: plans reminders in reminder_schedule (the source of truth).

schedule_reminder / cancel_reminders are implemented (foundation, keep semantics);
sync_task_reminders and register() are owned by the reminders module (stage A).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.models import ReminderSchedule, Task
from procbot.domain.enums import MessageCategory, ReminderState


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


async def sync_task_reminders(session: AsyncSession, ctx: AppContext, task: Task) -> None:
    """(Re)plan pre_task / start_delay reminders after a task's planned start,
    status or reminder settings changed. Cancels them for done/cancelled tasks.

    STUB (no-op) until the reminders module implements it."""
    return None


def register(ctx: AppContext) -> None:
    """Register default renderers and periodic jobs."""
