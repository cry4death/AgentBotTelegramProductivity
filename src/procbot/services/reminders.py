"""ReminderService: plans reminders in reminder_schedule (the source of truth).

CONTRACT STUB — implemented in stage A by the reminders module. Keep signatures.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.models import ReminderSchedule, Task
from procbot.domain.enums import MessageCategory


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
    """Idempotent on dedup_key: an existing pending row is updated, not duplicated."""
    raise NotImplementedError


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
    raise NotImplementedError


async def sync_task_reminders(session: AsyncSession, ctx: AppContext, task: Task) -> None:
    """(Re)plan pre_task / start_delay reminders after a task's planned start,
    status or reminder settings changed. Cancels them for done/cancelled tasks."""
    raise NotImplementedError


def register(ctx: AppContext) -> None:
    """Register default renderers and periodic jobs."""
