"""TaskService.

CONTRACT STUB — implemented in stage A by the tasks module. Keep signatures.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.models import Task
from procbot.domain.enums import Actor, SessionOutcome


async def get_task(session: AsyncSession, task_id: int) -> Task | None:
    return await session.get(Task, task_id)


async def record_work_outcome(
    session: AsyncSession,
    ctx: AppContext,
    *,
    task_id: int,
    outcome: SessionOutcome,
    actor: Actor,
    session_id: int | None = None,
    note: str | None = None,
) -> Task:
    """Apply a work result to a task: STARTED/PROGRESSED -> in_progress (never done),
    COMPLETED -> done. Writes task_events and re-syncs reminders."""
    raise NotImplementedError


def register(ctx: AppContext) -> None:
    """Register action appliers for task actions."""
