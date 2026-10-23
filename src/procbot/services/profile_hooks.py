"""Side effects of profile changes that affect planned reminders.

Called inside the same transaction as the profile update, so reminders always
match the saved settings (no 5-minute wait for the periodic re-seed).
"""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.models import Task
from procbot.domain.enums import TaskStatus

INITIATIVE_FIELDS = frozenset(
    {"timezone", "morning_time", "evening_time", "weekly_review_dow", "weekly_review_time"}
)
TASK_REMINDER_FIELDS = frozenset({"timezone", "pre_task_lead_min", "start_delay_min"})


async def after_profile_change(
    session: AsyncSession, ctx: AppContext, changed: Iterable[str]
) -> None:
    keys = set(changed)
    if keys & INITIATIVE_FIELDS:
        from procbot.services.initiative import seed_initiative_in

        await seed_initiative_in(session, ctx)
    if keys & TASK_REMINDER_FIELDS:
        from procbot.services.reminders import sync_task_reminders

        now = ctx.clock.now()
        tasks = (
            await session.execute(
                select(Task).where(
                    Task.planned_start_at > now,
                    Task.status.in_([TaskStatus.TODO, TaskStatus.IN_PROGRESS]),
                )
            )
        ).scalars()
        for task in tasks.all():
            await sync_task_reminders(session, ctx, task)
