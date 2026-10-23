"""Profile changes immediately re-plan initiative and task reminders."""

from __future__ import annotations

from datetime import time, timedelta

from sqlalchemy import select

from procbot.db.models import ReminderSchedule, Task
from procbot.domain.enums import ReminderState
from procbot.services.profile import update_profile
from procbot.services.profile_hooks import after_profile_change
from procbot.services.reminders import sync_task_reminders


async def _pending(ctx, prefix: str) -> list[ReminderSchedule]:
    async with ctx.db.transaction() as s:
        rows = (
            await s.execute(
                select(ReminderSchedule).where(
                    ReminderSchedule.dedup_key.like(f"{prefix}%"),
                    ReminderSchedule.state == ReminderState.PENDING,
                )
            )
        ).scalars()
        return list(rows.all())


async def test_setting_morning_time_seeds_reminder_at_once(ctx, profile) -> None:
    assert await _pending(ctx, "morning:") == []
    async with ctx.db.transaction() as s:
        await update_profile(s, now=ctx.clock.now(), morning_time=time(9, 30))
        await after_profile_change(s, ctx, {"morning_time"})
    rows = await _pending(ctx, "morning:")
    assert len(rows) == 1


async def test_zero_lead_means_off_even_if_app_yaml_has_default(ctx, profile) -> None:
    ctx.config.reminders.pre_task_lead_minutes = 10
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        t = Task(
            title="x", planned_start_at=now + timedelta(hours=2), created_at=now, updated_at=now
        )
        s.add(t)
        await s.flush()
        await sync_task_reminders(s, ctx, t)
    assert len(await _pending(ctx, "pre:")) == 1
    async with ctx.db.transaction() as s:
        await update_profile(s, now=now, pre_task_lead_min=0)
        await after_profile_change(s, ctx, {"pre_task_lead_min"})
    assert await _pending(ctx, "pre:") == []
