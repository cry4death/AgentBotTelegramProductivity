"""Stage-A intervention records (FR-04): what help was offered after a start delay
and what came of it. Stage B (procrastination analysis) extends this module.

Reasons are hypotheses the user picked (`reason_origin = user_said`), never
diagnoses.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.clock import SystemClock
from procbot.db.models import Intervention, InterventionOutcome
from procbot.domain.enums import (
    InterventionKind,
    InterventionResult,
    MemoryOrigin,
    ObstacleReason,
    TimeSource,
)


async def record_quick_intervention(
    session: AsyncSession,
    *,
    now: datetime,
    task_id: int | None,
    reminder_id: int | None,
    kind: InterventionKind,
    reason: ObstacleReason | None,
    suggestion: str | None,
) -> tuple[Intervention, bool]:
    """Record a quick (button) intervention. A repeated press for the same reminder
    and the same choice returns the existing row: (row, created)."""
    stmt = select(Intervention).where(
        Intervention.kind == str(kind),
        Intervention.depth == "quick",
    )
    stmt = stmt.where(
        Intervention.reminder_id == reminder_id
        if reminder_id is not None
        else Intervention.reminder_id.is_(None)
    )
    stmt = stmt.where(
        Intervention.task_id == task_id if task_id is not None else Intervention.task_id.is_(None)
    )
    stmt = stmt.where(
        Intervention.reason_category == str(reason)
        if reason is not None
        else Intervention.reason_category.is_(None)
    )
    if reminder_id is None:
        # without a reminder, only a press within the last minutes is a "repeat"
        stmt = stmt.where(Intervention.created_at >= now - timedelta(minutes=5))
    existing = (await session.execute(stmt.limit(1))).scalar_one_or_none()
    if existing is not None:
        return existing, False
    row = Intervention(
        task_id=task_id,
        reminder_id=reminder_id,
        kind=str(kind),
        reason_category=None if reason is None else str(reason),
        reason_origin=None if reason is None else str(MemoryOrigin.USER_SAID),
        depth="quick",
        suggestion=suggestion,
        used_ai=False,
        created_at=now,
    )
    session.add(row)
    await session.flush()
    return row, True


async def find_recent_intervention(
    session: AsyncSession, *, task_id: int, since: datetime
) -> Intervention | None:
    """Latest intervention for `task_id` created at or after `since` that has no
    outcome yet - for linking a work session the user starts afterwards."""
    rows = (
        (
            await session.execute(
                select(Intervention)
                .where(Intervention.task_id == task_id, Intervention.created_at >= since)
                .order_by(Intervention.created_at.desc(), Intervention.id.desc())
            )
        )
        .scalars()
        .all()
    )
    for iv in rows:
        has_outcome = (
            await session.execute(
                select(InterventionOutcome.id)
                .where(InterventionOutcome.intervention_id == iv.id)
                .limit(1)
            )
        ).first()
        if has_outcome is None:
            return iv
    return None


async def record_intervention_outcome(
    session: AsyncSession,
    *,
    intervention_id: int,
    result: InterventionResult | str,
    session_id: int | None = None,
    now: datetime | None = None,
    note: str | None = None,
    time_source: TimeSource | str = TimeSource.EXACT,
) -> InterventionOutcome:
    """Record what came of an intervention (e.g. the user started a short session).

    Idempotent per (intervention, result, session): a repeat returns the existing row.
    Pass `now=ctx.clock.now()`; the system clock is only a fallback.
    """
    res = str(InterventionResult(result))
    stmt = select(InterventionOutcome).where(
        InterventionOutcome.intervention_id == intervention_id,
        InterventionOutcome.result == res,
    )
    stmt = stmt.where(
        InterventionOutcome.session_id == session_id
        if session_id is not None
        else InterventionOutcome.session_id.is_(None)
    )
    existing = (await session.execute(stmt.limit(1))).scalar_one_or_none()
    if existing is not None:
        return existing
    row = InterventionOutcome(
        intervention_id=intervention_id,
        result=res,
        session_id=session_id,
        time_source=str(time_source),
        note=note,
        recorded_at=now or SystemClock().now(),
    )
    session.add(row)
    await session.flush()
    return row
