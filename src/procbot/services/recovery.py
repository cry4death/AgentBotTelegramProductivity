"""Startup recovery (scenario G, section 7 "Восстановление"). main() runs it after
wiring and before polling / the scheduler loop.

Steps:
1. release expired leases (a crashed process held them);
2. outbound rows stuck in ``sending``: older than the threshold -> ``maybe_sent``
   (never re-sent blindly; they keep counting toward the 4/60 window); younger
   rows may still be in flight in another instance and are left for the next
   sweep (the scheduler sweeps every tick);
3. reminders left in ``processing`` -> back to ``pending``;
4. re-seed the initiative reminders (morning / evening / weekly);
5. missed reminders (past due) go through the normal path: re-evaluated by the
   gate, stale ones folded into ONE digest sent at the next allowed moment - no
   avalanche of old messages.

Idempotent: a second run changes nothing more (everything is keyed and
state-checked).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy import select

from procbot.context import AppContext
from procbot.db.models import OutboundMessage, ReminderSchedule
from procbot.domain.enums import ReminderState, SendState
from procbot.services import gate, leases
from procbot.services.initiative import seed_initiative_reminders
from procbot.services.scheduler import (
    TickReport,
    process_due_reminders,
    stuck_sending_threshold,
)

log = logging.getLogger(__name__)


@dataclass
class RecoveryReport:
    leases_released: int = 0
    marked_maybe_sent: int = 0
    in_flight: int = 0  # recent `sending` rows left alone
    reminders_reset: int = 0
    initiative_planned: int = 0
    reminders: TickReport = field(default_factory=TickReport)


async def run_recovery(ctx: AppContext, *, process_missed: bool = True) -> RecoveryReport:
    report = RecoveryReport()
    now = ctx.clock.now()
    report.leases_released = await leases.expire_stale(ctx)
    report.marked_maybe_sent = await gate.sweep_stuck_sending(
        ctx, older_than=stuck_sending_threshold(ctx)
    )
    async with ctx.db.transaction() as s:
        report.in_flight = len(
            (
                await s.execute(
                    select(OutboundMessage.id).where(
                        OutboundMessage.send_state == SendState.SENDING
                    )
                )
            ).all()
        )
        stuck = (
            (
                await s.execute(
                    select(ReminderSchedule).where(
                        ReminderSchedule.state == ReminderState.PROCESSING
                    )
                )
            )
            .scalars()
            .all()
        )
        for r in stuck:
            r.state = ReminderState.PENDING
            r.last_reason = "recovered after restart"
            r.updated_at = now
        report.reminders_reset = len(stuck)
    report.initiative_planned = await seed_initiative_reminders(ctx)
    if process_missed:
        report.reminders = await process_due_reminders(ctx)
    log.info(
        "recovery done",
        extra={
            "leases_released": report.leases_released,
            "maybe_sent": report.marked_maybe_sent,
            "in_flight": report.in_flight,
            "reminders": report.reminders.results,
        },
    )
    return report
