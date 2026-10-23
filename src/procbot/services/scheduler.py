"""Scheduler loop: processes due reminders through the Outbound Gate and runs the
registered periodic jobs (services.jobs).

Per due reminder (architecture, "Фоновый планировщик и привратник"):

1. take the lease ``reminder:<id>`` (two schedulers never process one reminder);
2. re-read it; skip if it is no longer pending/due;
3. stale (``stale_at`` passed, or ``gate.default_stale_after_minutes`` after its
   original moment) -> FOLDED into ONE pending digest (no avalanche);
4. ``gate`` pre-check; blocked -> re-evaluate: reschedule to the earliest allowed
   moment (end of quiet hours / pause / busy block / session, or when the sliding
   window frees a slot), or fold if that moment is after it goes stale, or drop if
   it can never pass (category / notifications disabled);
5. render OUTSIDE transactions (renderers may call the AI later). None -> DROPPED;
6. ``gate.deliver``: sent -> DONE; blocked by a race -> step 4; transient error ->
   retry with exponential backoff (limited attempts); permanent error -> DROPPED.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.models import ReminderSchedule
from procbot.domain.enums import MessageCategory, ReminderState
from procbot.services import gate, leases
from procbot.services.jobs import PeriodicJob, registered_jobs
from procbot.services.outbound import get_renderer
from procbot.services.profile import get_profile
from procbot.services.reminders import LIVE_STATES, schedule_reminder
from procbot.timeutil import local_date

log = logging.getLogger(__name__)

# Technical retry policy for transient Telegram errors (spec section 7).
RETRY_BASE = timedelta(seconds=30)
RETRY_MAX = timedelta(minutes=30)
MAX_SEND_ATTEMPTS = 6


class ReminderResult(StrEnum):
    SENT = "sent"
    DUPLICATE = "duplicate"  # already sent / maybe sent earlier: not re-sent
    RESCHEDULED = "rescheduled"
    FOLDED = "folded"
    DROPPED = "dropped"
    RETRY = "retry"  # transient send error, backoff
    SKIPPED = "skipped"  # not due any more / lease held elsewhere


@dataclass
class TickReport:
    results: dict[str, int] = field(default_factory=dict)
    jobs_run: list[str] = field(default_factory=list)
    jobs_failed: list[str] = field(default_factory=list)
    maybe_sent: int = 0

    def add(self, result: ReminderResult) -> None:
        self.results[result] = self.results.get(result, 0) + 1

    def count(self, result: ReminderResult) -> int:
        return self.results.get(result, 0)

    def merge(self, other: TickReport) -> None:
        for k, v in other.results.items():
            self.results[k] = self.results.get(k, 0) + v
        self.jobs_run += other.jobs_run
        self.jobs_failed += other.jobs_failed
        self.maybe_sent += other.maybe_sent


def reminder_lease(reminder_id: int) -> str:
    return f"reminder:{reminder_id}"


def outbound_key(reminder_id: int) -> str:
    return f"rem:{reminder_id}"


def _category(raw: str) -> MessageCategory | None:
    try:
        return MessageCategory(raw)
    except ValueError:
        return None


def effective_stale_at(ctx: AppContext, r: ReminderSchedule) -> datetime | None:
    if r.category == MessageCategory.DIGEST:
        return None  # the digest itself never goes stale
    if r.stale_at is not None:
        return r.stale_at
    return r.due_at + timedelta(minutes=ctx.config.gate.default_stale_after_minutes)


# --------------------------------------------------------------------------- folding


async def _fold(
    s: AsyncSession, ctx: AppContext, r: ReminderSchedule, now: datetime, reason: str
) -> None:
    """Mark `r` folded and add it to the one pending digest (or start one)."""
    r.state = ReminderState.FOLDED
    r.last_reason = reason
    r.updated_at = now
    digests = (
        (
            await s.execute(
                select(ReminderSchedule)
                .where(
                    ReminderSchedule.category == MessageCategory.DIGEST,
                    ReminderSchedule.state.in_([str(x) for x in LIVE_STATES]),
                )
                .order_by(ReminderSchedule.id)
            )
        )
        .scalars()
        .all()
    )
    for d in digests:
        # a digest being rendered right now (lease held) is not extended
        if await leases.is_held_in(s, reminder_lease(d.id), now):
            continue
        payload = dict(d.payload or {})
        items = [int(x) for x in payload.get("items", [])]
        if r.id not in items:
            items.append(r.id)
        payload["items"] = items
        d.payload = payload
        d.updated_at = now
        return
    state = await gate.load_state(s, ctx, now)
    due = gate.next_allowed_in(state, MessageCategory.DIGEST, now) or now
    profile = await get_profile(s)
    day = local_date(now, profile.timezone if profile else None).isoformat()
    n = (
        await s.execute(
            select(func.count(ReminderSchedule.id)).where(
                ReminderSchedule.dedup_key.like(f"digest:{day}:%")
            )
        )
    ).scalar_one()
    await schedule_reminder(
        s,
        now=now,
        category=MessageCategory.DIGEST,
        due_at=due,
        dedup_key=f"digest:{day}:{n + 1}",
        payload={"items": [r.id]},
    )


async def _defer(
    s: AsyncSession,
    ctx: AppContext,
    r: ReminderSchedule,
    decision: gate.GateDecision,
    now: datetime,
) -> ReminderResult:
    """Blocked: reschedule to the next allowed moment, fold, or drop."""
    reason = str(decision.reason) if decision.reason else "blocked"
    next_at = decision.retry_at
    if next_at is None:
        r.state = ReminderState.DROPPED
        r.last_reason = reason
        r.updated_at = now
        return ReminderResult.DROPPED
    stale = effective_stale_at(ctx, r)
    if stale is not None and next_at > stale:
        await _fold(s, ctx, r, now, reason=f"{reason}; stale")
        return ReminderResult.FOLDED
    r.stale_at = stale
    r.due_at = next_at
    r.state = ReminderState.PENDING
    r.reschedule_count += 1
    r.last_reason = reason
    r.updated_at = now
    return ReminderResult.RESCHEDULED


async def _finish(
    ctx: AppContext, reminder_id: int, state: ReminderState, reason: str | None
) -> None:
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        r = await s.get(ReminderSchedule, reminder_id)
        if r is None:
            return
        # a sent message is a fact: DONE even if the reminder was cancelled meanwhile
        if state != ReminderState.DONE and r.state not in LIVE_STATES:
            return
        r.state = state
        r.last_reason = reason
        r.updated_at = now


async def _retry_later(ctx: AppContext, reminder_id: int, error: str) -> ReminderResult:
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        r = await s.get(ReminderSchedule, reminder_id)
        if r is None or r.state not in LIVE_STATES:
            return ReminderResult.SKIPPED
        r.attempts += 1
        if r.attempts >= MAX_SEND_ATTEMPTS:
            r.state = ReminderState.DROPPED
            r.last_reason = f"gave up after {r.attempts} attempts: {error}"[:500]
            r.updated_at = now
            log.error("reminder %s dropped after %s attempts", reminder_id, r.attempts)
            return ReminderResult.DROPPED
        delay = min(RETRY_BASE * (2 ** (r.attempts - 1)), RETRY_MAX)
        r.stale_at = effective_stale_at(ctx, r)
        r.due_at = now + delay
        r.last_reason = f"retry: {error}"[:500]
        r.updated_at = now
        return ReminderResult.RETRY


# --------------------------------------------------------------------------- processing


async def process_reminder(ctx: AppContext, reminder_id: int) -> ReminderResult:
    name = reminder_lease(reminder_id)
    owner = f"{ctx.instance_id}:{secrets.token_hex(4)}"  # unique per processing run
    if not await leases.acquire(ctx, name, owner=owner):
        return ReminderResult.SKIPPED
    try:
        return await _process_locked(ctx, reminder_id)
    finally:
        await leases.release(ctx, name, owner=owner)


async def _process_locked(ctx: AppContext, reminder_id: int) -> ReminderResult:
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        r = await s.get(ReminderSchedule, reminder_id)
        if r is None or r.state not in LIVE_STATES or r.due_at > now:
            return ReminderResult.SKIPPED
        cat = _category(r.category)
        if cat is None:
            r.state = ReminderState.DROPPED
            r.last_reason = "unknown category"
            return ReminderResult.DROPPED
        stale = effective_stale_at(ctx, r)
        if stale is not None and now > stale:
            await _fold(s, ctx, r, now, reason="stale")
            return ReminderResult.FOLDED
        decision = await gate.check_in(s, ctx, cat, now)
        if not decision.allowed:
            return await _defer(s, ctx, r, decision, now)

    renderer = get_renderer(ctx, cat)
    if renderer is None:
        log.warning("no renderer for %s; reminder %s dropped", cat, reminder_id)
        await _finish(ctx, reminder_id, ReminderState.DROPPED, "no renderer")
        return ReminderResult.DROPPED
    try:
        out = await renderer(ctx, r)
    except Exception as exc:
        log.exception("renderer for %s failed", cat)
        return await _retry_later(ctx, reminder_id, f"render: {exc}")
    if out is None:
        await _finish(ctx, reminder_id, ReminderState.DROPPED, "obsolete")
        return ReminderResult.DROPPED

    res = await gate.deliver(
        ctx,
        category=cat,
        text=out.text,
        reply_markup=out.reply_markup,
        dedup_key=outbound_key(reminder_id),
        reminder_id=reminder_id,
    )
    if res.outcome in (gate.Outcome.SENT, gate.Outcome.DUPLICATE):
        await _finish(ctx, reminder_id, ReminderState.DONE, None)
        return ReminderResult.SENT if res.outcome == gate.Outcome.SENT else ReminderResult.DUPLICATE
    if res.outcome == gate.Outcome.BLOCKED:  # state changed between check and deliver
        now = ctx.clock.now()
        async with ctx.db.transaction() as s:
            r2 = await s.get(ReminderSchedule, reminder_id)
            if r2 is None or r2.state not in LIVE_STATES:
                return ReminderResult.SKIPPED
            return await _defer(s, ctx, r2, res, now)
    if res.permanent:
        await _finish(ctx, reminder_id, ReminderState.DROPPED, f"send failed: {res.error}")
        return ReminderResult.DROPPED
    return await _retry_later(ctx, reminder_id, res.error or "send failed")


async def due_reminder_ids(
    ctx: AppContext, *, now: datetime, limit: int, exclude: set[int] | None = None
) -> list[int]:
    async with ctx.db.transaction() as s:
        stmt = (
            select(ReminderSchedule.id)
            .where(
                ReminderSchedule.state.in_([str(x) for x in LIVE_STATES]),
                ReminderSchedule.due_at <= now,
            )
            .order_by(ReminderSchedule.due_at, ReminderSchedule.id)
            .limit(limit + len(exclude or ()))
        )
        ids = list((await s.execute(stmt)).scalars().all())
    return [i for i in ids if i not in (exclude or set())][:limit]


async def process_due_reminders(ctx: AppContext, *, limit: int | None = None) -> TickReport:
    report = TickReport()
    cap = limit or ctx.config.scheduler.batch_size
    seen: set[int] = set()
    while len(seen) < cap:
        ids = await due_reminder_ids(ctx, now=ctx.clock.now(), limit=cap - len(seen), exclude=seen)
        if not ids:
            break
        for rid in ids:
            seen.add(rid)
            try:
                report.add(await process_reminder(ctx, rid))
            except Exception:
                log.exception("processing reminder %s failed", rid)
    return report


# --------------------------------------------------------------------------- scheduler


def stuck_sending_threshold(ctx: AppContext) -> timedelta:
    return timedelta(seconds=max(ctx.config.scheduler.lease_seconds, 60))


class Scheduler:
    """One loop per process. Job "last run" times live in memory (jobs are
    idempotent by contract); a lease per job name prevents parallel runs."""

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self._next_run: dict[str, datetime] = {}

    def _due_jobs(self, now: datetime) -> list[PeriodicJob]:
        due: list[PeriodicJob] = []
        for job in registered_jobs(self.ctx):
            nxt = self._next_run.get(job.name)
            if nxt is None:
                if job.run_on_start:
                    due.append(job)
                else:
                    self._next_run[job.name] = now + timedelta(seconds=job.interval_seconds)
            elif now >= nxt:
                due.append(job)
        return due

    async def run_jobs(self) -> TickReport:
        report = TickReport()
        now = self.ctx.clock.now()
        for job in self._due_jobs(now):
            name = f"job:{job.name}"
            if not await leases.acquire(self.ctx, name):
                continue
            try:
                await job.run(self.ctx)
                report.jobs_run.append(job.name)
            except Exception:
                log.exception("periodic job %s failed", job.name)
                report.jobs_failed.append(job.name)
            finally:
                self._next_run[job.name] = now + timedelta(seconds=job.interval_seconds)
                await leases.release(self.ctx, name)
        return report

    async def tick(self) -> TickReport:
        report = TickReport()
        try:
            report.maybe_sent = await gate.sweep_stuck_sending(
                self.ctx, older_than=stuck_sending_threshold(self.ctx)
            )
        except Exception:
            log.exception("sweep of stuck outbound messages failed")
        try:
            report.merge(await self.run_jobs())
        except Exception:
            log.exception("periodic jobs failed")
        try:
            report.merge(await process_due_reminders(self.ctx))
        except Exception:
            log.exception("reminder processing failed")
        return report

    async def run_forever(self, stop: asyncio.Event) -> None:
        interval = self.ctx.config.scheduler.tick_seconds
        while not stop.is_set():
            try:
                await self.tick()
            except Exception:  # pragma: no cover - tick already guards each step
                log.exception("scheduler tick failed")
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except TimeoutError:
                pass
