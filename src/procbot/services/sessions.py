"""WorkSessionService (FR-06): work sessions, timer signals, retro entries, stats.

Owns ``work_sessions``, ``session_pauses`` and ``session_corrections``.

Modes (the user always chooses; the bot may only recommend):
- FREE: free timer, optional target (``planned_minutes``) -> one "target" signal;
- POMODORO: ``work_minutes``/``break_minutes`` snapshot from the profile;
  ``planned_minutes`` is the net-minute at which the current work period ends;
- SHORT_START: ``work_minutes`` = the short-start step, ``planned_minutes`` = the
  cumulative net target ("Ещё N минут" moves it forward).

Durations come only from the owner profile (open question 5). When a value is
missing ``SettingsMissing`` is raised and the UI asks once, saving the answer.

Timer signals (the only messages allowed during a session, acceptance 14) are
derived deterministically from DB rows: ``expected_timer()`` computes the one
signal that should currently be pending (key ``timer:{sid}:{phase}:{marker}:{p}``
where ``p`` is the number of pauses), so restart recovery recomputes the same key
and the reminder dedup prevents duplicates. After a boundary is signalled no
further signal is scheduled until the user acts, so ignoring the bot never leads
to a stream of messages.

Pomodoro breaks are pauses: a pause that starts after the work period ended is a
break (it gets a "break over" signal after ``break_minutes``); resuming from it
starts a new work period. A pause in the middle of a period just stops the clock.

Times: live sessions are ``time_source=exact``. Planned-but-unconfirmed sessions
stay ``unconfirmed`` and are never counted (acceptance 15). Retro entries keep
``retro``/``approximate`` + confidence and a correction journal (acceptance 16).
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.models import (
    Intervention,
    OutboundMessage,
    ReminderSchedule,
    SessionCorrection,
    SessionPause,
    Task,
    WorkSession,
)
from procbot.domain.enums import (
    Actor,
    InterventionResult,
    MessageCategory,
    ObstacleReason,
    ReminderState,
    SessionMode,
    SessionOutcome,
    SessionStatus,
    TimeSource,
)
from procbot.services import interventions_basic
from procbot.services import tasks as task_service
from procbot.services.audit import audit
from procbot.services.outbound import Outgoing, register_renderer
from procbot.services.profile import get_profile
from procbot.services.reminders import cancel_reminders, schedule_reminder
from procbot.telegram.callbacks import SessionCb
from procbot.timeutil import local_date, local_day_bounds, resolve_local, to_local

log = logging.getLogger(__name__)

LIVE_STATUSES: tuple[SessionStatus, ...] = (SessionStatus.ACTIVE, SessionStatus.PAUSED)

MODE_LABELS: dict[SessionMode, str] = {
    SessionMode.FREE: "Свободный таймер",
    SessionMode.POMODORO: "Помодоро",
    SessionMode.SHORT_START: "Короткий старт",
}
OUTCOME_LABELS: dict[SessionOutcome, str] = {
    SessionOutcome.STARTED: "Начал",
    SessionOutcome.PROGRESSED: "Продвинулся, но не закончил",
    SessionOutcome.COMPLETED: "Завершил",
}
SOURCE_LABELS: dict[str, str] = {
    TimeSource.EXACT: "точно (таймер)",
    TimeSource.RETRO: "задним числом",
    TimeSource.APPROXIMATE: "приблизительно",
    TimeSource.UNKNOWN: "неизвестно",
}
CONFIDENCE_LABELS: dict[str, str] = {"high": "точно", "medium": "примерно", "low": "не уверен"}
CONFIDENCE_VALUES: tuple[str, ...] = ("high", "medium", "low")

# Profile fields each mode needs (open question 5: never invented, asked once).
MODE_SETTINGS: dict[SessionMode, tuple[str, ...]] = {
    SessionMode.FREE: (),
    SessionMode.POMODORO: ("pomodoro_work_min", "pomodoro_break_min"),
    SessionMode.SHORT_START: ("short_start_min",),
}
# Obstacles after which a short start is the natural recommendation (FR-06).
SHORT_START_OBSTACLES: frozenset[str] = frozenset(
    {ObstacleReason.UNCLEAR, ObstacleReason.TOO_BIG, ObstacleReason.UNPLEASANT}
)
# Validation bounds for user-entered durations (not defaults).
SETTING_BOUNDS: dict[str, tuple[int, int]] = {
    "pomodoro_work_min": (1, 240),
    "pomodoro_break_min": (1, 120),
    "short_start_min": (1, 120),
}
MAX_SESSION = timedelta(hours=24)  # sanity bound for retro entries / corrections
# "начинаю" typed after a recent pre-task / start-delay prompt, and linking a
# session to help offered shortly before (technical heuristics, not reminder
# intervals).
RECENT_PROMPT = timedelta(minutes=60)
UNCONFIRMED_WINDOW = timedelta(hours=3)

_SIGNALLED_STATES = (ReminderState.DONE, ReminderState.FOLDED, ReminderState.PROCESSING)
_PENDING_STATES = (ReminderState.PENDING, ReminderState.RESCHEDULED)


# --------------------------------------------------------------------------- errors


class SessionError(Exception):
    """Base class: the operation is not possible in the current state."""


class ActiveSessionExists(SessionError):
    """Only one active session at a time: finish the current one first."""

    def __init__(self, session: WorkSession) -> None:
        super().__init__(f"work session {session.id} is still {session.status}")
        # Plain values: the ORM object is expired once the transaction rolls back.
        self.session_id: int = session.id
        self.task_id: int | None = session.task_id
        self.status: str = session.status


class SettingsMissing(SessionError):
    """The mode needs durations the owner has not set yet (open question 5)."""

    def __init__(self, mode: SessionMode, fields: tuple[str, ...]) -> None:
        super().__init__(f"{mode} needs {', '.join(fields)}")
        self.mode = mode
        self.fields = fields


class CriteriaCheckRequired(SessionError):
    """COMPLETED for a task with a definition of done needs an explicit check."""


class InvalidSessionState(SessionError):
    pass


# --------------------------------------------------------------------------- queries


async def get_work_session(s: AsyncSession, session_id: int) -> WorkSession | None:
    return await s.get(WorkSession, session_id)


async def get_active_session(s: AsyncSession) -> WorkSession | None:
    """The single active or paused session, if any."""
    return (
        (
            await s.execute(
                select(WorkSession)
                .where(WorkSession.status.in_([str(x) for x in LIVE_STATUSES]))
                .order_by(WorkSession.started_at.desc())
                .limit(1)
            )
        )
        .scalars()
        .first()
    )


async def load_pauses(s: AsyncSession, session_id: int) -> list[SessionPause]:
    rows = (
        (
            await s.execute(
                select(SessionPause)
                .where(SessionPause.session_id == session_id)
                .order_by(SessionPause.started_at, SessionPause.id)
            )
        )
        .scalars()
        .all()
    )
    return list(rows)


async def _require(s: AsyncSession, session_id: int) -> WorkSession:
    ws = await s.get(WorkSession, session_id)
    if ws is None:
        raise InvalidSessionState(f"work session {session_id} not found")
    return ws


def task_needs_criteria(task: Task | None) -> bool:
    """Important tasks are compared with their definition of done (FR-06, scenario E)."""
    if task is None:
        return False
    return task.verification == "criteria" or bool((task.done_criteria or "").strip())


async def missing_settings(s: AsyncSession, mode: SessionMode) -> tuple[str, ...]:
    profile = await get_profile(s)
    needed = MODE_SETTINGS[SessionMode(mode)]
    if profile is None:
        return needed
    return tuple(f for f in needed if getattr(profile, f) is None)


# --------------------------------------------------------------------------- durations


def _overlap(pauses: Sequence[SessionPause], a: datetime, b: datetime) -> float:
    total = 0.0
    for p in pauses:
        p_end = p.ended_at or b
        lo, hi = max(p.started_at, a), min(p_end, b)
        if hi > lo:
            total += (hi - lo).total_seconds()
    return total


def net_seconds_at(ws: WorkSession, pauses: Sequence[SessionPause], moment: datetime) -> float:
    """Net worked seconds from start up to `moment` (pauses excluded)."""
    if ws.started_at is None:
        return 0.0
    end = min(moment, ws.ended_at) if ws.ended_at is not None else moment
    if end <= ws.started_at:
        return 0.0
    gross = (end - ws.started_at).total_seconds()
    return max(0.0, gross - _overlap(pauses, ws.started_at, end))


def net_seconds(ws: WorkSession, pauses: Sequence[SessionPause], now: datetime) -> float:
    """(end - start) - pauses; a live session is measured up to `now`."""
    return net_seconds_at(ws, pauses, ws.ended_at or now)


def pause_seconds(ws: WorkSession, pauses: Sequence[SessionPause], now: datetime) -> float:
    if ws.started_at is None:
        return 0.0
    return _overlap(pauses, ws.started_at, ws.ended_at or now)


def wall_time_at_net(
    ws: WorkSession, pauses: Sequence[SessionPause], target_seconds: float
) -> datetime | None:
    """The moment net time reaches `target_seconds` (None while an open pause blocks it)."""
    if ws.started_at is None:
        return None
    t = ws.started_at
    remaining = max(0.0, target_seconds)
    for p in sorted(pauses, key=lambda x: x.started_at):
        if p.ended_at is not None and p.ended_at <= t:
            continue
        seg = max(0.0, (p.started_at - t).total_seconds())
        if remaining <= seg:
            return t + timedelta(seconds=remaining)
        remaining -= seg
        if p.ended_at is None:
            return None
        t = max(t, p.ended_at)
    return t + timedelta(seconds=remaining)


def fmt_minutes(minutes: float) -> str:
    m = round(minutes)
    if m < 60:
        return f"{m} мин"
    h, rest = divmod(m, 60)
    return f"{h} ч" if rest == 0 else f"{h} ч {rest:02d} мин"


# --------------------------------------------------------------------------- timer signals


@dataclass(frozen=True)
class TimerSignal:
    phase: str  # work_end / break_end / short_end / target
    key: str
    prefix: str  # same boundary regardless of later pauses
    due_at: datetime
    payload: dict[str, Any]


PHASE_BY_MODE: dict[str, str] = {
    SessionMode.POMODORO: "work_end",
    SessionMode.SHORT_START: "short_end",
    SessionMode.FREE: "target",
}


def _is_break(ws: WorkSession, pauses: Sequence[SessionPause], pause: SessionPause) -> bool:
    """A pomodoro pause that began after the work period ended is a break."""
    if ws.mode != SessionMode.POMODORO or ws.planned_minutes is None:
        return False
    return net_seconds_at(ws, pauses, pause.started_at) >= ws.planned_minutes * 60 - 1


def expected_timer(ws: WorkSession, pauses: Sequence[SessionPause]) -> TimerSignal | None:
    """The single timer signal that should currently be pending for `ws` (or None).

    Pure function of DB rows, so restart recovery recomputes the same dedup key.
    """
    if ws.started_at is None or ws.status not in LIVE_STATUSES:
        return None
    p = len(pauses)
    open_pause = next((x for x in reversed(pauses) if x.ended_at is None), None)
    if ws.status == SessionStatus.PAUSED:
        if open_pause is not None and ws.break_minutes and _is_break(ws, pauses, open_pause):
            prefix = f"timer:{ws.id}:break_end:{open_pause.id}:"
            return TimerSignal(
                phase="break_end",
                key=f"{prefix}{p}",
                prefix=prefix,
                due_at=open_pause.started_at + timedelta(minutes=ws.break_minutes),
                payload={"phase": "break_end", "pause_id": open_pause.id, "pauses": p},
            )
        return None
    if open_pause is not None or ws.planned_minutes is None:
        return None
    phase = PHASE_BY_MODE[ws.mode]
    due = wall_time_at_net(ws, pauses, ws.planned_minutes * 60)
    if due is None:
        return None
    prefix = f"timer:{ws.id}:{phase}:{ws.planned_minutes}:"
    return TimerSignal(
        phase=phase,
        key=f"{prefix}{p}",
        prefix=prefix,
        due_at=due,
        payload={"phase": phase, "target_min": ws.planned_minutes, "pauses": p},
    )


async def _timer_rows(s: AsyncSession, session_id: int) -> list[ReminderSchedule]:
    rows = (
        (
            await s.execute(
                select(ReminderSchedule).where(
                    ReminderSchedule.session_id == session_id,
                    ReminderSchedule.category == str(MessageCategory.TIMER_SIGNAL),
                )
            )
        )
        .scalars()
        .all()
    )
    return list(rows)


async def resync_timer(
    s: AsyncSession, ctx: AppContext, ws: WorkSession, *, allow_past: bool = False
) -> ReminderSchedule | None:
    """Make the pending timer signal of `ws` match `expected_timer()`. Idempotent.

    allow_past=False (live transitions): a boundary already behind us is not
    signalled again. allow_past=True (restart recovery): a boundary that passed
    during downtime and was never signalled is scheduled now with its original
    due time; the gate/scheduler decide whether it is still worth sending.
    """
    now = ctx.clock.now()
    pauses = await load_pauses(s, ws.id)
    exp = expected_timer(ws, pauses)
    rows = await _timer_rows(s, ws.id)
    pending = [r for r in rows if r.state in _PENDING_STATES]
    if (exp is None or not any(r.dedup_key == exp.key for r in pending)) and pending:
        await cancel_reminders(
            s,
            now=now,
            reason="session timer changed",
            session_id=ws.id,
            categories={MessageCategory.TIMER_SIGNAL},
        )
    if exp is None:
        return None
    if any(r.dedup_key == exp.key for r in rows):
        # Pending -> refresh due time; final -> schedule_reminder leaves it alone.
        return await schedule_reminder(
            s,
            now=now,
            category=MessageCategory.TIMER_SIGNAL,
            due_at=exp.due_at,
            dedup_key=exp.key,
            task_id=ws.task_id,
            session_id=ws.id,
            payload=exp.payload,
        )
    if any(r.dedup_key.startswith(exp.prefix) and r.state in _SIGNALLED_STATES for r in rows):
        return None  # this boundary was already signalled before a later pause
    if exp.due_at <= now and not allow_past:
        return None
    return await schedule_reminder(
        s,
        now=now,
        category=MessageCategory.TIMER_SIGNAL,
        due_at=exp.due_at,
        dedup_key=exp.key,
        task_id=ws.task_id,
        session_id=ws.id,
        payload=exp.payload,
    )


async def _cancel_timer(s: AsyncSession, ctx: AppContext, ws: WorkSession, reason: str) -> None:
    await cancel_reminders(
        s,
        now=ctx.clock.now(),
        reason=reason,
        session_id=ws.id,
        categories={MessageCategory.TIMER_SIGNAL},
    )


# --------------------------------------------------------------------------- lifecycle


async def create_unconfirmed_session(
    s: AsyncSession,
    ctx: AppContext,
    *,
    task_id: int | None,
    mode: SessionMode = SessionMode.FREE,
    planned_start_at: datetime | None = None,
    planned_minutes: int | None = None,
    intervention_id: int | None = None,
    actor: Actor = Actor.SYSTEM,
) -> WorkSession:
    """A planned/proposed session. It stays `unconfirmed` (never counted as worked
    time) until the user confirms the start (acceptance 15)."""
    now = ctx.clock.now()
    ws = WorkSession(
        task_id=task_id,
        mode=str(SessionMode(mode)),
        status=SessionStatus.UNCONFIRMED,
        planned_start_at=planned_start_at,
        time_source=TimeSource.EXACT,
        planned_minutes=planned_minutes,
        intervention_id=intervention_id,
        created_at=now,
        updated_at=now,
    )
    s.add(ws)
    await s.flush()
    await audit(
        s,
        now=now,
        actor=actor,
        action="session.planned",
        entity="work_session",
        entity_id=ws.id,
        details={"task_id": task_id, "mode": str(mode)},
    )
    return ws


async def start_session(
    s: AsyncSession,
    ctx: AppContext,
    *,
    task_id: int | None,
    mode: SessionMode,
    actor: Actor = Actor.USER,
    session_id: int | None = None,
    planned_minutes: int | None = None,
    intervention_id: int | None = None,
) -> WorkSession:
    """Start a session the user confirmed: status active, started_at=now, exact.

    `session_id` confirms an existing unconfirmed session instead of creating one.
    Raises ActiveSessionExists (one at a time) or SettingsMissing (ask the user).
    The task (if any) gets STARTED via services.tasks.record_work_outcome.
    """
    mode = SessionMode(mode)
    now = ctx.clock.now()
    active = await get_active_session(s)
    if active is not None:
        raise ActiveSessionExists(active)
    missing = await missing_settings(s, mode)
    if missing:
        raise SettingsMissing(mode, missing)
    profile = await get_profile(s)

    ws: WorkSession | None = None
    if session_id is not None:
        ws = await _require(s, session_id)
        if ws.status != SessionStatus.UNCONFIRMED:
            raise InvalidSessionState(f"session {session_id} is {ws.status}")
        task_id = ws.task_id if task_id is None else task_id
        intervention_id = intervention_id or ws.intervention_id
        planned_minutes = planned_minutes if planned_minutes is not None else ws.planned_minutes
    if intervention_id is None and task_id is not None:
        # Link the help that was just offered for this task (scenario C -> D).
        iv = await interventions_basic.find_recent_intervention(
            s, task_id=task_id, since=now - RECENT_PROMPT
        )
        intervention_id = iv.id if iv is not None else None
    if ws is None:
        ws = WorkSession(
            created_at=now, updated_at=now, mode=str(mode), status=SessionStatus.UNCONFIRMED
        )
        s.add(ws)

    ws.task_id = task_id
    ws.mode = str(mode)
    ws.status = SessionStatus.ACTIVE
    ws.started_at = now
    ws.ended_at = None
    ws.time_source = TimeSource.EXACT
    ws.intervention_id = intervention_id
    ws.updated_at = now
    if mode == SessionMode.POMODORO:
        assert profile is not None and profile.pomodoro_work_min is not None
        ws.work_minutes = profile.pomodoro_work_min
        ws.break_minutes = profile.pomodoro_break_min
        ws.planned_minutes = profile.pomodoro_work_min
    elif mode == SessionMode.SHORT_START:
        assert profile is not None and profile.short_start_min is not None
        ws.work_minutes = profile.short_start_min
        ws.break_minutes = None
        ws.planned_minutes = profile.short_start_min
    else:
        ws.work_minutes = None
        ws.break_minutes = None
        ws.planned_minutes = planned_minutes if planned_minutes and planned_minutes > 0 else None
    await s.flush()

    if task_id is not None:
        await task_service.record_work_outcome(
            s,
            ctx,
            task_id=task_id,
            outcome=SessionOutcome.STARTED,
            actor=actor,
            session_id=ws.id,
        )
    if intervention_id is not None:
        await interventions_basic.record_intervention_outcome(
            s,
            intervention_id=intervention_id,
            result=InterventionResult.STARTED,
            session_id=ws.id,
            now=now,
        )
    await resync_timer(s, ctx, ws)
    await audit(
        s,
        now=now,
        actor=actor,
        action="session.started",
        entity="work_session",
        entity_id=ws.id,
        details={"task_id": task_id, "mode": str(mode)},
    )
    return ws


async def pause_session(
    s: AsyncSession, ctx: AppContext, *, session_id: int, actor: Actor = Actor.USER
) -> WorkSession:
    """Pause (no-op if already paused). Pending timer signals are cancelled."""
    ws = await _require(s, session_id)
    if ws.status == SessionStatus.PAUSED:
        return ws
    if ws.status != SessionStatus.ACTIVE:
        raise InvalidSessionState(f"session {session_id} is {ws.status}")
    now = ctx.clock.now()
    s.add(SessionPause(session_id=ws.id, started_at=now, time_source=TimeSource.EXACT))
    ws.status = SessionStatus.PAUSED
    ws.updated_at = now
    await s.flush()
    await _cancel_timer(s, ctx, ws, "session paused")
    await resync_timer(s, ctx, ws)  # a pomodoro break gets its "break over" signal
    await audit(
        s, now=now, actor=actor, action="session.paused", entity="work_session", entity_id=ws.id
    )
    return ws


async def resume_session(
    s: AsyncSession, ctx: AppContext, *, session_id: int, actor: Actor = Actor.USER
) -> WorkSession:
    """Resume (no-op if already active). Timer signals are rescheduled; after a
    pomodoro break a new work period starts."""
    ws = await _require(s, session_id)
    if ws.status == SessionStatus.ACTIVE:
        return ws
    if ws.status != SessionStatus.PAUSED:
        raise InvalidSessionState(f"session {session_id} is {ws.status}")
    now = ctx.clock.now()
    pauses = await load_pauses(s, ws.id)
    was_break = False
    for p in pauses:
        if p.ended_at is None:
            was_break = was_break or _is_break(ws, pauses, p)
            p.ended_at = now
    ws.status = SessionStatus.ACTIVE
    ws.updated_at = now
    if was_break and ws.work_minutes:
        ws.planned_minutes = _ceil_min(net_seconds_at(ws, pauses, now)) + ws.work_minutes
    await s.flush()
    await resync_timer(s, ctx, ws)
    await audit(
        s,
        now=now,
        actor=actor,
        action="session.resumed",
        entity="work_session",
        entity_id=ws.id,
        details={"after_break": was_break},
    )
    return ws


async def stop_session(
    s: AsyncSession, ctx: AppContext, *, session_id: int, actor: Actor = Actor.USER
) -> WorkSession:
    """Finish the session (no-op if already finished). The outcome is asked next."""
    ws = await _require(s, session_id)
    if ws.status == SessionStatus.FINISHED:
        return ws
    if ws.status not in LIVE_STATUSES:
        raise InvalidSessionState(f"session {session_id} is {ws.status}")
    now = ctx.clock.now()
    for p in await load_pauses(s, ws.id):
        if p.ended_at is None:
            p.ended_at = now
    ws.status = SessionStatus.FINISHED
    ws.ended_at = now
    ws.updated_at = now
    await s.flush()
    await _cancel_timer(s, ctx, ws, "session finished")
    pauses = await load_pauses(s, ws.id)
    await audit(
        s,
        now=now,
        actor=actor,
        action="session.finished",
        entity="work_session",
        entity_id=ws.id,
        details={"net_minutes": round(net_seconds(ws, pauses, now) / 60, 1)},
    )
    return ws


async def cancel_session(
    s: AsyncSession, ctx: AppContext, *, session_id: int, actor: Actor = Actor.USER
) -> WorkSession:
    """Drop a planned (unconfirmed) session. It is never counted."""
    ws = await _require(s, session_id)
    if ws.status == SessionStatus.CANCELLED:
        return ws
    if ws.status != SessionStatus.UNCONFIRMED:
        raise InvalidSessionState(f"session {session_id} is {ws.status}")
    now = ctx.clock.now()
    ws.status = SessionStatus.CANCELLED
    ws.updated_at = now
    await _cancel_timer(s, ctx, ws, "session cancelled")
    await audit(
        s, now=now, actor=actor, action="session.cancelled", entity="work_session", entity_id=ws.id
    )
    return ws


def _ceil_min(seconds: float) -> int:
    return max(0, math.ceil(seconds / 60 - 1e-9))


async def _require_live_active(s: AsyncSession, session_id: int) -> WorkSession:
    ws = await _require(s, session_id)
    if ws.status != SessionStatus.ACTIVE:
        raise InvalidSessionState(f"session {session_id} is {ws.status}")
    return ws


async def extend_session(
    s: AsyncSession, ctx: AppContext, *, session_id: int, minutes: int | None = None
) -> WorkSession:
    """ "Ещё N минут" after a short start (or a free-timer target): new target =
    now + N. N defaults to the session's short-start step."""
    ws = await _require_live_active(s, session_id)
    step = minutes or ws.work_minutes
    if not step or step <= 0:
        raise InvalidSessionState("no step to extend by")
    now = ctx.clock.now()
    pauses = await load_pauses(s, ws.id)
    ws.planned_minutes = _ceil_min(net_seconds_at(ws, pauses, now)) + step
    ws.updated_at = now
    await s.flush()
    await resync_timer(s, ctx, ws)
    return ws


async def switch_to_free(s: AsyncSession, ctx: AppContext, *, session_id: int) -> WorkSession:
    """Continue the same session as a free timer (no more signals)."""
    ws = await _require_live_active(s, session_id)
    if ws.mode != SessionMode.FREE:
        old = ws.mode
        ws.mode = str(SessionMode.FREE)
        ws.planned_minutes = None
        ws.updated_at = ctx.clock.now()
        await s.flush()
        await resync_timer(s, ctx, ws)
        await audit(
            s,
            now=ctx.clock.now(),
            actor=Actor.USER,
            action="session.mode_changed",
            entity="work_session",
            entity_id=ws.id,
            details={"from": old, "to": str(SessionMode.FREE)},
        )
    return ws


async def continue_without_break(
    s: AsyncSession, ctx: AppContext, *, session_id: int
) -> WorkSession:
    """Pomodoro: skip the break, a new work period starts now."""
    ws = await _require_live_active(s, session_id)
    if ws.mode != SessionMode.POMODORO or not ws.work_minutes or ws.planned_minutes is None:
        return ws
    now = ctx.clock.now()
    pauses = await load_pauses(s, ws.id)
    net = net_seconds_at(ws, pauses, now)
    if net < ws.planned_minutes * 60 - 1:
        return ws  # the work period is still running: nothing to skip (repeat press)
    ws.planned_minutes = _ceil_min(net) + ws.work_minutes
    ws.updated_at = now
    await s.flush()
    await resync_timer(s, ctx, ws)
    return ws


async def take_break(s: AsyncSession, ctx: AppContext, *, session_id: int) -> WorkSession:
    """Pomodoro: start the break (a pause after the work period ended)."""
    return await pause_session(s, ctx, session_id=session_id)


def current_phase(ws: WorkSession, pauses: Sequence[SessionPause], now: datetime) -> str:
    """Human-oriented phase: work / overtime / break / paused / finished / ..."""
    if ws.status == SessionStatus.PAUSED:
        open_pause = next((x for x in reversed(pauses) if x.ended_at is None), None)
        if open_pause is not None and _is_break(ws, pauses, open_pause):
            return "break"
        return "paused"
    if ws.status != SessionStatus.ACTIVE:
        return str(ws.status)
    if (
        ws.planned_minutes is not None
        and net_seconds_at(ws, pauses, now) >= ws.planned_minutes * 60
    ):
        return "overtime"
    return "work"


# --------------------------------------------------------------------------- outcome


async def record_outcome(
    s: AsyncSession,
    ctx: AppContext,
    *,
    session_id: int,
    outcome: SessionOutcome,
    actor: Actor = Actor.USER,
    criteria_confirmed: bool | None = None,
    note: str | None = None,
) -> tuple[WorkSession, bool]:
    """Store the FR-06 result of a finished session and apply it to the task.

    COMPLETED on a task with a definition of done requires criteria_confirmed=True
    (the user compared the result with the criterion); otherwise
    CriteriaCheckRequired is raised. A partial result is PROGRESSED, never done.
    Returns (session, applied); applied=False when an outcome was already stored
    (repeated button press).
    """
    outcome = SessionOutcome(outcome)
    ws = await _require(s, session_id)
    if ws.status != SessionStatus.FINISHED:
        raise InvalidSessionState("outcome is recorded after the session is finished")
    if ws.outcome is not None:
        return ws, False
    task = await s.get(Task, ws.task_id) if ws.task_id is not None else None
    if outcome == SessionOutcome.COMPLETED and task_needs_criteria(task) and not criteria_confirmed:
        raise CriteriaCheckRequired("compare the result with the definition of done first")
    now = ctx.clock.now()
    ws.outcome = str(outcome)
    if note:
        ws.result_note = note
    ws.updated_at = now
    await s.flush()
    if task is not None:
        await task_service.record_work_outcome(
            s, ctx, task_id=task.id, outcome=outcome, actor=actor, session_id=ws.id, note=note
        )
    if ws.intervention_id is not None and outcome != SessionOutcome.STARTED:
        await interventions_basic.record_intervention_outcome(
            s,
            intervention_id=ws.intervention_id,
            result=InterventionResult(str(outcome)),
            session_id=ws.id,
            now=now,
            note=note,
        )
    await audit(
        s,
        now=now,
        actor=actor,
        action="session.outcome",
        entity="work_session",
        entity_id=ws.id,
        details={"outcome": str(outcome), "criteria": criteria_confirmed},
    )
    return ws, True


async def save_evidence(
    s: AsyncSession, ctx: AppContext, *, session_id: int, evidence: str
) -> WorkSession:
    """Store what the user gave as a result (text/link/photo ref). Never verified by
    the bot: an open IDE on a screenshot is not proof of done (FR-06)."""
    ws = await _require(s, session_id)
    evidence = evidence.strip()
    ws.evidence = evidence if not ws.evidence else f"{ws.evidence}\n{evidence}"
    ws.updated_at = ctx.clock.now()
    await s.flush()
    return ws


# --------------------------------------------------------------------------- retro


@dataclass(frozen=True)
class RetroBreak:
    started_at: datetime
    ended_at: datetime
    time_source: TimeSource = TimeSource.RETRO


def _iso(dt: datetime | None) -> str | None:
    return None if dt is None else dt.isoformat()


async def create_retro_session(
    s: AsyncSession,
    ctx: AppContext,
    *,
    task_id: int | None,
    started_at: datetime,
    ended_at: datetime,
    time_source: TimeSource,
    confidence: str | None,
    breaks: Sequence[RetroBreak] = (),
    actor: Actor = Actor.USER,
    note: str | None = None,
    session_id: int | None = None,
) -> WorkSession:
    """A forgotten session reported afterwards (scenario F, acceptance 16).

    Stored as finished with time_source retro/approximate + confidence and a
    `created_retro` correction entry. `session_id` fills a planned (unconfirmed)
    session instead of creating a new one.
    """
    time_source = TimeSource(time_source)
    if time_source == TimeSource.EXACT:
        raise ValueError("retro entries are never exact")
    if confidence is not None and confidence not in CONFIDENCE_VALUES:
        raise ValueError(f"bad confidence {confidence!r}")
    now = ctx.clock.now()
    _validate_interval(started_at, ended_at, now)
    for b in breaks:
        if not (started_at <= b.started_at < b.ended_at <= ended_at):
            raise ValueError("break outside the session")
    ws: WorkSession | None = None
    if session_id is not None:
        ws = await _require(s, session_id)
        if ws.status != SessionStatus.UNCONFIRMED:
            raise InvalidSessionState(f"session {session_id} is {ws.status}")
        task_id = ws.task_id if task_id is None else task_id
    if ws is None:
        ws = WorkSession(created_at=now, mode=str(SessionMode.FREE))
        s.add(ws)
    ws.task_id = task_id
    ws.status = SessionStatus.FINISHED
    ws.started_at = started_at
    ws.ended_at = ended_at
    ws.time_source = str(time_source)
    ws.confidence = confidence
    ws.updated_at = now
    if note:
        ws.result_note = note
    await s.flush()
    for b in breaks:
        s.add(
            SessionPause(
                session_id=ws.id,
                started_at=b.started_at,
                ended_at=b.ended_at,
                time_source=str(b.time_source),
            )
        )
    s.add(
        SessionCorrection(
            session_id=ws.id,
            field="created_retro",
            old_value=None,
            new_value=json.dumps(
                {
                    "started_at": _iso(started_at),
                    "ended_at": _iso(ended_at),
                    "breaks": [[_iso(b.started_at), _iso(b.ended_at)] for b in breaks],
                    "time_source": str(time_source),
                    "confidence": confidence,
                },
                ensure_ascii=False,
            ),
            actor=str(actor),
            reason=note or "retro entry",
            created_at=now,
        )
    )
    await s.flush()
    await audit(
        s,
        now=now,
        actor=actor,
        action="session.retro_created",
        entity="work_session",
        entity_id=ws.id,
        details={"task_id": task_id, "time_source": str(time_source)},
    )
    return ws


def _validate_interval(start: datetime, end: datetime, now: datetime) -> None:
    if end <= start:
        raise ValueError("end must be after start")
    if end > now + timedelta(minutes=1):
        raise ValueError("end is in the future")
    if end - start > MAX_SESSION:
        raise ValueError("session longer than 24 hours")


async def correct_session_times(
    s: AsyncSession,
    ctx: AppContext,
    *,
    session_id: int,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
    actor: Actor = Actor.USER,
    reason: str | None = None,
) -> list[SessionCorrection]:
    """Edit a finished session's times. Every change is journaled (old/new, actor,
    reason); the time source is kept as is (an edited exact session stays exact but
    is flagged `corrected` in stats)."""
    ws = await _require(s, session_id)
    if ws.status != SessionStatus.FINISHED or ws.started_at is None or ws.ended_at is None:
        raise InvalidSessionState("only finished sessions can be corrected")
    now = ctx.clock.now()
    new_start = started_at or ws.started_at
    new_end = ended_at or ws.ended_at
    _validate_interval(new_start, new_end, now)
    out: list[SessionCorrection] = []
    for name, old, new in (
        ("started_at", ws.started_at, new_start),
        ("ended_at", ws.ended_at, new_end),
    ):
        if old == new:
            continue
        row = SessionCorrection(
            session_id=ws.id,
            field=name,
            old_value=_iso(old),
            new_value=_iso(new),
            actor=str(actor),
            reason=reason,
            created_at=now,
        )
        s.add(row)
        out.append(row)
    if out:
        ws.started_at = new_start
        ws.ended_at = new_end
        ws.updated_at = now
        # Pauses outside the new interval no longer apply; clip them.
        for p in await load_pauses(s, ws.id):
            p_end = p.ended_at or new_end
            if p_end <= new_start or p.started_at >= new_end:
                await s.delete(p)
            else:
                p.started_at = max(p.started_at, new_start)
                p.ended_at = min(p_end, new_end)
        await s.flush()
        await audit(
            s,
            now=now,
            actor=actor,
            action="session.corrected",
            entity="work_session",
            entity_id=ws.id,
            details={"fields": [c.field for c in out]},
        )
    return out


# --------------------------------------------------------------------------- stats


@dataclass(frozen=True)
class SessionStat:
    session_id: int
    task_id: int | None
    mode: str
    status: str
    started_at: datetime
    ended_at: datetime | None
    gross_minutes: float
    pause_minutes: float
    net_minutes: float
    time_source: str
    confidence: str | None
    outcome: str | None
    corrected: bool
    live: bool

    @property
    def is_exact(self) -> bool:
        """Only live timer sessions are exact; retro/approximate never are."""
        return self.time_source == TimeSource.EXACT

    @property
    def source_label(self) -> str:
        label = SOURCE_LABELS.get(self.time_source, self.time_source)
        if self.confidence and not self.is_exact:
            label += f", уверенность: {CONFIDENCE_LABELS.get(self.confidence, self.confidence)}"
        return label + (" (исправлено)" if self.corrected else "")


@dataclass
class SessionStats:
    items: list[SessionStat] = field(default_factory=list)
    unconfirmed_count: int = 0  # planned, never confirmed: NOT worked time

    def minutes_by_source(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for it in self.items:
            out[it.time_source] = out.get(it.time_source, 0.0) + it.net_minutes
        return out

    @property
    def total_minutes(self) -> float:
        return sum(it.net_minutes for it in self.items)

    @property
    def exact_minutes(self) -> float:
        return sum(it.net_minutes for it in self.items if it.is_exact)

    @property
    def not_exact_minutes(self) -> float:
        return sum(it.net_minutes for it in self.items if not it.is_exact)

    @property
    def all_exact(self) -> bool:
        return all(it.is_exact for it in self.items)


async def session_stats(
    s: AsyncSession,
    *,
    start: datetime,
    end: datetime,
    now: datetime,
    task_id: int | None = None,
) -> SessionStats:
    """Worked time of sessions that started in [start, end), for analytics/reports.

    Unconfirmed sessions are only counted in `unconfirmed_count`, never as minutes
    (acceptance 15). Every item carries time_source/confidence/corrected so reports
    never present retro/approximate time as an exact timer (acceptance 16).
    """
    stmt = select(WorkSession).where(
        WorkSession.status.in_(
            [str(SessionStatus.ACTIVE), str(SessionStatus.PAUSED), str(SessionStatus.FINISHED)]
        ),
        WorkSession.started_at.is_not(None),
        WorkSession.started_at >= start,
        WorkSession.started_at < end,
    )
    unconf = select(WorkSession).where(
        WorkSession.status == str(SessionStatus.UNCONFIRMED),
        or_(
            (WorkSession.planned_start_at >= start) & (WorkSession.planned_start_at < end),
            WorkSession.planned_start_at.is_(None)
            & (WorkSession.created_at >= start)
            & (WorkSession.created_at < end),
        ),
    )
    if task_id is not None:
        stmt = stmt.where(WorkSession.task_id == task_id)
        unconf = unconf.where(WorkSession.task_id == task_id)
    rows = (await s.execute(stmt.order_by(WorkSession.started_at))).scalars().all()
    unconfirmed = (await s.execute(unconf)).scalars().all()
    out = SessionStats(unconfirmed_count=len(unconfirmed))
    ids = [r.id for r in rows]
    corrected: set[int] = set()
    if ids:
        corrected = set(
            (
                await s.execute(
                    select(SessionCorrection.session_id).where(
                        SessionCorrection.session_id.in_(ids),
                        SessionCorrection.field != "created_retro",
                    )
                )
            )
            .scalars()
            .all()
        )
    for ws in rows:
        assert ws.started_at is not None
        pauses = await load_pauses(s, ws.id)
        stop = ws.ended_at or now
        gross = max(0.0, (stop - ws.started_at).total_seconds())
        net = net_seconds(ws, pauses, now)
        out.items.append(
            SessionStat(
                session_id=ws.id,
                task_id=ws.task_id,
                mode=ws.mode,
                status=ws.status,
                started_at=ws.started_at,
                ended_at=ws.ended_at,
                gross_minutes=gross / 60,
                pause_minutes=pause_seconds(ws, pauses, now) / 60,
                net_minutes=net / 60,
                time_source=ws.time_source,
                confidence=ws.confidence,
                outcome=ws.outcome,
                corrected=ws.id in corrected,
                live=ws.status in LIVE_STATUSES,
            )
        )
    return out


# --------------------------------------------------------------------------- recommendation


@dataclass(frozen=True)
class ModeRecommendation:
    mode: SessionMode
    reason: str


async def recommend_mode(
    s: AsyncSession,
    ctx: AppContext,
    *,
    task_id: int | None,
    obstacle: ObstacleReason | str | None = None,
) -> ModeRecommendation | None:
    """Simple rules (stage A): after an "unclear / too big / unpleasant" obstacle ->
    short start; otherwise the user's last choice; no history -> no recommendation."""
    now = ctx.clock.now()
    if obstacle is None and task_id is not None:
        profile = await get_profile(s)
        tz = profile.timezone if profile else None
        day_start, _ = local_day_bounds(local_date(now, tz), tz)
        iv = (
            (
                await s.execute(
                    select(Intervention)
                    .where(Intervention.task_id == task_id, Intervention.created_at >= day_start)
                    .order_by(Intervention.created_at.desc())
                    .limit(1)
                )
            )
            .scalars()
            .first()
        )
        if iv is not None:
            obstacle = iv.reason_category
    if obstacle is not None and str(obstacle) in SHORT_START_OBSTACLES:
        return ModeRecommendation(
            SessionMode.SHORT_START, "когда непонятно, много или неприятно — легче начать с малого"
        )
    last = (
        (
            await s.execute(
                select(WorkSession.mode)
                .where(
                    WorkSession.status.in_(
                        [str(SessionStatus.FINISHED), *[str(x) for x in LIVE_STATUSES]]
                    ),
                    WorkSession.time_source == str(TimeSource.EXACT),
                    WorkSession.started_at.is_not(None),
                )
                .order_by(WorkSession.started_at.desc())
                .limit(1)
            )
        )
        .scalars()
        .first()
    )
    if last is not None:
        return ModeRecommendation(SessionMode(last), "ты выбирал этот режим в прошлый раз")
    return None


# --------------------------------------------------------------------------- start candidates


@dataclass(frozen=True)
class StartCandidate:
    session_id: int | None
    task_id: int | None
    mode: SessionMode | None


async def find_start_candidate(s: AsyncSession, ctx: AppContext) -> StartCandidate | None:
    """What "начал / начинаю" refers to: a planned unconfirmed session near now, or
    the task of a pre-task / start-delay prompt sent recently."""
    now = ctx.clock.now()
    planned = (
        (
            await s.execute(
                select(WorkSession).where(
                    WorkSession.status == str(SessionStatus.UNCONFIRMED),
                    or_(
                        (WorkSession.planned_start_at >= now - UNCONFIRMED_WINDOW)
                        & (WorkSession.planned_start_at <= now + UNCONFIRMED_WINDOW),
                        WorkSession.planned_start_at.is_(None)
                        & (WorkSession.created_at >= now - UNCONFIRMED_WINDOW),
                    ),
                )
            )
        )
        .scalars()
        .all()
    )
    if planned:
        best = min(planned, key=lambda w: abs((w.planned_start_at or w.created_at) - now))
        return StartCandidate(best.id, best.task_id, SessionMode(best.mode))
    row = (
        await s.execute(
            select(ReminderSchedule.task_id)
            .join(OutboundMessage, OutboundMessage.reminder_id == ReminderSchedule.id)
            .where(
                ReminderSchedule.task_id.is_not(None),
                ReminderSchedule.category.in_(
                    [str(MessageCategory.PRE_TASK), str(MessageCategory.START_DELAY)]
                ),
                OutboundMessage.created_at >= now - RECENT_PROMPT,
            )
            .order_by(OutboundMessage.created_at.desc())
            .limit(1)
        )
    ).first()
    prompted_task: int | None = row[0] if row is not None else None
    if prompted_task is not None:
        return StartCandidate(None, prompted_task, None)
    return None


# --------------------------------------------------------------------------- texts / keyboards


def _btn(text: str, cb: SessionCb) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=cb.pack())


def control_keyboard(ws: WorkSession) -> InlineKeyboardMarkup:
    if ws.status == SessionStatus.PAUSED:
        first = _btn("▶️ Продолжить", SessionCb(action="resume", session_id=ws.id))
    else:
        first = _btn("⏸ Пауза", SessionCb(action="pause", session_id=ws.id))
    return InlineKeyboardMarkup(
        inline_keyboard=[[first, _btn("⏹ Завершить", SessionCb(action="stop", session_id=ws.id))]]
    )


def outcome_keyboard(ws: WorkSession) -> InlineKeyboardMarkup:
    rows = [
        [_btn(label, SessionCb(action="outcome", session_id=ws.id, arg=str(o)))]
        for o, label in OUTCOME_LABELS.items()
    ]
    rows.append([_btn("✏️ Исправить время", SessionCb(action="edit", session_id=ws.id))])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def mode_label(ws: WorkSession) -> str:
    mode = SessionMode(ws.mode)
    if mode == SessionMode.POMODORO:
        return f"Помодоро {ws.work_minutes}/{ws.break_minutes} мин"
    if mode == SessionMode.SHORT_START:
        return f"Короткий старт {ws.work_minutes} мин"
    if ws.planned_minutes:
        return f"Свободный таймер, цель {ws.planned_minutes} мин"
    return "Свободный таймер"


async def task_title(s: AsyncSession, task_id: int | None) -> str | None:
    if task_id is None:
        return None
    t = await s.get(Task, task_id)
    return t.title if t is not None else None


async def describe_session(s: AsyncSession, ctx: AppContext, ws: WorkSession) -> str:
    now = ctx.clock.now()
    pauses = await load_pauses(s, ws.id)
    title = await task_title(s, ws.task_id)
    head = f"«{title}»" if title else "без задачи"
    phase = current_phase(ws, pauses, now)
    state = {
        "work": "идёт работа",
        "overtime": "идёт работа",
        "break": "перерыв",
        "paused": "на паузе",
    }.get(phase, phase)
    lines = [
        f"⏱ Сессия: {head}",
        f"Режим: {mode_label(ws)} · {state}",
        f"Чистое время: {fmt_minutes(net_seconds(ws, pauses, now) / 60)}",
    ]
    exp = expected_timer(ws, pauses)
    if exp is not None and exp.due_at > now:
        profile = await get_profile(s)
        tz = profile.timezone if profile else None
        lines.append(f"Сигнал: в {to_local(exp.due_at, tz):%H:%M}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- renderer


async def render_timer_signal(ctx: AppContext, reminder: ReminderSchedule) -> Outgoing | None:
    """Text + buttons for a due timer signal; None when the session moved on."""
    if reminder.session_id is None:
        return None
    async with ctx.db.transaction() as s:
        ws = await s.get(WorkSession, reminder.session_id)
        if ws is None:
            return None
        pauses = await load_pauses(s, ws.id)
        exp = expected_timer(ws, pauses)
        if exp is None or exp.key != reminder.dedup_key:
            return None
        title = await task_title(s, ws.task_id)
    sid = ws.id
    suffix = f"\nЗадача: «{title}»" if title else ""
    stop = _btn("⏹ Завершить", SessionCb(action="stop", session_id=sid))
    if exp.phase == "work_end":
        text = f"🍅 Помодоро готово ({ws.work_minutes} мин). Время перерыва — {ws.break_minutes} мин.{suffix}"
        kb = [
            [_btn(f"☕ Перерыв {ws.break_minutes} мин", SessionCb(action="brk", session_id=sid))],
            [_btn("▶️ Работать дальше", SessionCb(action="skipbrk", session_id=sid))],
            [stop],
        ]
    elif exp.phase == "break_end":
        text = f"☕ Перерыв окончен. Продолжаем?{suffix}"
        kb = [[_btn("▶️ Продолжить", SessionCb(action="resume", session_id=sid))], [stop]]
    elif exp.phase == "short_end":
        text = f"⏱ {ws.planned_minutes} мин позади — старт есть! Продолжаем?{suffix}"
        kb = [
            [_btn(f"➕ Ещё {ws.work_minutes} мин", SessionCb(action="more", session_id=sid))],
            [_btn("▶️ Свободный режим", SessionCb(action="tofree", session_id=sid))],
            [stop],
        ]
    else:
        text = f"⏰ Прошло {ws.planned_minutes} мин — цель по времени достигнута.{suffix}"
        kb = [[_btn("▶️ Продолжаю", SessionCb(action="ack", session_id=sid))], [stop]]
    return Outgoing(text=text, reply_markup=InlineKeyboardMarkup(inline_keyboard=kb))


# --------------------------------------------------------------------------- restart


async def restore_sessions(ctx: AppContext) -> int:
    """Scenario G / acceptance 2: reload live sessions after a restart and recompute
    their timer signals from the saved started_at/pauses. Idempotent (dedup keys):
    calling it twice never duplicates a signal. A boundary that passed during the
    downtime gets its signal with the original due time; the gate folds stale ones.
    Returns the number of live sessions found."""
    async with ctx.db.transaction() as s:
        rows = (
            (
                await s.execute(
                    select(WorkSession).where(
                        WorkSession.status.in_([str(x) for x in LIVE_STATUSES])
                    )
                )
            )
            .scalars()
            .all()
        )
        for ws in rows:
            await resync_timer(s, ctx, ws, allow_past=True)
    if rows:
        log.info("restored %d work session(s)", len(rows))
    return len(rows)


# --------------------------------------------------------------------------- retro parsing

_DASHES = re.compile(r"[–—−]")
_T = r"(\d{1,2})(?:[:.](\d{2}))?"
_INTERVAL = re.compile(rf"(?:\bс\s*)?{_T}\s*(?:-|\bдо\b|\bпо\b)\s*{_T}(?!\s*(?:ч|мин|м\b))")
_APPROX_WORDS = ("примерно", "около", "где-то", "приблизительно", "~", "наверное", "вроде")


@dataclass(frozen=True)
class ParsedInterval:
    """Result of parsing a retro answer: either explicit clock times or a duration."""

    start: datetime | None
    end: datetime | None
    duration: timedelta | None
    explicit: bool  # clock times were given
    approximate: bool  # hedge words ("примерно", "около") were used

    @property
    def time_source(self) -> TimeSource:
        return (
            TimeSource.RETRO if self.explicit and not self.approximate else TimeSource.APPROXIMATE
        )


def _norm(text: str) -> str:
    t = _DASHES.sub("-", text.lower().replace("ё", "е"))
    return re.sub(r"\s+", " ", t).strip()


def _day_offset(t: str) -> int | None:
    if "позавчера" in t:
        return -2
    if "вчера" in t:
        return -1
    if "сегодня" in t:
        return 0
    return None


def _hm(h: str, m: str | None) -> time | None:
    hh, mm = int(h), int(m or 0)
    if hh == 24 and mm == 0:
        hh = 0
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None
    return time(hh, mm)


def parse_duration(text: str) -> timedelta | None:
    """'2 часа', '1.5 ч', 'полтора часа', 'полчаса', '90 минут', 'час', '2ч 30м'."""
    t = _norm(text)
    total = 0.0
    found = False
    if re.search(r"\bполтора\s+час", t):
        total += 90
        found = True
    elif re.search(r"\bполчаса\b", t):
        total += 30
        found = True
    else:
        mh = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:ч\b|час|часа|часов)", t)
        if mh:
            total += float(mh.group(1).replace(",", ".")) * 60
            found = True
        elif re.search(r"\bпар[ау] час", t):
            total += 120
            found = True
        elif re.search(r"(?:^|\s)час(?:ик)?\b", t):
            total += 60
            found = True
    mm = re.search(r"(\d+)\s*(?:м\b|мин)", t)
    if mm:
        total += int(mm.group(1))
        found = True
    if not found or total <= 0:
        return None
    return timedelta(minutes=total)


def parse_interval(text: str, *, now: datetime, tz: str | None) -> ParsedInterval | None:
    """Parse '14:00-16:30', 'с 14 до 16', 'вчера 10-12', 'с 2 до 4 дня', '2 часа',
    'примерно час утром'. Explicit times are placed today (or yesterday when they
    would be in the future). Returns None when nothing usable was found."""
    t = _norm(text)
    approx = any(w in t for w in _APPROX_WORDS)
    m = _INTERVAL.search(t)
    if m:
        t1, t2 = _hm(m.group(1), m.group(2)), _hm(m.group(3), m.group(4))
        if t1 is not None and t2 is not None:
            tail = t[m.end() :]
            if re.match(r"\s*(?:дня|вечера)\b", tail):
                if t1.hour < 12 and m.group(2) is None:
                    t1 = t1.replace(hour=t1.hour + 12)
                if t2.hour < 12:
                    t2 = t2.replace(hour=t2.hour + 12)
            offset = _day_offset(t)
            day = local_date(now, tz) + timedelta(days=offset or 0)
            start = resolve_local(day, t1, tz)
            end_day = day if t2 > t1 else day + timedelta(days=1)
            end = resolve_local(end_day, t2, tz)
            if offset is None and end > now + timedelta(minutes=1):
                start = resolve_local(day - timedelta(days=1), t1, tz)
                end = resolve_local(end_day - timedelta(days=1), t2, tz)
            return ParsedInterval(start, end, end - start, True, approx)
    d = parse_duration(t)
    if d is not None:
        return ParsedInterval(None, None, d, False, True)
    return None


def parse_clock(text: str, *, now: datetime, tz: str | None) -> datetime | None:
    """End time answer: 'только что', 'сейчас', '12:30', 'в 12'. Future -> yesterday."""
    t = _norm(text)
    if any(w in t for w in ("только что", "сейчас", "щас", "прямо сейчас")):
        return now
    m = re.search(rf"(?:\bв\s*)?{_T}\b", t)
    if not m:
        return None
    hm = _hm(m.group(1), m.group(2))
    if hm is None:
        return None
    offset = _day_offset(t)
    day = local_date(now, tz) + timedelta(days=offset or 0)
    moment = resolve_local(day, hm, tz)
    if offset is None and moment > now + timedelta(minutes=1):
        moment = resolve_local(day - timedelta(days=1), hm, tz)
    return moment


def parse_breaks(
    text: str, *, start: datetime, end: datetime, tz: str | None
) -> list[RetroBreak] | None:
    """'без перерывов' / 'нет' / '0' -> []; '13:00-13:20' -> explicit break;
    '15', '15 минут', 'полчаса' -> one break of that length placed mid-session
    (marked approximate). None = not understood or does not fit the session."""
    t = _norm(text)
    if t in ("0", "нет", "не было", "без перерывов", "без перерыва", "никаких") or t.startswith(
        "без перерыв"
    ):
        return []
    breaks: list[RetroBreak] = []
    for m in _INTERVAL.finditer(t):
        t1, t2 = _hm(m.group(1), m.group(2)), _hm(m.group(3), m.group(4))
        if t1 is None or t2 is None:
            return None
        day = local_date(start, tz)
        b_start = resolve_local(day, t1, tz)
        if b_start < start:
            b_start = resolve_local(day + timedelta(days=1), t1, tz)
        b_end = resolve_local(local_date(b_start, tz), t2, tz)
        if b_end <= b_start:
            b_end = resolve_local(local_date(b_start, tz) + timedelta(days=1), t2, tz)
        if not (start <= b_start < b_end <= end):
            return None
        breaks.append(RetroBreak(b_start, b_end, TimeSource.RETRO))
    if breaks:
        return breaks
    d = parse_duration(t)
    if d is None and re.fullmatch(r"\d{1,3}", t):
        d = timedelta(minutes=int(t))
    if d is None:
        return None
    if d <= timedelta(0):
        return []
    if d >= end - start:
        return None
    mid = start + (end - start) / 2
    b_start = mid - d / 2
    return [RetroBreak(b_start, b_start + d, TimeSource.APPROXIMATE)]


# --------------------------------------------------------------------------- NL phrases

_PHRASES: dict[str, frozenset[str]] = {
    "pause": frozenset(
        {
            "пауза",
            "на паузу",
            "поставь на паузу",
            "поставь паузу",
            "пауза пожалуйста",
            "сделаю паузу",
            "перерыв",
            "беру перерыв",
            "ставлю на паузу",
        }
    ),
    "resume": frozenset(
        {
            "продолжаю",
            "продолжить",
            "продолжим",
            "продолжаем",
            "вернулся",
            "вернулась",
            "я вернулся",
            "я вернулась",
            "снова работаю",
            "возобнови",
            "возобновить",
            "сними с паузы",
            "снять с паузы",
        }
    ),
    "stop": frozenset(
        {
            "стоп",
            "закончил",
            "закончила",
            "я закончил",
            "я закончила",
            "готово",
            "завершить",
            "завершил",
            "завершила",
            "заканчиваю",
            "хватит",
            "стоп таймер",
            "останови таймер",
            "закончить",
        }
    ),
    "start": frozenset(
        {
            "начал",
            "начала",
            "я начал",
            "я начала",
            "начинаю",
            "приступаю",
            "приступил",
            "приступила",
            "сажусь",
            "сажусь работать",
            "начинаю работать",
            "поехали",
        }
    ),
}


def classify_phrase(text: str | None) -> str | None:
    """Keyword control for stage A: only short unambiguous phrases are recognised;
    everything else returns None and falls through to other handlers."""
    if not text or text.startswith("/") or len(text) > 40:
        return None
    t = re.sub(r"[!.,?…)(]+", " ", _norm(text))
    t = re.sub(r"\s+", " ", t).strip()
    if not t or len(t.split()) > 4:
        return None
    for intent, phrases in _PHRASES.items():
        if t in phrases:
            return intent
    return None


# --------------------------------------------------------------------------- wiring


def register(ctx: AppContext) -> None:
    """Register the timer-signal renderer."""
    register_renderer(ctx, MessageCategory.TIMER_SIGNAL, render_timer_signal)
