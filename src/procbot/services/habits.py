"""HabitService: habits, schedules, daily occurrences, stability (FR-09).

Owner of the `habits` and `habit_occurrences` tables. Rules:

- habits are separate from one-off tasks; a habit has a schedule
  (`daily` / `weekly` on given weekdays / flexible `times_per_week`), an optional
  preferred local time and a flexible target ({"unit", "amount", "min_amount"}:
  the minimum version counts as `partial`);
- one occurrence row per habit and local day (unique); marking the same status
  again is a no-op, a different status is a correction (audited);
- `skipped` is a conscious decision; `missed` is a scheduled day that passed
  without a mark (derived, no row needed) or a row the user wrote when telling
  what happened (reason code in the note);
- a single miss never resets anything: progress is regularity over a window
  (share of scheduled days kept) and how fast the owner comes back after a miss.
  There are no streaks to zero;
- changing a habit's volume, schedule or time by the bot/AI is never autonomous
  (FR-15): such changes are CHANGE_HABIT proposals -> pending approval. A change
  the owner makes in the habits menu is an explicit command (origin USER);
- volume increase is only *suggested*: on the owner's request, or in the weekly
  review when the stability threshold is configured (undecided: None = the
  weekly review suggests nothing).

Public helpers for other modules: `due_habits` (planner), `habit_stability`
(analytics), `day_section` (day view).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

from aiogram.types import InlineKeyboardButton
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.models import Goal, Habit, HabitOccurrence, PendingApproval
from procbot.domain.actions import ActionType, ChangeHabit, CreateHabit
from procbot.domain.enums import (
    Actor,
    ApprovalStatus,
    EntityStatus,
    HabitOccurrenceStatus,
    TimeSource,
)
from procbot.services import actions
from procbot.services.actions import ActionError, ApplyOutcome
from procbot.services.audit import audit
from procbot.services.profile import get_profile
from procbot.telegram.callbacks import HabitCb
from procbot.timeutil import local_date, resolve_local

# --------------------------------------------------------------------------- constants

DAILY = "daily"
WEEKLY = "weekly"
TIMES_PER_WEEK = "times_per_week"
SCHEDULE_TYPES: frozenset[str] = frozenset({DAILY, WEEKLY, TIMES_PER_WEEK})

DONE = HabitOccurrenceStatus.DONE.value
PARTIAL = HabitOccurrenceStatus.PARTIAL.value
SKIPPED = HabitOccurrenceStatus.SKIPPED.value
MISSED = HabitOccurrenceStatus.MISSED.value
KEPT: frozenset[str] = frozenset({DONE, PARTIAL})

# How far back the owner may mark a day (later marks are stored as RETRO).
RETRO_DAYS = 14
DEFAULT_WINDOWS: tuple[int, ...] = (14, 28)
# "Enough data" for an honest metric: at least this many counted days/expected.
MIN_DATA_POINTS = 4

# UNCONFIRMED (spec FR-09, docs/open-questions.md): the stability level X from
# which the weekly review may suggest more volume is not decided. None = the
# weekly review suggests nothing; the owner can still ask in the habits menu.
INCREASE_MIN_REGULARITY: float | None = None
# UNCONFIRMED: size of one suggested step (+20% of the amount, at least +1 unit,
# or +1 time per week). Only a suggestion; applied after explicit consent.
INCREASE_STEP_RATIO = 0.2

WEEKDAY_SHORT = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")
WEEKDAY_PARSE: dict[str, int] = {
    "пн": 0,
    "пон": 0,
    "понедельник": 0,
    "вт": 1,
    "вторник": 1,
    "ср": 2,
    "среда": 2,
    "среду": 2,
    "чт": 3,
    "четверг": 3,
    "пт": 4,
    "пятница": 4,
    "пятницу": 4,
    "сб": 5,
    "суббота": 5,
    "субботу": 5,
    "вс": 6,
    "воскресенье": 6,
}

# Quick reasons for a missed day (hypotheses of the owner, not diagnoses).
MISS_REASONS: dict[str, str] = {
    "forgot": "Забыл",
    "no_time": "Не было времени",
    "low_energy": "Мало сил",
    "unwell": "Плохо себя чувствовал",
    "circumstances": "Обстоятельства",
    "no_wish": "Не хотелось",
    "other": "Другое",
}

STATUS_ICON: dict[str | None, str] = {
    DONE: "✅",
    PARTIAL: "◐",
    SKIPPED: "⏭",
    MISSED: "·",
    None: "○",
}
STATUS_LABEL: dict[str, str] = {
    DONE: "сделано",
    PARTIAL: "минимум / частично",
    SKIPPED: "осознанно пропущено",
    MISSED: "не получилось",
}

_NOTE_RE = re.compile(r"^\[(\w+)\]\s*(.*)$", re.S)


class HabitError(ActionError):
    """Habit operation not possible; str() is a short Russian message."""


# --------------------------------------------------------------------------- schedule


@dataclass(frozen=True)
class Schedule:
    type: str
    days: frozenset[int] = frozenset()
    n: int = 0

    @property
    def flexible(self) -> bool:
        return self.type == TIMES_PER_WEEK

    def is_scheduled(self, d: date) -> bool:
        """Fixed schedules: is `d` a habit day? Flexible: every day is eligible."""
        if self.type == WEEKLY:
            return d.weekday() in self.days
        return True

    def as_dict(self) -> dict[str, Any]:
        if self.type == WEEKLY:
            return {"type": WEEKLY, "days": sorted(self.days)}
        if self.type == TIMES_PER_WEEK:
            return {"type": TIMES_PER_WEEK, "n": self.n}
        return {"type": DAILY}


def parse_schedule(raw: dict[str, Any] | None) -> Schedule:
    """Validate a schedule dict (raises HabitError with a Russian message)."""
    raw = dict(raw or {})
    kind = str(raw.get("type") or DAILY)
    if kind not in SCHEDULE_TYPES:
        raise HabitError("Неизвестный тип графика привычки.")
    if kind == WEEKLY:
        days_raw = raw.get("days")
        if not isinstance(days_raw, list | tuple | set | frozenset):
            raise HabitError("Укажи дни недели.")
        try:
            days = frozenset(int(x) for x in days_raw)
        except (TypeError, ValueError):
            raise HabitError("Дни недели указаны неверно.") from None
        if not days or any(d < 0 or d > 6 for d in days):
            raise HabitError("Дни недели указаны неверно.")
        if len(days) == 7:
            return Schedule(DAILY)
        return Schedule(WEEKLY, days=days)
    if kind == TIMES_PER_WEEK:
        try:
            n = int(raw.get("n") or 0)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            n = 0
        if n < 1 or n > 7:
            raise HabitError("Сколько раз в неделю: от 1 до 7.")
        if n == 7:
            return Schedule(DAILY)
        return Schedule(TIMES_PER_WEEK, n=n)
    return Schedule(DAILY)


def habit_schedule(habit: Habit) -> Schedule:
    try:
        return parse_schedule(habit.schedule)
    except HabitError:
        return Schedule(DAILY)


def schedule_text(s: Schedule) -> str:
    if s.type == WEEKLY:
        return "по " + ", ".join(WEEKDAY_SHORT[d] for d in sorted(s.days))
    if s.type == TIMES_PER_WEEK:
        return f"{s.n} раз(а) в неделю, дни свободные"
    return "каждый день"


def parse_weekdays(text: str) -> frozenset[int] | None:
    """ "пн ср пт" / "пн, ср, пт" / "пн-пт" -> weekdays; None if not understood."""
    clean = text.lower().replace(",", " ").replace(";", " ").strip()
    if not clean:
        return None
    out: set[int] = set()
    for token in clean.split():
        if "-" in token or "–" in token:
            a, _, b = token.replace("–", "-").partition("-")
            if a not in WEEKDAY_PARSE or b not in WEEKDAY_PARSE:
                return None
            i, j = WEEKDAY_PARSE[a], WEEKDAY_PARSE[b]
            if i <= j:
                out.update(range(i, j + 1))
            else:
                out.update(list(range(i, 7)) + list(range(0, j + 1)))
        elif token in WEEKDAY_PARSE:
            out.add(WEEKDAY_PARSE[token])
        else:
            return None
    return frozenset(out) or None


# --------------------------------------------------------------------------- target


def _num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise HabitError("Объём должен быть числом.") from None
    if f <= 0 or f > 100_000:
        raise HabitError("Объём должен быть больше нуля.")
    return f


def normalize_target(raw: dict[str, Any] | None) -> dict[str, Any]:
    """{"unit": str|None, "amount": float|None, "min_amount": float|None}."""
    raw = dict(raw or {})
    amount = _num(raw.get("amount"))
    min_amount = _num(raw.get("min_amount"))
    unit = str(raw.get("unit") or "").strip()[:30] or None
    if amount is not None and min_amount is not None and min_amount > amount:
        raise HabitError("Минимум не может быть больше цели.")
    out: dict[str, Any] = {}
    if amount is not None:
        out["amount"] = _int_if_whole(amount)
    if min_amount is not None:
        out["min_amount"] = _int_if_whole(min_amount)
    if unit and (amount is not None or min_amount is not None):
        out["unit"] = unit
    return out


def _int_if_whole(f: float) -> float | int:
    return int(f) if float(f).is_integer() else round(f, 2)


_TARGET_NUM_RE = re.compile(r"(\d+(?:[.,]\d+)?)")
_TARGET_MIN_RE = re.compile(r"(?:миним\w*|мин\.?\s*объ\w*|хотя\s+бы)\s*(\d+(?:[.,]\d+)?)")


def parse_target(text: str) -> dict[str, Any] | None:
    """ "10 мин, минимум 2" / "20 отжиманий минимум 5" / "15" -> target dict.
    None when no number was found."""
    low = " ".join(text.lower().split())
    min_amount: float | None = None
    m = _TARGET_MIN_RE.search(low)
    if m:
        min_amount = float(m.group(1).replace(",", "."))
        low = (low[: m.start()] + low[m.end() :]).strip(" ,;")
    m = _TARGET_NUM_RE.search(low)
    if not m:
        if min_amount is None:
            return None
        return normalize_target({"min_amount": min_amount})
    amount = float(m.group(1).replace(",", "."))
    unit = low[m.end() :].strip(" ,;.()") or None
    if unit:
        unit = unit.split(",")[0].strip()
    return normalize_target({"amount": amount, "min_amount": min_amount, "unit": unit})


def fmt_num(v: Any) -> str:
    return str(_int_if_whole(float(v))) if v is not None else ""


def target_text(target: dict[str, Any] | None) -> str:
    t = target or {}
    unit = t.get("unit") or ""
    unit_sfx = f" {unit}" if unit else ""
    parts: list[str] = []
    if t.get("amount") is not None:
        parts.append(f"{fmt_num(t['amount'])}{unit_sfx}")
    if t.get("min_amount") is not None:
        mn = f"минимум {fmt_num(t['min_amount'])}{unit_sfx}"
        parts.append(f"({mn})" if parts else mn)
    return " ".join(parts) if parts else "без объёма"


def is_minutes_unit(unit: str | None) -> bool:
    return bool(unit) and str(unit).lower().startswith(("мин", "min"))


def classify_amount(target: dict[str, Any] | None, amount: float) -> str:
    """Status for a reported amount: the full target -> done, otherwise partial
    (the minimum version counts, and something is never "nothing")."""
    t = target or {}
    goal = t.get("amount")
    if goal is not None and amount >= float(goal):
        return DONE
    if goal is None and t.get("min_amount") is not None and amount >= float(t["min_amount"]):
        return DONE
    if goal is None and t.get("min_amount") is None:
        return DONE
    return PARTIAL


def parse_hhmm(value: str | None) -> time | None:
    if value is None or value == "":
        return None
    m = re.fullmatch(r"\s*(\d{1,2})[:.](\d{2})\s*", value)
    if not m:
        raise HabitError("Время в формате ЧЧ:ММ, например 07:30.")
    hh, mm = int(m.group(1)), int(m.group(2))
    if hh > 23 or mm > 59:
        raise HabitError("Время в формате ЧЧ:ММ, например 07:30.")
    return time(hh, mm)


# --------------------------------------------------------------------------- basics


async def owner_tz(session: AsyncSession) -> str:
    profile = await get_profile(session)
    return (profile.timezone if profile is not None else None) or "UTC"


async def local_today(session: AsyncSession, ctx: AppContext) -> date:
    return local_date(ctx.clock.now(), await owner_tz(session))


async def get_habit(session: AsyncSession, habit_id: int) -> Habit | None:
    return await session.get(Habit, habit_id)


async def require_habit(session: AsyncSession, habit_id: int) -> Habit:
    row = await session.get(Habit, habit_id)
    if row is None:
        raise HabitError(f"Привычка #{habit_id} не найдена.")
    return row


async def list_habits(
    session: AsyncSession, *, statuses: Sequence[str] | None = (EntityStatus.ACTIVE,)
) -> list[Habit]:
    stmt = select(Habit).order_by(Habit.id)
    if statuses is not None:
        stmt = stmt.where(Habit.status.in_([str(s) for s in statuses]))
    return list((await session.execute(stmt)).scalars())


def _clean_title(title: str) -> str:
    clean = " ".join(str(title).split()).strip()
    if not clean:
        raise HabitError("Пустое название привычки.")
    return clean[:300]


def _pauses(habit: Habit) -> list[dict[str, Any]]:
    raw = (habit.schedule or {}).get("pauses")
    return list(raw) if isinstance(raw, list) else []


def is_paused_on(habit: Habit, d: date) -> bool:
    """Paused ranges are [from, to) in local dates; `to` None = still paused."""
    for p in _pauses(habit):
        try:
            start = date.fromisoformat(str(p["from"]))
            end = date.fromisoformat(str(p["to"])) if p.get("to") else None
        except (KeyError, ValueError):
            continue
        if start <= d and (end is None or d < end):
            return True
    return False


def start_date(habit: Habit, tz: str) -> date:
    return local_date(habit.created_at, tz)


def _snapshot(habit: Habit) -> dict[str, Any]:
    return {
        "title": habit.title,
        "schedule": habit_schedule(habit).as_dict(),
        "target": dict(habit.target or {}),
        "preferred_time": habit.preferred_time.strftime("%H:%M") if habit.preferred_time else None,
        "goal_id": habit.goal_id,
        "status": habit.status,
    }


async def create_habit(
    session: AsyncSession,
    ctx: AppContext,
    *,
    title: str,
    actor: Actor,
    schedule: dict[str, Any] | None = None,
    target: dict[str, Any] | None = None,
    preferred_time: time | None = None,
    goal_id: int | None = None,
    description: str | None = None,
) -> Habit:
    sched = parse_schedule(schedule)
    tgt = normalize_target(target)
    if goal_id is not None and await session.get(Goal, goal_id) is None:
        raise HabitError(f"Цель #{goal_id} не найдена.")
    now = ctx.clock.now()
    row = Habit(
        title=_clean_title(title),
        description=description,
        goal_id=goal_id,
        schedule=sched.as_dict(),
        preferred_time=preferred_time,
        target=tgt,
        status=EntityStatus.ACTIVE.value,
        created_at=now,
        updated_at=now,
    )
    session.add(row)
    await session.flush()
    await audit(
        session,
        now=now,
        actor=actor,
        action="habit.created",
        entity="habit",
        entity_id=row.id,
        details=_snapshot(row),
    )
    return row


_UNSET: Any = object()


async def update_habit(
    session: AsyncSession,
    ctx: AppContext,
    habit_id: int,
    *,
    actor: Actor,
    reason: str | None = None,
    title: str | None = None,
    schedule: dict[str, Any] | None = None,
    target: dict[str, Any] | None = None,
    preferred_time: time | None = _UNSET,
    goal_id: int | None = _UNSET,
) -> tuple[Habit, dict[str, Any]]:
    """Apply changes; returns (habit, {"before":..., "after":...}) of what changed.

    Callers: the owner's explicit edit (actor USER) or an approved CHANGE_HABIT
    card. The bot never calls this on its own initiative (FR-15)."""
    habit = await require_habit(session, habit_id)
    if habit.status == EntityStatus.ARCHIVED:
        raise HabitError("Привычка в архиве — сначала верни её.")
    before = _snapshot(habit)
    if title is not None:
        habit.title = _clean_title(title)
    if schedule is not None:
        sched = parse_schedule(schedule)
        new_sched = sched.as_dict()
        pauses = _pauses(habit)
        if pauses:
            new_sched["pauses"] = pauses
        habit.schedule = new_sched
    if target is not None:
        habit.target = normalize_target(target)
    if preferred_time is not _UNSET:
        habit.preferred_time = preferred_time
    if goal_id is not _UNSET:
        if goal_id is not None and await session.get(Goal, goal_id) is None:
            raise HabitError(f"Цель #{goal_id} не найдена.")
        habit.goal_id = goal_id
    after = _snapshot(habit)
    changed = {k: (before[k], after[k]) for k in after if before[k] != after[k]}
    if not changed:
        return habit, {}
    habit.updated_at = ctx.clock.now()
    diff = {
        "before": {k: v[0] for k, v in changed.items()},
        "after": {k: v[1] for k, v in changed.items()},
    }
    await session.flush()
    await audit(
        session,
        now=habit.updated_at,
        actor=actor,
        action="habit.updated",
        entity="habit",
        entity_id=habit.id,
        details={**diff, "reason": reason},
    )
    return habit, diff


async def set_status(
    session: AsyncSession, ctx: AppContext, habit_id: int, status: EntityStatus, *, actor: Actor
) -> Habit:
    """Pause / resume / archive (the owner's explicit command). Paused days are
    excluded from the metrics, so a pause never looks like a run of misses."""
    habit = await require_habit(session, habit_id)
    status = EntityStatus(status)
    if habit.status == status:
        return habit
    tz = await owner_tz(session)
    today = local_date(ctx.clock.now(), tz)
    sched = dict(habit.schedule or {})
    pauses = _pauses(habit)
    if status in (EntityStatus.PAUSED, EntityStatus.ARCHIVED) and habit.status == "active":
        pauses.append({"from": today.isoformat(), "to": None})
    elif status == EntityStatus.ACTIVE:
        for p in pauses:
            if not p.get("to"):
                p["to"] = today.isoformat()
    sched["pauses"] = pauses
    before = habit.status
    habit.schedule = sched
    habit.status = status.value
    habit.updated_at = ctx.clock.now()
    await session.flush()
    await audit(
        session,
        now=habit.updated_at,
        actor=actor,
        action="habit.status",
        entity="habit",
        entity_id=habit.id,
        details={"before": before, "after": habit.status},
    )
    return habit


# --------------------------------------------------------------------------- occurrences


async def get_occurrence(session: AsyncSession, habit_id: int, d: date) -> HabitOccurrence | None:
    stmt = select(HabitOccurrence).where(
        HabitOccurrence.habit_id == habit_id, HabitOccurrence.local_date == d
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def occurrences(
    session: AsyncSession, habit_id: int, start: date, end: date
) -> dict[date, HabitOccurrence]:
    stmt = select(HabitOccurrence).where(
        HabitOccurrence.habit_id == habit_id,
        HabitOccurrence.local_date >= start,
        HabitOccurrence.local_date <= end,
    )
    return {o.local_date: o for o in (await session.execute(stmt)).scalars()}


def reason_of(occ: HabitOccurrence | None) -> tuple[str | None, str | None]:
    """(reason code, free text) from an occurrence note."""
    if occ is None or not occ.note:
        return None, None
    m = _NOTE_RE.match(occ.note)
    if m and m.group(1) in MISS_REASONS:
        return m.group(1), (m.group(2) or None)
    return None, occ.note


@dataclass
class MarkResult:
    occurrence: HabitOccurrence
    changed: bool
    previous: str | None = None


async def mark(
    session: AsyncSession,
    ctx: AppContext,
    habit_id: int,
    status: str,
    *,
    actor: Actor = Actor.USER,
    day: date | None = None,
    amount: float | None = None,
    note: str | None = None,
    time_source: TimeSource | None = None,
) -> MarkResult:
    """Mark a habit for one local day. Idempotent: the same status (and amount)
    again changes nothing; a different status is a correction (audited)."""
    status = HabitOccurrenceStatus(status).value
    habit = await require_habit(session, habit_id)
    if habit.status == EntityStatus.ARCHIVED:
        raise HabitError("Привычка в архиве.")
    tz = await owner_tz(session)
    today = local_date(ctx.clock.now(), tz)
    day = day or today
    if day > today:
        raise HabitError("Нельзя отметить будущий день.")
    if (today - day).days > RETRO_DAYS:
        raise HabitError(f"Отмечать можно только последние {RETRO_DAYS} дней.")
    if amount is None and status == DONE:
        amount = (habit.target or {}).get("amount")
    if amount is None and status == PARTIAL:
        amount = (habit.target or {}).get("min_amount")
    src = time_source or (TimeSource.EXACT if day == today else TimeSource.RETRO)
    occ = await get_occurrence(session, habit_id, day)
    now = ctx.clock.now()
    if occ is not None:
        same = occ.status == status and (amount is None or occ.amount == amount)
        if same and (note is None or note == occ.note):
            return MarkResult(occ, False, occ.status)
        previous = occ.status
        before = {"status": occ.status, "amount": occ.amount, "note": occ.note}
        occ.status = status
        if amount is not None:
            occ.amount = float(amount)
        if note is not None:
            occ.note = note
        await session.flush()
        await audit(
            session,
            now=now,
            actor=actor,
            action="habit.occurrence_corrected",
            entity="habit",
            entity_id=habit_id,
            details={
                "date": day.isoformat(),
                "before": before,
                "after": {"status": occ.status, "amount": occ.amount, "note": occ.note},
            },
        )
        return MarkResult(occ, True, previous)
    occ = HabitOccurrence(
        habit_id=habit_id,
        local_date=day,
        status=status,
        amount=None if amount is None else float(amount),
        note=note,
        time_source=str(src),
        created_at=now,
    )
    session.add(occ)
    await session.flush()
    await audit(
        session,
        now=now,
        actor=actor,
        action="habit.marked",
        entity="habit",
        entity_id=habit_id,
        details={"date": day.isoformat(), "status": status, "amount": occ.amount},
    )
    return MarkResult(occ, True, None)


async def mark_amount(
    session: AsyncSession,
    ctx: AppContext,
    habit_id: int,
    amount: float,
    *,
    day: date | None = None,
    actor: Actor = Actor.USER,
) -> MarkResult:
    habit = await require_habit(session, habit_id)
    if amount <= 0:
        raise HabitError("Объём должен быть больше нуля.")
    return await mark(
        session,
        ctx,
        habit_id,
        classify_amount(habit.target, amount),
        day=day,
        amount=amount,
        actor=actor,
    )


async def record_miss(
    session: AsyncSession,
    ctx: AppContext,
    habit_id: int,
    day: date,
    reason: str,
    *,
    text: str | None = None,
    actor: Actor = Actor.USER,
) -> MarkResult:
    """The owner tells what happened on a missed day (quick reason + note)."""
    if reason not in MISS_REASONS:
        raise HabitError("Неизвестная причина.")
    tz = await owner_tz(session)
    if day >= local_date(ctx.clock.now(), tz):
        raise HabitError("Разбирать можно только прошедшие дни.")
    note = f"[{reason}] {text.strip()}" if text and text.strip() else f"[{reason}]"
    existing = await get_occurrence(session, habit_id, day)
    if existing is not None and existing.status in KEPT:
        raise HabitError("Этот день уже отмечен как выполненный.")
    status = SKIPPED if existing is not None and existing.status == SKIPPED else MISSED
    return await mark(
        session,
        ctx,
        habit_id,
        status,
        day=day,
        note=note,
        actor=actor,
        time_source=TimeSource.RETRO,
    )


# --------------------------------------------------------------------------- due habits


@dataclass(frozen=True)
class DueHabit:
    """A habit to do on a local day (read by the planner and the day view).

    `status` is today's mark (None = not marked yet); `start_at` is the preferred
    local time resolved to UTC (DST-safe) or None; `estimate_minutes` comes from
    a target in minutes."""

    habit_id: int
    title: str
    local_date: date
    preferred_time: time | None
    start_at: datetime | None
    target: dict[str, Any]
    target_text: str
    estimate_minutes: int | None
    flexible: bool
    status: str | None
    week_kept: int | None = None  # flexible habits: kept days this week (other days)
    week_target: int | None = None

    @property
    def pending(self) -> bool:
        return self.status is None


def _week_bounds(d: date) -> tuple[date, date]:
    monday = d - timedelta(days=d.weekday())
    return monday, monday + timedelta(days=6)


async def due_habits(
    session: AsyncSession, ctx: AppContext, local_day: date | None = None
) -> list[DueHabit]:
    """Active habits that are due on `local_day` (default: today), including the
    ones already marked that day (see `DueHabit.pending`).

    Fixed schedules: due on scheduled weekdays. Flexible N/week: due while fewer
    than N days of this ISO week (Mon-Sun, other days) are kept. Paused days and
    days before the habit existed are never due. Sorted by preferred time."""
    tz = await owner_tz(session)
    day = local_day or local_date(ctx.clock.now(), tz)
    out: list[DueHabit] = []
    for habit in await list_habits(session):
        if day < start_date(habit, tz) or is_paused_on(habit, day):
            continue
        sched = habit_schedule(habit)
        if not sched.is_scheduled(day):
            continue
        occ = await get_occurrence(session, habit.id, day)
        week_kept: int | None = None
        if sched.flexible:
            monday, sunday = _week_bounds(day)
            week = await occurrences(session, habit.id, monday, sunday)
            week_kept = sum(1 for d, o in week.items() if d != day and o.status in KEPT)
            if week_kept >= sched.n and occ is None:
                continue  # weekly target already reached: not due, no pressure
        target = dict(habit.target or {})
        minutes = None
        if is_minutes_unit(target.get("unit")) and target.get("amount") is not None:
            minutes = max(1, math.ceil(float(target["amount"])))
        start_at = (
            resolve_local(day, habit.preferred_time, tz)
            if habit.preferred_time is not None
            else None
        )
        out.append(
            DueHabit(
                habit_id=habit.id,
                title=habit.title,
                local_date=day,
                preferred_time=habit.preferred_time,
                start_at=start_at,
                target=target,
                target_text=target_text(target),
                estimate_minutes=minutes,
                flexible=sched.flexible,
                status=occ.status if occ is not None else None,
                week_kept=week_kept,
                week_target=sched.n if sched.flexible else None,
            )
        )
    out.sort(key=lambda h: (h.preferred_time is None, h.preferred_time or time(0), h.habit_id))
    return out


# --------------------------------------------------------------------------- stability


@dataclass
class HabitStability:
    """Regularity over a window ending today (no streaks, nothing is zeroed).

    Fixed schedules count scheduled days; flexible habits count expected days
    per ISO week (the current week only counts what is already certain).
    `regularity` = kept (done + partial) / counted, where conscious skips and
    today's not-yet-marked day are left out of `counted`. Paused days never
    count. `avg_return_days`: calendar days from the first missed day to the
    next kept day, averaged over the window (fixed schedules)."""

    habit_id: int
    days: int
    window_start: date
    window_end: date
    flexible: bool
    counted: int = 0
    kept: int = 0
    done: int = 0
    partial: int = 0
    skipped: int = 0
    missed: int = 0
    extra: int = 0  # kept days outside the schedule (bonus, not in the ratio)
    regularity: float | None = None
    full_share: float | None = None
    returns: list[int] = field(default_factory=list)
    open_gap_since: date | None = None  # currently after a miss, not yet back

    @property
    def avg_return_days(self) -> float | None:
        return sum(self.returns) / len(self.returns) if self.returns else None

    @property
    def enough_data(self) -> bool:
        return self.counted >= MIN_DATA_POINTS

    def as_dict(self) -> dict[str, Any]:
        return {
            "habit_id": self.habit_id,
            "days": self.days,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "flexible": self.flexible,
            "counted": self.counted,
            "kept": self.kept,
            "done": self.done,
            "partial": self.partial,
            "skipped": self.skipped,
            "missed": self.missed,
            "extra": self.extra,
            "regularity": self.regularity,
            "full_share": self.full_share,
            "avg_return_days": self.avg_return_days,
            "open_gap_since": self.open_gap_since.isoformat() if self.open_gap_since else None,
            "enough_data": self.enough_data,
        }


def _first_day(habit: Habit, tz: str, occs: dict[date, HabitOccurrence]) -> date:
    first = start_date(habit, tz)
    if occs:
        first = min(first, min(occs))
    return first


async def habit_stability(
    session: AsyncSession,
    ctx: AppContext,
    habit_id: int,
    *,
    days: int = 28,
    end: date | None = None,
) -> HabitStability:
    """Stability of one habit over the last `days` local days ending `end`
    (default today). Called by the weekly analytics."""
    habit = await require_habit(session, habit_id)
    tz = await owner_tz(session)
    today = local_date(ctx.clock.now(), tz)
    end = end or today
    days = max(1, int(days))
    sched = habit_schedule(habit)
    start = end - timedelta(days=days - 1)
    if sched.flexible:
        # Align to whole ISO weeks: the last ceil(days/7) weeks including `end`.
        weeks = max(1, math.ceil(days / 7))
        start = _week_bounds(end)[0] - timedelta(days=7 * (weeks - 1))
    occs = await occurrences(session, habit.id, start - timedelta(days=7), end)
    first = _first_day(habit, tz, occs)
    st = HabitStability(
        habit_id=habit.id,
        days=days,
        window_start=max(start, first),
        window_end=end,
        flexible=sched.flexible,
    )
    if sched.flexible:
        _flexible_stats(st, habit, sched, occs, start, end, today, first)
    else:
        _fixed_stats(st, habit, sched, occs, start, end, today, first)
    if st.counted > 0:
        st.regularity = round(st.kept / st.counted, 4)
        st.full_share = round(st.done / st.counted, 4)
    return st


def _fixed_stats(
    st: HabitStability,
    habit: Habit,
    sched: Schedule,
    occs: dict[date, HabitOccurrence],
    start: date,
    end: date,
    today: date,
    first: date,
) -> None:
    gap_start: date | None = None
    d = max(start, first)
    while d <= end:
        occ = occs.get(d)
        status = occ.status if occ is not None else None
        if is_paused_on(habit, d) and status is None:
            d += timedelta(days=1)
            continue
        if not sched.is_scheduled(d):
            if status in KEPT:
                st.extra += 1
                if gap_start is not None:  # coming back on an off-day counts
                    st.returns.append((d - gap_start).days)
                    gap_start = None
            d += timedelta(days=1)
            continue
        if status is None and d >= today:
            d += timedelta(days=1)
            continue  # today is not over yet: neither kept nor missed
        if status is None or status == MISSED:
            st.missed += 1
            st.counted += 1
            if gap_start is None:
                gap_start = d
        elif status == SKIPPED:
            st.skipped += 1  # a conscious decision: not in the ratio, gap unchanged
        else:
            st.counted += 1
            st.kept += 1
            if status == DONE:
                st.done += 1
            else:
                st.partial += 1
            if gap_start is not None:
                st.returns.append((d - gap_start).days)
                gap_start = None
        d += timedelta(days=1)
    st.open_gap_since = gap_start


def _flexible_stats(
    st: HabitStability,
    habit: Habit,
    sched: Schedule,
    occs: dict[date, HabitOccurrence],
    start: date,
    end: date,
    today: date,
    first: date,
) -> None:
    monday = start
    while monday <= end:
        sunday = monday + timedelta(days=6)
        active_days = [
            monday + timedelta(days=i)
            for i in range(7)
            if first <= monday + timedelta(days=i)
            and not is_paused_on(habit, monday + timedelta(days=i))
        ]
        week = {d: o for d, o in occs.items() if monday <= d <= min(sunday, end)}
        kept_days = [d for d, o in week.items() if o.status in KEPT]
        st.skipped += sum(1 for o in week.values() if o.status == SKIPPED)
        if not active_days:
            st.extra += len(kept_days)
            monday += timedelta(days=7)
            continue
        if sunday < today:
            # A finished week: prorate the target if the habit started/paused mid-week.
            expected = min(sched.n, round(sched.n * len(active_days) / 7))
        else:
            # The current week only counts what is already certain: kept days, plus
            # the deficit that can no longer be made up in the days left.
            left = sum(1 for d in active_days if d >= today and week.get(d) is None)
            expected = max(min(len(kept_days), sched.n), sched.n - left)
            expected = max(0, min(sched.n, expected))
        kept = min(len(kept_days), expected)
        st.extra += len(kept_days) - kept
        st.counted += expected
        st.kept += kept
        done_days = sorted(d for d in kept_days if week[d].status == DONE)
        st.done += min(len(done_days), kept)
        st.partial += kept - min(len(done_days), kept)
        st.missed += expected - kept
        monday += timedelta(days=7)


def pct(v: float | None) -> str:
    return "—" if v is None else f"{round(v * 100)}%"


def stability_text(st: HabitStability) -> str:
    """One honest line, no streak language."""
    if st.counted == 0:
        return f"за {st.days} дн.: пока нет данных"
    unit = "раз" if st.flexible else "дн."
    parts = [f"за {st.days} дн.: {st.kept} из {st.counted} {unit} ({pct(st.regularity)})"]
    if st.partial:
        parts.append(f"из них минимум — {st.partial}")
    if st.skipped:
        parts.append(f"осознанных пропусков — {st.skipped}")
    if st.extra:
        parts.append(f"сверх графика — {st.extra}")
    text = ", ".join(parts)
    if not st.enough_data:
        text += " · данных пока мало"
    return text


def return_text(st: HabitStability) -> str | None:
    avg = st.avg_return_days
    if avg is None:
        return None
    return f"после пропуска возвращаешься в среднем через {avg:.1f} дн.".replace(".0 ", " ")


# --------------------------------------------------------------------------- miss analysis


@dataclass
class MissAnalysis:
    habit_id: int
    days: int
    unexplained: list[date] = field(default_factory=list)  # newest first
    by_reason: Counter[str] = field(default_factory=Counter)
    by_weekday: Counter[int] = field(default_factory=Counter)
    notes: list[tuple[date, str]] = field(default_factory=list)
    skipped: int = 0
    missed: int = 0


async def missed_analysis(
    session: AsyncSession, ctx: AppContext, habit_id: int, *, days: int = 28
) -> MissAnalysis:
    """What happened on missed days: reasons the owner gave, weekdays with most
    misses, and recent scheduled days without any mark (to ask about)."""
    habit = await require_habit(session, habit_id)
    tz = await owner_tz(session)
    today = local_date(ctx.clock.now(), tz)
    start = today - timedelta(days=days)
    occs = await occurrences(session, habit_id, start, today)
    first = _first_day(habit, tz, occs)
    sched = habit_schedule(habit)
    out = MissAnalysis(habit_id=habit_id, days=days)
    d = max(start, first)
    while d < today:
        occ = occs.get(d)
        if occ is not None and occ.status in (MISSED, SKIPPED):
            code, text = reason_of(occ)
            if occ.status == MISSED:
                out.missed += 1
                out.by_weekday[d.weekday()] += 1
            else:
                out.skipped += 1
            if code:
                out.by_reason[code] += 1
            if text:
                out.notes.append((d, text))
        elif (
            occ is None
            and not sched.flexible
            and sched.is_scheduled(d)
            and not is_paused_on(habit, d)
        ):
            out.missed += 1
            out.by_weekday[d.weekday()] += 1
            if (today - d).days <= RETRO_DAYS:
                out.unexplained.append(d)
        d += timedelta(days=1)
    out.unexplained.sort(reverse=True)
    return out


def analysis_text(a: MissAnalysis) -> str:
    lines: list[str] = []
    if not a.missed and not a.skipped:
        return f"За {a.days} дн. пропусков нет."
    lines.append(f"За {a.days} дн.: не получилось — {a.missed}, осознанно пропущено — {a.skipped}.")
    if a.by_reason:
        reasons = ", ".join(f"{MISS_REASONS[c]} — {n}" for c, n in a.by_reason.most_common())
        lines.append(f"Причины (с твоих слов): {reasons}.")
    if a.by_weekday and a.missed >= 2:
        top, cnt = a.by_weekday.most_common(1)[0]
        if cnt >= 2:
            lines.append(
                f"Чаще всего не получается по {WEEKDAY_SHORT[top]} ({cnt}) — "
                "возможно, в этот день стоит уменьшить объём или сдвинуть время. "
                "Это лишь наблюдение."
            )
    if a.unexplained:
        lines.append(f"Без пояснений: {len(a.unexplained)} дн.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- increase


@dataclass(frozen=True)
class IncreaseSuggestion:
    habit_id: int
    action: ChangeHabit
    text: str
    stability: HabitStability


def increase_threshold(ctx: AppContext) -> float | None:
    """Undecided X (see INCREASE_MIN_REGULARITY); app config may set it later."""
    value = getattr(ctx.config, "habit_increase_min_regularity", None)
    return value if value is not None else INCREASE_MIN_REGULARITY


def _next_volume(habit: Habit) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    """(new target, new schedule, text) of one gradual step; (None, None, "")
    when there is nothing sensible to suggest."""
    sched = habit_schedule(habit)
    target = dict(habit.target or {})
    amount = target.get("amount")
    if amount is not None:
        step = max(1.0, math.ceil(float(amount) * INCREASE_STEP_RATIO))
        new = dict(target)
        new["amount"] = _int_if_whole(float(amount) + step)
        return new, None, f"объём {target_text(target)} → {target_text(new)}"
    if sched.flexible and sched.n < 7:
        new_s = Schedule(TIMES_PER_WEEK, n=sched.n + 1) if sched.n + 1 < 7 else Schedule(DAILY)
        return None, new_s.as_dict(), f"график {schedule_text(sched)} → {schedule_text(new_s)}"
    return None, None, ""


async def suggest_increase(
    session: AsyncSession, ctx: AppContext, habit_id: int, *, days: int = 28
) -> IncreaseSuggestion | None:
    """A rule-based suggestion of one gradual step (never applied by itself)."""
    habit = await require_habit(session, habit_id)
    st = await habit_stability(session, ctx, habit_id, days=days)
    new_target, new_schedule, text = _next_volume(habit)
    if not text:
        return None
    reason = f"Регулярность {stability_text(st)}. Предложение — небольшой шаг: {text}."
    changes: dict[str, Any] = {}
    if new_target is not None:
        changes["target"] = new_target
    if new_schedule is not None:
        changes["schedule"] = new_schedule
    action = ChangeHabit(habit_id=habit.id, reason=reason, **changes)
    return IncreaseSuggestion(habit_id=habit.id, action=action, text=text, stability=st)


async def increase_candidates(
    session: AsyncSession, ctx: AppContext, *, days: int = 28
) -> list[IncreaseSuggestion]:
    """For the weekly review: habits stable enough for a suggested step.
    Empty while the threshold is not configured (open question)."""
    threshold = increase_threshold(ctx)
    if threshold is None:
        return []
    out: list[IncreaseSuggestion] = []
    for habit in await list_habits(session):
        if await has_pending_change(session, habit.id):
            continue
        sug = await suggest_increase(session, ctx, habit.id, days=days)
        if (
            sug is not None
            and sug.stability.enough_data
            and sug.stability.regularity is not None
            and sug.stability.regularity >= threshold
        ):
            out.append(sug)
    return out


async def has_pending_change(session: AsyncSession, habit_id: int) -> bool:
    stmt = select(PendingApproval).where(
        PendingApproval.action_type == ActionType.CHANGE_HABIT.value,
        PendingApproval.status == ApprovalStatus.PENDING.value,
    )
    for row in (await session.execute(stmt)).scalars():
        if int(row.action_payload.get("habit_id") or 0) == habit_id:
            return True
    return False


async def propose_change(
    session: AsyncSession,
    ctx: AppContext,
    action: ChangeHabit,
    *,
    origin: Actor,
    op_key: str,
    ai_call_id: int | None = None,
) -> actions.ExecResult:
    """Route a habit change through the executor. Bot/AI origin -> approval card;
    origin USER (the owner's explicit edit) -> applied."""
    return await actions.execute(
        session, ctx, action, origin=origin, op_key=op_key, ai_call_id=ai_call_id
    )


# --------------------------------------------------------------------------- day view


def mark_cb(habit_id: int, status: str, day: date, back: str) -> str:
    """Callback for a quick mark; carries the day so a late press marks the
    right day. back: l (list) / c (card) / d (day view)."""
    code = {DONE: "d", PARTIAL: "p", SKIPPED: "s"}[status]
    return HabitCb(action="m", habit_id=habit_id, arg=f"{code}.{day:%Y%m%d}.{back}").pack()


async def day_section(
    session: AsyncSession, ctx: AppContext, day: date | None = None
) -> tuple[list[str], list[list[InlineKeyboardButton]]]:
    """Lines + quick-mark buttons for habits due on `day`, for "Мой день"."""
    due = await due_habits(session, ctx, day)
    if not due:
        return [], []
    lines = ["🔁 Привычки"]
    rows: list[list[InlineKeyboardButton]] = []
    for h in due:
        at = f"{h.preferred_time:%H:%M} " if h.preferred_time else ""
        week = f" · {h.week_kept}/{h.week_target} за неделю" if h.flexible else ""
        lines.append(f"{STATUS_ICON.get(h.status, '○')} {at}{h.title} — {h.target_text}{week}")
        if h.pending:
            title = h.title if len(h.title) <= 24 else h.title[:23] + "…"
            rows.append(
                [
                    InlineKeyboardButton(
                        text=f"✅ {title}",
                        callback_data=mark_cb(h.habit_id, DONE, h.local_date, "d"),
                    ),
                    InlineKeyboardButton(
                        text="◐", callback_data=mark_cb(h.habit_id, PARTIAL, h.local_date, "d")
                    ),
                    InlineKeyboardButton(
                        text="⏭", callback_data=mark_cb(h.habit_id, SKIPPED, h.local_date, "d")
                    ),
                ]
            )
    return lines, rows


# --------------------------------------------------------------------------- appliers


async def _apply_create(
    session: AsyncSession, ctx: AppContext, action: CreateHabit, actor: Actor
) -> ApplyOutcome:
    habit = await create_habit(
        session,
        ctx,
        title=action.title,
        actor=actor,
        schedule=dict(action.schedule),
        target=dict(action.target),
        preferred_time=parse_hhmm(action.preferred_time),
        goal_id=action.goal_id,
    )
    return ApplyOutcome(result={"habit_id": habit.id, "title": habit.title})


async def _describe_create(
    session: AsyncSession, ctx: AppContext, action: CreateHabit
) -> dict[str, Any]:
    sched = parse_schedule(dict(action.schedule))
    tgt = normalize_target(dict(action.target))
    when = parse_hhmm(action.preferred_time)
    lines = [f"График: {schedule_text(sched)}", f"Объём: {target_text(tgt)}"]
    if when is not None:
        lines.append(f"Время: {when:%H:%M}")
    if action.goal_id is not None:
        goal = await session.get(Goal, action.goal_id)
        if goal is None:
            raise HabitError(f"Цель #{action.goal_id} не найдена.")
        lines.append(f"Цель: {goal.title}")
    return {"summary": f"Новая привычка «{_clean_title(action.title)}»", "lines": lines}


def _time_arg(value: str | None) -> Any:
    if value is None:
        return _UNSET
    return parse_hhmm(value)  # "" clears the preferred time


async def _apply_change(
    session: AsyncSession, ctx: AppContext, action: ChangeHabit, actor: Actor
) -> ApplyOutcome:
    habit, diff = await update_habit(
        session,
        ctx,
        action.habit_id,
        actor=actor,
        reason=action.reason,
        schedule=None if action.schedule is None else dict(action.schedule),
        target=None if action.target is None else dict(action.target),
        preferred_time=_time_arg(action.preferred_time),
    )
    return ApplyOutcome(result={"habit_id": habit.id, "changed": sorted(diff.get("after", {}))})


async def _describe_change(
    session: AsyncSession, ctx: AppContext, action: ChangeHabit
) -> dict[str, Any]:
    habit = await require_habit(session, action.habit_id)
    if habit.status == EntityStatus.ARCHIVED:
        raise HabitError("Привычка в архиве.")
    lines: list[str] = []
    if action.schedule is not None:
        new = parse_schedule(dict(action.schedule))
        lines.append(f"График: {schedule_text(habit_schedule(habit))} → {schedule_text(new)}")
    if action.target is not None:
        new_t = normalize_target(dict(action.target))
        lines.append(f"Объём: {target_text(habit.target)} → {target_text(new_t)}")
    if action.preferred_time is not None:
        old = f"{habit.preferred_time:%H:%M}" if habit.preferred_time else "не задано"
        new_time = parse_hhmm(action.preferred_time)
        lines.append(f"Время: {old} → {new_time.strftime('%H:%M') if new_time else 'не задано'}")
    if not lines:
        raise HabitError("Предложение ничего не меняет.")
    if action.reason:
        lines.append(f"Почему: {action.reason}")
    return {"summary": f"Изменить привычку «{habit.title}»", "lines": lines}


def register(ctx: AppContext) -> None:
    """Register appliers for habit actions (both NEEDS_APPROVAL in FR-15)."""
    actions.register_applier(ctx, ActionType.CREATE_HABIT, _apply_create, describe=_describe_create)
    actions.register_applier(ctx, ActionType.CHANGE_HABIT, _apply_change, describe=_describe_change)
