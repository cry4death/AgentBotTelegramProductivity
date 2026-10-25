"""Recurring tasks: the rule format, its validation and the next date.

A rule is stored in ``tasks.recurrence`` (JSON) and set only by the owner:

- ``{"type": "daily", "every": N}``: every N days (N = 1..365);
- ``{"type": "weekly", "days": [0..6]}``: on these weekdays (0 = Monday);
- ``{"type": "monthly", "day": D}``: on day D of the month (1..31; a short
  month uses its last day).

When a recurring task is done, `services.tasks.complete_task` creates the next
copy on the next matching day (pure code, no AI). Nothing here touches the
database.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from typing import Any

WEEKDAYS_SHORT = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")
MAX_EVERY = 365
# Search horizon for the next date (covers every rule above).
_MAX_STEPS = 400


class RecurrenceError(ValueError):
    pass


def parse_rule(raw: Any) -> dict[str, Any] | None:
    """Validate and normalise a rule. None / {} / {"type": "none"} -> None."""
    if raw is None or raw == {}:
        return None
    if not isinstance(raw, dict):
        raise RecurrenceError("Повтор: ожидается объект.")
    kind = raw.get("type")
    if kind in (None, "none", ""):
        return None
    if kind == "daily":
        every = raw.get("every", 1)
        if not isinstance(every, int) or isinstance(every, bool) or not 1 <= every <= MAX_EVERY:
            raise RecurrenceError(f"Повтор: «каждые N дней» — от 1 до {MAX_EVERY}.")
        return {"type": "daily", "every": every}
    if kind == "weekly":
        days = raw.get("days")
        if (
            not isinstance(days, list)
            or not days
            or not all(isinstance(d, int) and not isinstance(d, bool) and 0 <= d <= 6 for d in days)
        ):
            raise RecurrenceError("Повтор: выбери хотя бы один день недели.")
        return {"type": "weekly", "days": sorted(set(days))}
    if kind == "monthly":
        day = raw.get("day")
        if not isinstance(day, int) or isinstance(day, bool) or not 1 <= day <= 31:
            raise RecurrenceError("Повтор: число месяца — от 1 до 31.")
        return {"type": "monthly", "day": day}
    raise RecurrenceError("Повтор: неизвестный тип.")


def _monthly_day(year: int, month: int, day: int) -> date:
    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


def next_date(rule: dict[str, Any], *, base: date, after: date) -> date:
    """First date strictly after `after` that matches the rule. `base` is the
    day the finished copy was planned for: daily steps count from it."""
    kind = rule["type"]
    if kind == "daily":
        every = int(rule.get("every", 1))
        d = base + timedelta(days=every)
        while d <= after:
            d += timedelta(days=every)
        return d
    if kind == "weekly":
        days = set(rule["days"])
        d = after + timedelta(days=1)
        for _ in range(_MAX_STEPS):
            if d.weekday() in days:
                return d
            d += timedelta(days=1)
        raise RecurrenceError("Повтор: нет подходящего дня.")  # pragma: no cover
    if kind == "monthly":
        day = int(rule["day"])
        year, month = after.year, after.month
        for _ in range(_MAX_STEPS):
            d = _monthly_day(year, month, day)
            if d > after:
                return d
            month += 1
            if month > 12:
                year, month = year + 1, 1
        raise RecurrenceError("Повтор: нет подходящего дня.")  # pragma: no cover
    raise RecurrenceError("Повтор: неизвестный тип.")


def rule_text(rule: dict[str, Any] | None) -> str | None:
    if not rule:
        return None
    kind = rule.get("type")
    if kind == "daily":
        every = int(rule.get("every", 1))
        return "каждый день" if every == 1 else f"каждые {every} дн."
    if kind == "weekly":
        days = rule.get("days") or []
        if sorted(days) == [0, 1, 2, 3, 4]:
            return "по будням"
        if len(days) == 7:
            return "каждый день"
        return "по " + ", ".join(WEEKDAYS_SHORT[d] for d in days if 0 <= d <= 6)
    if kind == "monthly":
        return f"каждый месяц, {rule.get('day')}-го"
    return None
