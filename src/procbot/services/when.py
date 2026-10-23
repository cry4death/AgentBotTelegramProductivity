"""Deterministic parsing of short Russian date/time input (no AI).

Accepted forms (case-insensitive, combinable "date time"):
- dates: "сегодня", "завтра", "послезавтра", weekday names ("пн", "среда"...),
  "25.10", "25.10.2026", "25.10.26", "2026-10-25";
- times: "14:00", "14.30", "9:05", "в 14:00", "14ч".
A weekday means its next occurrence (today counts if it is that weekday).
A day/month without a year is this year, or next year if that day already passed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, time, timedelta

_WEEKDAYS: dict[str, int] = {
    "пн": 0,
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
_RELATIVE: dict[str, int] = {"сегодня": 0, "завтра": 1, "послезавтра": 2}

_TIME_RE = re.compile(r"(?:^|\s)(?:в\s*)?(\d{1,2})(?:[:.](\d{2})|\s*ч)(?=\s|$)")
_ISO_RE = re.compile(r"(?:^|\s)(\d{4})-(\d{2})-(\d{2})(?=\s|$)")
_DMY_RE = re.compile(r"(?:^|\s)(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?(?=\s|$)")


@dataclass(frozen=True)
class When:
    day: date | None = None
    at: time | None = None

    @property
    def empty(self) -> bool:
        return self.day is None and self.at is None


def parse_when(text: str, *, today: date) -> When | None:
    """Parse `text`; returns None when nothing recognisable or leftovers remain."""
    raw = " " + " ".join(text.lower().replace(",", " ").split()) + " "
    day: date | None = None
    at: time | None = None

    m = _ISO_RE.search(raw)
    if m:
        try:
            day = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
        raw = raw[: m.start()] + " " + raw[m.end() :]
    else:
        # Dates first, so "25.10" is not mistaken for the time 25:10.
        m = _DMY_RE.search(raw)
        if m and (m.group(3) is not None or not _looks_like_time(m.group(1), m.group(2))):
            day = _dmy(m.group(1), m.group(2), m.group(3), today)
            if day is None:
                return None
            raw = raw[: m.start()] + " " + raw[m.end() :]

    m = _TIME_RE.search(raw)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2) or 0)
        if hh > 23 or mm > 59:
            return None
        at = time(hh, mm)
        raw = raw[: m.start()] + " " + raw[m.end() :]

    words = raw.split()
    rest: list[str] = []
    for w in words:
        if day is None and w in _RELATIVE:
            day = today + timedelta(days=_RELATIVE[w])
        elif day is None and w in _WEEKDAYS:
            day = today + timedelta(days=(_WEEKDAYS[w] - today.weekday()) % 7)
        elif w in {"в", "во", "на"}:
            continue
        else:
            rest.append(w)
    if rest:
        return None
    result = When(day=day, at=at)
    return None if result.empty else result


def _looks_like_time(a: str, b: str) -> bool:
    """ "14.30" is a time; "25.10" is a date. Ambiguous "10.11" -> date."""
    hh, mm = int(a), int(b)
    return hh <= 23 and mm <= 59 and len(b) == 2 and (mm > 12 or mm == 0)


def _dmy(d: str, m: str, y: str | None, today: date) -> date | None:
    try:
        if y is None:
            candidate = date(today.year, int(m), int(d))
            if candidate < today:
                candidate = date(today.year + 1, int(m), int(d))
            return candidate
        year = int(y)
        if year < 100:
            year += 2000
        return date(year, int(m), int(d))
    except ValueError:
        return None


def parse_time(text: str) -> time | None:
    w = parse_when(text, today=date(2000, 1, 1))
    if w is None or w.day is not None:
        return None
    return w.at
