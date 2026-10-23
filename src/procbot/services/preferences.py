"""Owner preferences: parsing, validation and display helpers for onboarding and
settings (FR-01, FR-07, FR-08). Storage is `owner_profile` via services.profile.

Nothing here invents values for spec section 10 questions: quick-pick buttons
are options the owner chooses from; unset values stay None ("не задано").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from typing import Any
from zoneinfo import ZoneInfo, available_timezones

from procbot.timeutil import local_date, resolve_local, to_local

WEEKDAYS_SHORT = ("Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс")
WEEKDAYS_FULL = (
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
)
NOT_SET = "не задано"

# ------------------------------------------------------------------ personality (FR-08)

PRESETS: dict[str, str] = {
    "neutral": "Нейтральный",
    "coach": "Строгий тренер",
    "ironic": "Ироничный",
    "soft": "Мягкий",
}
KNOBS: dict[str, str] = {
    "strictness": "Строгость",
    "humor": "Юмор",
    "verbosity": "Подробность",
}
KNOB_MIN, KNOB_MAX = 0, 4

# ------------------------------------------------------------------ memory / autonomy
# Read by later stages (memory service, AI action executor). Missing key = True:
# the spec allows these by default; the owner may switch them off.

MEMORY_TOGGLES: dict[str, str] = {
    "enabled": "Долговременная память",
    "use_history": "Учитывать историю диалогов",
    "hypotheses": "Строить гипотезы о закономерностях",
}
# The four actions FR-15 allows without confirmation.
AUTONOMY_TOGGLES: dict[str, str] = {
    "reorder_tasks": "Менять порядок задач",
    "adjust_reminders": "Подстраивать время/частоту напоминаний",
    "manage_subtasks": "Создавать и перестраивать подзадачи",
    "choose_interventions": "Выбирать способы помощи",
}


def flag(settings: dict[str, Any] | None, key: str) -> bool:
    value = (settings or {}).get(key)
    return True if value is None else bool(value)


def toggled(settings: dict[str, Any] | None, key: str) -> dict[str, Any]:
    out = dict(settings or {})
    out[key] = not flag(settings, key)
    return out


# ------------------------------------------------------------------ time parsing

_HHMM = re.compile(r"^\s*(\d{1,2})(?:\s*[:.\s]\s*(\d{2}))?\s*$")
_HHMM_COMPACT = re.compile(r"^\s*(\d{2})(\d{2})\s*$")


def parse_hhmm(text: str) -> time | None:
    m = _HHMM_COMPACT.match(text) or _HHMM.match(text)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2) or 0)
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        return None
    return time(h, mi)


def hhmm(t: time | None) -> str:
    return NOT_SET if t is None else t.strftime("%H:%M")


def compact(t: time) -> str:
    """Callback-safe time (no ':' which is the callback separator)."""
    return t.strftime("%H%M")


def from_compact(s: str) -> time | None:
    return parse_hhmm(s)


_DAY_WORDS: dict[str, frozenset[int]] = {
    "будни": frozenset(range(5)),
    "будние": frozenset(range(5)),
    "выходные": frozenset({5, 6}),
    "ежедневно": frozenset(range(7)),
    "каждый день": frozenset(range(7)),
}
_DAY_TOKENS = {
    "пн": 0,
    "вт": 1,
    "ср": 2,
    "чт": 3,
    "пт": 4,
    "сб": 5,
    "вс": 6,
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}


def _day_token(s: str) -> int | None:
    s = s.strip().lower()
    for i, full in enumerate(WEEKDAYS_FULL):
        if len(s) >= 2 and full.startswith(s):
            return i
    return _DAY_TOKENS.get(s[:2], _DAY_TOKENS.get(s[:3]))


def parse_days(text: str) -> frozenset[int] | None:
    """'пн-пт', 'сб,вс', 'будни', 'выходные' -> weekday set; '' -> None (every day)."""
    t = text.strip().lower().replace("–", "-").replace("—", "-")
    if not t:
        return None
    if t in _DAY_WORDS:
        days = _DAY_WORDS[t]
        return None if len(days) == 7 else days
    out: set[int] = set()
    for part in re.split(r"[,\s]+", t):
        if not part:
            continue
        a, sep, b = part.partition("-")
        start = _day_token(a)
        end = _day_token(b) if sep else start
        if start is None or end is None:
            raise ValueError(f"unknown day: {part}")
        d = start
        out.add(d)
        while d != end:
            d = (d + 1) % 7
            out.add(d)
    return None if len(out) == 7 else frozenset(out)


def format_days(days: Any) -> str:
    if days is None:
        return "каждый день"
    ds = sorted(int(d) for d in days)
    if ds == [0, 1, 2, 3, 4]:
        return "Пн–Пт"
    if ds == [5, 6]:
        return "Сб–Вс"
    return ", ".join(WEEKDAYS_SHORT[d] for d in ds) or "—"


_WINDOW = re.compile(
    r"^\s*(\d{1,2}(?:[:.]\d{2})?)\s*[-–—]\s*(\d{1,2}(?:[:.]\d{2})?)\s*(.*)$", re.IGNORECASE
)


def parse_window(text: str) -> dict[str, Any]:
    """'23:00-08:00', '23-8 будни', '22:30–07:00 пн-пт' -> quiet-hours dict."""
    m = _WINDOW.match(text)
    if not m:
        raise ValueError("expected HH:MM-HH:MM")
    start, end = parse_hhmm(m.group(1)), parse_hhmm(m.group(2))
    if start is None or end is None:
        raise ValueError("bad time")
    if start == end:
        raise ValueError("empty window")
    days = parse_days(m.group(3))
    return {
        "start": start.strftime("%H:%M"),
        "end": end.strftime("%H:%M"),
        "days": None if days is None else sorted(days),
    }


def format_window(raw: dict[str, Any]) -> str:
    base = f"{raw.get('start')}–{raw.get('end')}"
    days = raw.get("days")
    return base if days is None else f"{base} ({format_days(days)})"


def format_windows(raws: list[Any] | None) -> str:
    return "; ".join(format_window(r) for r in raws) if raws else NOT_SET


# ------------------------------------------------------------------ timezone parsing

QUICK_TIMEZONES: tuple[tuple[str, str], ...] = (
    ("Europe/Moscow", "Москва"),
    ("Europe/Kaliningrad", "Калининград"),
    ("Europe/Samara", "Самара"),
    ("Asia/Yekaterinburg", "Екатеринбург"),
    ("Asia/Novosibirsk", "Новосибирск"),
    ("Asia/Vladivostok", "Владивосток"),
    ("Europe/Kyiv", "Киев"),
    ("Europe/Minsk", "Минск"),
    ("Asia/Almaty", "Алматы"),
    ("Europe/Berlin", "Берлин"),
    ("Europe/London", "Лондон"),
    ("UTC", "UTC"),
)

CITY_ZONES: dict[str, str] = {
    "москва": "Europe/Moscow",
    "мск": "Europe/Moscow",
    "санкт-петербург": "Europe/Moscow",
    "петербург": "Europe/Moscow",
    "спб": "Europe/Moscow",
    "питер": "Europe/Moscow",
    "казань": "Europe/Moscow",
    "нижний новгород": "Europe/Moscow",
    "калининград": "Europe/Kaliningrad",
    "самара": "Europe/Samara",
    "волгоград": "Europe/Volgograd",
    "екатеринбург": "Asia/Yekaterinburg",
    "екб": "Asia/Yekaterinburg",
    "челябинск": "Asia/Yekaterinburg",
    "пермь": "Asia/Yekaterinburg",
    "уфа": "Asia/Yekaterinburg",
    "тюмень": "Asia/Yekaterinburg",
    "омск": "Asia/Omsk",
    "новосибирск": "Asia/Novosibirsk",
    "красноярск": "Asia/Krasnoyarsk",
    "иркутск": "Asia/Irkutsk",
    "якутск": "Asia/Yakutsk",
    "владивосток": "Asia/Vladivostok",
    "хабаровск": "Asia/Vladivostok",
    "магадан": "Asia/Magadan",
    "киев": "Europe/Kyiv",
    "київ": "Europe/Kyiv",
    "kiev": "Europe/Kyiv",
    "минск": "Europe/Minsk",
    "алматы": "Asia/Almaty",
    "ташкент": "Asia/Tashkent",
    "бишкек": "Asia/Bishkek",
    "тбилиси": "Asia/Tbilisi",
    "ереван": "Asia/Yerevan",
    "баку": "Asia/Baku",
    "кишинев": "Europe/Chisinau",
    "рига": "Europe/Riga",
    "вильнюс": "Europe/Vilnius",
    "таллин": "Europe/Tallinn",
    "варшава": "Europe/Warsaw",
    "берлин": "Europe/Berlin",
    "прага": "Europe/Prague",
    "вена": "Europe/Vienna",
    "париж": "Europe/Paris",
    "мадрид": "Europe/Madrid",
    "рим": "Europe/Rome",
    "лондон": "Europe/London",
    "лиссабон": "Europe/Lisbon",
    "стамбул": "Europe/Istanbul",
    "дубай": "Asia/Dubai",
    "бангкок": "Asia/Bangkok",
    "токио": "Asia/Tokyo",
    "нью-йорк": "America/New_York",
    "лос-анджелес": "America/Los_Angeles",
    "торонто": "America/Toronto",
}


@lru_cache(maxsize=1)
def _zones() -> dict[str, str]:
    """lower-case IANA name -> canonical name."""
    return {z.lower(): z for z in available_timezones()}


@lru_cache(maxsize=1)
def _zone_by_city() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for z in available_timezones():
        if "/" not in z or z.startswith(("Etc/", "SystemV/", "posix/", "right/")):
            continue
        city = z.rsplit("/", 1)[-1].replace("_", " ").lower()
        out.setdefault(city, []).append(z)
    return out


def is_valid_timezone(name: str) -> bool:
    return name in available_timezones() or name == "UTC"


def utc_offset_label(zone_name: str, now: datetime) -> str:
    off = now.astimezone(ZoneInfo(zone_name)).utcoffset() or timedelta(0)
    total = int(off.total_seconds() // 60)
    sign = "+" if total >= 0 else "−"
    h, m = divmod(abs(total), 60)
    return f"UTC{sign}{h}" + (f":{m:02d}" if m else "")


@dataclass
class TzMatch:
    zone: str | None = None
    candidates: list[str] = field(default_factory=list)


_OFFSET = re.compile(r"^(?:utc|gmt)?\s*([+\-−])\s*(\d{1,2})(?:[:.]?(\d{2}))?$", re.IGNORECASE)


def parse_timezone(text: str, now: datetime) -> TzMatch:
    """IANA name or an unambiguous city -> zone; offsets/ambiguous -> candidates."""
    raw = text.strip()
    key = raw.lower().replace("ё", "е")
    if not key:
        return TzMatch()
    if key in ("utc", "gmt", "z"):
        return TzMatch(zone="UTC")
    canonical = _zones().get(key)
    if canonical:
        return TzMatch(zone=canonical)
    if key in CITY_ZONES:
        return TzMatch(zone=CITY_ZONES[key])
    by_city = _zone_by_city().get(key.replace("_", " "))
    if by_city:
        if len(by_city) == 1:
            return TzMatch(zone=by_city[0])
        return TzMatch(candidates=sorted(by_city)[:6])
    m = _OFFSET.match(key.replace(" ", ""))
    if m:
        sign = -1 if m.group(1) in "-−" else 1
        minutes = sign * (int(m.group(2)) * 60 + int(m.group(3) or 0))
        if abs(minutes) > 14 * 60:
            return TzMatch()
        return TzMatch(candidates=_zones_with_offset(minutes, now))
    return TzMatch()


def _zones_with_offset(minutes: int, now: datetime) -> list[str]:
    """An offset alone is ambiguous (DST rules differ): offer matching zones."""
    pool = [z for z, _ in QUICK_TIMEZONES] + sorted(set(CITY_ZONES.values()))
    out: list[str] = []
    for z in pool:
        off = now.astimezone(ZoneInfo(z)).utcoffset()
        if off is not None and int(off.total_seconds() // 60) == minutes and z not in out:
            out.append(z)
    if minutes % 60 == 0:
        h = minutes // 60
        fixed = "UTC" if h == 0 else f"Etc/GMT{'-' if h > 0 else '+'}{abs(h)}"  # POSIX sign
        if fixed not in out:
            out.append(fixed)
    return out[:6]


# ------------------------------------------------------------------ pause (FR-07)


def pause_for(choice: str, now: datetime, tz: str | None, morning: time | None) -> datetime:
    """Quick pause choices -> aware UTC end moment."""
    if choice == "1h":
        return now + timedelta(hours=1)
    if choice == "3h":
        return now + timedelta(hours=3)
    if choice == "tomorrow":
        tomorrow = local_date(now, tz) + timedelta(days=1)
        return resolve_local(tomorrow, morning or time(0, 0), tz)
    raise ValueError(f"unknown pause choice: {choice}")


_DATE = re.compile(r"^\s*(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?(?:\s+(\d{1,2}[:.]\d{2}))?\s*$")


def parse_pause_until(text: str, now: datetime, tz: str | None, morning: time | None) -> datetime:
    """'ДД.ММ', 'ДД.ММ ЧЧ:ММ', 'ДД.ММ.ГГГГ ЧЧ:ММ' or 'ЧЧ:ММ' -> future UTC moment."""
    stripped = text.strip()
    if re.fullmatch(r"\d{1,2}:\d{2}", stripped):
        t = parse_hhmm(stripped)
        if t is None:
            raise ValueError("bad time")
        d = to_local(now, tz).date()
        candidate = resolve_local(d, t, tz)
        if candidate <= now:
            candidate = resolve_local(d + timedelta(days=1), t, tz)
        return candidate
    m = _DATE.match(text)
    if not m:
        raise ValueError("expected ДД.ММ [ЧЧ:ММ]")
    day, month = int(m.group(1)), int(m.group(2))
    year_raw = m.group(3)
    today = local_date(now, tz)
    year = today.year if year_raw is None else int(year_raw) + (2000 if len(year_raw) == 2 else 0)
    try:
        d = date(year, month, day)
    except ValueError as exc:
        raise ValueError("bad date") from exc
    if year_raw is None and d < today:
        d = date(year + 1, month, day)
    at = parse_hhmm(m.group(4)) if m.group(4) else (morning or time(0, 0))
    if at is None:
        raise ValueError("bad time")
    result = resolve_local(d, at, tz)
    if result <= now:
        raise ValueError("moment is in the past")
    if result - now > timedelta(days=366):
        raise ValueError("too far in the future")
    return result


def parse_weekly(text: str) -> tuple[int, time]:
    """'вс 18:00', 'пятница 17:30' -> (weekday, time)."""
    parts = text.strip().lower().split()
    if len(parts) != 2:
        raise ValueError("expected '<день> ЧЧ:ММ'")
    day = _day_token(parts[0])
    t = parse_hhmm(parts[1])
    if day is None or t is None:
        raise ValueError("bad weekly review value")
    return day, t
