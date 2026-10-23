"""Timezone helpers. Storage is UTC; rules are local wall-clock time + IANA zone.

DST policy (acceptance criterion 23):
- a local time that does not exist (spring-forward gap) resolves to the first
  valid instant after the gap (e.g. 02:30 -> 03:00 local);
- an ambiguous local time (fall-back overlap) resolves to the first occurrence
  (fold=0).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo


def zone(tz: str | None) -> ZoneInfo:
    return ZoneInfo(tz or "UTC")


def to_local(dt: datetime, tz: str | None) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("naive datetime")
    return dt.astimezone(zone(tz))


def local_date(dt: datetime, tz: str | None) -> date:
    return to_local(dt, tz).date()


def _exists(local_naive: datetime, z: ZoneInfo) -> bool:
    aware = local_naive.replace(tzinfo=z)
    roundtrip = aware.astimezone(UTC).astimezone(z).replace(tzinfo=None)
    return roundtrip == local_naive


def resolve_local(d: date, t: time, tz: str | None) -> datetime:
    """Local date+time in `tz` -> aware UTC datetime, DST-safe (see module doc)."""
    z = zone(tz)
    naive = datetime.combine(d, t.replace(tzinfo=None))
    if _exists(naive, z):
        return naive.replace(tzinfo=z, fold=0).astimezone(UTC)
    # Inside a spring-forward gap: walk forward minute by minute (gaps are < 3h).
    probe = naive
    for _ in range(180):
        probe += timedelta(minutes=1)
        if _exists(probe, z):
            return probe.replace(tzinfo=z, fold=0).astimezone(UTC)
    raise ValueError(f"cannot resolve {naive} in {tz}")  # pragma: no cover


def start_of_local_day(d: date, tz: str | None) -> datetime:
    return resolve_local(d, time(0, 0), tz)


def local_day_bounds(d: date, tz: str | None) -> tuple[datetime, datetime]:
    """[start, end) of a local calendar day in UTC (23/24/25 hours long)."""
    return start_of_local_day(d, tz), start_of_local_day(d + timedelta(days=1), tz)


@dataclass(frozen=True)
class LocalWindow:
    """A recurring local window, e.g. quiet hours 23:00-08:00 on some weekdays.

    `days` are weekdays (0=Mon) on which the window STARTS; None = every day.
    If end <= start the window crosses midnight.
    """

    start: time
    end: time
    days: frozenset[int] | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> LocalWindow:
        days = raw.get("days")
        return cls(
            start=time.fromisoformat(str(raw["start"])),
            end=time.fromisoformat(str(raw["end"])),
            days=frozenset(int(x) for x in days) if isinstance(days, Iterable) else None,  # type: ignore[arg-type]
        )

    def occurrences(self, around: datetime, tz: str | None) -> list[tuple[datetime, datetime]]:
        """UTC intervals of this window starting on local days around `around`."""
        base = local_date(around, tz)
        out: list[tuple[datetime, datetime]] = []
        for offset in (-1, 0, 1):
            d = base + timedelta(days=offset)
            if self.days is not None and d.weekday() not in self.days:
                continue
            start = resolve_local(d, self.start, tz)
            end_day = d if self.end > self.start else d + timedelta(days=1)
            end = resolve_local(end_day, self.end, tz)
            out.append((start, end))
        return out


def windows_from_raw(raw: Sequence[dict[str, object]] | None) -> list[LocalWindow]:
    return [LocalWindow.from_dict(r) for r in (raw or [])]


def active_window_end(
    now: datetime, windows: Sequence[LocalWindow], tz: str | None
) -> datetime | None:
    """If `now` is inside any window, return the UTC moment when all overlapping
    windows have ended; otherwise None."""
    end: datetime | None = None
    moment = now
    # Chained windows (e.g. 22-00 and 00-08) are followed until a free moment.
    for _ in range(10):
        current_end: datetime | None = None
        for w in windows:
            for s, e in w.occurrences(moment, tz):
                if s <= moment < e and (current_end is None or e > current_end):
                    current_end = e
        if current_end is None:
            break
        end = current_end
        moment = current_end
    return end


def is_in_windows(now: datetime, windows: Sequence[LocalWindow], tz: str | None) -> bool:
    return active_window_end(now, windows, tz) is not None


def next_local_occurrence(
    after: datetime, t: time, tz: str | None, weekday: int | None = None
) -> datetime:
    """Next UTC instant strictly after `after` when local time is `t`
    (optionally only on `weekday`, 0=Mon)."""
    d = local_date(after, tz)
    for offset in range(0, 15):
        day = d + timedelta(days=offset)
        if weekday is not None and day.weekday() != weekday:
            continue
        candidate = resolve_local(day, t, tz)
        if candidate > after:
            return candidate
    raise ValueError("no occurrence found")  # pragma: no cover


def fmt_local(dt: datetime | None, tz: str | None, fmt: str = "%d.%m %H:%M") -> str:
    if dt is None:
        return "—"
    return to_local(dt, tz).strftime(fmt)
