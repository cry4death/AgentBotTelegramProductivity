"""Injectable clock. All services read time through a Clock so tests can freeze it."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime:
        """Current time, timezone-aware, in UTC."""
        ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FrozenClock:
    """Manually advanced clock for tests."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("FrozenClock needs an aware datetime")
        self._now = start.astimezone(UTC)

    def now(self) -> datetime:
        return self._now

    def set(self, value: datetime) -> None:
        if value.tzinfo is None:
            raise ValueError("aware datetime required")
        self._now = value.astimezone(UTC)

    def advance(self, **kwargs: float) -> datetime:
        self._now = self._now + timedelta(**kwargs)
        return self._now


def utc(dt: datetime) -> datetime:
    """Normalise an aware datetime to UTC; reject naive values."""
    if dt.tzinfo is None:
        raise ValueError("naive datetime is not allowed; attach a timezone first")
    return dt.astimezone(UTC)
