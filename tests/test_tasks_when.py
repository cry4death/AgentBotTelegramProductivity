"""Deterministic date/time input parsing (services.when)."""

from __future__ import annotations

from datetime import date, time

import pytest

from procbot.services.when import When, parse_time, parse_when

TODAY = date(2026, 10, 26)  # Monday


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("сегодня", When(TODAY, None)),
        ("завтра 15:30", When(date(2026, 10, 27), time(15, 30))),
        ("послезавтра", When(date(2026, 10, 28), None)),
        ("в 9:05", When(None, time(9, 5))),
        ("14.30", When(None, time(14, 30))),
        ("14ч", When(None, time(14, 0))),
        ("пт 10:00", When(date(2026, 10, 30), time(10, 0))),
        ("понедельник", When(TODAY, None)),
        ("27.10", When(date(2026, 10, 27), None)),
        ("25.10", When(date(2027, 10, 25), None)),  # already passed -> next year
        ("27.10.2026 8:00", When(date(2026, 10, 27), time(8, 0))),
        ("2026-11-01", When(date(2026, 11, 1), None)),
    ],
)
def test_parse_when(text: str, expected: When) -> None:
    assert parse_when(text, today=TODAY) == expected


@pytest.mark.parametrize("text", ["хлеб", "31.02", "25:00", "завтра купить", ""])
def test_parse_when_rejects(text: str) -> None:
    assert parse_when(text, today=TODAY) is None


def test_parse_time_only() -> None:
    assert parse_time("18:15") == time(18, 15)
    assert parse_time("завтра 18:15") is None
