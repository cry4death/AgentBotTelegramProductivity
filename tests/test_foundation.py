"""Foundation: time handling, storage types, transactions, auth and dedup."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

import pytest
from aiogram import F, Router
from sqlalchemy import func, select

from procbot.db.models import ProcessedUpdate, Task
from procbot.telegram.middlewares import NOT_ALLOWED_TEXT
from procbot.testing import STRANGER_ID, callback_update, message_update
from procbot.timeutil import (
    LocalWindow,
    active_window_end,
    local_day_bounds,
    next_local_occurrence,
    resolve_local,
)

BERLIN = "Europe/Berlin"


def test_resolve_local_regular_and_dst_gap_and_overlap() -> None:
    # 2026-03-29: Berlin springs forward 02:00 -> 03:00
    assert resolve_local(date(2026, 3, 29), time(2, 30), BERLIN) == datetime(
        2026, 3, 29, 1, 0, tzinfo=UTC
    )  # 03:00 local CEST
    # 2026-10-25: falls back 03:00 -> 02:00; 02:30 is ambiguous -> first (CEST)
    assert resolve_local(date(2026, 10, 25), time(2, 30), BERLIN) == datetime(
        2026, 10, 25, 0, 30, tzinfo=UTC
    )
    assert resolve_local(date(2026, 10, 26), time(9, 0), BERLIN) == datetime(
        2026, 10, 26, 8, 0, tzinfo=UTC
    )


def test_day_bounds_length_on_dst_days() -> None:
    s, e = local_day_bounds(date(2026, 10, 25), BERLIN)
    assert e - s == timedelta(hours=25)
    s, e = local_day_bounds(date(2026, 3, 29), BERLIN)
    assert e - s == timedelta(hours=23)


def test_quiet_window_crossing_midnight_and_dst() -> None:
    w = LocalWindow(start=time(23, 0), end=time(8, 0))
    # 2026-10-25 01:00 UTC = 03:00 CEST? no: after fallback at 01:00 UTC it's 02:00 CET
    inside = datetime(2026, 10, 25, 1, 30, tzinfo=UTC)
    end = active_window_end(inside, [w], BERLIN)
    assert end == datetime(2026, 10, 25, 7, 0, tzinfo=UTC)  # 08:00 CET
    assert active_window_end(datetime(2026, 10, 25, 12, 0, tzinfo=UTC), [w], BERLIN) is None


def test_next_local_occurrence_is_local_time_stable_over_dst() -> None:
    before = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
    first = next_local_occurrence(before, time(9, 0), BERLIN)
    second = next_local_occurrence(first, time(9, 0), BERLIN)
    assert first == datetime(2026, 10, 25, 8, 0, tzinfo=UTC)  # CET after fallback
    assert second == datetime(2026, 10, 26, 8, 0, tzinfo=UTC)


async def test_utc_roundtrip_and_naive_rejected(ctx) -> None:
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        s.add(Task(title="x", created_at=now, updated_at=now, planned_start_at=now))
    async with ctx.db.transaction() as s:
        t = (await s.execute(select(Task))).scalar_one()
        assert t.planned_start_at == now and t.planned_start_at.tzinfo is not None
    with pytest.raises(Exception, match="naive"):
        async with ctx.db.transaction() as s:
            s.add(Task(title="y", created_at=datetime(2026, 1, 1), updated_at=now))
            await s.flush()


async def test_nested_transaction_joins_outer_and_rollback_is_atomic(ctx) -> None:
    now = ctx.clock.now()
    with pytest.raises(RuntimeError):
        async with ctx.db.transaction() as s:
            s.add(Task(title="a", created_at=now, updated_at=now))
            async with ctx.db.transaction() as inner:
                assert inner is s
            raise RuntimeError("boom")
    async with ctx.db.transaction() as s:
        assert (await s.execute(select(func.count(Task.id)))).scalar_one() == 0


async def test_stranger_gets_neutral_refusal_and_no_handler(ctx, bot_and_session) -> None:
    from procbot.telegram.app import build_dispatcher

    bot, session = bot_and_session
    dp = build_dispatcher(ctx)
    probe = Router(name="probe")
    hits: list[str] = []

    @probe.message(F.text)
    async def _h(message) -> None:  # type: ignore[no-untyped-def]
        hits.append(message.text)

    dp.include_router(probe)
    await dp.feed_update(bot, message_update("/start", uid=STRANGER_ID))
    await dp.feed_update(bot, callback_update("tk:list:0:", uid=STRANGER_ID))
    assert hits == []
    assert session.sent_texts() == [NOT_ALLOWED_TEXT]


async def test_duplicate_update_id_processed_once(ctx, bot_and_session) -> None:
    from procbot.telegram.app import build_dispatcher

    bot, _ = bot_and_session
    dp = build_dispatcher(ctx)
    probe = Router(name="probe")
    hits: list[int] = []

    @probe.message(F.text == "ping")
    async def _h(message) -> None:  # type: ignore[no-untyped-def]
        hits.append(message.message_id)

    dp.include_router(probe)
    dp.sub_routers.insert(0, dp.sub_routers.pop())  # before the free-text fallback (chat)
    upd = message_update("ping", update_id=777)
    await dp.feed_update(bot, upd)
    await dp.feed_update(bot, upd)
    assert len(hits) == 1
    async with ctx.db.transaction() as s:
        assert await s.get(ProcessedUpdate, 777) is not None
