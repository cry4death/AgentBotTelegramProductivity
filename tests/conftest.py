from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from procbot.clock import FrozenClock
from procbot.context import AppContext
from procbot.services.profile import get_or_create_profile, update_profile
from procbot.testing import OWNER_ID, FakeMessenger, make_bot, make_context


@pytest.fixture
async def ctx(tmp_path: Path) -> AsyncIterator[AppContext]:
    c = await make_context(tmp_path, now=datetime(2026, 10, 26, 9, 0, tzinfo=UTC))
    yield c
    await c.db.dispose()


@pytest.fixture
def clock(ctx: AppContext) -> FrozenClock:
    assert isinstance(ctx.clock, FrozenClock)
    return ctx.clock


@pytest.fixture
def messenger(ctx: AppContext) -> FakeMessenger:
    assert isinstance(ctx.messenger, FakeMessenger)
    return ctx.messenger


@pytest.fixture
async def profile(ctx: AppContext):
    """Owner profile with timezone set and onboarding finished."""
    async with ctx.db.transaction() as s:
        await get_or_create_profile(
            s, telegram_user_id=OWNER_ID, now=ctx.clock.now(), chat_id=OWNER_ID
        )
        p = await update_profile(
            s, now=ctx.clock.now(), timezone="Europe/Berlin", onboarding_step=None
        )
    return p


@pytest.fixture
def bot_and_session():
    return make_bot()


@pytest.fixture
def dp(ctx: AppContext):
    from procbot.telegram.app import build_dispatcher

    return build_dispatcher(ctx)
