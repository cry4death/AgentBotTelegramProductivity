"""Acceptance 2 (partial, this module's share): after a restart the profile,
settings, dialogue state (FSM) and conversation history are intact."""

from __future__ import annotations

from datetime import time
from pathlib import Path

from sqlalchemy import func, select

from procbot.db.models import AuditLog, ConversationMessage, ConversationState
from procbot.services.profile import get_profile
from procbot.telegram.app import build_dispatcher
from procbot.telegram.callbacks import OnboardingCb, SettingsCb
from procbot.testing import callback_update, make_bot, make_context, message_update


async def test_profile_settings_and_dialogue_state_survive_restart(tmp_path: Path) -> None:
    ctx = await make_context(tmp_path)
    bot, _ = make_bot()
    dp = build_dispatcher(ctx)
    await dp.feed_update(bot, message_update("/start"))
    await dp.feed_update(bot, callback_update(OnboardingCb(step="timezone", arg="skip").pack()))
    await dp.feed_update(bot, callback_update(OnboardingCb(step="morning", arg="0730").pack()))
    await dp.feed_update(bot, callback_update(SettingsCb(action="kn", arg="humor.3").pack()))
    await dp.feed_update(bot, callback_update(SettingsCb(action="q_add", arg="2300-0700").pack()))
    # Waiting for a typed answer when the process dies:
    await dp.feed_update(bot, callback_update(SettingsCb(action="input", arg="timezone").pack()))
    async with ctx.db.transaction() as s:
        states = (await s.execute(select(ConversationState.state))).scalars().all()
        assert "SettingsForm:timezone" in states
        history_before = (await s.execute(select(func.count(ConversationMessage.id)))).scalar_one()
        audits_before = (await s.execute(select(func.count(AuditLog.id)))).scalar_one()
    await ctx.db.dispose()

    # "Restart": a fresh context, engine and dispatcher on the same file.
    ctx2 = await make_context(tmp_path)
    try:
        bot2, session2 = make_bot()
        dp2 = build_dispatcher(ctx2)
        async with ctx2.db.transaction() as s:
            p = await get_profile(s)
            assert p is not None
            assert p.morning_time == time(7, 30) and p.onboarding_step == "evening"
            assert p.humor == 3
            assert p.quiet_hours == [{"start": "23:00", "end": "07:00", "days": None}]
            assert (
                await s.execute(select(func.count(ConversationMessage.id)))
            ).scalar_one() == history_before
            assert (await s.execute(select(func.count(AuditLog.id)))).scalar_one() == audits_before
        # The pending question is still awaited after the restart.
        await dp2.feed_update(bot2, message_update("Europe/Berlin"))
        assert session2.sent_texts()[0].startswith("Сохранено")
        async with ctx2.db.transaction() as s:
            p = await get_profile(s)
            assert p is not None and p.timezone == "Europe/Berlin"
    finally:
        await ctx2.db.dispose()
