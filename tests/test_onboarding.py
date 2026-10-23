"""FR-01 onboarding: owner-only, short, skippable, usable before completion."""

from __future__ import annotations

from datetime import UTC, datetime, time

from aiogram.methods import SendMessage
from aiogram.types import ReplyKeyboardMarkup
from sqlalchemy import func, select

from procbot.db.models import OwnerProfile
from procbot.services.preferences import parse_timezone
from procbot.services.profile import get_profile
from procbot.telegram.callbacks import OnboardingCb
from procbot.telegram.handlers import start
from procbot.telegram.keyboards import Menu
from procbot.telegram.middlewares import NOT_ALLOWED_TEXT
from procbot.testing import STRANGER_ID, callback_update, message_update

NOW = datetime(2026, 10, 26, 9, 0, tzinfo=UTC)


def ob(step: str, arg: str) -> str:
    return OnboardingCb(step=step, arg=arg).pack()


async def _profile(ctx) -> OwnerProfile | None:
    async with ctx.db.transaction() as s:
        return await get_profile(s)


def _has_main_menu(session) -> bool:
    return any(
        isinstance(m, SendMessage) and isinstance(m.reply_markup, ReplyKeyboardMarkup)
        for m in session.requests
    )


async def test_stranger_start_creates_no_profile_and_gets_no_menu(ctx, dp, bot_and_session):
    bot, session = bot_and_session
    await dp.feed_update(bot, message_update("/start", uid=STRANGER_ID))
    await dp.feed_update(bot, callback_update(ob("timezone", "UTC"), uid=STRANGER_ID))
    async with ctx.db.transaction() as s:
        assert (await s.execute(select(func.count(OwnerProfile.id)))).scalar_one() == 0
    assert session.sent_texts() == [NOT_ALLOWED_TEXT]
    assert not _has_main_menu(session)


async def test_start_creates_profile_shows_menu_and_first_question(ctx, dp, bot_and_session):
    bot, session = bot_and_session
    await dp.feed_update(bot, message_update("/start"))
    p = await _profile(ctx)
    assert p is not None and p.onboarding_step == "timezone" and p.chat_id is not None
    texts = session.sent_texts()
    assert texts[0] == start.WELCOME
    assert texts[1] == start.QUESTIONS["timezone"]
    assert _has_main_menu(session)  # usable right away


async def test_short_path_with_buttons_and_text(ctx, dp, bot_and_session):
    bot, session = bot_and_session
    await dp.feed_update(bot, message_update("/start"))
    await dp.feed_update(bot, callback_update(ob("timezone", "Europe/Moscow")))
    await dp.feed_update(bot, callback_update(ob("morning", "0800")))
    await dp.feed_update(bot, callback_update(ob("evening", "skip")))
    await dp.feed_update(bot, callback_update(ob("quiet", "2300-0700")))
    await dp.feed_update(bot, message_update("Сдать диплом, бегать 3 раза в неделю"))
    p = await _profile(ctx)
    assert p is not None
    assert p.timezone == "Europe/Moscow"
    assert p.morning_time == time(8, 0)
    assert p.evening_time is None  # skipped: nothing invented
    assert p.quiet_hours == [{"start": "23:00", "end": "07:00", "days": None}]
    assert p.goals_note == "Сдать диплом, бегать 3 раза в неделю"
    assert p.onboarding_step is None
    assert session.sent_texts()[-1] == start.DONE


async def test_skip_everything_leaves_values_unset(ctx, dp, bot_and_session):
    bot, _ = bot_and_session
    await dp.feed_update(bot, message_update("/start"))
    for step in start.STEPS:
        await dp.feed_update(bot, callback_update(ob(step, "skip")))
    p = await _profile(ctx)
    assert p is not None and p.onboarding_step is None
    assert p.timezone is None and p.morning_time is None and p.evening_time is None
    assert p.quiet_hours == [] and p.goals_note is None
    # work-session defaults are not asked during onboarding (open question 5)
    assert p.pomodoro_work_min is None and p.short_start_min is None


async def test_menu_usable_before_onboarding_complete(ctx, dp, bot_and_session):
    bot, session = bot_and_session
    await dp.feed_update(bot, message_update("/start"))
    session.clear()
    await dp.feed_update(bot, message_update(Menu.SETTINGS))
    assert session.sent_texts() and session.sent_texts()[0].startswith("⚙️ Настройки")
    session.clear()
    await dp.feed_update(bot, message_update("/help"))
    assert "Разделы меню" in session.sent_texts()[0]
    p = await _profile(ctx)
    assert p is not None and p.onboarding_step == "timezone"  # still pending, not forced


async def test_rerun_start_never_resets_data(ctx, dp, bot_and_session):
    bot, session = bot_and_session
    await dp.feed_update(bot, message_update("/start"))
    await dp.feed_update(bot, callback_update(ob("timezone", "Europe/Berlin")))
    session.clear()
    await dp.feed_update(bot, message_update("/start"))
    p = await _profile(ctx)
    assert p is not None and p.timezone == "Europe/Berlin" and p.onboarding_step == "morning"
    assert session.sent_texts()[:2] == [start.RESUME, start.QUESTIONS["morning"]]
    for step in start.STEPS[1:]:
        await dp.feed_update(bot, callback_update(ob(step, "skip")))
    session.clear()
    await dp.feed_update(bot, message_update("/start"))
    assert session.sent_texts() == [start.WELCOME_BACK]
    p = await _profile(ctx)
    assert p is not None and p.timezone == "Europe/Berlin"


async def test_stale_button_is_a_noop(ctx, dp, bot_and_session):
    bot, _ = bot_and_session
    await dp.feed_update(bot, message_update("/start"))
    await dp.feed_update(bot, callback_update(ob("timezone", "Europe/Berlin")))
    # Old timezone card pressed again with another value: ignored.
    await dp.feed_update(bot, callback_update(ob("timezone", "Asia/Tokyo")))
    p = await _profile(ctx)
    assert p is not None and p.timezone == "Europe/Berlin" and p.onboarding_step == "morning"


async def test_manual_timezone_validation(ctx, dp, bot_and_session):
    bot, session = bot_and_session
    await dp.feed_update(bot, message_update("/start"))
    await dp.feed_update(bot, callback_update(ob("timezone", "manual")))
    session.clear()
    await dp.feed_update(bot, message_update("Марс/Олимп"))
    assert "Не узнаю" in session.sent_texts()[0]
    session.clear()
    await dp.feed_update(bot, message_update("UTC+3"))  # ambiguous: offer choices
    assert "несколько поясов" in session.sent_texts()[0]
    p = await _profile(ctx)
    assert p is not None and p.timezone is None
    await dp.feed_update(bot, message_update("Новосибирск"))
    p = await _profile(ctx)
    assert p is not None and p.timezone == "Asia/Novosibirsk" and p.onboarding_step == "morning"


async def test_invalid_timezone_button_rejected(ctx, dp, bot_and_session):
    bot, _ = bot_and_session
    await dp.feed_update(bot, message_update("/start"))
    await dp.feed_update(bot, callback_update(ob("timezone", "Not/AZone")))
    p = await _profile(ctx)
    assert p is not None and p.timezone is None and p.onboarding_step == "timezone"


async def test_menu_button_during_manual_input_is_not_swallowed(ctx, dp, bot_and_session):
    bot, session = bot_and_session
    await dp.feed_update(bot, message_update("/start"))
    await dp.feed_update(bot, callback_update(ob("timezone", "manual")))
    session.clear()
    await dp.feed_update(bot, message_update(Menu.SETTINGS))
    assert session.sent_texts()[0].startswith("⚙️ Настройки")
    p = await _profile(ctx)
    assert p is not None and p.timezone is None


async def test_manual_time_and_quiet_validation(ctx, dp, bot_and_session):
    bot, session = bot_and_session
    await dp.feed_update(bot, message_update("/start"))
    await dp.feed_update(bot, callback_update(ob("timezone", "UTC")))
    await dp.feed_update(bot, callback_update(ob("morning", "manual")))
    session.clear()
    await dp.feed_update(bot, message_update("25:99"))
    assert "Не понял время" in session.sent_texts()[0]
    await dp.feed_update(bot, message_update("7:45"))
    await dp.feed_update(bot, callback_update(ob("evening", "skip")))
    await dp.feed_update(bot, callback_update(ob("quiet", "manual")))
    await dp.feed_update(bot, message_update("22:30-06:30 будни"))
    p = await _profile(ctx)
    assert p is not None
    assert p.morning_time == time(7, 45)
    assert p.quiet_hours == [{"start": "22:30", "end": "06:30", "days": [0, 1, 2, 3, 4]}]
    assert p.onboarding_step == "goals"


def test_parse_timezone_rules() -> None:
    assert parse_timezone("Europe/Moscow", NOW).zone == "Europe/Moscow"
    assert parse_timezone("europe/berlin", NOW).zone == "Europe/Berlin"
    assert parse_timezone("Москва", NOW).zone == "Europe/Moscow"
    assert parse_timezone("Berlin", NOW).zone == "Europe/Berlin"
    assert parse_timezone("utc", NOW).zone == "UTC"
    offset = parse_timezone("+3", NOW)
    assert offset.zone is None and "Europe/Moscow" in offset.candidates
    assert "Etc/GMT-3" in offset.candidates
    assert parse_timezone("Атлантида", NOW).zone is None
    assert parse_timezone("UTC+20", NOW).candidates == []
