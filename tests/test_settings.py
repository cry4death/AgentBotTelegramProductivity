"""Settings (FR-01/07/08): view, edit, audit; personality keeps history (acceptance 21)."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta

from sqlalchemy import func, select

from procbot.db.models import AuditLog, MemoryItem, OwnerProfile, Task, TaskEvent
from procbot.services.preferences import parse_pause_until, parse_window
from procbot.services.profile import get_profile, quiet_windows
from procbot.telegram.callbacks import SettingsCb
from procbot.telegram.keyboards import Menu
from procbot.testing import callback_update, message_update
from procbot.timeutil import is_in_windows


def st(action: str, arg: str = "") -> str:
    return SettingsCb(action=action, arg=arg).pack()


async def _p(ctx) -> OwnerProfile:
    async with ctx.db.transaction() as s:
        p = await get_profile(s)
    assert p is not None
    return p


async def test_overview_shows_not_set_values(ctx, dp, bot_and_session, profile):
    bot, session = bot_and_session
    await dp.feed_update(bot, message_update("/settings"))
    text = session.sent_texts()[0]
    assert "Часовой пояс: Europe/Berlin" in text
    assert "Утренний план: не задано" in text
    assert "Часы тишины: не задано" in text
    assert "Недельный обзор: не задано" in text
    assert "Помодоро: не задано" in text
    assert "Напоминание перед задачей: выкл" in text
    session.clear()
    await dp.feed_update(bot, message_update(Menu.SETTINGS))
    assert session.sent_texts()[0].startswith("⚙️ Настройки")


async def test_personality_change_keeps_tasks_and_memory(ctx, dp, bot_and_session, profile):
    bot, _ = bot_and_session
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        task = Task(title="Отчёт", created_at=now, updated_at=now)
        s.add(task)
        await s.flush()
        s.add(TaskEvent(task_id=task.id, event_type="created", actor="user", created_at=now))
        s.add(
            MemoryItem(
                kind="fact",
                origin="user_said",
                content="Люблю утро",
                created_at=now,
                updated_at=now,
            )
        )

    async def snapshot():
        async with ctx.db.transaction() as s:
            tasks = [(t.id, t.title, t.status) for t in (await s.execute(select(Task))).scalars()]
            events = (await s.execute(select(func.count(TaskEvent.id)))).scalar_one()
            mem = [
                (m.id, m.content, m.status) for m in (await s.execute(select(MemoryItem))).scalars()
            ]
        return tasks, events, mem

    before = await snapshot()
    await dp.feed_update(bot, callback_update(st("ps", "coach")))
    await dp.feed_update(bot, callback_update(st("kn", "strictness.4")))
    await dp.feed_update(bot, callback_update(st("kn", "humor.0")))
    await dp.feed_update(bot, callback_update(st("kn", "verbosity.1")))
    await dp.feed_update(bot, callback_update(st("prof", "1")))
    p = await _p(ctx)
    assert (p.style_preset, p.strictness, p.humor, p.verbosity, p.profanity_allowed) == (
        "coach",
        4,
        0,
        1,
        True,
    )
    assert await snapshot() == before
    async with ctx.db.transaction() as s:
        audits = (
            await s.execute(
                select(func.count(AuditLog.id)).where(AuditLog.action == "profile.updated")
            )
        ).scalar_one()
    assert audits >= 5


async def test_invalid_knob_value_is_rejected(ctx, dp, bot_and_session, profile):
    bot, _ = bot_and_session
    await dp.feed_update(bot, callback_update(st("kn", "strictness.9")))
    await dp.feed_update(bot, callback_update(st("kn", "timezone.1")))
    p = await _p(ctx)
    assert p.strictness == 2 and p.timezone == "Europe/Berlin"


async def test_quiet_hours_round_trip(ctx, dp, bot_and_session, profile):
    bot, _ = bot_and_session
    await dp.feed_update(bot, callback_update(st("q_add", "2300-0700")))
    await dp.feed_update(bot, callback_update(st("q_add", "2300-0700")))  # repeated: no dup
    await dp.feed_update(bot, callback_update(st("input", "quiet")))
    await dp.feed_update(bot, message_update("13:00-14:30 выходные"))
    p = await _p(ctx)
    assert p.quiet_hours == [
        {"start": "23:00", "end": "07:00", "days": None},
        {"start": "13:00", "end": "14:30", "days": [5, 6]},
    ]
    windows = quiet_windows(p)
    # Monday 2026-10-26 23:30 Berlin (22:30 UTC) is quiet; Saturday 13:30 local too.
    assert is_in_windows(datetime(2026, 10, 26, 22, 30, tzinfo=UTC), windows, p.timezone)
    assert is_in_windows(datetime(2026, 10, 31, 12, 30, tzinfo=UTC), windows, p.timezone)
    assert not is_in_windows(datetime(2026, 10, 27, 12, 30, tzinfo=UTC), windows, p.timezone)
    await dp.feed_update(bot, callback_update(st("q_del", "1300-1430-56")))
    await dp.feed_update(bot, callback_update(st("q_del", "1300-1430-56")))
    p = await _p(ctx)
    assert p.quiet_hours == [{"start": "23:00", "end": "07:00", "days": None}]
    await dp.feed_update(bot, callback_update(st("q_clr")))
    assert (await _p(ctx)).quiet_hours == []


async def test_bad_quiet_input_keeps_waiting(ctx, dp, bot_and_session, profile):
    bot, session = bot_and_session
    await dp.feed_update(bot, callback_update(st("input", "quiet")))
    session.clear()
    await dp.feed_update(bot, message_update("с утра до вечера"))
    assert session.sent_texts()[0].startswith("Не понял")
    await dp.feed_update(bot, message_update("23-8"))
    assert (await _p(ctx)).quiet_hours == [{"start": "23:00", "end": "08:00", "days": None}]


async def test_manual_pause_options(ctx, dp, bot_and_session, profile, clock):
    bot, _ = bot_and_session
    now = clock.now()  # Monday 2026-10-26 10:00 Berlin
    await dp.feed_update(bot, callback_update(st("pz", "1h")))
    assert (await _p(ctx)).pause_until == now + timedelta(hours=1)
    await dp.feed_update(bot, callback_update(st("pz", "3h")))
    assert (await _p(ctx)).pause_until == now + timedelta(hours=3)
    await dp.feed_update(bot, callback_update(st("mo_set", "0830")))
    await dp.feed_update(bot, callback_update(st("pz", "tomorrow")))
    # until tomorrow's morning time 08:30 Berlin (CET, UTC+1)
    assert (await _p(ctx)).pause_until == datetime(2026, 10, 27, 7, 30, tzinfo=UTC)
    await dp.feed_update(bot, callback_update(st("input", "pause")))
    await dp.feed_update(bot, message_update("05.11 12:00"))
    assert (await _p(ctx)).pause_until == datetime(2026, 11, 5, 11, 0, tzinfo=UTC)
    await dp.feed_update(bot, callback_update(st("pz", "clear")))
    assert (await _p(ctx)).pause_until is None


def test_pause_until_parsing_rejects_past() -> None:
    now = datetime(2026, 10, 26, 9, 0, tzinfo=UTC)
    assert parse_pause_until("27.10", now, "UTC", None) == datetime(2026, 10, 27, tzinfo=UTC)
    assert parse_pause_until("18:00", now, "UTC", None) == datetime(2026, 10, 26, 18, tzinfo=UTC)
    assert parse_pause_until("08:00", now, "UTC", None) == datetime(2026, 10, 27, 8, tzinfo=UTC)
    # date without a year that already passed -> next year
    assert parse_pause_until("01.01", now, "UTC", None).year == 2027
    try:
        parse_pause_until("26.10.2026 08:00", now, "UTC", None)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("past moment accepted")


def test_parse_window_forms() -> None:
    assert parse_window("23:00-08:00") == {"start": "23:00", "end": "08:00", "days": None}
    assert parse_window("22.30–7.15 пн-пт")["days"] == [0, 1, 2, 3, 4]
    assert parse_window("10-12 сб,вс")["days"] == [5, 6]
    assert parse_window("10-12 пт-пн")["days"] == [0, 4, 5, 6]
    for bad in ("10-10", "25:00-08:00", "утро", "10-12 xx"):
        try:
            parse_window(bad)
        except ValueError:
            continue
        raise AssertionError(bad)  # pragma: no cover


async def test_rhythm_weekly_weekdays_and_notifications(ctx, dp, bot_and_session, profile):
    bot, _ = bot_and_session
    await dp.feed_update(bot, callback_update(st("ev_set", "2130")))
    await dp.feed_update(bot, callback_update(st("wk_d", "6")))
    await dp.feed_update(bot, callback_update(st("wk_t", "1800")))
    for d in range(5):
        await dp.feed_update(bot, callback_update(st("wd_t", f"{d}.1")))
    await dp.feed_update(bot, callback_update(st("wd_t", "4.1")))  # repeated press: no-op
    await dp.feed_update(bot, callback_update(st("sleep_set", "2300-0700")))
    await dp.feed_update(bot, callback_update(st("notif", "off")))
    await dp.feed_update(bot, callback_update(st("notif", "off")))
    p = await _p(ctx)
    assert p.evening_time == time(21, 30)
    assert (p.weekly_review_dow, p.weekly_review_time) == (6, time(18, 0))
    assert p.routine == {"weekdays": [0, 1, 2, 3, 4], "sleep": {"start": "23:00", "end": "07:00"}}
    assert p.notifications_enabled is False
    await dp.feed_update(bot, callback_update(st("ev_set", "off")))
    await dp.feed_update(bot, callback_update(st("wk_off")))
    p = await _p(ctx)
    assert p.evening_time is None and p.weekly_review_dow is None


async def test_manual_timezone_and_weekly_text(ctx, dp, bot_and_session, profile):
    bot, session = bot_and_session
    await dp.feed_update(bot, callback_update(st("input", "timezone")))
    await dp.feed_update(bot, message_update("Asia/Tokyo"))
    await dp.feed_update(bot, callback_update(st("input", "weekly")))
    await dp.feed_update(bot, message_update("пятница 17:30"))
    p = await _p(ctx)
    assert p.timezone == "Asia/Tokyo"
    assert (p.weekly_review_dow, p.weekly_review_time) == (4, time(17, 30))
    session.clear()
    await dp.feed_update(bot, callback_update(st("tz_set", "Bad/Zone")))
    assert (await _p(ctx)).timezone == "Asia/Tokyo"


async def test_reminder_work_and_memory_toggles(ctx, dp, bot_and_session, profile):
    bot, _ = bot_and_session
    await dp.feed_update(bot, callback_update(st("rl", "10")))
    await dp.feed_update(bot, callback_update(st("rd", "15")))
    await dp.feed_update(bot, callback_update(st("rd", "7")))  # not offered: rejected
    await dp.feed_update(bot, callback_update(st("pw", "25")))
    await dp.feed_update(bot, callback_update(st("pb", "5")))
    await dp.feed_update(bot, callback_update(st("ss", "2")))
    await dp.feed_update(bot, callback_update(st("mt", "hypotheses.0")))
    await dp.feed_update(bot, callback_update(st("at", "reorder_tasks.0")))
    await dp.feed_update(bot, callback_update(st("at", "unknown.0")))
    p = await _p(ctx)
    assert (p.pre_task_lead_min, p.start_delay_min) == (10, 15)
    assert (p.pomodoro_work_min, p.pomodoro_break_min, p.short_start_min) == (25, 5, 2)
    assert p.memory_settings == {"hypotheses": False}
    assert p.autonomy_settings == {"reorder_tasks": False}
    await dp.feed_update(bot, callback_update(st("rl", "off")))
    assert (await _p(ctx)).pre_task_lead_min == 0  # 0 = explicitly off


async def test_settings_sections_render(ctx, dp, bot_and_session, profile):
    bot, session = bot_and_session
    for section in (
        "tz",
        "wd",
        "sleep",
        "quiet",
        "morning",
        "evening",
        "weekly",
        "pause",
        "persona",
        "remind",
        "work",
        "mem",
        "home",
    ):
        session.clear()
        await dp.feed_update(bot, callback_update(st(section)))
        assert session.sent_texts(), section
