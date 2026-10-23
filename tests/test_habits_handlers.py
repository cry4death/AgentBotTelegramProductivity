"""«🔁 Привычки» through the real dispatcher: create (FSM), quick marks are
idempotent per day, owner's edit applies, "suggest a step" is an approval card."""

from __future__ import annotations

from datetime import date, time

import pytest
from aiogram.methods import EditMessageText, SendMessage
from aiogram.types import InlineKeyboardMarkup
from sqlalchemy import func, select

from procbot.db.models import AuditLog, Habit, HabitOccurrence, PendingApproval
from procbot.domain.enums import Actor
from procbot.services import approvals
from procbot.services import habits as hs
from procbot.services import tasks as ts
from procbot.telegram.callbacks import HabitCb
from procbot.telegram.keyboards import Menu
from procbot.testing import callback_update, message_update

TODAY = date(2026, 10, 26)


@pytest.fixture
def wired(ctx, profile):
    ts.register(ctx)
    approvals.register(ctx)
    hs.register(ctx)
    return ctx


@pytest.fixture
def bot(bot_and_session):
    return bot_and_session[0]


@pytest.fixture
def tg(bot_and_session):
    return bot_and_session[1]


def last_markup(tg) -> InlineKeyboardMarkup:
    for r in reversed(tg.requests):
        if isinstance(r, SendMessage | EditMessageText) and isinstance(
            r.reply_markup, InlineKeyboardMarkup
        ):
            return r.reply_markup
    raise AssertionError("no inline keyboard sent")


def button(markup: InlineKeyboardMarkup, prefix: str) -> str:
    for row in markup.inline_keyboard:
        for b in row:
            if b.text.startswith(prefix):
                assert b.callback_data is not None
                return b.callback_data
    raise AssertionError(
        f"no {prefix!r} in {[[b.text for b in r] for r in markup.inline_keyboard]}"
    )


def last_text(tg) -> str:
    texts = tg.sent_texts()
    assert texts
    return texts[-1]


async def habits(ctx) -> list[Habit]:
    async with ctx.db.transaction() as s:
        return list((await s.execute(select(Habit).order_by(Habit.id))).scalars())


async def count(ctx, model) -> int:
    async with ctx.db.transaction() as s:
        return (await s.execute(select(func.count()).select_from(model))).scalar_one()


async def test_create_habit_flow_and_idempotent_marks(wired, dp, bot, tg) -> None:
    ctx = wired
    await dp.feed_update(bot, message_update(Menu.HABITS))
    assert "Привычек пока нет" in last_text(tg)
    await dp.feed_update(bot, callback_update(button(last_markup(tg), "➕ Новая")))
    assert "Как назовём" in last_text(tg)
    await dp.feed_update(bot, message_update("Зарядка"))
    await dp.feed_update(bot, callback_update(button(last_markup(tg), "По дням недели")))
    await dp.feed_update(bot, message_update("пн ср пт"))
    assert "Какой объём" in last_text(tg)
    await dp.feed_update(bot, message_update("10 мин, минимум 2"))
    await dp.feed_update(bot, message_update("07:30"))
    assert "Привычка добавлена" in last_text(tg)
    (h,) = await habits(ctx)
    assert h.title == "Зарядка" and h.schedule == {"type": "weekly", "days": [0, 2, 4]}
    assert h.target == {"amount": 10, "min_amount": 2, "unit": "мин"}
    assert h.preferred_time == time(7, 30)

    await dp.feed_update(bot, message_update("/habits"))
    assert "сегодня по графику" in last_text(tg)
    done_cb = button(last_markup(tg), "✅")
    await dp.feed_update(bot, callback_update(done_cb))
    await dp.feed_update(bot, callback_update(done_cb))  # double press from the same message
    assert "Уже отмечено: сделано" in last_text(tg)
    assert await count(ctx, HabitOccurrence) == 1
    # Same update delivered twice -> handled once.
    upd = callback_update(hs.mark_cb(h.id, hs.PARTIAL, TODAY, "c"), update_id=777_001)
    await dp.feed_update(bot, upd)
    await dp.feed_update(bot, upd)
    async with ctx.db.transaction() as s:
        occ = await hs.get_occurrence(s, h.id, TODAY)
        assert occ is not None and occ.status == hs.PARTIAL
        corrections = (
            await s.execute(
                select(func.count())
                .select_from(AuditLog)
                .where(AuditLog.action == "habit.occurrence_corrected")
            )
        ).scalar_one()
    assert corrections == 1


async def test_user_edit_applies_and_suggestion_waits_for_approval(wired, dp, bot, tg) -> None:
    ctx = wired
    async with ctx.db.transaction() as s:
        h = await hs.create_habit(
            s, ctx, title="Чтение", actor=Actor.USER, target={"amount": 10, "unit": "мин"}
        )
    await dp.feed_update(bot, callback_update(HabitCb(action="view", habit_id=h.id).pack()))
    assert "Устойчивость" in last_text(tg)
    await dp.feed_update(bot, callback_update(button(last_markup(tg), "✏️ Изменить")))
    await dp.feed_update(bot, callback_update(button(last_markup(tg), "Объём")))
    await dp.feed_update(bot, message_update("15 мин минимум 5"))
    assert "Изменил" in last_text(tg)
    (row,) = await habits(ctx)
    assert row.target == {"amount": 15, "min_amount": 5, "unit": "мин"}
    assert await count(ctx, PendingApproval) == 0

    await dp.feed_update(bot, callback_update(HabitCb(action="inc", habit_id=h.id).pack()))
    texts = tg.sent_texts()
    assert any("только предложение" in t for t in texts)
    assert "Нужно твоё подтверждение" in last_text(tg)
    assert "15 мин (минимум 5 мин) → 18 мин (минимум 5 мин)" in last_text(tg)
    (row,) = await habits(ctx)
    assert row.target["amount"] == 15  # unchanged until accepted
    await dp.feed_update(bot, callback_update(button(last_markup(tg), "✅ Принять")))
    (row,) = await habits(ctx)
    assert row.target["amount"] == 18


async def test_miss_reason_and_pause_through_menu(wired, dp, bot, tg, clock) -> None:
    ctx = wired
    from procbot.timeutil import resolve_local

    saved = clock.now()
    clock.set(resolve_local(date(2026, 10, 20), time(8, 0), "Europe/Berlin"))
    async with ctx.db.transaction() as s:
        h = await hs.create_habit(s, ctx, title="Медитация", actor=Actor.USER)
    clock.set(saved)
    await dp.feed_update(bot, callback_update(HabitCb(action="miss", habit_id=h.id).pack()))
    assert "Что случилось" in last_text(tg)
    await dp.feed_update(bot, callback_update(button(last_markup(tg), "25.10")))
    await dp.feed_update(bot, callback_update(button(last_markup(tg), "Не было времени")))
    assert "Не было времени" in last_text(tg)
    async with ctx.db.transaction() as s:
        occ = await hs.get_occurrence(s, h.id, date(2026, 10, 25))
        assert occ is not None and occ.status == hs.MISSED and occ.time_source == "retro"
    await dp.feed_update(
        bot, callback_update(HabitCb(action="st", habit_id=h.id, arg="paused").pack())
    )
    assert "На паузе" in last_text(tg)
    async with ctx.db.transaction() as s:
        assert await hs.due_habits(s, ctx) == []
