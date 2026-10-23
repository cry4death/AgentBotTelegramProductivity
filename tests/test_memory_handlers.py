"""«🧠 Память» through the dispatcher: list by kind with origin labels, item card
with sources, confirm / refute / edit / delete, repeated presses are no-ops."""

from __future__ import annotations

from aiogram.methods import EditMessageText, SendMessage
from aiogram.types import InlineKeyboardMarkup

from procbot.db.models import MemoryItem, Task
from procbot.services import memory
from procbot.services.memory import SourceRef, SourceType
from procbot.telegram.callbacks import MemoryCb
from procbot.telegram.keyboards import Menu
from procbot.testing import callback_update, message_update

HYP = "Возможно, вечером ты устаёшь сильнее"


def last_markup(tg) -> InlineKeyboardMarkup:
    for r in reversed(tg.requests):
        if isinstance(r, SendMessage | EditMessageText) and isinstance(
            r.reply_markup, InlineKeyboardMarkup
        ):
            return r.reply_markup
    raise AssertionError("no keyboard")


def btn(markup: InlineKeyboardMarkup, prefix: str) -> str:
    for row in markup.inline_keyboard:
        for b in row:
            if b.text.startswith(prefix) and b.callback_data:
                return b.callback_data
    raise AssertionError(
        f"{prefix!r} not in {[[b.text for b in r] for r in markup.inline_keyboard]}"
    )


async def seed(ctx) -> tuple[int, int]:
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        t = Task(title="Вечерний отчёт", created_at=now, updated_at=now)
        s.add(t)
        await s.flush()
        hyp = await memory.add_hypothesis(
            s,
            ctx,
            HYP,
            verification_hint="Неделю ставить сложное на утро",
            confidence=0.4,
            sources=[SourceRef(SourceType.TASK, t.id)],
        )
        fact = await memory.add_fact(s, ctx, "Работаю из дома")
        assert hyp and fact
        return hyp.id, fact.id


async def item(ctx, item_id) -> MemoryItem:
    async with ctx.db.transaction() as s:
        row = await s.get(MemoryItem, item_id)
        assert row is not None
        return row


async def test_memory_screen_lists_and_shows_card(ctx, profile, dp, bot_and_session):
    bot, tg = bot_and_session
    hid, _ = await seed(ctx)
    await dp.feed_update(bot, message_update(Menu.MEMORY))
    home = tg.sent_texts()[-1]
    assert "Действующих записей: 2" in home
    assert "предположение ИИ — 1" in home and "вы сказали — 1" in home
    await dp.feed_update(bot, callback_update(btn(last_markup(tg), "Гипотезы")))
    listing = tg.sent_texts()[-1]
    assert f"#{hid} {HYP} — предположение ИИ" in listing
    await dp.feed_update(bot, callback_update(btn(last_markup(tg), f"#{hid}")))
    card = tg.sent_texts()[-1]
    assert "Происхождение: предположение ИИ" in card
    assert "Как проверить: Неделю ставить сложное на утро" in card
    assert "Уверенность: 40%" in card
    assert "задача «Вечерний отчёт»" in card


async def test_confirm_refute_edit_delete_through_buttons(ctx, profile, dp, bot_and_session):
    bot, tg = bot_and_session
    hid, _ = await seed(ctx)
    view = MemoryCb(action="view", item_id=hid).pack()
    await dp.feed_update(bot, callback_update(view))
    await dp.feed_update(bot, callback_update(btn(last_markup(tg), "✅ Подтвердить")))
    assert (await item(ctx, hid)).origin == "user_confirmed"
    assert "Происхождение: подтверждено" in tg.sent_texts()[-1]

    refute = MemoryCb(action="refute", item_id=hid).pack()
    await dp.feed_update(bot, callback_update(refute))
    await dp.feed_update(bot, callback_update(refute))  # repeated press: no-op
    row = await item(ctx, hid)
    assert (row.status, row.origin) == ("refuted", "user_refuted")
    assert "Статус: опровергнуто" in tg.sent_texts()[-1]

    await dp.feed_update(bot, callback_update(MemoryCb(action="list", arg="refuted").pack()))
    assert HYP in tg.sent_texts()[-1]

    await dp.feed_update(bot, callback_update(MemoryCb(action="edit", item_id=hid).pack()))
    await dp.feed_update(bot, message_update("Вечером устаю, если не было обеда"))
    assert "Исправлено" in tg.sent_texts()[-1]
    row = await item(ctx, hid)
    assert (row.status, row.origin, row.content) == (
        "active",
        "user_confirmed",
        "Вечером устаю, если не было обеда",
    )

    await dp.feed_update(bot, callback_update(MemoryCb(action="del", item_id=hid).pack()))
    assert "мгновенно стереть" in tg.sent_texts()[-1]
    ok = btn(last_markup(tg), "🗑 Да, удалить")
    await dp.feed_update(bot, callback_update(ok))
    await dp.feed_update(bot, callback_update(ok))
    row = await item(ctx, hid)
    assert row.status == "deleted" and row.content == ""
    await dp.feed_update(bot, callback_update(view))
    assert "не найдена" in tg.sent_texts()[-1]


async def test_menu_text_in_edit_state_is_not_saved(ctx, profile, dp, bot_and_session):
    bot, tg = bot_and_session
    hid, _ = await seed(ctx)
    await dp.feed_update(bot, callback_update(MemoryCb(action="edit", item_id=hid).pack()))
    await dp.feed_update(bot, message_update(Menu.MEMORY))
    assert (await item(ctx, hid)).content == HYP
    assert "🧠 Память" in tg.sent_texts()[-1]


async def test_profile_view(ctx, profile, dp, bot_and_session):
    bot, tg = bot_and_session
    await dp.feed_update(bot, message_update("/memory"))
    await dp.feed_update(bot, callback_update(btn(last_markup(tg), "👤 Профиль")))
    text = tg.sent_texts()[-1]
    assert text.startswith("👤 Профиль") and "Europe/Berlin" in text
