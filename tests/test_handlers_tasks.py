"""Tasks / approvals / day / chat handlers through the real dispatcher.

Covers acceptance 3 (update dedup, double press), 4 (multi-intent text ->
candidates, nothing created until confirmed; reject; expiry), 17 (button vs
criteria completion, partial != done)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from aiogram.methods import EditMessageText, SendMessage
from aiogram.types import Chat, InlineKeyboardMarkup, Message, Update
from sqlalchemy import func, select

from procbot.db.models import PendingApproval, Task, TaskEvent
from procbot.domain.actions import TaskDraft
from procbot.domain.enums import Actor, ApprovalStatus, TaskStatus
from procbot.services import approvals
from procbot.services import tasks as ts
from procbot.telegram.keyboards import Menu
from procbot.testing import OWNER_ID, callback_update, message_update, user


@pytest.fixture
def wired(ctx):
    ts.register(ctx)
    approvals.register(ctx)
    return ctx


@pytest.fixture
def bot(bot_and_session):
    return bot_and_session[0]


@pytest.fixture
def tg(bot_and_session):
    return bot_and_session[1]


async def feed(dp, bot, upd: Update) -> None:
    await dp.feed_update(bot, upd)


def last_markup(tg) -> InlineKeyboardMarkup:
    for r in reversed(tg.requests):
        if isinstance(r, SendMessage | EditMessageText) and isinstance(
            r.reply_markup, InlineKeyboardMarkup
        ):
            return r.reply_markup
    raise AssertionError("no inline keyboard sent")


def last_text(tg) -> str:
    texts = tg.sent_texts()
    assert texts, "nothing sent"
    return texts[-1]


def button(markup: InlineKeyboardMarkup, prefix: str) -> str:
    for row in markup.inline_keyboard:
        for b in row:
            if b.text.startswith(prefix):
                assert b.callback_data is not None
                return b.callback_data
    raise AssertionError(
        f"no button {prefix!r} in {[[b.text for b in r] for r in markup.inline_keyboard]}"
    )


async def count(ctx, model, *where) -> int:
    async with ctx.db.transaction() as s:
        stmt = select(func.count()).select_from(model)
        for w in where:
            stmt = stmt.where(w)
        return (await s.execute(stmt)).scalar_one()


async def all_tasks(ctx) -> list[Task]:
    async with ctx.db.transaction() as s:
        return list((await s.execute(select(Task).order_by(Task.id))).scalars())


async def make_task(ctx, **kw) -> Task:
    async with ctx.db.transaction() as s:
        return await ts.create_task(s, ctx, TaskDraft(**kw), actor=Actor.USER)


# --------------------------------------------------------------------------- add


async def test_add_menu_creates_directly_and_quick_plan(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    await feed(dp, bot, message_update(Menu.ADD))
    assert "Напиши задачу" in last_text(tg)
    await feed(dp, bot, message_update("Купить хлеб"))
    tasks = await all_tasks(ctx)
    assert [t.title for t in tasks] == ["Купить хлеб"] and tasks[0].created_by == "user"
    assert "Добавил" in last_text(tg)
    await feed(dp, bot, callback_update(button(last_markup(tg), "Сегодня")))
    t = (await all_tasks(ctx))[0]
    assert t.planned_date == date(2026, 10, 26)
    assert "Запланировал на сегодня" in last_text(tg)


async def test_add_command_same_update_id_twice_creates_one_task(wired, profile, dp, bot) -> None:
    """Acceptance 3."""
    ctx = wired
    upd = message_update("/add Позвонить маме", update_id=50_001)
    await feed(dp, bot, upd)
    await feed(dp, bot, upd)
    assert [t.title for t in await all_tasks(ctx)] == ["Позвонить маме"]


async def test_add_time_via_text(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    await feed(dp, bot, message_update("/add Отчёт"))
    await feed(dp, bot, callback_update(button(last_markup(tg), "🕘 Указать время")))
    await feed(dp, bot, message_update("завтра 15:30"))
    t = (await all_tasks(ctx))[0]
    assert t.planned_start_at == datetime(2026, 10, 27, 14, 30, tzinfo=UTC)  # 15:30 CET
    assert t.planned_date == date(2026, 10, 27)


# --------------------------------------------------------------------------- free text


async def test_multi_intent_text_gives_candidates_not_tasks(wired, profile, dp, bot, tg) -> None:
    """Acceptance 4 + scenario A: nothing is created until confirmed."""
    ctx = wired
    upd = message_update("Завтра доделать сайт; на выходных позаниматься проектом")
    await feed(dp, bot, upd)
    await feed(dp, bot, upd)  # repeated delivery: no duplicate proposals
    assert await count(ctx, Task) == 0
    assert await count(ctx, PendingApproval) == 2
    card = last_text(tg)
    assert "Завтра доделать сайт" in card and "на выходных позаниматься проектом" in card
    assert "Без твоего подтверждения ничего не изменится" in card
    kb = last_markup(tg)
    accept_1 = button(kb, "✅ Создать 1")
    button(kb, "✅ Создать все")
    button(kb, "✖️ Ничего")
    await feed(dp, bot, callback_update(accept_1))
    await feed(dp, bot, callback_update(accept_1))  # double press -> still one task
    assert [t.title for t in await all_tasks(ctx)] == ["Завтра доделать сайт"]
    await feed(dp, bot, callback_update(button(last_markup(tg), "✖️ Не надо 2")))
    assert await count(ctx, Task) == 1
    assert await count(ctx, PendingApproval, PendingApproval.status == "rejected") == 1


async def test_accept_all_and_reject_all(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    await feed(dp, bot, message_update("- молоко\n- хлеб\n- сыр"))
    kb = last_markup(tg)
    reject_all = button(kb, "✖️ Ничего")
    accept_all = button(kb, "✅ Создать все")
    await feed(dp, bot, callback_update(reject_all))
    await feed(dp, bot, callback_update(accept_all))  # too late, all rejected
    assert await count(ctx, Task) == 0
    await feed(dp, bot, message_update("кофе\nчай"))
    accept_all = button(last_markup(tg), "✅ Создать все")
    await feed(dp, bot, callback_update(accept_all))
    await feed(dp, bot, callback_update(accept_all))  # double press
    assert sorted(t.title for t in await all_tasks(ctx)) == ["кофе", "чай"]
    edited = [r for r in tg.requests if isinstance(r, EditMessageText)][-1]
    assert edited.reply_markup is None  # nothing pending -> no buttons left


async def test_expired_card_cannot_be_applied(wired, profile, dp, bot, tg, clock) -> None:
    ctx = wired
    await feed(dp, bot, message_update("разобрать почту"))
    accept = button(last_markup(tg), "✅ Создать")
    clock.advance(hours=25)
    await feed(dp, bot, callback_update(accept))
    assert await count(ctx, Task) == 0
    async with ctx.db.transaction() as s:
        row = (await s.execute(select(PendingApproval))).scalar_one()
    assert row.status == ApprovalStatus.EXPIRED
    assert "срок истёк" in last_text(tg)


async def test_yes_reply_to_card_accepts_it(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    await feed(dp, bot, message_update("помыть окна"))
    card = next(r for r in reversed(tg.requests) if isinstance(r, SendMessage))
    async with ctx.db.transaction() as s:
        row = (await s.execute(select(PendingApproval))).scalar_one()
    assert row.telegram_message_id is not None
    reply = Message(
        message_id=777_001,
        date=datetime.now(UTC),
        chat=Chat(id=OWNER_ID, type="private"),
        from_user=user(),
        text="да",
        reply_to_message=Message(
            message_id=row.telegram_message_id,
            date=datetime.now(UTC),
            chat=Chat(id=OWNER_ID, type="private"),
            text=card.text,
        ),
    )
    await feed(dp, bot, Update(update_id=777_001, message=reply))
    assert [t.title for t in await all_tasks(ctx)] == ["помыть окна"]
    assert await count(ctx, PendingApproval, PendingApproval.status == "pending") == 0


async def test_plain_yes_without_card_is_not_consent(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    await feed(dp, bot, message_update("помыть окна"))
    await feed(dp, bot, message_update("да"))  # not a reply to the card
    assert await count(ctx, Task) == 0


async def test_unknown_command_and_menu_not_tasks(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    await feed(dp, bot, message_update("/nonexistent"))
    assert "Не знаю такой команды" in last_text(tg)
    assert await count(ctx, PendingApproval) == 0


# --------------------------------------------------------------------------- completion


async def test_button_task_closes_with_one_tap(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    t = await make_task(ctx, title="Вынести мусор")
    await feed(dp, bot, callback_update(f"tk:done:{t.id}:"))
    await feed(dp, bot, callback_update(f"tk:done:{t.id}:"))
    async with ctx.db.transaction() as s:
        task = await ts.require_task(s, t.id)
        kinds = [e.event_type for e in await ts.task_events(s, t.id)]
    assert task.status == TaskStatus.DONE
    assert kinds.count("completed") == 1
    assert "уже было отмечено как готово" in last_text(tg).lower()


async def test_criteria_task_asks_and_partial_is_not_done(wired, profile, dp, bot, tg) -> None:
    """Acceptance 17 / scenario E."""
    ctx = wired
    t = await make_task(ctx, title="Статья", done_criteria="Опубликована на сайте")
    await feed(dp, bot, callback_update(f"tk:done:{t.id}:"))
    assert "Опубликована на сайте" in last_text(tg)
    kb = last_markup(tg)
    async with ctx.db.transaction() as s:
        assert (await ts.require_task(s, t.id)).status == TaskStatus.TODO
    await feed(dp, bot, callback_update(button(kb, "◐ Частично")))
    await feed(dp, bot, message_update("черновик готов"))
    async with ctx.db.transaction() as s:
        task = await ts.require_task(s, t.id)
    assert task.status == TaskStatus.IN_PROGRESS and task.completed_at is None
    assert task.progress_note == "черновик готов"
    await feed(dp, bot, callback_update(f"tk:done:{t.id}:"))
    await feed(dp, bot, callback_update(button(last_markup(tg), "✖️ Нет")))
    async with ctx.db.transaction() as s:
        assert (await ts.require_task(s, t.id)).status == TaskStatus.IN_PROGRESS
    await feed(dp, bot, callback_update(f"tk:done:{t.id}:"))
    await feed(dp, bot, callback_update(button(last_markup(tg), "✅ Соответствует полностью")))
    async with ctx.db.transaction() as s:
        task = await ts.require_task(s, t.id)
        last = (await ts.task_events(s, t.id))[-1]
    assert task.status == TaskStatus.DONE
    assert last.event_type == "completed" and last.payload["criteria_confirmed"] is True


# --------------------------------------------------------------------------- postpone


async def test_postpone_tomorrow_with_reason_and_double_press(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    t = await make_task(ctx, title="Налоги", planned_date=date(2026, 10, 26))
    await feed(dp, bot, callback_update(f"tk:postpone:{t.id}:"))
    tomorrow = button(last_markup(tg), "Завтра")
    await feed(dp, bot, callback_update(tomorrow))
    await feed(dp, bot, callback_update(tomorrow))  # stale button: no second postpone
    assert "Что помешало" in tg.sent_texts()[-1] or "Что помешало" in tg.sent_texts()[-2]
    reasons = None
    for r in reversed(tg.requests):
        if isinstance(r, EditMessageText) and r.text and "Что помешало" in r.text:
            reasons = r.reply_markup
            break
    assert isinstance(reasons, InlineKeyboardMarkup)
    low = button(reasons, "Мало сил")
    await feed(dp, bot, callback_update(low))
    await feed(dp, bot, callback_update(low))
    async with ctx.db.transaction() as s:
        task = await ts.require_task(s, t.id)
        events = await ts.task_events(s, t.id)
    assert task.planned_date == date(2026, 10, 27) and task.postpone_count == 1
    kinds = [e.event_type for e in events]
    assert kinds.count("postponed") == 1 and kinds.count("obstacle_reported") == 1
    ob = next(e for e in events if e.event_type == "obstacle_reported")
    assert ob.reason == "low_energy" and ob.actor == "user"


async def test_postpone_to_backlog(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    t = await make_task(ctx, title="Полка", planned_date=date(2026, 10, 26))
    await feed(dp, bot, callback_update(f"tk:postpone:{t.id}:"))
    await feed(dp, bot, callback_update(button(last_markup(tg), "📥 В бэклог")))
    await feed(dp, bot, callback_update(button(last_markup(tg), "Пропустить")))
    async with ctx.db.transaction() as s:
        task = await ts.require_task(s, t.id)
    assert task.is_backlog and task.planned_date is None and task.postpone_count == 1
    assert await count(ctx, TaskEvent, TaskEvent.event_type == "obstacle_reported") == 0


# --------------------------------------------------------------------------- lists / day / misc


async def test_tasks_list_and_filters(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    await make_task(ctx, title="Сегодняшняя", planned_date=date(2026, 10, 26))
    await make_task(ctx, title="Без даты")
    await feed(dp, bot, message_update(Menu.TASKS))
    assert "Сегодняшняя" in last_text(tg) and "Без даты" not in last_text(tg)
    await feed(dp, bot, callback_update(button(last_markup(tg), "Бэклог")))
    assert "Без даты" in last_text(tg)


async def test_today_command_shows_day_view(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    await make_task(ctx, title="Созвон", planned_start_at=datetime(2026, 10, 26, 12, 0, tzinfo=UTC))
    await feed(dp, bot, message_update("/today"))
    text = last_text(tg)
    assert "Мой день" in text and "13:00 Созвон" in text
    assert button(last_markup(tg), "▶️ Созвон").startswith("ws:pick")


async def test_subtasks_added_manually(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    t = await make_task(ctx, title="Ремонт", due_date=date(2026, 11, 30))
    await feed(dp, bot, callback_update(f"tk:subtasks:{t.id}:"))
    await feed(dp, bot, callback_update(button(last_markup(tg), "➕ Добавить шаги")))
    await feed(dp, bot, message_update("купить краску\nпокрасить стену"))
    async with ctx.db.transaction() as s:
        subs = await ts.list_subtasks(s, t.id)
        parent = await ts.require_task(s, t.id)
    assert [x.title for x in subs] == ["купить краску", "покрасить стену"]
    assert parent.title == "Ремонт" and parent.due_date == date(2026, 11, 30)
    assert await count(ctx, PendingApproval) == 0


async def test_delete_requires_confirmation(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    t = await make_task(ctx, title="Лишнее")
    await feed(dp, bot, callback_update(f"tk:delete:{t.id}:"))
    assert await count(ctx, Task) == 1
    ok = button(last_markup(tg), "Да, удалить")
    await feed(dp, bot, callback_update(ok))
    await feed(dp, bot, callback_update(ok))
    assert await count(ctx, Task) == 0
    assert "удален" in last_text(tg).lower() or "Удалил" in last_text(tg)


async def test_edit_candidate_title_before_accepting(wired, profile, dp, bot, tg) -> None:
    ctx = wired
    await feed(dp, bot, message_update("сделать штуку"))
    await feed(dp, bot, callback_update(button(last_markup(tg), "✏️")))
    await feed(dp, bot, message_update("Сделать презентацию к пятнице"))
    assert "Сделать презентацию к пятнице" in last_text(tg)
    assert await count(ctx, Task) == 0
    await feed(dp, bot, callback_update(button(last_markup(tg), "✅ Создать")))
    assert [t.title for t in await all_tasks(ctx)] == ["Сделать презентацию к пятнице"]
