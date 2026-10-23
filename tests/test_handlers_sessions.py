"""Telegram flows for work sessions: menu -> task -> mode -> start, ask-once settings,
keyword control, stop -> outcome (criteria), retro entry (scenario F)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from aiogram.methods import SendMessage
from aiogram.types import InlineKeyboardMarkup
from sqlalchemy import select

from procbot.context import AppContext
from procbot.db.models import SessionCorrection, Task, WorkSession
from procbot.domain.enums import SessionMode, SessionStatus, TaskStatus, TimeSource
from procbot.services import sessions as svc
from procbot.services.profile import get_profile
from procbot.telegram.callbacks import SessionCb
from procbot.telegram.keyboards import Menu
from procbot.testing import callback_update, message_update

T0 = datetime(2026, 10, 26, 9, 0, tzinfo=UTC)  # 10:00 Berlin


async def _task(ctx: AppContext, title: str = "Отчёт", **kw: object) -> int:
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        t = Task(title=title, created_at=now, updated_at=now, planned_date=T0.date(), **kw)
        s.add(t)
        await s.flush()
        return t.id


def _last(session) -> SendMessage:  # type: ignore[no-untyped-def]
    return session.sent_messages()[-1]


def _buttons(msg: SendMessage) -> list[tuple[str, str]]:
    markup = msg.reply_markup
    assert isinstance(markup, InlineKeyboardMarkup)
    return [(b.text, b.callback_data or "") for row in markup.inline_keyboard for b in row]


def _data(msg: SendMessage, label_part: str) -> str:
    for text, data in _buttons(msg):
        if label_part in text:
            return data
    raise AssertionError(f"no button {label_part!r} in {_buttons(msg)}")


async def _active(ctx: AppContext) -> WorkSession | None:
    async with ctx.db.transaction() as s:
        return await svc.get_active_session(s)


async def _only_session(ctx: AppContext) -> WorkSession:
    async with ctx.db.transaction() as s:
        return (await s.execute(select(WorkSession))).scalar_one()


async def test_menu_pick_mode_ask_settings_once_then_start(
    ctx, profile, dp, bot_and_session
) -> None:
    bot, session = bot_and_session
    tid = await _task(ctx, "Сайт")
    await dp.feed_update(bot, message_update(Menu.WORK))
    menu = _last(session)
    assert menu.text.startswith("Над чем работаем")
    labels = [t for t, _ in _buttons(menu)]
    assert "▶️ Сайт" in labels and "Без задачи" in labels and "🕓 Отметить задним числом" in labels

    await dp.feed_update(bot, callback_update(_data(menu, "Сайт")))
    modes = _last(session)
    assert "Выбери режим" in modes.text
    assert not any("рекомендую" in t for t, _ in _buttons(modes))  # no history, no guess

    await dp.feed_update(bot, callback_update(_data(modes, "Помодоро")))
    ask_work = _last(session)
    assert "рабочий период" in ask_work.text
    await dp.feed_update(bot, callback_update(_data(ask_work, "25 мин")))
    ask_break = _last(session)
    assert "перерыв" in ask_break.text
    await dp.feed_update(bot, message_update("5"))  # free number input
    started = _last(session)
    assert "Поехали" in started.text and "Помодоро 25/5" in started.text
    async with ctx.db.transaction() as s:
        p = await get_profile(s)
        assert p is not None and (p.pomodoro_work_min, p.pomodoro_break_min) == (25, 5)
        task = await s.get(Task, tid)
        assert task is not None and task.status == TaskStatus.IN_PROGRESS
    ws = await _active(ctx)
    assert ws is not None and ws.mode == SessionMode.POMODORO and ws.task_id == tid

    # second time: no questions, and the last choice is recommended
    await dp.feed_update(bot, message_update("стоп"))
    await dp.feed_update(bot, callback_update(SessionCb(action="pick", task_id=tid).pack()))
    modes = _last(session)
    assert any("Помодоро" in t and "рекомендую" in t for t, _ in _buttons(modes))
    await dp.feed_update(bot, callback_update(_data(modes, "Помодоро")))
    assert "Поехали" in _last(session).text


async def test_pick_with_mode_from_other_modules_and_conflict(
    ctx, profile, dp, bot_and_session
) -> None:
    bot, session = bot_and_session
    a = await _task(ctx, "A")
    b = await _task(ctx, "B")
    await dp.feed_update(
        bot, callback_update(SessionCb(action="pick", task_id=a, arg="free").pack())
    )
    assert "Поехали" in _last(session).text
    first = await _active(ctx)
    assert first is not None and first.task_id == a
    # same task again: no-op, shows the running session
    await dp.feed_update(
        bot, callback_update(SessionCb(action="pick", task_id=a, arg="free").pack())
    )
    assert "уже идёт" in _last(session).text
    # another task: asks to finish the current one
    await dp.feed_update(
        bot, callback_update(SessionCb(action="pick", task_id=b, arg="free").pack())
    )
    conflict = _last(session)
    assert "только одна" in conflict.text
    await dp.feed_update(bot, callback_update(_data(conflict, "Завершить текущую")))
    texts = [m.text for m in session.sent_messages()[-2:]]
    assert "Что получилось с «A»" in texts[0] and "Поехали" in texts[1]
    now_active = await _active(ctx)
    assert now_active is not None and now_active.task_id == b


async def test_keyword_control_and_fallthrough(ctx, profile, dp, bot_and_session, clock) -> None:
    bot, session = bot_and_session
    # nothing running: "пауза" is not ours and falls through to later routers
    await dp.feed_update(bot, message_update("пауза"))
    assert not any("⏸" in (m.text or "") for m in session.sent_messages())
    assert await _active(ctx) is None
    await dp.feed_update(bot, message_update("/work"))
    await dp.feed_update(bot, callback_update(SessionCb(action="pick", arg="free").pack()))
    assert "Поехали" in _last(session).text
    clock.advance(minutes=10)
    await dp.feed_update(bot, message_update("Пауза!"))
    assert (await _active(ctx)).status == SessionStatus.PAUSED  # type: ignore[union-attr]
    n = len(session.sent_messages())
    await dp.feed_update(bot, message_update("пауза бывает полезна, расскажи подробнее"))
    later = session.sent_messages()[n:]  # long text falls through to other routers
    assert not any("Пауза" in (m.text or "") or "Продолжаем" in (m.text or "") for m in later)
    clock.advance(minutes=5)
    await dp.feed_update(bot, message_update("продолжаю"))
    assert (await _active(ctx)).status == SessionStatus.ACTIVE  # type: ignore[union-attr]
    clock.advance(minutes=20)
    await dp.feed_update(bot, message_update("готово"))
    stop = _last(session)
    assert "Чистое время: 30 мин" in stop.text and "паузы 5 мин" in stop.text
    assert await _active(ctx) is None
    # no task -> no outcome question, only an edit button
    assert [t for t, _ in _buttons(stop)] == ["✏️ Исправить время"]


async def test_start_word_confirms_planned_session(ctx, profile, dp, bot_and_session) -> None:
    bot, session = bot_and_session
    tid = await _task(ctx)
    async with ctx.db.transaction() as s:
        planned = await svc.create_unconfirmed_session(
            s, ctx, task_id=tid, planned_start_at=T0 + timedelta(minutes=5)
        )
    await dp.feed_update(bot, message_update("начинаю"))
    assert "Поехали" in _last(session).text
    ws = await _active(ctx)
    assert ws is not None and ws.id == planned.id and ws.started_at == T0
    assert ws.time_source == TimeSource.EXACT


async def test_stop_outcome_criteria_partial_is_not_done(ctx, profile, dp, bot_and_session) -> None:
    """Acceptance 17 through the Telegram flow."""
    bot, session = bot_and_session
    tid = await _task(ctx, "Релиз", verification="criteria", done_criteria="Тесты зелёные")
    await dp.feed_update(
        bot, callback_update(SessionCb(action="pick", task_id=tid, arg="free").pack())
    )
    await dp.feed_update(bot, message_update("/stop"))
    outcome = _last(session)
    labels = [t for t, _ in _buttons(outcome)]
    assert labels[:3] == ["Начал", "Продвинулся, но не закончил", "Завершил"]
    await dp.feed_update(bot, callback_update(_data(outcome, "Завершил")))
    crit = _last(session)
    assert "Тесты зелёные" in crit.text
    await dp.feed_update(bot, callback_update(_data(crit, "Частично")))
    assert "не считается выполнением" in _last(session).text
    async with ctx.db.transaction() as s:
        task = await s.get(Task, tid)
        assert task is not None and task.status == TaskStatus.IN_PROGRESS
    ws = await _only_session(ctx)
    assert ws.outcome == "progressed" and ws.result_note == "criteria: partial"
    # repeated press does nothing
    await dp.feed_update(bot, callback_update(_data(outcome, "Завершил")))
    assert "уже записан" in _last(session).text


async def test_full_criteria_then_evidence_is_stored_not_verified(
    ctx, profile, dp, bot_and_session
) -> None:
    bot, session = bot_and_session
    tid = await _task(ctx, "Релиз", verification="criteria")
    await dp.feed_update(
        bot, callback_update(SessionCb(action="pick", task_id=tid, arg="free").pack())
    )
    await dp.feed_update(bot, message_update("стоп"))
    await dp.feed_update(bot, callback_update(_data(_last(session), "Завершил")))
    crit = _last(session)
    assert "критерий готовности не записан" in crit.text
    await dp.feed_update(bot, callback_update(_data(crit, "Полностью")))
    assert "не проверяю" in _last(session).text
    await dp.feed_update(bot, message_update("https://github.com/x/y/pull/1"))
    assert "не проверял" in _last(session).text
    ws = await _only_session(ctx)
    assert ws.evidence == "https://github.com/x/y/pull/1" and ws.outcome == "completed"
    async with ctx.db.transaction() as s:
        task = await s.get(Task, tid)
        assert task is not None and task.status == TaskStatus.DONE


async def test_simple_task_closes_by_tap(ctx, profile, dp, bot_and_session) -> None:
    bot, session = bot_and_session
    tid = await _task(ctx, "Посуда")
    await dp.feed_update(
        bot, callback_update(SessionCb(action="pick", task_id=tid, arg="free").pack())
    )
    await dp.feed_update(bot, message_update("закончил"))
    await dp.feed_update(bot, callback_update(_data(_last(session), "Завершил")))
    assert "выполненное" in _last(session).text
    async with ctx.db.transaction() as s:
        task = await s.get(Task, tid)
        assert task is not None and task.status == TaskStatus.DONE


async def test_scenario_f_retro_entry(ctx, profile, dp, bot_and_session, clock) -> None:
    """Hours later the user reports earlier work: interval, breaks, confidence."""
    bot, session = bot_and_session
    clock.set(datetime(2026, 10, 26, 17, 0, tzinfo=UTC))  # 18:00 Berlin
    tid = await _task(ctx, "Сайт")
    await dp.feed_update(bot, message_update("/retro"))
    await dp.feed_update(bot, callback_update(_data(_last(session), "Сайт")))
    assert "Когда работал" in _last(session).text
    await dp.feed_update(bot, message_update("непонятно что"))
    assert "Не понял" in _last(session).text
    await dp.feed_update(bot, message_update("с 14 до 16:30"))
    assert "перерывы" in _last(session).text
    await dp.feed_update(bot, message_update("15"))
    conf = _last(session)
    assert "уверен" in conf.text
    await dp.feed_update(bot, callback_update(_data(conf, "Примерно")))
    done = _last(session)
    assert "задним числом" in done.text and "не будет выдано за точный таймер" in done.text
    assert "2 ч 15 мин" in done.text and "Что получилось с «Сайт»" in done.text
    ws = await _only_session(ctx)
    assert ws.status == SessionStatus.FINISHED
    assert ws.time_source == TimeSource.RETRO and ws.confidence == "medium"
    assert ws.started_at == datetime(2026, 10, 26, 13, 0, tzinfo=UTC)
    async with ctx.db.transaction() as s:
        corr = (await s.execute(select(SessionCorrection))).scalars().all()
        assert [c.field for c in corr] == ["created_retro"]
        stats = await svc.session_stats(
            s, start=T0 - timedelta(days=1), end=T0 + timedelta(days=1), now=clock.now()
        )
    [item] = stats.items
    assert not item.is_exact and item.net_minutes == 135
    # outcome after the retro entry
    await dp.feed_update(bot, callback_update(_data(done, "Продвинулся")))
    async with ctx.db.transaction() as s:
        task = await s.get(Task, tid)
        assert task is not None and task.status == TaskStatus.IN_PROGRESS
    # later edit of the times is journaled
    await dp.feed_update(bot, callback_update(_data(done, "Исправить время")))
    await dp.feed_update(bot, message_update("14:00-17:00"))
    assert "журнал" in _last(session).text
    async with ctx.db.transaction() as s:
        fields = [c.field for c in (await s.execute(select(SessionCorrection))).scalars()]
    assert fields == ["created_retro", "ended_at"]


async def test_retro_duration_only_is_approximate(ctx, profile, dp, bot_and_session, clock) -> None:
    bot, session = bot_and_session
    await dp.feed_update(bot, message_update("/retro"))
    await dp.feed_update(bot, callback_update(_data(_last(session), "Без задачи")))
    await dp.feed_update(bot, message_update("примерно час утром"))
    assert "Во сколько" in _last(session).text
    await dp.feed_update(bot, callback_update(_data(_last(session), "Только что")))
    await dp.feed_update(bot, callback_update(_data(_last(session), "Без перерывов")))
    await dp.feed_update(bot, callback_update(_data(_last(session), "Не уверен")))
    ws = await _only_session(ctx)
    assert ws.time_source == TimeSource.APPROXIMATE and ws.confidence == "low"
    assert ws.ended_at == T0 and ws.started_at == T0 - timedelta(hours=1)
    assert ws.task_id is None


async def test_timer_signal_buttons(ctx, profile, dp, bot_and_session, clock) -> None:
    bot, session = bot_and_session
    async with ctx.db.transaction() as s:
        from procbot.services.profile import update_profile

        await update_profile(s, now=clock.now(), short_start_min=5)
    await dp.feed_update(bot, callback_update(SessionCb(action="pick", arg="short_start").pack()))
    ws = await _active(ctx)
    assert ws is not None
    clock.advance(minutes=5)
    await dp.feed_update(bot, callback_update(SessionCb(action="more", session_id=ws.id).pack()))
    assert "Ещё 5 мин" in _last(session).text
    await dp.feed_update(bot, callback_update(SessionCb(action="tofree", session_id=ws.id).pack()))
    assert "свободный таймер" in _last(session).text
    ws = await _active(ctx)
    assert ws is not None and ws.mode == SessionMode.FREE


async def test_menu_and_commands_escape_open_questions(ctx, profile, dp, bot_and_session) -> None:
    bot, session = bot_and_session
    await dp.feed_update(bot, message_update("/retro"))
    await dp.feed_update(bot, callback_update(_data(_last(session), "Без задачи")))
    await dp.feed_update(bot, message_update(Menu.WORK))  # not taken as an interval
    assert _last(session).text.startswith("Над чем работаем")
    await dp.feed_update(bot, message_update("/retro"))
    await dp.feed_update(bot, message_update("/cancel"))
    assert "отменил" in _last(session).text
    await dp.feed_update(bot, message_update("14-16"))
    async with ctx.db.transaction() as s:
        assert (await s.execute(select(WorkSession))).first() is None


async def test_pomodoro_break_button(ctx, profile, dp, bot_and_session, clock) -> None:
    bot, session = bot_and_session
    async with ctx.db.transaction() as s:
        from procbot.services.profile import update_profile

        await update_profile(s, now=clock.now(), pomodoro_work_min=25, pomodoro_break_min=5)
    await dp.feed_update(bot, callback_update(SessionCb(action="pick", arg="pomodoro").pack()))
    ws = await _active(ctx)
    assert ws is not None
    clock.advance(minutes=25)
    await dp.feed_update(bot, callback_update(SessionCb(action="brk", session_id=ws.id).pack()))
    assert "Перерыв 5 мин" in _last(session).text
    clock.advance(minutes=5)
    await dp.feed_update(bot, callback_update(SessionCb(action="resume", session_id=ws.id).pack()))
    assert "Продолжаем" in _last(session).text
    ws = await _active(ctx)
    assert ws is not None and ws.status == SessionStatus.ACTIVE and ws.planned_minutes == 50
