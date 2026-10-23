"""Telegram handlers: work sessions (FR-06), retro entries (scenario F).

Entry points:
- menu "▶️ Начать работу", /work -> pick a task -> choose mode -> start;
- SessionCb(action="pick", task_id, arg=mode?) used by other modules' buttons;
- /pause /resume /stop /retro, and short keyword phrases while a session runs
  ("пауза", "продолжаю", "стоп", "готово", "начинаю"...). Anything else falls
  through to later routers.

Handlers are thin: parse input, call services.sessions in one transaction, reply
after it committed. No messages are sent during a session except replies to the
user's own actions (timer signals are bot-initiated and go through the gate).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from aiogram import F, Router
from aiogram.filters import Command, Filter, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.models import Task, WorkSession
from procbot.domain.enums import (
    SessionMode,
    SessionOutcome,
    SessionStatus,
    TaskStatus,
    TimeSource,
)
from procbot.services import sessions as svc
from procbot.services import tasks as task_service
from procbot.services.profile import get_profile, update_profile
from procbot.telegram.callbacks import SessionCb
from procbot.telegram.keyboards import Menu
from procbot.timeutil import to_local

log = logging.getLogger(__name__)

router = Router(name="sessions")


class SetupStates(StatesGroup):
    value = State()  # free number input for a missing duration setting


class RetroStates(StatesGroup):
    task = State()
    interval = State()
    end_time = State()
    breaks = State()
    confidence = State()


class EvidenceStates(StatesGroup):
    text = State()


class EditStates(StatesGroup):
    interval = State()


MY_STATES = (SetupStates, RetroStates, EvidenceStates, EditStates)

FIELD_CODES: dict[str, str] = {
    "w": "pomodoro_work_min",
    "b": "pomodoro_break_min",
    "s": "short_start_min",
}
CODE_BY_FIELD = {v: k for k, v in FIELD_CODES.items()}
# Quick-pick options shown when a duration is not set yet (open question 5).
# They are choices for the user, not defaults: nothing is saved without a tap/answer.
QUICK_PICKS: dict[str, tuple[int, ...]] = {
    "pomodoro_work_min": (15, 25, 45, 60),
    "pomodoro_break_min": (5, 10, 15),
    "short_start_min": (2, 5, 10),
}
SETTING_PROMPTS: dict[str, str] = {
    "pomodoro_work_min": "Сколько минут длится рабочий период Помодоро?",
    "pomodoro_break_min": "Сколько минут перерыв между периодами Помодоро?",
    "short_start_min": "Сколько минут длится короткий старт?",
}
CANCEL_WORDS = frozenset({"отмена", "/cancel", "отменить", "стоп отмена"})
PICK_LIMIT = 8
OUTCOME_REPLIES: dict[SessionOutcome, str] = {
    SessionOutcome.STARTED: "Записал: начал. Задача в работе.",
    SessionOutcome.PROGRESSED: (
        "Записал прогресс. Задача остаётся в работе — частичный результат не считается выполнением."
    ),
    SessionOutcome.COMPLETED: "✅ Отмечено как выполненное.",
}


# --------------------------------------------------------------------------- helpers


def _is_plain_input(message: Message) -> bool:
    """Free input for our questions: not a command and not a main-menu button, so
    those still reach their own handlers while one of our questions is open."""
    text = message.text or message.caption or ""
    return not text.startswith("/") and text not in Menu.ALL


PLAIN_INPUT = F.func(_is_plain_input)


def _btn(text: str, cb: SessionCb) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=cb.pack())


def _kb(*rows: list[InlineKeyboardButton]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=list(rows))


def _short(text: str, n: int = 40) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


async def _send(
    event: Message | CallbackQuery, text: str, kb: InlineKeyboardMarkup | None = None
) -> None:
    if isinstance(event, CallbackQuery):
        msg = event.message
        if isinstance(msg, Message):
            await msg.answer(text, reply_markup=kb)
        elif event.bot is not None:
            await event.bot.send_message(event.from_user.id, text, reply_markup=kb)
        return
    await event.answer(text, reply_markup=kb)


async def _done(cb: CallbackQuery, text: str | None = None) -> None:
    try:
        await cb.answer(text)
    except Exception:  # pragma: no cover - expired callback ids are harmless
        log.debug("answer_callback_query failed", exc_info=True)


async def _tz(s: AsyncSession) -> str | None:
    profile = await get_profile(s)
    return profile.timezone if profile else None


def _parse_mode(arg: str) -> SessionMode | None:
    try:
        return SessionMode(arg)
    except ValueError:
        return None


async def _clear_if_mine(state: FSMContext) -> None:
    current = await state.get_state()
    if current is not None and any(current.startswith(g.__name__ + ":") for g in MY_STATES):
        await state.clear()


async def _pick_tasks(s: AsyncSession, ctx: AppContext) -> list[Task]:
    """Today's tasks first, then other tasks in progress."""
    today = (await task_service.today_buckets(s, ctx)).all()
    seen = {t.id for t in today}
    extra = (
        (
            await s.execute(
                select(Task)
                .where(Task.status == str(TaskStatus.IN_PROGRESS))
                .order_by(Task.updated_at.desc())
                .limit(PICK_LIMIT)
            )
        )
        .scalars()
        .all()
    )
    return [*today, *(t for t in extra if t.id not in seen)][:PICK_LIMIT]


async def _session_card(
    s: AsyncSession, ctx: AppContext, ws: WorkSession
) -> tuple[str, InlineKeyboardMarkup]:
    return await svc.describe_session(s, ctx, ws), svc.control_keyboard(ws)


# --------------------------------------------------------------------------- menu


@router.message(StateFilter(*MY_STATES), Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Ок, отменил.")


@router.message(Command("work"))
@router.message(F.text == Menu.WORK)
async def work_menu(message: Message, ctx: AppContext, state: FSMContext) -> None:
    await _clear_if_mine(state)
    tasks: list[Task] = []
    cand: svc.StartCandidate | None = None
    planned_title: str | None = None
    async with ctx.db.transaction() as s:
        active = await svc.get_active_session(s)
        if active is not None:
            text, kb = await _session_card(s, ctx, active)
            card: tuple[str, InlineKeyboardMarkup] | None = (text, kb)
        else:
            card = None
            tasks = await _pick_tasks(s, ctx)
            cand = await svc.find_start_candidate(s, ctx)
            planned_title = (
                await svc.task_title(s, cand.task_id)
                if cand is not None and cand.session_id is not None
                else None
            )
    if card is not None:
        await message.answer("Сессия уже идёт:\n\n" + card[0], reply_markup=card[1])
        return
    rows: list[list[InlineKeyboardButton]] = []
    if cand is not None and cand.session_id is not None:
        label = f"✅ Подтвердить старт: {_short(planned_title or 'без задачи', 30)}"
        rows.append([_btn(label, SessionCb(action="start", session_id=cand.session_id))])
    for t in tasks:
        rows.append([_btn(f"▶️ {_short(t.title)}", SessionCb(action="pick", task_id=t.id))])
    rows.append([_btn("Без задачи", SessionCb(action="pick", task_id=0))])
    rows.append([_btn("🕓 Отметить задним числом", SessionCb(action="retro"))])
    text = "Над чем работаем?" if tasks else "Над чем работаем? Задач на сегодня нет."
    await message.answer(text, reply_markup=_kb(*rows))


@router.callback_query(SessionCb.filter(F.action == "pick"))
async def on_pick(
    cb: CallbackQuery, callback_data: SessionCb, ctx: AppContext, state: FSMContext
) -> None:
    await _done(cb)
    task_id = callback_data.task_id or None
    mode = _parse_mode(callback_data.arg)
    if mode is not None:
        await _begin(cb, ctx, state, task_id=task_id, mode=mode)
        return
    await _show_modes(cb, ctx, task_id)


async def _show_modes(event: Message | CallbackQuery, ctx: AppContext, task_id: int | None) -> None:
    async with ctx.db.transaction() as s:
        task = await s.get(Task, task_id) if task_id else None
        rec = await svc.recommend_mode(s, ctx, task_id=task_id)
    if task_id and task is None:
        await _send(event, "Не нашёл эту задачу.")
        return
    head = f"Задача: «{task.title}»\n" if task is not None else "Без задачи.\n"
    rows = []
    for mode, label in svc.MODE_LABELS.items():
        text = f"⭐ {label} (рекомендую)" if rec is not None and rec.mode == mode else label
        rows.append([_btn(text, SessionCb(action="mode", task_id=task_id or 0, arg=str(mode)))])
    hint = f"\nРекомендую: {svc.MODE_LABELS[rec.mode].lower()} — {rec.reason}." if rec else ""
    await _send(event, f"{head}Выбери режим.{hint}", _kb(*rows))


@router.callback_query(SessionCb.filter(F.action == "mode"))
async def on_mode(
    cb: CallbackQuery, callback_data: SessionCb, ctx: AppContext, state: FSMContext
) -> None:
    await _done(cb)
    mode = _parse_mode(callback_data.arg)
    if mode is None:
        return
    await _begin(cb, ctx, state, task_id=callback_data.task_id or None, mode=mode)


@router.callback_query(SessionCb.filter(F.action == "start"))
async def on_confirm_planned(
    cb: CallbackQuery, callback_data: SessionCb, ctx: AppContext, state: FSMContext
) -> None:
    """Confirm a planned (unconfirmed) session."""
    await _done(cb)
    async with ctx.db.transaction() as s:
        ws = await svc.get_work_session(s, callback_data.session_id)
        info = (ws.status, ws.task_id, ws.mode) if ws is not None else None
    if info is None:
        return
    status, task_id, mode = info
    if status != SessionStatus.UNCONFIRMED:
        await _show_current(cb, ctx, note="Эта сессия уже начата или закрыта.")
        return
    await _begin(
        cb, ctx, state, task_id=task_id, mode=SessionMode(mode), session_id=callback_data.session_id
    )


async def _show_current(event: Message | CallbackQuery, ctx: AppContext, note: str = "") -> None:
    async with ctx.db.transaction() as s:
        ws = await svc.get_active_session(s)
        card = await _session_card(s, ctx, ws) if ws is not None else None
    if card is None:
        await _send(event, note or "Сейчас нет активной сессии.")
        return
    await _send(event, (note + "\n\n" if note else "") + card[0], card[1])


async def _begin(
    event: Message | CallbackQuery,
    ctx: AppContext,
    state: FSMContext,
    *,
    task_id: int | None,
    mode: SessionMode,
    session_id: int | None = None,
) -> None:
    """Start the session the user chose, or ask what is missing / conflicting."""
    try:
        async with ctx.db.transaction() as s:
            if task_id is not None:
                task = await s.get(Task, task_id)
                if task is None:
                    closed = "missing"
                elif not task_service.is_open(task):
                    closed = "closed"
                else:
                    closed = None
                if closed is not None:
                    raise _TaskUnavailable(closed)
            ws = await svc.start_session(s, ctx, task_id=task_id, mode=mode, session_id=session_id)
            text, kb = await _session_card(s, ctx, ws)
    except _TaskUnavailable as exc:
        msg = "Не нашёл эту задачу." if exc.kind == "missing" else "Эта задача уже закрыта."
        await _send(event, msg)
        return
    except svc.SettingsMissing as exc:
        await _ask_setting(event, state, exc.fields[0], task_id, mode, session_id)
        return
    except svc.ActiveSessionExists as exc:
        await _conflict(event, ctx, exc, task_id, mode)
        return
    await _clear_if_mine(state)
    await _send(
        event,
        f"▶️ Поехали!\n\n{text}\n\nВо время работы я не пишу — только сигнал таймера, "
        "если он предусмотрен режимом.",
        kb,
    )


class _TaskUnavailable(Exception):
    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


async def _conflict(
    event: Message | CallbackQuery,
    ctx: AppContext,
    current: svc.ActiveSessionExists,
    task_id: int | None,
    mode: SessionMode,
) -> None:
    async with ctx.db.transaction() as s:
        title = await svc.task_title(s, current.task_id)
        ws = await svc.get_work_session(s, current.session_id)
        card = (
            await _session_card(s, ctx, ws)
            if ws is not None and current.task_id == task_id
            else None
        )
    if card is not None:
        await _send(event, "Эта сессия уже идёт.\n\n" + card[0], card[1])
        return
    name = f"«{title}»" if title else "без задачи"
    await _send(
        event,
        f"Сейчас уже идёт сессия {name}. Одновременно — только одна.\nЗавершить её и начать новую?",
        _kb(
            [
                _btn(
                    "⏹ Завершить текущую и начать",
                    SessionCb(
                        action="switch",
                        session_id=current.session_id,
                        task_id=task_id or 0,
                        arg=str(mode),
                    ),
                )
            ],
            [
                _btn(
                    "↩️ Вернуться к текущей",
                    SessionCb(action="show", session_id=current.session_id),
                )
            ],
        ),
    )


@router.callback_query(SessionCb.filter(F.action == "show"))
async def on_show(cb: CallbackQuery, ctx: AppContext) -> None:
    await _done(cb)
    await _show_current(cb, ctx)


@router.callback_query(SessionCb.filter(F.action == "switch"))
async def on_switch(
    cb: CallbackQuery, callback_data: SessionCb, ctx: AppContext, state: FSMContext
) -> None:
    await _done(cb)
    mode = _parse_mode(callback_data.arg) or SessionMode.FREE
    try:
        async with ctx.db.transaction() as s:
            ws = await svc.stop_session(s, ctx, session_id=callback_data.session_id)
            summary = await _stop_summary(s, ctx, ws)
    except svc.InvalidSessionState:
        summary = None
    if summary is not None:
        await _send(cb, summary[0], summary[1])
    await _begin(cb, ctx, state, task_id=callback_data.task_id or None, mode=mode)


# --------------------------------------------------------------------------- settings (ask once)


async def _ask_setting(
    event: Message | CallbackQuery,
    state: FSMContext,
    field: str,
    task_id: int | None,
    mode: SessionMode,
    session_id: int | None,
) -> None:
    await state.set_state(SetupStates.value)
    await state.update_data(
        setup={
            "field": field,
            "task_id": task_id,
            "mode": str(mode),
            "session_id": session_id,
        }
    )
    code = CODE_BY_FIELD[field]
    buttons = [
        _btn(
            f"{n} мин",
            SessionCb(
                action="set",
                task_id=task_id or 0,
                session_id=session_id or 0,
                arg=f"{code}.{n}.{mode}",
            ),
        )
        for n in QUICK_PICKS[field]
    ]
    await _send(
        event,
        f"{SETTING_PROMPTS[field]}\nВыбери или напиши число минут. Сохраню в настройках — "
        "в следующий раз спрашивать не буду.",
        _kb(buttons),
    )


async def _save_setting(ctx: AppContext, field: str, value: int) -> bool:
    lo, hi = svc.SETTING_BOUNDS[field]
    if not lo <= value <= hi:
        return False
    async with ctx.db.transaction() as s:
        changes: dict[str, Any] = {field: value}
        await update_profile(s, now=ctx.clock.now(), **changes)
    return True


@router.callback_query(SessionCb.filter(F.action == "set"))
async def on_setting_pick(
    cb: CallbackQuery, callback_data: SessionCb, ctx: AppContext, state: FSMContext
) -> None:
    await _done(cb)
    try:
        code, raw, mode_raw = callback_data.arg.split(".", 2)
        field, value, mode = FIELD_CODES[code], int(raw), SessionMode(mode_raw)
    except (ValueError, KeyError):
        return
    if not await _save_setting(ctx, field, value):
        return
    await state.clear()
    await _begin(
        cb,
        ctx,
        state,
        task_id=callback_data.task_id or None,
        mode=mode,
        session_id=callback_data.session_id or None,
    )


@router.message(SetupStates.value, F.text, PLAIN_INPUT)
async def on_setting_text(message: Message, ctx: AppContext, state: FSMContext) -> None:
    text = (message.text or "").strip().lower()
    data = (await state.get_data()).get("setup") or {}
    if text in CANCEL_WORDS or not data:
        await state.clear()
        await message.answer("Ок, отменил.")
        return
    field = str(data.get("field"))
    digits = text.replace("мин", "").replace("минут", "").strip()
    if not digits.isdigit() or not await _save_setting(ctx, field, int(digits)):
        lo, hi = svc.SETTING_BOUNDS.get(field, (1, 240))
        await message.answer(
            f"Напиши число минут от {lo} до {hi}, например {QUICK_PICKS[field][0]}."
        )
        return
    await state.clear()
    await _begin(
        message,
        ctx,
        state,
        task_id=data.get("task_id"),
        mode=SessionMode(str(data.get("mode"))),
        session_id=data.get("session_id"),
    )


# --------------------------------------------------------------------------- control


async def _active_id(ctx: AppContext) -> int | None:
    async with ctx.db.transaction() as s:
        ws = await svc.get_active_session(s)
        return ws.id if ws is not None else None


async def _do_pause(event: Message | CallbackQuery, ctx: AppContext, session_id: int) -> None:
    try:
        async with ctx.db.transaction() as s:
            ws = await svc.pause_session(s, ctx, session_id=session_id)
            text, kb = await _session_card(s, ctx, ws)
            phase = svc.current_phase(ws, await svc.load_pauses(s, ws.id), ctx.clock.now())
    except svc.InvalidSessionState:
        await _send(event, "Эта сессия уже не идёт.")
        return
    if phase == "break":
        head = f"☕ Перерыв {ws.break_minutes} мин. Напишу, когда он закончится."
    else:
        head = "⏸ Пауза. Время не идёт."
    await _send(event, f"{head}\n\n{text}", kb)


async def _do_resume(event: Message | CallbackQuery, ctx: AppContext, session_id: int) -> None:
    try:
        async with ctx.db.transaction() as s:
            ws = await svc.resume_session(s, ctx, session_id=session_id)
            text, kb = await _session_card(s, ctx, ws)
    except svc.InvalidSessionState:
        await _send(event, "Эта сессия уже не идёт.")
        return
    await _send(event, "▶️ Продолжаем.\n\n" + text, kb)


async def _stop_summary(
    s: AsyncSession, ctx: AppContext, ws: WorkSession
) -> tuple[str, InlineKeyboardMarkup]:
    now = ctx.clock.now()
    pauses = await svc.load_pauses(s, ws.id)
    net = svc.net_seconds(ws, pauses, now) / 60
    paused = svc.pause_seconds(ws, pauses, now) / 60
    title = await svc.task_title(s, ws.task_id)
    text = f"⏹ Сессия завершена. Чистое время: {svc.fmt_minutes(net)}"
    if paused >= 1:
        text += f" (паузы {svc.fmt_minutes(paused)})"
    text += "."
    if ws.task_id is not None and ws.outcome is None:
        text += f"\nЧто получилось с «{title}»?"
        return text, svc.outcome_keyboard(ws)
    return text, _kb([_btn("✏️ Исправить время", SessionCb(action="edit", session_id=ws.id))])


async def _do_stop(event: Message | CallbackQuery, ctx: AppContext, session_id: int) -> None:
    try:
        async with ctx.db.transaction() as s:
            ws = await svc.stop_session(s, ctx, session_id=session_id)
            text, kb = await _stop_summary(s, ctx, ws)
    except svc.InvalidSessionState:
        await _send(event, "Эта сессия уже не идёт.")
        return
    await _send(event, text, kb)


@router.callback_query(SessionCb.filter(F.action.in_({"pause", "brk"})))
async def on_pause(cb: CallbackQuery, callback_data: SessionCb, ctx: AppContext) -> None:
    await _done(cb)
    await _do_pause(cb, ctx, callback_data.session_id)


@router.callback_query(SessionCb.filter(F.action == "resume"))
async def on_resume(cb: CallbackQuery, callback_data: SessionCb, ctx: AppContext) -> None:
    await _done(cb)
    await _do_resume(cb, ctx, callback_data.session_id)


@router.callback_query(SessionCb.filter(F.action == "stop"))
async def on_stop(cb: CallbackQuery, callback_data: SessionCb, ctx: AppContext) -> None:
    await _done(cb)
    await _do_stop(cb, ctx, callback_data.session_id)


@router.callback_query(SessionCb.filter(F.action.in_({"skipbrk", "more", "tofree", "ack"})))
async def on_timer_action(cb: CallbackQuery, callback_data: SessionCb, ctx: AppContext) -> None:
    """Buttons of timer signals: skip break / +N minutes / free mode / acknowledge."""
    await _done(cb)
    action, sid = callback_data.action, callback_data.session_id
    try:
        async with ctx.db.transaction() as s:
            if action == "skipbrk":
                ws = await svc.continue_without_break(s, ctx, session_id=sid)
                head = "▶️ Работаем дальше без перерыва."
            elif action == "more":
                ws = await svc.extend_session(s, ctx, session_id=sid)
                head = f"➕ Ещё {ws.work_minutes} мин."
            elif action == "tofree":
                ws = await svc.switch_to_free(s, ctx, session_id=sid)
                head = "▶️ Дальше — свободный таймер, без сигналов."
            else:
                ws = await svc.get_work_session(s, sid)
                if ws is None or ws.status not in svc.LIVE_STATUSES:
                    raise svc.InvalidSessionState("not live")
                head = "▶️ Продолжаем."
            text, kb = await _session_card(s, ctx, ws)
    except svc.InvalidSessionState:
        await _send(cb, "Эта сессия уже не идёт.")
        return
    await _send(cb, f"{head}\n\n{text}", kb)


@router.message(Command("pause"))
async def cmd_pause(message: Message, ctx: AppContext) -> None:
    sid = await _active_id(ctx)
    if sid is None:
        await message.answer("Сейчас нет активной сессии.")
        return
    await _do_pause(message, ctx, sid)


@router.message(Command("resume"))
async def cmd_resume(message: Message, ctx: AppContext) -> None:
    sid = await _active_id(ctx)
    if sid is None:
        await message.answer("Сейчас нет активной сессии.")
        return
    await _do_resume(message, ctx, sid)


@router.message(Command("stop"))
async def cmd_stop(message: Message, ctx: AppContext) -> None:
    sid = await _active_id(ctx)
    if sid is None:
        await message.answer("Сейчас нет активной сессии.")
        return
    await _do_stop(message, ctx, sid)


# --------------------------------------------------------------------------- outcome (FR-06)


@router.callback_query(SessionCb.filter(F.action == "outcome"))
async def on_outcome(cb: CallbackQuery, callback_data: SessionCb, ctx: AppContext) -> None:
    await _done(cb)
    try:
        outcome = SessionOutcome(callback_data.arg)
    except ValueError:
        return
    sid = callback_data.session_id
    criteria: tuple[str, str] | None = None  # (task title, definition of done)
    try:
        async with ctx.db.transaction() as s:
            ws = await svc.get_work_session(s, sid)
            if ws is None:
                return
            task = await s.get(Task, ws.task_id) if ws.task_id else None
            if ws.outcome is not None:
                label = svc.OUTCOME_LABELS[SessionOutcome(ws.outcome)].lower()
                reply = f"Результат уже записан: {label}."
            elif (
                outcome == SessionOutcome.COMPLETED
                and task is not None
                and svc.task_needs_criteria(task)
            ):
                criteria = (task.title, (task.done_criteria or "").strip())
                reply = ""
            else:
                await svc.record_outcome(s, ctx, session_id=sid, outcome=outcome)
                reply = OUTCOME_REPLIES[outcome]
    except svc.InvalidSessionState:
        await _send(cb, "Сначала завершим сессию.")
        return
    if criteria is None:
        await _send(cb, reply)
        return
    title, done_criteria = criteria
    if done_criteria:
        text = (
            f"Критерий готовности для «{title}»:\n«{done_criteria}»\n\nРезультат ему соответствует?"
        )
    else:
        text = (
            f"«{title}» — важная задача, но критерий готовности не записан.\nОна сделана полностью?"
        )
    await _send(
        cb,
        text,
        _kb(
            [_btn("✅ Полностью", SessionCb(action="crit", session_id=sid, arg="full"))],
            [_btn("◐ Частично", SessionCb(action="crit", session_id=sid, arg="partial"))],
            [_btn("✖️ Нет", SessionCb(action="crit", session_id=sid, arg="no"))],
        ),
    )


@router.callback_query(SessionCb.filter(F.action == "crit"))
async def on_criteria(
    cb: CallbackQuery, callback_data: SessionCb, ctx: AppContext, state: FSMContext
) -> None:
    await _done(cb)
    sid, answer = callback_data.session_id, callback_data.arg
    if answer not in ("full", "partial", "no"):
        return
    outcome = SessionOutcome.COMPLETED if answer == "full" else SessionOutcome.PROGRESSED
    note = {"full": "criteria: met", "partial": "criteria: partial", "no": "criteria: not met"}[
        answer
    ]
    try:
        async with ctx.db.transaction() as s:
            _, applied = await svc.record_outcome(
                s,
                ctx,
                session_id=sid,
                outcome=outcome,
                criteria_confirmed=answer == "full",
                note=note,
            )
    except svc.InvalidSessionState:
        return
    if not applied:
        await _send(cb, "Результат уже записан.")
        return
    if answer != "full":
        await _send(cb, OUTCOME_REPLIES[SessionOutcome.PROGRESSED])
        return
    await state.set_state(EvidenceStates.text)
    await state.update_data(evidence_session=sid)
    await _send(
        cb,
        "✅ Отмечено как выполненное.\n\nЕсли хочешь, пришли ссылку или пару слов о "
        "результате — сохраню рядом с отметкой. Я это не проверяю, а скриншот открытой "
        "IDE сам по себе выполнение не подтверждает.",
        _kb([_btn("Пропустить", SessionCb(action="evskip", session_id=sid))]),
    )


@router.callback_query(SessionCb.filter(F.action == "evskip"))
async def on_evidence_skip(cb: CallbackQuery, state: FSMContext) -> None:
    await _done(cb)
    await _clear_if_mine(state)
    await _send(cb, "Ок, без подтверждения.")


@router.message(EvidenceStates.text, PLAIN_INPUT)
async def on_evidence(message: Message, ctx: AppContext, state: FSMContext) -> None:
    sid = (await state.get_data()).get("evidence_session")
    text = (message.text or message.caption or "").strip()
    if text.lower() in CANCEL_WORDS:
        await state.clear()
        await message.answer("Ок, без подтверждения.")
        return
    parts = [text] if text else []
    if message.photo:
        parts.append(f"[фото {message.photo[-1].file_id}]")
    if message.document:
        parts.append(f"[файл {message.document.file_id}]")
    if not parts or not sid:
        await message.answer("Пришли текст или ссылку — или нажми «Пропустить».")
        return
    async with ctx.db.transaction() as s:
        await svc.save_evidence(s, ctx, session_id=int(sid), evidence=" ".join(parts))
    await state.clear()
    await message.answer("Сохранил как твоё описание результата (я его не проверял).")


# --------------------------------------------------------------------------- edit times


@router.callback_query(SessionCb.filter(F.action == "edit"))
async def on_edit(cb: CallbackQuery, callback_data: SessionCb, state: FSMContext) -> None:
    await _done(cb)
    await state.set_state(EditStates.interval)
    await state.update_data(edit_session=callback_data.session_id)
    await _send(
        cb,
        "Укажи правильный интервал, например «14:00-16:30» или «вчера 10-12». "
        "Исправление будет записано в журнал.",
    )


@router.message(EditStates.interval, F.text, PLAIN_INPUT)
async def on_edit_text(message: Message, ctx: AppContext, state: FSMContext) -> None:
    text = (message.text or "").strip()
    if text.lower() in CANCEL_WORDS:
        await state.clear()
        await message.answer("Ок, оставил как было.")
        return
    sid = (await state.get_data()).get("edit_session")
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        tz = await _tz(s)
    parsed = svc.parse_interval(text, now=now, tz=tz)
    if parsed is None or parsed.start is None or parsed.end is None:
        await message.answer("Нужны время начала и конца, например «14:00-16:30».")
        return
    try:
        async with ctx.db.transaction() as s:
            changes = await svc.correct_session_times(
                s,
                ctx,
                session_id=int(sid or 0),
                started_at=parsed.start,
                ended_at=parsed.end,
                reason="user edit",
            )
    except (ValueError, svc.InvalidSessionState) as exc:
        await message.answer(f"Не получилось исправить: {_error_text(exc)}")
        return
    await state.clear()
    if not changes:
        await message.answer("Время и так такое — ничего не менял.")
        return
    await message.answer(
        f"Исправил: {_fmt_range(parsed.start, parsed.end, tz)}. Изменение записано в журнал."
    )


def _fmt_range(start: datetime, end: datetime, tz: str | None) -> str:
    a, b = to_local(start, tz), to_local(end, tz)
    if a.date() == b.date():
        return f"{a:%d.%m} {a:%H:%M}–{b:%H:%M}"
    return f"{a:%d.%m %H:%M} – {b:%d.%m %H:%M}"


def _error_text(exc: Exception) -> str:
    msg = str(exc)
    return {
        "end must be after start": "конец раньше начала",
        "end is in the future": "конец в будущем",
        "session longer than 24 hours": "больше 24 часов",
        "break outside the session": "перерыв вне интервала",
    }.get(msg, "неверные данные")


# --------------------------------------------------------------------------- retro (scenario F)


@router.message(Command("retro"))
async def cmd_retro(message: Message, ctx: AppContext, state: FSMContext) -> None:
    await _retro_start(message, ctx, state)


@router.callback_query(SessionCb.filter(F.action == "retro"))
async def on_retro(cb: CallbackQuery, ctx: AppContext, state: FSMContext) -> None:
    await _done(cb)
    await _retro_start(cb, ctx, state)


async def _retro_start(event: Message | CallbackQuery, ctx: AppContext, state: FSMContext) -> None:
    await _clear_if_mine(state)
    async with ctx.db.transaction() as s:
        tasks = await _pick_tasks(s, ctx)
    await state.set_state(RetroStates.task)
    rows = [[_btn(_short(t.title), SessionCb(action="rtask", task_id=t.id))] for t in tasks]
    rows.append([_btn("Без задачи", SessionCb(action="rtask", task_id=0))])
    await _send(event, "🕓 Отметим работу задним числом. Над чем работал?", _kb(*rows))


@router.callback_query(SessionCb.filter(F.action == "rtask"))
async def on_retro_task(cb: CallbackQuery, callback_data: SessionCb, state: FSMContext) -> None:
    await _done(cb)
    await state.set_state(RetroStates.interval)
    await state.update_data(retro={"task_id": callback_data.task_id or None})
    await _send(
        cb,
        "Когда работал? Например: «14:00-16:30», «с 14 до 16», «вчера 10-12» — "
        "или длительность: «2 часа», «примерно час».",
    )


async def _retro_data(state: FSMContext) -> dict[str, Any]:
    return dict((await state.get_data()).get("retro") or {})


async def _retro_cancelled(message: Message, state: FSMContext) -> bool:
    if (message.text or "").strip().lower() in CANCEL_WORDS:
        await state.clear()
        await message.answer("Ок, ничего не записал.")
        return True
    return False


@router.message(RetroStates.task, F.text, PLAIN_INPUT)
async def on_retro_task_text(message: Message, state: FSMContext) -> None:
    if await _retro_cancelled(message, state):
        return
    await message.answer("Выбери задачу кнопкой выше (или «Без задачи»), либо напиши «отмена».")


@router.message(RetroStates.interval, F.text, PLAIN_INPUT)
async def on_retro_interval(message: Message, ctx: AppContext, state: FSMContext) -> None:
    if await _retro_cancelled(message, state):
        return
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        tz = await _tz(s)
    parsed = svc.parse_interval(message.text or "", now=now, tz=tz)
    if parsed is None:
        await message.answer(
            "Не понял интервал. Напиши, например, «14:00-16:30», «с 14 до 16» или «2 часа»."
        )
        return
    data = await _retro_data(state)
    data["source"] = str(parsed.time_source)
    if parsed.start is not None and parsed.end is not None:
        if parsed.end > now + timedelta(minutes=1):
            await message.answer("Конец получается в будущем — уточни интервал.")
            return
        data["start"], data["end"] = parsed.start.isoformat(), parsed.end.isoformat()
        await state.update_data(retro=data)
        await _ask_breaks(message, state)
        return
    assert parsed.duration is not None
    data["duration_min"] = parsed.duration.total_seconds() / 60
    await state.update_data(retro=data)
    await state.set_state(RetroStates.end_time)
    await message.answer(
        f"Длительность: {svc.fmt_minutes(data['duration_min'])}. Во сколько примерно закончил? "
        "Например «12:30».",
        reply_markup=_kb([_btn("Только что", SessionCb(action="rend"))]),
    )


async def _set_end(event: Message | CallbackQuery, state: FSMContext, end: datetime) -> None:
    data = await _retro_data(state)
    duration = timedelta(minutes=float(data.get("duration_min") or 0))
    if duration <= timedelta(0):
        await state.clear()
        return
    data["start"], data["end"] = (end - duration).isoformat(), end.isoformat()
    await state.update_data(retro=data)
    await _ask_breaks(event, state)


@router.callback_query(SessionCb.filter(F.action == "rend"), RetroStates.end_time)
async def on_retro_end_now(cb: CallbackQuery, ctx: AppContext, state: FSMContext) -> None:
    await _done(cb)
    await _set_end(cb, state, ctx.clock.now())


@router.message(RetroStates.end_time, F.text, PLAIN_INPUT)
async def on_retro_end_text(message: Message, ctx: AppContext, state: FSMContext) -> None:
    if await _retro_cancelled(message, state):
        return
    async with ctx.db.transaction() as s:
        tz = await _tz(s)
    end = svc.parse_clock(message.text or "", now=ctx.clock.now(), tz=tz)
    if end is None:
        await message.answer("Напиши время, например «12:30», или нажми «Только что».")
        return
    await _set_end(message, state, end)


async def _ask_breaks(event: Message | CallbackQuery, state: FSMContext) -> None:
    await state.set_state(RetroStates.breaks)
    await _send(
        event,
        "Были перерывы? Напиши, сколько минут (например «15»), или интервал «13:00-13:20».",
        _kb([_btn("Без перерывов", SessionCb(action="rbrk"))]),
    )


async def _ask_confidence(event: Message | CallbackQuery, state: FSMContext) -> None:
    await state.set_state(RetroStates.confidence)
    await _send(
        event,
        "Насколько ты уверен во времени?",
        _kb(
            [
                _btn("Точно", SessionCb(action="rconf", arg="high")),
                _btn("Примерно", SessionCb(action="rconf", arg="medium")),
                _btn("Не уверен", SessionCb(action="rconf", arg="low")),
            ]
        ),
    )


@router.callback_query(SessionCb.filter(F.action == "rbrk"), RetroStates.breaks)
async def on_retro_no_breaks(cb: CallbackQuery, state: FSMContext) -> None:
    await _done(cb)
    data = await _retro_data(state)
    data["breaks"] = []
    await state.update_data(retro=data)
    await _ask_confidence(cb, state)


@router.message(RetroStates.breaks, F.text, PLAIN_INPUT)
async def on_retro_breaks(message: Message, ctx: AppContext, state: FSMContext) -> None:
    if await _retro_cancelled(message, state):
        return
    data = await _retro_data(state)
    async with ctx.db.transaction() as s:
        tz = await _tz(s)
    start, end = datetime.fromisoformat(data["start"]), datetime.fromisoformat(data["end"])
    breaks = svc.parse_breaks(message.text or "", start=start, end=end, tz=tz)
    if breaks is None:
        await message.answer(
            "Не понял. Напиши минуты (например «15»), интервал «13:00-13:20» "
            "внутри времени работы, или нажми «Без перерывов»."
        )
        return
    data["breaks"] = [
        [b.started_at.isoformat(), b.ended_at.isoformat(), str(b.time_source)] for b in breaks
    ]
    await state.update_data(retro=data)
    await _ask_confidence(message, state)


@router.callback_query(SessionCb.filter(F.action == "rconf"), RetroStates.confidence)
async def on_retro_confidence(
    cb: CallbackQuery, callback_data: SessionCb, ctx: AppContext, state: FSMContext
) -> None:
    await _done(cb)
    confidence = callback_data.arg
    if confidence not in svc.CONFIDENCE_VALUES:
        return
    data = await _retro_data(state)
    task_id: int | None = data.get("task_id")
    breaks = [
        svc.RetroBreak(datetime.fromisoformat(a), datetime.fromisoformat(b), TimeSource(src))
        for a, b, src in data.get("breaks") or []
    ]
    start, end = datetime.fromisoformat(data["start"]), datetime.fromisoformat(data["end"])
    try:
        async with ctx.db.transaction() as s:
            planned = await _planned_for(s, task_id) if task_id else None
            ws = await svc.create_retro_session(
                s,
                ctx,
                task_id=task_id,
                started_at=start,
                ended_at=end,
                time_source=TimeSource(data.get("source") or TimeSource.APPROXIMATE),
                confidence=confidence,
                breaks=breaks,
                session_id=planned,
            )
            pauses = await svc.load_pauses(s, ws.id)
            net = svc.net_seconds(ws, pauses, ctx.clock.now()) / 60
            title = await svc.task_title(s, task_id)
            tz = await _tz(s)
            kb = (
                svc.outcome_keyboard(ws)
                if task_id
                else _kb([_btn("✏️ Исправить время", SessionCb(action="edit", session_id=ws.id))])
            )
            source = svc.SOURCE_LABELS[ws.time_source]
    except (ValueError, svc.InvalidSessionState) as exc:
        await state.clear()
        await _send(cb, f"Не получилось записать: {_error_text(exc)}. Попробуй /retro ещё раз.")
        return
    await state.clear()
    name = f"«{title}»" if title else "без задачи"
    text = (
        f"🕓 Записал задним числом: {name}, {_fmt_range(start, end, tz)}, "
        f"чистое время {svc.fmt_minutes(net)}.\n"
        f"Источник: {source}, уверенность: {svc.CONFIDENCE_LABELS[confidence]}. "
        "В отчётах это не будет выдано за точный таймер."
    )
    if task_id:
        text += f"\n\nЧто получилось с «{title}»?"
    await _send(cb, text, kb)


async def _planned_for(s: AsyncSession, task_id: int) -> int | None:
    """A planned unconfirmed session of this task that the retro entry fills."""
    row = (
        await s.execute(
            select(WorkSession.id)
            .where(
                WorkSession.task_id == task_id,
                WorkSession.status == str(SessionStatus.UNCONFIRMED),
            )
            .order_by(WorkSession.created_at.desc())
            .limit(1)
        )
    ).first()
    return row[0] if row is not None else None


# --------------------------------------------------------------------------- keyword control


class SessionPhrase(Filter):
    """Matches a short control phrase AND a session state where it makes sense.

    Returns False for everything else so the text reaches later routers.
    """

    def __init__(self, intent: str) -> None:
        self.intent = intent

    async def __call__(self, message: Message, ctx: AppContext) -> bool | dict[str, Any]:
        if svc.classify_phrase(message.text) != self.intent:
            return False
        async with ctx.db.transaction() as s:
            ws = await svc.get_active_session(s)
            if self.intent == "pause":
                ok = ws is not None and ws.status == SessionStatus.ACTIVE
            elif self.intent == "resume":
                ok = ws is not None and ws.status == SessionStatus.PAUSED
            elif self.intent == "stop":
                ok = ws is not None
            else:  # start
                if ws is not None:
                    return False
                cand = await svc.find_start_candidate(s, ctx)
                if cand is None:
                    return False
                return {"start_candidate": cand}
        if not ok or ws is None:
            return False
        return {"work_session_id": ws.id}


@router.message(StateFilter(None), F.text, SessionPhrase("pause"))
async def nl_pause(message: Message, ctx: AppContext, work_session_id: int) -> None:
    await _do_pause(message, ctx, work_session_id)


@router.message(StateFilter(None), F.text, SessionPhrase("resume"))
async def nl_resume(message: Message, ctx: AppContext, work_session_id: int) -> None:
    await _do_resume(message, ctx, work_session_id)


@router.message(StateFilter(None), F.text, SessionPhrase("stop"))
async def nl_stop(message: Message, ctx: AppContext, work_session_id: int) -> None:
    await _do_stop(message, ctx, work_session_id)


@router.message(StateFilter(None), F.text, SessionPhrase("start"))
async def nl_start(
    message: Message, ctx: AppContext, state: FSMContext, start_candidate: svc.StartCandidate
) -> None:
    cand = start_candidate
    if cand.session_id is not None and cand.mode is not None:
        await _begin(
            message, ctx, state, task_id=cand.task_id, mode=cand.mode, session_id=cand.session_id
        )
        return
    await _show_modes(message, ctx, cand.task_id)
