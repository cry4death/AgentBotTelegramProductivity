"""Telegram handlers: buttons under reminders (ReminderCb, prefix "rm").

- reason: the user picked a likely obstacle after a start delay. Recorded as a quick
  intervention (hypothesis the user said, no AI) and answered with a template next
  step. Rest and conscious postponement are valid outcomes (FR-04).
- rest: the user chose to rest; recorded, no pressure.
- later / snooze: plan a new reminder at a moment the user picked (gate rules apply).
- evening: feedback on the evening review.

Replies to button presses are answers to the user, not bot-initiated messages, so
they do not go through the Outbound Gate.
"""

from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.models import Task
from procbot.domain.enums import Actor, InterventionKind, ObstacleReason
from procbot.services.audit import audit
from procbot.services.interventions_basic import record_quick_intervention
from procbot.services.profile import get_profile
from procbot.services.reminders import snooze_reminder
from procbot.telegram.callbacks import DayCb, ReminderCb, SessionCb, TaskCb
from procbot.timeutil import fmt_local

log = logging.getLogger(__name__)

router = Router(name="reminders")

SNOOZE_CHOICES: tuple[tuple[str, str], ...] = (
    ("15", "через 15 мин"),
    ("30", "через 30 мин"),
    ("60", "через час"),
)


def _btn(text: str, data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=data)


def _start(task_id: int, text: str = "▶️ Начать") -> InlineKeyboardButton:
    return _btn(text, SessionCb(action="pick", task_id=task_id).pack())


def _short_start(task_id: int) -> InlineKeyboardButton:
    return _btn(
        "⏱ Короткий старт", SessionCb(action="pick", task_id=task_id, arg="short_start").pack()
    )


def _postpone(task_id: int) -> InlineKeyboardButton:
    return _btn("📆 Перенести", TaskCb(action="postpone", task_id=task_id).pack())


def _split(task_id: int) -> InlineKeyboardButton:
    return _btn("✂️ Разбить на шаги", TaskCb(action="subtasks", task_id=task_id).pack())


def _rest(reminder_id: int, task_id: int) -> InlineKeyboardButton:
    return _btn(
        "☕ Отдохнуть", ReminderCb(action="rest", reminder_id=reminder_id, task_id=task_id).pack()
    )


def next_step(
    reason: ObstacleReason, *, reminder_id: int, task_id: int
) -> tuple[InterventionKind, str, list[list[InlineKeyboardButton]]]:
    """Stage-A template answer per obstacle (no AI). Worded as a guess the user can
    reject, never as a diagnosis or a reproach (acceptance 10)."""
    R = ObstacleReason
    if reason == R.UNCLEAR:
        return (
            InterventionKind.TINY_STEP,
            "Похоже, пока непонятно, с какого конца взяться — это нормально. "
            "Назови самый первый физический шаг, совсем маленький, на пару минут: "
            "открыть файл, записать три пункта, найти нужную ссылку. "
            "А потом можно запустить короткий старт.",
            [[_short_start(task_id)], [_split(task_id), _postpone(task_id)]],
        )
    if reason == R.TOO_BIG:
        return (
            InterventionKind.SPLIT,
            "Возможно, задача просто слишком крупная, чтобы за неё браться целиком. "
            "Давай разобьём её на шаги и начнём с самого маленького.",
            [[_split(task_id)], [_short_start(task_id), _postpone(task_id)]],
        )
    if reason == R.UNPLEASANT:
        return (
            InterventionKind.SHORT_START,
            "Бывает, что задача просто неприятная. Иногда помогает договориться с собой "
            "только начать — коротко, без обязательства закончить.",
            [[_short_start(task_id)], [_split(task_id), _postpone(task_id)]],
        )
    if reason == R.DISTRACTIONS:
        return (
            InterventionKind.REMOVE_DISTRACTIONS,
            "Если что-то отвлекает — попробуй убрать телефон подальше, закрыть лишние "
            "вкладки и включить таймер. Во время сессии я не буду писать с проверками.",
            [[_start(task_id, "▶️ Начать сессию")], [_short_start(task_id), _postpone(task_id)]],
        )
    if reason == R.LOW_ENERGY:
        return (
            InterventionKind.REST,
            "Если сил мало, это стоит учитывать, а не преодолевать любой ценой. "
            "Можно отдохнуть, сделать совсем короткий старт или сознательно перенести задачу.",
            [[_rest(reminder_id, task_id)], [_short_start(task_id), _postpone(task_id)]],
        )
    if reason == R.PRIORITY_CONFLICT:
        return (
            InterventionKind.RESCHEDULE,
            "Если сейчас важнее другое — лучше сознательно перенести эту задачу, "
            "чтобы она не висела. Можно открыть день и пересмотреть порядок.",
            [[_postpone(task_id)], [_btn("📅 Мой день", DayCb(action="show").pack())]],
        )
    if reason == R.EXTERNAL:
        return (
            InterventionKind.RESCHEDULE,
            "Внешние дела — понятная причина. Перенести задачу на другое время?",
            [[_postpone(task_id)], [_start(task_id, "▶️ Всё-таки начать")]],
        )
    return (
        InterventionKind.TINY_STEP,
        "Понял. Если хочешь, напиши в паре слов, что мешает, — подумаем вместе. "
        "Или выбери, что сделать сейчас:",
        [[_short_start(task_id)], [_split(task_id), _postpone(task_id)]],
    )


async def _reply(
    query: CallbackQuery, text: str, markup: InlineKeyboardMarkup | None = None
) -> None:
    if isinstance(query.message, Message):
        await query.message.answer(text, reply_markup=markup)
    elif query.bot is not None:
        await query.bot.send_message(query.from_user.id, text, reply_markup=markup)
    await query.answer()


async def _report_obstacle(
    ctx: AppContext, s: AsyncSession, task_id: int, reason: ObstacleReason
) -> None:
    """Journal the obstacle on the task (task_events, owned by TaskService)."""
    try:
        from procbot.services.tasks import report_obstacle
    except ImportError:  # pragma: no cover
        return
    try:
        await report_obstacle(s, ctx, task_id, reason=reason, actor=Actor.USER)
    except NotImplementedError:  # pragma: no cover
        return


@router.callback_query(ReminderCb.filter(F.action == "reason"))
async def on_reason(query: CallbackQuery, callback_data: ReminderCb, ctx: AppContext) -> None:
    try:
        reason = ObstacleReason(callback_data.arg)
    except ValueError:
        await query.answer()
        return
    task_id = callback_data.task_id
    kind, text, rows = next_step(reason, reminder_id=callback_data.reminder_id, task_id=task_id)
    async with ctx.db.transaction() as s:
        task = await s.get(Task, task_id) if task_id else None
        _, created = await record_quick_intervention(
            s,
            now=ctx.clock.now(),
            task_id=task.id if task else None,
            reminder_id=callback_data.reminder_id or None,
            kind=kind,
            reason=reason,
            suggestion=text,
        )
        if created and task is not None:
            await _report_obstacle(ctx, s, task.id, reason)
    if task is None:
        await _reply(query, "Эта задача больше не найдена.")
        return
    await _reply(query, text, InlineKeyboardMarkup(inline_keyboard=rows))


@router.callback_query(ReminderCb.filter(F.action == "rest"))
async def on_rest(query: CallbackQuery, callback_data: ReminderCb, ctx: AppContext) -> None:
    task_id = callback_data.task_id
    async with ctx.db.transaction() as s:
        task = await s.get(Task, task_id) if task_id else None
        await record_quick_intervention(
            s,
            now=ctx.clock.now(),
            task_id=task.id if task else None,
            reminder_id=callback_data.reminder_id or None,
            kind=InterventionKind.REST,
            reason=None,
            suggestion="rest",
        )
    text = (
        "Отдых — нормальный выбор. Когда будешь готов, начни задачу или перенеси её — без спешки."
    )
    rows = [[_start(task.id), _postpone(task.id)]] if task is not None else []
    await _reply(query, text, InlineKeyboardMarkup(inline_keyboard=rows) if rows else None)


@router.callback_query(ReminderCb.filter(F.action == "later"))
async def on_later(query: CallbackQuery, callback_data: ReminderCb) -> None:
    rows = [
        [
            _btn(
                label,
                ReminderCb(
                    action="snooze",
                    reminder_id=callback_data.reminder_id,
                    task_id=callback_data.task_id,
                    arg=minutes,
                ).pack(),
            )
            for minutes, label in SNOOZE_CHOICES
        ]
    ]
    await _reply(query, "Когда напомнить ещё раз?", InlineKeyboardMarkup(inline_keyboard=rows))


@router.callback_query(ReminderCb.filter(F.action == "snooze"))
async def on_snooze(query: CallbackQuery, callback_data: ReminderCb, ctx: AppContext) -> None:
    allowed = {m for m, _ in SNOOZE_CHOICES}
    if callback_data.arg not in allowed or not callback_data.reminder_id:
        await query.answer()
        return
    async with ctx.db.transaction() as s:
        row, created = await snooze_reminder(
            s, ctx, reminder_id=callback_data.reminder_id, minutes=int(callback_data.arg)
        )
        profile = await get_profile(s)
        tz = profile.timezone if profile else None
    if row is None:
        await _reply(query, "Это напоминание уже неактуально.")
        return
    at = fmt_local(row.due_at, tz, "%H:%M")
    if created:
        text = (
            f"Хорошо, напомню около {at} — если в это время можно писать "
            "(часы тишины, пауза и лимит напоминаний соблюдаются)."
        )
    else:
        text = f"Уже отложено — напомню около {at}."
    await _reply(query, text)


@router.callback_query(ReminderCb.filter(F.action == "evening"))
async def on_evening(query: CallbackQuery, callback_data: ReminderCb, ctx: AppContext) -> None:
    deviation = callback_data.arg == "deviation"
    async with ctx.db.transaction() as s:
        await audit(
            s,
            now=ctx.clock.now(),
            actor=Actor.USER,
            action="evening.feedback",
            entity="reminder",
            entity_id=callback_data.reminder_id or None,
            details={"deviation": deviation},
        )
    if deviation:
        text = (
            "Понял. Если хочешь, напиши в паре слов, что пошло не так или что помешало, — "
            "учту при следующем планировании. Перенести незаконченное можно в «📅 Мой день»."
        )
        markup = InlineKeyboardMarkup(
            inline_keyboard=[[_btn("📅 Мой день", DayCb(action="show").pack())]]
        )
    else:
        text = "Отлично. Хорошего вечера!"
        markup = None
    await _reply(query, text, markup)
