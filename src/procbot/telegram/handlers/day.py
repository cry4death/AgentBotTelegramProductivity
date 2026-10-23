"""Handlers for "📅 Мой день" and /today (DayCb, prefix "dy").

- With a plan for today (proposed or approved): the current revision with its
  explanation, "≈" approximate marks and calendar staleness; buttons Утвердить /
  Изменить / Отклонить (proposal) or Изменить / Пересобрать (approved).
- Without a plan: the stage-A deterministic day view + "🗓 Составить план".
- "Изменить" takes free text ("перенеси отчёт на вечер") -> an AI revision
  that is again only a proposal; without AI nothing is guessed.
- "Утвердить" = the owner's explicit consent: ApplyDayPlan (services.planning).
"""

from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from procbot.ai.errors import AIUnavailable
from procbot.context import AppContext
from procbot.domain.enums import PlanStatus
from procbot.services import planning
from procbot.services.actions import ExecStatus
from procbot.services.dayview import build_day_view, render_day_keyboard, render_day_text
from procbot.services.tasks import local_today
from procbot.telegram.callbacks import DayCb
from procbot.telegram.keyboards import Menu

log = logging.getLogger(__name__)

router = Router(name="day")

TEXT_INPUT = (F.text, ~F.text.startswith("/"), ~F.text.in_(set(Menu.ALL)))
MAX_TEXT = 4000


class DayForm(StatesGroup):
    change = State()  # data: plan_id, revision_id


def _make_button() -> InlineKeyboardButton:
    return InlineKeyboardButton(text="🗓 Составить план", callback_data=DayCb(action="make").pack())


async def _stage_a_view(
    ctx: AppContext, note: str | None = None
) -> tuple[str, InlineKeyboardMarkup]:
    async with ctx.db.transaction() as s:
        view = await build_day_view(s, ctx)
    text = render_day_text(view, note=note)
    kb = render_day_keyboard(view)
    rows = [*kb.inline_keyboard, [_make_button()]]
    return text, InlineKeyboardMarkup(inline_keyboard=rows)


async def day_screen(ctx: AppContext) -> tuple[str, InlineKeyboardMarkup]:
    """The current plan revision of today, or the stage-A view + "Составить план"."""
    async with ctx.db.transaction() as s:
        view = await planning.load_plan_view(s, ctx)
    if view is not None and view.revision.status in (PlanStatus.PROPOSED, PlanStatus.APPROVED):
        text, kb = planning.render_plan(view)
        return text[:MAX_TEXT], kb
    note = None
    if view is not None and view.revision.status == PlanStatus.REJECTED:
        note = "Предложенный план отклонён — можно составить новый."
    return await _stage_a_view(ctx, note=note)


async def _show(query: CallbackQuery, text: str, kb: InlineKeyboardMarkup | None) -> None:
    """Edit the message under the button if possible, else send a new one."""
    msg = query.message
    sent = False
    if isinstance(msg, Message):
        try:
            await msg.edit_text(text[:MAX_TEXT], reply_markup=kb)
            sent = True
        except TelegramBadRequest as exc:
            sent = "not modified" in str(exc)
    if not sent and query.bot is not None:
        await query.bot.send_message(query.from_user.id, text[:MAX_TEXT], reply_markup=kb)


async def _send(query: CallbackQuery, text: str, kb: InlineKeyboardMarkup | None = None) -> None:
    if isinstance(query.message, Message):
        await query.message.answer(text[:MAX_TEXT], reply_markup=kb)
    elif query.bot is not None:
        await query.bot.send_message(query.from_user.id, text[:MAX_TEXT], reply_markup=kb)


async def _answer(query: CallbackQuery, text: str | None = None) -> None:
    try:
        await query.answer(text)
    except Exception:  # pragma: no cover - an expired query must not break the reply
        log.debug("callback answer failed", exc_info=True)


async def _revision_screen(ctx: AppContext, revision_id: int, header: str | None = None):
    async with ctx.db.transaction() as s:
        view = await planning.load_plan_view(s, ctx, revision_id=revision_id)
    if view is None:
        return await day_screen(ctx)
    text, kb = planning.render_plan(view, header=header)
    return text[:MAX_TEXT], kb


@router.message(Command("today"))
@router.message(F.text == Menu.DAY)
async def on_day(message: Message, ctx: AppContext, state: FSMContext) -> None:
    await state.clear()
    text, kb = await day_screen(ctx)
    await message.answer(text, reply_markup=kb)


@router.callback_query(DayCb.filter(F.action == "show"))
async def on_show(query: CallbackQuery, ctx: AppContext) -> None:
    text, kb = await day_screen(ctx)
    await _show(query, text, kb)
    await _answer(query)


@router.callback_query(DayCb.filter(F.action == "tasks"))
async def on_tasks(query: CallbackQuery, ctx: AppContext) -> None:
    text, kb = await _stage_a_view(ctx)
    await _show(query, text, kb)
    await _answer(query)


@router.callback_query(DayCb.filter(F.action == "make"))
async def on_make(query: CallbackQuery, ctx: AppContext, update_id: int) -> None:
    await _answer(query, "Составляю план…")
    async with ctx.db.transaction() as s:
        day = await local_today(s, ctx)
        recent = await planning.recent_proposal(s, ctx, day)
    rev_id = (
        recent.id
        if recent is not None
        else await planning.propose(
            ctx, day=day, change_reason="по запросу", op_key=f"u{update_id}:plan"
        )
    )
    text, kb = await _revision_screen(ctx, rev_id)
    await _send(query, text, kb)


@router.callback_query(DayCb.filter(F.action == "replan"))
async def on_replan(query: CallbackQuery, ctx: AppContext, update_id: int) -> None:
    await _answer(query, "Пересобираю план…")
    async with ctx.db.transaction() as s:
        day = await local_today(s, ctx)
        recent = await planning.recent_proposal(s, ctx, day)
    rev_id = (
        recent.id
        if recent is not None
        else await planning.propose(
            ctx,
            day=day,
            mode="replan",
            change_reason="пересборка по запросу",
            op_key=f"u{update_id}:replan",
        )
    )
    text, kb = await _revision_screen(ctx, rev_id)
    await _send(query, text, kb)


@router.callback_query(DayCb.filter(F.action == "approve"))
async def on_approve(query: CallbackQuery, callback_data: DayCb, ctx: AppContext) -> None:
    async with ctx.db.transaction() as s:
        res = await planning.approve_revision(
            s, ctx, plan_id=callback_data.plan_id, revision_id=callback_data.revision_id
        )
    if res.status == ExecStatus.APPLIED:
        moved = len(res.result.get("moved", []))
        note = "План утверждён ✅ Время задач и напоминания перестроены по плану."
        if moved:
            note += f" Переносов применено: {moved}."
        if res.result.get("skipped_past"):
            note += " Пункты, время которых уже прошло, не назначались."
        if res.repeated or res.result.get("already"):
            note = "План уже утверждён."
        await _answer(query, "Утверждено")
        text, kb = await _revision_screen(ctx, callback_data.revision_id, header=note)
        await _show(query, text, kb)
        return
    await _answer(query)
    await _send(query, res.error or "Не удалось утвердить план — ничего не изменено.")


@router.callback_query(DayCb.filter(F.action == "reject"))
async def on_reject(query: CallbackQuery, callback_data: DayCb, ctx: AppContext) -> None:
    async with ctx.db.transaction() as s:
        rev = await planning.reject_revision(s, ctx, revision_id=callback_data.revision_id)
    if rev is None:
        await _answer(query, "План не найден.")
        return
    if rev.status != PlanStatus.REJECTED:
        await _answer(query, "Эта версия уже не ждёт решения.")
        return
    await _answer(query, "Отклонено")
    text, kb = await day_screen(ctx)
    await _show(query, "Предложение отклонено — ничего не изменено.\n\n" + text, kb)


@router.callback_query(DayCb.filter(F.action == "revise"))
async def on_revise(
    query: CallbackQuery, callback_data: DayCb, ctx: AppContext, state: FSMContext
) -> None:
    await state.set_state(DayForm.change)
    await state.update_data(plan_id=callback_data.plan_id, revision_id=callback_data.revision_id)
    await _answer(query)
    await _send(
        query,
        "Напиши, что поменять в плане. Например: «перенеси отчёт на вечер», «сегодня мало "
        "сил — оставь главное», «добавь время на спорт». Я предложу новую версию — "
        "применится только после «Утвердить».",
    )


@router.message(DayForm.change, *TEXT_INPUT)
async def on_change_text(
    message: Message, ctx: AppContext, state: FSMContext, update_id: int
) -> None:
    await state.clear()
    text = (message.text or "").strip()
    try:
        rev_id = await planning.propose(
            ctx,
            mode="replan",
            user_request=text[:1000],
            change_reason=f"просьба: {text[:200]}",
            op_key=f"u{update_id}:revise",
        )
    except AIUnavailable:
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="🔄 Пересобрать план", callback_data=DayCb(action="replan").pack()
                    ),
                    InlineKeyboardButton(
                        text="📋 Все дела дня", callback_data=DayCb(action="tasks").pack()
                    ),
                ]
            ]
        )
        await message.answer(
            "Без ИИ не могу надёжно разобрать такую просьбу — ничего не изменено. "
            "Перенести задачу можно из её карточки, а план — пересобрать кнопкой.",
            reply_markup=kb,
        )
        return
    body, kb = await _revision_screen(ctx, rev_id, header="Новая версия плана по твоей просьбе:")
    await message.answer(body, reply_markup=kb)


@router.callback_query(DayCb.filter(F.action == "history"))
async def on_history(query: CallbackQuery, callback_data: DayCb, ctx: AppContext) -> None:
    async with ctx.db.transaction() as s:
        text = await planning.render_history(s, ctx, callback_data.plan_id)
    await _answer(query)
    await _send(
        query,
        text,
        InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="📅 Мой день", callback_data=DayCb(action="show").pack()
                    )
                ]
            ]
        ),
    )


@router.callback_query(DayCb.filter())
async def on_other(query: CallbackQuery) -> None:
    await _answer(query)
