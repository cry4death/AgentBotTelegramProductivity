"""Handlers for "📅 Мой день" and /today. Stage A: deterministic view without AI
(services.dayview); the AI morning plan with approve/revise arrives in stage B."""

from __future__ import annotations

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from procbot.context import AppContext
from procbot.services.dayview import build_day_view, render_day_view
from procbot.telegram.callbacks import DayCb
from procbot.telegram.keyboards import Menu

router = Router(name="day")


@router.message(Command("today"))
@router.message(F.text == Menu.DAY)
async def on_day(message: Message, ctx: AppContext, state: FSMContext) -> None:
    await state.clear()
    async with ctx.db.transaction() as s:
        view = await build_day_view(s, ctx)
    text, kb = render_day_view(view)
    await message.answer(text, reply_markup=kb)


@router.callback_query(DayCb.filter(F.action == "show"))
async def on_show(query: CallbackQuery, ctx: AppContext) -> None:
    async with ctx.db.transaction() as s:
        view = await build_day_view(s, ctx)
    text, kb = render_day_view(view)
    msg = query.message
    sent = False
    if isinstance(msg, Message):
        try:
            await msg.edit_text(text, reply_markup=kb)
            sent = True
        except TelegramBadRequest as exc:
            # "not modified" = already up to date; otherwise send a fresh message
            sent = "not modified" in str(exc)
    if not sent and query.bot is not None:
        await query.bot.send_message(query.from_user.id, text, reply_markup=kb)
    await query.answer()


@router.callback_query(DayCb.filter())
async def on_other(query: CallbackQuery) -> None:
    # approve / revise / reject / replan belong to the morning plan (stage B)
    await query.answer("Утвердить план можно будет, когда появится план от ИИ (этап B).")
