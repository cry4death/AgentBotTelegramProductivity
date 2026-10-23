"""/start, /help and the short, skippable onboarding (FR-01).

- The main menu is shown right away: the bot is usable before the profile is
  complete ("не превращать первый запуск в длинный обязательный опрос").
- Steps: timezone -> morning time -> evening time -> quiet hours -> base goals.
  Every step has "Пропустить". Work-session defaults are asked on first use
  (open question 5), not here.
- Re-running /start never resets data: it resumes the current step or just
  shows the menu.
- Stale buttons (a step that is already passed) are no-ops.
"""

from __future__ import annotations

from typing import Any

from aiogram import F, Router
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from procbot.context import AppContext
from procbot.services.preferences import (
    QUICK_TIMEZONES,
    format_window,
    from_compact,
    is_valid_timezone,
    parse_hhmm,
    parse_timezone,
    parse_window,
    utc_offset_label,
)
from procbot.services.profile import get_or_create_profile, get_profile, update_profile
from procbot.telegram.callbacks import OnboardingCb
from procbot.telegram.keyboards import Menu, main_menu

router = Router(name="start")

STEPS: tuple[str, ...] = ("timezone", "morning", "evening", "quiet", "goals")
GOALS_MAX_LEN = 2000

WELCOME = (
    "Привет! Я помогаю планировать день, вовремя начинать дела и разбираться, что мешает.\n"
    "Меню внизу уже работает — пользоваться можно сразу.\n"
    "Пара коротких вопросов для настройки; любой можно пропустить."
)
WELCOME_BACK = "С возвращением! Меню внизу. Настройки — «⚙️ Настройки» или /settings."
RESUME = "Продолжим знакомство с того места, где остановились (можно пропустить)."
DONE = (
    "Готово! Дальше можно просто пользоваться меню.\n"
    "Будни, сон, недельный обзор, характер бота и напоминания — в «⚙️ Настройки». "
    "Длительность рабочих сессий спрошу при первом запуске таймера."
)
STALE = "Этот шаг уже пройден. Изменить можно в «⚙️ Настройки»."

HELP = (
    "Разделы меню:\n"
    "📅 Мой день — план на сегодня\n"
    "➕ Добавить — новая задача\n"
    "📋 Задачи — список задач\n"
    "▶️ Начать работу — таймер, Помодоро, короткий старт\n"
    "🎯 Проекты/цели · 🔁 Привычки · 📊 Отчёт · 🧠 Память\n"
    "⚙️ Настройки — часовой пояс, тишина, пауза, характер, напоминания\n\n"
    "Команды:\n"
    "/start — меню и знакомство (данные не сбрасываются)\n"
    "/settings — настройки\n"
    "/status — состояние бота\n"
    "/help — эта справка"
)


class OnboardingForm(StatesGroup):
    timezone = State()
    morning = State()
    evening = State()
    quiet = State()
    goals = State()


_FORM_STATES: dict[str, State] = {
    "timezone": OnboardingForm.timezone,
    "morning": OnboardingForm.morning,
    "evening": OnboardingForm.evening,
    "quiet": OnboardingForm.quiet,
    "goals": OnboardingForm.goals,
}

QUESTIONS: dict[str, str] = {
    "timezone": "1/5. Какой у тебя часовой пояс? Выбери или напиши (город или Europe/Moscow).",
    "morning": "2/5. Во сколько присылать утренний план?",
    "evening": "3/5. Во сколько удобно подводить итоги дня?",
    "quiet": "4/5. Часы тишины — когда не беспокоить?",
    "goals": "5/5. Есть 1–3 главные цели на ближайшее время? Напиши одним сообщением.",
}
MANUAL_PROMPTS: dict[str, str] = {
    "timezone": "Напиши часовой пояс: город (Москва, Berlin) или IANA-имя (Europe/Moscow).",
    "morning": "Напиши время в формате ЧЧ:ММ, например 08:30.",
    "evening": "Напиши время в формате ЧЧ:ММ, например 21:30.",
    "quiet": "Напиши интервал ЧЧ:ММ-ЧЧ:ММ, например 23:00-08:00 (можно добавить «будни»).",
}
MORNING_PICKS = ("0700", "0800", "0900", "1000")
EVENING_PICKS = ("2000", "2100", "2200", "2300")
QUIET_PICKS = ("2200-0800", "2300-0700", "2300-0900", "0000-0800")


def next_step(step: str) -> str | None:
    i = STEPS.index(step)
    return STEPS[i + 1] if i + 1 < len(STEPS) else None


def _btn(text: str, step: str, arg: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=OnboardingCb(step=step, arg=arg).pack())


def _tail(step: str, *, manual: bool = True) -> list[list[InlineKeyboardButton]]:
    row = [_btn("✏️ Ввести вручную", step, "manual")] if manual else []
    row.append(_btn("Пропустить", step, "skip"))
    return [row]


def _rows(buttons: list[InlineKeyboardButton], per_row: int) -> list[list[InlineKeyboardButton]]:
    return [buttons[i : i + per_row] for i in range(0, len(buttons), per_row)]


def timezone_keyboard(ctx: AppContext, zones: list[str] | None = None) -> InlineKeyboardMarkup:
    now = ctx.clock.now()
    labels = dict(QUICK_TIMEZONES)
    picks = zones if zones is not None else [z for z, _ in QUICK_TIMEZONES]
    buttons = [
        _btn(f"{labels.get(z, z)} ({utc_offset_label(z, now)})", "timezone", z) for z in picks
    ]
    return InlineKeyboardMarkup(inline_keyboard=_rows(buttons, 2) + _tail("timezone"))


def question_keyboard(ctx: AppContext, step: str) -> InlineKeyboardMarkup:
    if step == "timezone":
        return timezone_keyboard(ctx)
    if step in ("morning", "evening"):
        picks = MORNING_PICKS if step == "morning" else EVENING_PICKS
        buttons = [_btn(f"{p[:2]}:{p[2:]}", step, p) for p in picks]
        return InlineKeyboardMarkup(inline_keyboard=[buttons, *_tail(step)])
    if step == "quiet":
        buttons = [_btn(f"{p[:2]}:{p[2:4]}–{p[5:7]}:{p[7:]}", step, p) for p in QUIET_PICKS]
        return InlineKeyboardMarkup(inline_keyboard=_rows(buttons, 2) + _tail(step))
    return InlineKeyboardMarkup(inline_keyboard=_tail(step, manual=False))


async def ask_step(message: Message, ctx: AppContext, state: FSMContext, step: str) -> None:
    if step == "goals":
        await state.set_state(OnboardingForm.goals)  # free text expected right away
    await message.answer(QUESTIONS[step], reply_markup=question_keyboard(ctx, step))


def _changes_for(step: str, arg: str) -> tuple[dict[str, Any], str] | None:
    """Callback arg -> (profile changes, human summary). None = invalid value."""
    if arg == "skip":
        return {}, "пропущено"
    if step == "timezone":
        return ({"timezone": arg}, arg) if is_valid_timezone(arg) else None
    if step in ("morning", "evening"):
        t = from_compact(arg)
        if t is None:
            return None
        return {f"{step}_time": t}, t.strftime("%H:%M")
    if step == "quiet":
        start, _, end = arg.partition("-")
        try:
            window = parse_window(f"{start[:2]}:{start[2:]}-{end[:2]}:{end[2:]}")
        except ValueError:
            return None
        return {"quiet_hours": [window]}, format_window(window)
    return None


STEP_TITLES = {
    "timezone": "Часовой пояс",
    "morning": "Утренний план",
    "evening": "Итоги дня",
    "quiet": "Часы тишины",
    "goals": "Цели",
}


async def apply_step(ctx: AppContext, step: str, changes: dict[str, Any]) -> bool:
    """Save the answer and advance. False if the step is not current (stale button)."""
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        profile = await get_profile(s)
        if profile is None or profile.onboarding_step != step:
            return False
        await update_profile(s, now=now, **changes, onboarding_step=next_step(step))
    return True


async def _after_step(
    message: Message, ctx: AppContext, state: FSMContext, step: str, summary: str
) -> None:
    await state.clear()
    await message.answer(f"✓ {STEP_TITLES[step]}: {summary}")
    nxt = next_step(step)
    if nxt is None:
        await message.answer(DONE, reply_markup=main_menu())
    else:
        await ask_step(message, ctx, state, nxt)


# --------------------------------------------------------------------------- commands


@router.message(CommandStart())
async def cmd_start(message: Message, ctx: AppContext, state: FSMContext) -> None:
    now = ctx.clock.now()
    uid = message.from_user.id if message.from_user else ctx.owner_id
    async with ctx.db.transaction() as s:
        profile, created = await get_or_create_profile(
            s, telegram_user_id=uid, now=now, chat_id=message.chat.id
        )
        step = profile.onboarding_step
    await state.clear()  # /start is also the way out of any pending question
    if step is None or step not in STEPS:
        await message.answer(WELCOME_BACK, reply_markup=main_menu())
        return
    await message.answer(WELCOME if created else RESUME, reply_markup=main_menu())
    await ask_step(message, ctx, state, step)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP, reply_markup=main_menu())


# --------------------------------------------------------------------------- buttons


@router.callback_query(OnboardingCb.filter())
async def on_onboarding(
    cq: CallbackQuery, callback_data: OnboardingCb, ctx: AppContext, state: FSMContext
) -> None:
    step, arg = callback_data.step, callback_data.arg
    msg = cq.message if isinstance(cq.message, Message) else None
    if step not in STEPS:
        await cq.answer()
        return
    async with ctx.db.transaction() as s:
        profile = await get_profile(s)
        current = profile.onboarding_step if profile else None
    if current != step:
        await cq.answer(STALE, show_alert=False)
        return
    if arg == "manual" and step in MANUAL_PROMPTS:
        await state.set_state(_FORM_STATES[step])
        await cq.answer()
        if msg is not None:
            await msg.answer(MANUAL_PROMPTS[step])
        return
    parsed = _changes_for(step, arg)
    if parsed is None:
        await cq.answer("Неизвестное значение", show_alert=False)
        return
    changes, summary = parsed
    if not await apply_step(ctx, step, changes):
        await cq.answer(STALE, show_alert=False)
        return
    await cq.answer()
    if msg is not None:
        await msg.edit_reply_markup(reply_markup=None)
        await _after_step(msg, ctx, state, step, summary)


# --------------------------------------------------------------------------- text answers


def _passthrough(message: Message) -> bool:
    """Menu buttons and commands leave the onboarding question for other handlers."""
    text = message.text or ""
    return text in Menu.ALL or text.startswith("/")


@router.message(StateFilter(OnboardingForm), F.text)
async def on_onboarding_text(message: Message, ctx: AppContext, state: FSMContext) -> None:
    current_state = await state.get_state()
    step = next((k for k, v in _FORM_STATES.items() if v.state == current_state), None)
    if _passthrough(message) or step is None:
        await state.clear()
        raise SkipHandler()
    text = (message.text or "").strip()
    changes: dict[str, Any]
    summary: str
    if step == "timezone":
        match = parse_timezone(text, ctx.clock.now())
        if match.zone is None:
            if match.candidates:
                await message.answer(
                    "Под это подходит несколько поясов — выбери свой:",
                    reply_markup=timezone_keyboard(ctx, match.candidates),
                )
            else:
                await message.answer(
                    "Не узнаю такой часовой пояс. Напиши город (Москва, Berlin) "
                    "или IANA-имя вроде Europe/Moscow."
                )
            return
        changes, summary = {"timezone": match.zone}, match.zone
    elif step in ("morning", "evening"):
        t = parse_hhmm(text)
        if t is None:
            await message.answer("Не понял время. Формат ЧЧ:ММ, например 08:30.")
            return
        changes, summary = {f"{step}_time": t}, t.strftime("%H:%M")
    elif step == "quiet":
        try:
            window = parse_window(text)
        except ValueError:
            await message.answer("Не понял интервал. Пример: 23:00-08:00 или 23:00-08:00 будни.")
            return
        changes, summary = {"quiet_hours": [window]}, format_window(window)
    else:  # goals
        note = text[:GOALS_MAX_LEN]
        if not note:
            return
        changes, summary = {"goals_note": note}, "сохранено"
    if not await apply_step(ctx, step, changes):
        await state.clear()
        await message.answer(STALE)
        return
    await _after_step(message, ctx, state, step, summary)


__all__ = ["STEPS", "OnboardingForm", "next_step", "router"]
