"""«📊 Отчёт» (FR-11): today / week / custom period, evening answers, disputes.

- Today = the evening review on demand; week / period = the six directions.
- "Не согласен с выводом" (scenario H): the owner's correction is saved as
  their own statement and hypotheses can be refuted right there.
- Proposals from the weekly AI comment are pending approvals: shown on a normal
  approval card (handlers.approvals), never applied without a press.
- AI costs live in /ai (button «💰 Расходы ИИ»).
"""

from __future__ import annotations

from datetime import date, timedelta

from aiogram import F, Router
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from procbot.context import AppContext
from procbot.db.models import Task
from procbot.domain.enums import ObstacleReason
from procbot.services import analytics, approvals, memory, reports
from procbot.services.tasks import TaskError
from procbot.telegram.callbacks import AIKeyCb, MemoryCb, ReportCb
from procbot.telegram.handlers.approvals import send_approval_card
from procbot.telegram.keyboards import Menu
from procbot.telegram.middlewares import mark_processed
from procbot.timeutil import local_date

router = Router(name="reports")

REASON_BUTTONS: dict[str, str] = {
    ObstacleReason.UNCLEAR: "🤔 Не понимал, с чего начать",
    ObstacleReason.TOO_BIG: "🐘 Слишком большая",
    ObstacleReason.UNPLEASANT: "😖 Неприятно",
    ObstacleReason.DISTRACTIONS: "📱 Отвлёкся",
    ObstacleReason.LOW_ENERGY: "🔋 Не было сил",
    ObstacleReason.PRIORITY_CONFLICT: "⚖️ Другое было важнее",
    ObstacleReason.EXTERNAL: "🚪 Внешние дела",
    ObstacleReason.OTHER: "🤷 Другое",
}
PERIOD_PROMPT = (
    "Напиши период в формате ДД.ММ-ДД.ММ (например 01.10-15.10), не длиннее "
    f"{reports.MAX_PERIOD_DAYS} дней. /cancel — отмена."
)


class ReportForm(StatesGroup):
    period = State()
    dispute = State()


def _b(text: str, action: str, arg: str = "") -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=ReportCb(action=action, arg=arg).pack())


def menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [_b("📅 Сегодня", "today"), _b("🗓 Неделя", "week")],
            [_b("📆 Свой период", "period")],
            [
                InlineKeyboardButton(
                    text="💰 Расходы ИИ", callback_data=AIKeyCb(action="spend", arg="month").pack()
                )
            ],
        ]
    )


MENU_TEXT = (
    "📊 Отчёты\n"
    "• Сегодня — короткий обзор дня.\n"
    "• Неделя — шесть направлений: выполнение и планы, задержка старта, причины "
    "прокрастинации, помощь, цели, гипотезы.\n"
    "• Свой период — то же за выбранные даты.\n"
    "Расходы на ИИ — /ai."
)


async def _today(ctx: AppContext) -> date:
    async with ctx.db.transaction() as s:
        tz = await analytics.owner_tz(s)
    return local_date(ctx.clock.now(), tz)


# --------------------------------------------------------------------------- entry


@router.message(Command("report"))
@router.message(F.text == Menu.REPORT)
async def on_report(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(MENU_TEXT, reply_markup=menu_keyboard())


@router.message(Command("week"))
async def on_week_cmd(message: Message, ctx: AppContext, update_id: int) -> None:
    await _send_week(message, ctx, update_id)


async def _send_week(message: Message, ctx: AppContext, update_id: int) -> None:
    today = await _today(ctx)
    rep = await reports.build_weekly(
        ctx, first_day=today - timedelta(days=6), last_day=today, op_prefix=f"u{update_id}"
    )
    await message.answer(rep.text, reply_markup=rep.markup)


# --------------------------------------------------------------------------- callbacks


@router.callback_query(ReportCb.filter())
async def on_report_cb(
    query: CallbackQuery,
    callback_data: ReportCb,
    ctx: AppContext,
    state: FSMContext,
    update_id: int,
) -> None:
    msg = query.message if isinstance(query.message, Message) else None
    action, arg = callback_data.action, callback_data.arg
    if msg is None:
        await query.answer()
        return
    if action == "menu":
        await query.answer()
        await msg.answer(MENU_TEXT, reply_markup=menu_keyboard())
    elif action == "today":
        await query.answer()
        text, kb = await reports.build_evening(ctx)
        await msg.answer(text, reply_markup=kb)
    elif action == "week":
        await query.answer("Считаю…")
        await _send_week(msg, ctx, update_id)
    elif action == "period":
        await state.set_state(ReportForm.period)
        await query.answer()
        await msg.answer(PERIOD_PROMPT)
    elif action == "why":
        await query.answer()
        await _ask_reason(msg, ctx, arg)
    elif action == "rsn":
        await _record_reason(query, msg, ctx, arg, update_id)
    elif action == "props":
        await query.answer()
        async with ctx.db.transaction() as s:
            items = await approvals.list_group(s, arg)
            tz = await analytics.owner_tz(s)
        if not items:
            await msg.answer("Предложений не найдено (возможно, срок карточки истёк).")
        else:
            await send_approval_card(
                msg, ctx, items, tz=tz, header="Предложения на следующую неделю:"
            )
    elif action == "dispute":
        await query.answer()
        await state.set_state(ReportForm.dispute)
        await msg.answer(await _dispute_prompt(ctx), reply_markup=await _hypotheses_keyboard(ctx))
    else:
        await query.answer()


async def _ask_reason(msg: Message, ctx: AppContext, arg: str) -> None:
    if not arg.isdigit():
        return
    rows = []
    codes = list(REASON_BUTTONS)
    for i in range(0, len(codes), 2):
        rows.append([_b(REASON_BUTTONS[c], "rsn", f"{arg}.{c}") for c in codes[i : i + 2]])
    async with ctx.db.transaction() as s:
        task = await s.get(Task, int(arg))
    title = f"«{task.title}»" if task else "задачей"
    await msg.answer(
        f"Что помешало с {title}? Это предположение, а не диагноз — можно выбрать «Другое».",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=rows),
    )


async def _record_reason(
    query: CallbackQuery, msg: Message, ctx: AppContext, arg: str, update_id: int
) -> None:
    task_part, _, code = arg.partition(".")
    if not task_part.isdigit() or code not in {r.value for r in ObstacleReason}:
        await query.answer()
        return
    try:
        async with ctx.db.transaction() as s:
            created = await reports.record_evening_reason(s, ctx, int(task_part), code)
            await mark_processed(s, update_id, ctx.clock.now())
    except TaskError as exc:
        await query.answer(str(exc)[:200], show_alert=True)
        return
    await query.answer("Записал" if created else "Уже записано")
    label = REASON_BUTTONS.get(code, code)
    text = (
        f"Записал: {label}. Учту в недельном разборе — это твоя отметка, а не диагноз."
        if created
        else "Эта причина уже отмечена сегодня."
    )
    try:
        await msg.edit_text(text)
    except TelegramBadRequest:
        await msg.answer(text)


async def _dispute_prompt(ctx: AppContext) -> str:
    return (
        "С чем ты не согласен? Напиши своими словами — сохраню как твоё утверждение и "
        "буду опираться на него. Если неверна конкретная гипотеза — нажми её кнопку ниже, "
        "она будет опровергнута и больше не используется. /cancel — отмена."
    )


async def _hypotheses_keyboard(ctx: AppContext) -> InlineKeyboardMarkup | None:
    async with ctx.db.transaction() as s:
        hyps = await memory.active_hypotheses(s, limit=reports.MAX_DISPUTE_BUTTONS)
    if not hyps:
        return None
    rows = [
        [
            InlineKeyboardButton(
                text=f"🙅 {h.content[:40]}",
                callback_data=MemoryCb(action="refute", item_id=h.id, arg="wk").pack(),
            )
        ]
        for h in hyps
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


# --------------------------------------------------------------------------- text input


@router.message(StateFilter(ReportForm), F.text)
async def on_report_text(
    message: Message, ctx: AppContext, state: FSMContext, update_id: int
) -> None:
    text = (message.text or "").strip()
    current = await state.get_state()
    if text in Menu.ALL or text.startswith("/"):
        await state.clear()
        raise SkipHandler()
    if current == ReportForm.period.state:
        today = await _today(ctx)
        try:
            first, last = reports.parse_period(text, today)
        except ValueError:
            await message.answer("Не понял период. " + PERIOD_PROMPT)
            return
        await state.clear()
        title = f"📊 Отчёт за {first.strftime('%d.%m.%Y')}–{last.strftime('%d.%m.%Y')}"
        rep = await reports.build_weekly(
            ctx, first_day=first, last_day=last, op_prefix=f"u{update_id}", title=title
        )
        await message.answer(rep.text, reply_markup=rep.markup)
        return
    await state.clear()
    async with ctx.db.transaction() as s:
        item = await reports.record_dispute(s, ctx, text, update_id=update_id)
        await mark_processed(s, update_id, ctx.clock.now())
    if item is None:
        await message.answer(
            "Память выключена (⚙️ Настройки → Память), поэтому поправку сохранить не могу. "
            "Опровергнуть гипотезу можно кнопкой под отчётом."
        )
        return
    await message.answer(
        "Спасибо, записал как твоё утверждение («вы сказали»). Сводки памяти обновятся с учётом "
        "поправки; неверную гипотезу можно опровергнуть кнопкой.",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="Открыть запись",
                        callback_data=MemoryCb(action="view", item_id=item.id).pack(),
                    )
                ]
            ]
        ),
    )
