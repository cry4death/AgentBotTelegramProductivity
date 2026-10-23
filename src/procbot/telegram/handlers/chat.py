"""Free-text fallback (must stay the LAST router) and the shared text pipeline.

`process_text` is used for typed text and voice transcripts (handlers.voice):

1. "да" as a reply to an approval card accepts what is pending on that card;
2. no AI configured -> stage A: one CREATE_TASK proposal per line / ';' on ONE
   card ("Создать"/"Не надо" per item, "Создать все"/"Ничего");
3. AI: `route_message` decides the intent, then
   - new_tasks -> `extract_tasks` -> the same confirmation card (uncertainties
     shown per item; reflection-only/ambiguous -> clarifying question or draft);
   - session_control -> the existing session buttons (a tap confirms);
   - retro_report -> the /retro flow prefilled with the understood interval;
   - postpone/obstacle -> buttons of that task (nothing changes without a tap);
   - chat/question -> `chat_reply` text (persona + context), never executed;
   - settings_request -> pointer to /settings; unclear -> clarifying question;
   - AIUnavailable anywhere -> stage A + a short note (acceptance 19).

Question 36 is open: new tasks are NEVER created silently (rule 3, acceptance 4).
"""

from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.ai.errors import AIUnavailable
from procbot.context import AppContext
from procbot.db.models import PendingApproval
from procbot.domain.enums import ApprovalStatus, SessionStatus
from procbot.services import actions, approvals, capture
from procbot.services import sessions as session_service
from procbot.services import tasks as task_service
from procbot.telegram.callbacks import SessionCb, TaskCb
from procbot.telegram.handlers import sessions as session_handlers
from procbot.telegram.handlers.approvals import render_approval_card, send_approval_card
from procbot.telegram.keyboards import Menu
from procbot.telegram.middlewares import mark_processed

log = logging.getLogger(__name__)

router = Router(name="chat")

YES_WORDS = frozenset({"да", "да!", "да.", "ок", "окей", "ok", "yes", "давай", "создавай"})
NOT_YET_TEXT = "Этот раздел появится на следующих этапах."
UNKNOWN_COMMAND = "Не знаю такой команды. Меню — внизу, задачи — /tasks, день — /today."
UNCLEAR_TEXT = (
    "Не совсем понял. Это новые дела, вопрос или отметка о работе? "
    "Напиши чуть подробнее — или /tasks, /work, /retro."
)
SETTINGS_TEXT = "Настройки — в «⚙️ Настройки» (/settings). Сам я их не меняю."
NO_SESSION_TEXT = "Сейчас нет активной рабочей сессии. Начать — /work."


@router.message(F.text.in_(set(Menu.ALL)))
async def on_unhandled_menu(message: Message) -> None:
    # Menu sections whose module is not built yet; never treat them as tasks.
    await message.answer(NOT_YET_TEXT)


@router.message(Command("cancel"))
async def on_cancel(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Ок, отменил. Ничего не изменено.")


@router.message(F.text.startswith("/"))
async def on_unknown_command(message: Message) -> None:
    await message.answer(UNKNOWN_COMMAND)


@router.message(StateFilter(None), F.text)
async def on_free_text(
    message: Message, ctx: AppContext, state: FSMContext, update_id: int
) -> None:
    text = (message.text or "").strip()
    if not text:
        return
    reply_to = message.reply_to_message
    if (
        reply_to is not None
        and text.lower() in YES_WORDS
        and await _approve_by_reply(message, ctx, reply_to.message_id, update_id)
    ):
        return
    await process_text(message, ctx, state, update_id, text)


# --------------------------------------------------------------------------- pipeline


async def process_text(
    message: Message, ctx: AppContext, state: FSMContext, update_id: int, text: str
) -> None:
    """Route one owner text (typed or transcribed) and answer with text/buttons."""
    if not capture.ai_configured(ctx):
        await _stage_a(message, ctx, text, update_id, note=None)
        return
    try:
        decision = await capture.route_message(ctx, text, update_id=update_id)
    except AIUnavailable as exc:
        log.info("route_message unavailable: %s", exc.reason)
        await _stage_a(message, ctx, text, update_id, note=capture.AI_DOWN_NOTE)
        return
    intent = decision.intent
    if intent == "new_tasks":
        await _new_tasks(message, ctx, text, update_id, decision)
    elif intent == "session_control":
        await _session_control(message, ctx, state, decision)
    elif intent == "retro_report":
        await _retro(message, ctx, state, text, decision)
    elif intent in ("postpone", "obstacle"):
        await _task_trouble(message, ctx, state, text, update_id, decision)
    elif intent == "settings_request":
        await message.answer(SETTINGS_TEXT)
    elif intent in ("chat", "question"):
        await _chat(message, ctx, text, update_id)
    else:
        await message.answer((decision.clarifying_question or "").strip() or UNCLEAR_TEXT)


def _marker(ctx: AppContext, update_id: int) -> capture.TxHook:
    async def mark(s: AsyncSession) -> None:
        await mark_processed(s, update_id, ctx.clock.now())

    return mark


async def _send_result(message: Message, ctx: AppContext, res: capture.CaptureResult) -> None:
    if res.kind != "proposed" or not res.approvals:
        await message.answer(res.text or "Не нашёл, что можно оформить задачей.")
        return
    await send_approval_card(message, ctx, res.approvals, tz=res.tz, header=res.header)


async def _stage_a(
    message: Message, ctx: AppContext, text: str, update_id: int, *, note: str | None
) -> None:
    res = await capture.fallback_capture(
        ctx, text, update_id=update_id, in_tx=_marker(ctx, update_id)
    )
    if note:
        if res.kind == "proposed":
            res.header = f"{note}\n\n{res.header or 'Создать эти задачи?'}"
        else:
            res.text = f"{note}\n{res.text or ''}".strip()
    await _send_result(message, ctx, res)


async def _new_tasks(
    message: Message,
    ctx: AppContext,
    text: str,
    update_id: int,
    decision: capture.RouteDecision,
) -> None:
    try:
        res = await capture.capture_tasks(
            ctx, text, update_id=update_id, in_tx=_marker(ctx, update_id), route=decision
        )
    except AIUnavailable as exc:
        log.info("extract_tasks unavailable: %s", exc.reason)
        await _stage_a(message, ctx, text, update_id, note=capture.AI_DOWN_NOTE)
        return
    await _send_result(message, ctx, res)


async def _chat(message: Message, ctx: AppContext, text: str, update_id: int) -> None:
    try:
        answer = await capture.chat_answer(ctx, text, update_id=update_id)
    except AIUnavailable as exc:
        log.info("chat_reply unavailable: %s", exc.reason)
        await _stage_a(message, ctx, text, update_id, note=capture.AI_DOWN_NOTE)
        return
    # Shown as text only; nothing in it is ever executed.
    ctx.registry.setdefault("update_replies", {}).setdefault(update_id, []).append(answer.text)
    await message.answer(answer.text)


def _btn(text: str, cb: SessionCb | TaskCb) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=cb.pack())


async def _session_control(
    message: Message, ctx: AppContext, state: FSMContext, decision: capture.RouteDecision
) -> None:
    """AI-routed session commands are offered as the existing buttons: one tap
    confirms, so a misread phrase never stops or starts work by itself."""
    action = decision.session_action
    if action == "start":
        title = await capture.task_title(ctx, decision.task_id)
        if decision.task_id and title:
            kb = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        _btn(
                            f"▶️ Начать: {title[:40]}",
                            SessionCb(action="pick", task_id=decision.task_id),
                        )
                    ],
                    [_btn("Другая задача", SessionCb(action="pick", task_id=0))],
                ]
            )
            await message.answer(f"Начинаем работу над «{title}»?", reply_markup=kb)
            return
        await session_handlers.work_menu(message, ctx, state)
        return
    async with ctx.db.transaction() as s:
        ws = await session_service.get_active_session(s)
        card = await session_service.describe_session(s, ctx, ws) if ws is not None else None
    if ws is None or card is None:
        await message.answer(NO_SESSION_TEXT)
        return
    buttons: dict[str, tuple[str, str]] = {
        "pause": ("⏸ Пауза", "pause"),
        "resume": ("▶️ Продолжить", "resume"),
        "stop": ("⏹ Завершить", "stop"),
    }
    if action == "pause" and ws.status != SessionStatus.ACTIVE:
        await message.answer("Сессия уже на паузе.\n\n" + card)
        return
    if action == "resume" and ws.status != SessionStatus.PAUSED:
        await message.answer("Сессия и так идёт.\n\n" + card)
        return
    label, cb_action = buttons.get(action or "", buttons["stop"])
    kb = InlineKeyboardMarkup(
        inline_keyboard=[[_btn(label, SessionCb(action=cb_action, session_id=ws.id))]]
    )
    await message.answer(f"{card}\n\nПодтверди кнопкой:", reply_markup=kb)


async def _retro(
    message: Message,
    ctx: AppContext,
    state: FSMContext,
    text: str,
    decision: capture.RouteDecision,
) -> None:
    async with ctx.db.transaction() as s:
        tz = await task_service.owner_tz(s)
    interval = capture.resolve_retro(decision, text, now=ctx.clock.now(), tz=tz)
    await session_handlers.start_retro_prefilled(
        message, ctx, state, interval=interval, task_id=decision.task_id
    )


async def _task_trouble(
    message: Message,
    ctx: AppContext,
    state: FSMContext,
    text: str,
    update_id: int,
    decision: capture.RouteDecision,
) -> None:
    """Postpone / obstacle about a known task -> that task's buttons (consent by tap)."""
    title = await capture.task_title(ctx, decision.task_id)
    if not decision.task_id or not title:
        await _chat(message, ctx, text, update_id)
        return
    tid = decision.task_id
    rows = [
        [_btn("📆 Перенести", TaskCb(action="postpone", task_id=tid))],
        [_btn("▶️ Начать с малого шага", SessionCb(action="pick", task_id=tid))],
        [_btn("Открыть карточку", TaskCb(action="view", task_id=tid))],
    ]
    head = (
        f"Понял, речь о «{title}». Перенести или попробовать начать с короткого шага?"
        if decision.intent == "postpone"
        else f"Понял, с «{title}» что-то мешает. Можно начать с очень маленького шага или перенести."
    )
    await message.answer(head, reply_markup=InlineKeyboardMarkup(inline_keyboard=rows))


# --------------------------------------------------------------------------- "да" reply


async def _approve_by_reply(
    message: Message, ctx: AppContext, card_message_id: int, update_id: int
) -> bool:
    """'да' as a reply to a card accepts everything pending on that card."""
    async with ctx.db.transaction() as s:
        items: list[PendingApproval] = await approvals.find_by_message(s, card_message_id)
        if not items:
            return False
        pending = [a for a in items if a.status == ApprovalStatus.PENDING]
        results = [await approvals.approve(s, ctx, a.id) for a in pending]
        hooks = [h for r in results for h in r.after_commit]
        group = await approvals.group_of(s, items[0])
        tz = await task_service.owner_tz(s)
        await mark_processed(s, update_id, ctx.clock.now())
    await actions.run_after_commit(hooks)
    text, kb = render_approval_card(group, tz=tz)
    summary = approvals.summarize(results) if results else "Эта карточка уже решена."
    await message.answer(f"{summary}\n\n{text}", reply_markup=kb)
    return True
