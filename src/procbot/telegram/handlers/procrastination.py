"""Telegram handlers: procrastination analysis and interventions (InterventionCb,
prefix "iv"; FR-04, FR-05, scenario C, acceptance 8/9/10).

- reason: a reason button of the start-delay card -> one concrete next action
  (AI-tailored, template fallback) + the "Разобраться подробнее" button; the deep
  dialog starts by itself only when the owner configured a threshold;
- deep / text answers / opt / end: the capped deep dialog; every decision is the
  owner's (deferring / dropping go through an approval card);
- refute / confirm: the owner rejects or accepts a hypothesis;
- split / step: adaptive decomposition (AUTO subtasks), or — without AI — the
  owner types the first physical step;
- outcome: "Не помогло" (no_effect) / "Не сейчас" (declined).

Replies to button presses are answers, not bot-initiated messages: no gate.
"""

from __future__ import annotations

import logging
from typing import Any

from aiogram import F, Router
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from procbot.context import AppContext
from procbot.db.models import PendingApproval
from procbot.domain.enums import InterventionKind, ObstacleReason
from procbot.services import decompose as dec
from procbot.services import interventions as iv
from procbot.services.actions import ActionError
from procbot.services.tasks import owner_tz
from procbot.telegram.callbacks import DayCb, InterventionCb, ReminderCb, SessionCb, TaskCb
from procbot.telegram.keyboards import Menu

log = logging.getLogger(__name__)

router = Router(name="procrastination")

TEXT_INPUT = (F.text, ~F.text.startswith("/"), ~F.text.in_(set(Menu.ALL)))


class DeepDialogForm(StatesGroup):
    answer = State()  # data: {"dialog": DialogTurn.state}


class FirstStepForm(StatesGroup):
    step = State()  # data: {"task_id": int}


# --------------------------------------------------------------------------- buttons


def _btn(text: str, data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=data)


def _short_start(task_id: int) -> InlineKeyboardButton:
    return _btn(
        "⏱ Короткий старт", SessionCb(action="pick", task_id=task_id, arg="short_start").pack()
    )


def _start(task_id: int, text: str = "▶️ Начать") -> InlineKeyboardButton:
    return _btn(text, SessionCb(action="pick", task_id=task_id).pack())


def _postpone(task_id: int) -> InlineKeyboardButton:
    return _btn("📆 Перенести", TaskCb(action="postpone", task_id=task_id).pack())


def _split(task_id: int, intervention_id: int = 0) -> InlineKeyboardButton:
    return _btn(
        "✂️ Разбить на шаги",
        InterventionCb(action="split", task_id=task_id, intervention_id=intervention_id).pack(),
    )


def _deep(task_id: int, intervention_id: int = 0, reminder_id: int = 0) -> InlineKeyboardButton:
    return _btn(
        "🔍 Разобраться подробнее",
        InterventionCb(
            action="deep", task_id=task_id, intervention_id=intervention_id, reminder_id=reminder_id
        ).pack(),
    )


def quick_keyboard(sug: iv.QuickSuggestion, reminder_id: int) -> InlineKeyboardMarkup:
    t, i, K = sug.task_id, sug.intervention_id, InterventionKind
    first_step = _btn("✍️ Записать первый шаг", InterventionCb(action="step", task_id=t).pack())
    rows: list[list[InlineKeyboardButton]]
    if sug.kind == K.TINY_STEP:
        rows = [[first_step, _short_start(t)], [_split(t, i), _postpone(t)]]
    elif sug.kind in (K.SPLIT, K.SIMPLIFY, K.CHANGE_APPROACH):
        rows = [[_split(t, i)], [_short_start(t), _postpone(t)]]
    elif sug.kind == K.SHORT_START:
        rows = [[_short_start(t)], [_split(t, i), _postpone(t)]]
    elif sug.kind == K.REMOVE_DISTRACTIONS:
        rows = [[_start(t, "▶️ Начать сессию")], [_short_start(t), _postpone(t)]]
    elif sug.kind == K.REST:
        rest = _btn(
            "☕ Отдохнуть", ReminderCb(action="rest", reminder_id=reminder_id, task_id=t).pack()
        )
        rows = [[rest], [_short_start(t), _postpone(t)]]
    else:  # reschedule
        rows = [[_postpone(t)], [_btn("📅 Мой день", DayCb(action="show").pack())]]
    rows.append([_deep(t, i, reminder_id)])
    rows.append(
        [
            _btn(
                "🙅 Не помогло",
                InterventionCb(action="outcome", intervention_id=i, arg="no_effect").pack(),
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def dialog_keyboard(turn: iv.DialogTurn) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for n, _ in enumerate(turn.hypotheses):
        ref = f"{turn.turn}.{n}"
        rows.append(
            [
                _btn(
                    f"❌ {n + 1}: это не так",
                    InterventionCb(
                        action="refute", intervention_id=turn.intervention_id, arg=ref
                    ).pack(),
                ),
                _btn(
                    f"✔️ {n + 1}: похоже",
                    InterventionCb(
                        action="confirm", intervention_id=turn.intervention_id, arg=ref
                    ).pack(),
                ),
            ]
        )
    pair: list[InlineKeyboardButton] = []
    for opt in turn.options:
        pair.append(
            _btn(
                opt.label,
                InterventionCb(
                    action="opt", intervention_id=turn.intervention_id, arg=opt.kind
                ).pack(),
            )
        )
        if len(pair) == 2:
            rows.append(pair)
            pair = []
    if pair:
        rows.append(pair)
    rows.append(
        [
            _btn(
                "Закончить разговор",
                InterventionCb(action="end", intervention_id=turn.intervention_id).pack(),
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def dialog_text(turn: iv.DialogTurn) -> str:
    parts = [turn.message]
    if turn.hypotheses:
        parts.append(
            "Возможные причины — это только предположения, поправь, если не так:\n"
            + "\n".join(f"{n}. {h}" for n, h in enumerate(turn.hypotheses, 1))
        )
    if turn.question:
        parts.append(f"❓ {turn.question}\n(ответь текстом или закончи разговор кнопкой)")
    elif turn.options:
        details = [f"• {o.label}: {o.details}" for o in turn.options if o.details]
        if details:
            parts.append("\n".join(details))
        parts.append("Что выберешь? Решение за тобой.")
    if not turn.used_ai:
        parts.append("(без ИИ — по истории задачи)")
    return "\n\n".join(parts)


async def _reply(
    query: CallbackQuery, text: str, markup: InlineKeyboardMarkup | None = None
) -> Message | None:
    if isinstance(query.message, Message):
        return await query.message.answer(text, reply_markup=markup)
    if query.bot is not None:
        return await query.bot.send_message(query.from_user.id, text, reply_markup=markup)
    return None


async def _answer(query: CallbackQuery, text: str | None = None) -> None:
    try:
        await query.answer(text)
    except Exception:  # pragma: no cover - an expired query must not break the reply
        log.debug("callback answer failed", exc_info=True)


# --------------------------------------------------------------------------- reason


@router.callback_query(InterventionCb.filter(F.action == "reason"))
async def on_reason(
    query: CallbackQuery, callback_data: InterventionCb, ctx: AppContext, state: FSMContext
) -> None:
    try:
        reason = ObstacleReason(callback_data.arg)
    except ValueError:
        await _answer(query)
        return
    await _answer(query)
    sug = await iv.suggest_after_reason(
        ctx,
        task_id=callback_data.task_id,
        reason=reason,
        reminder_id=callback_data.reminder_id or None,
    )
    if sug is None:
        await _reply(query, "Эта задача больше не найдена.")
        return
    text = sug.text
    hint = sug.signal.hint() if sug.signal.offer else None
    if hint:
        text += "\n\n" + hint
    await _reply(query, text, quick_keyboard(sug, callback_data.reminder_id))
    if sug.signal.auto and not sug.repeated:
        # owner-configured threshold (open question 4): the dialog opens by itself
        await _open_dialog(
            query,
            ctx,
            state,
            task_id=sug.task_id,
            reason=reason,
            reminder_id=callback_data.reminder_id or None,
            key=f"auto:{sug.intervention_id}",
        )


# --------------------------------------------------------------------------- deep dialog


async def _open_dialog(
    query: CallbackQuery,
    ctx: AppContext,
    state: FSMContext,
    *,
    task_id: int,
    reason: ObstacleReason | None,
    reminder_id: int | None,
    key: str,
) -> None:
    data = await state.get_data()
    current = data.get("dialog") or {}
    if await state.get_state() == DeepDialogForm.answer.state and current.get("key") == key:
        await _reply(query, "Разбор уже идёт — ответь на вопрос выше или закончи разговор.")
        return
    try:
        turn = await iv.start_deep_dialog(
            ctx, task_id=task_id, reason=reason, reminder_id=reminder_id, op_key=key
        )
    except ActionError as exc:
        await _reply(query, str(exc))
        return
    await _after_turn(state, turn, key)
    await _reply(query, dialog_text(turn), dialog_keyboard(turn))


async def _after_turn(state: FSMContext, turn: iv.DialogTurn, key: str) -> None:
    dialog = {**turn.state, "key": key}
    if turn.final:
        await state.set_state(None)
    else:
        await state.set_state(DeepDialogForm.answer)
    await state.update_data(dialog=dialog)


@router.callback_query(InterventionCb.filter(F.action == "deep"))
async def on_deep(
    query: CallbackQuery,
    callback_data: InterventionCb,
    ctx: AppContext,
    state: FSMContext,
    update_id: int,
) -> None:
    await _answer(query)
    reason: ObstacleReason | None = None
    if callback_data.intervention_id:
        async with ctx.db.transaction() as s:
            from procbot.db.models import Intervention

            row = await s.get(Intervention, callback_data.intervention_id)
            if row is not None and row.reason_category in set(ObstacleReason):
                reason = ObstacleReason(row.reason_category)
    if callback_data.intervention_id or callback_data.reminder_id:
        key = f"card:{callback_data.reminder_id}:{callback_data.intervention_id}"
    else:
        key = f"u{update_id}"
    await _open_dialog(
        query,
        ctx,
        state,
        task_id=callback_data.task_id,
        reason=reason,
        reminder_id=callback_data.reminder_id or None,
        key=f"{key}:{callback_data.task_id}",
    )


@router.message(DeepDialogForm.answer, *TEXT_INPUT)
async def on_dialog_answer(message: Message, ctx: AppContext, state: FSMContext) -> None:
    data = await state.get_data()
    dialog: dict[str, Any] = dict(data.get("dialog") or {})
    if not dialog.get("intervention_id"):
        await state.set_state(None)
        await message.answer("Разбор уже закончен. Можно начать заново из напоминания о задаче.")
        return
    try:
        turn = await iv.continue_deep_dialog(ctx, dialog, message.text or "")
    except ActionError as exc:
        await state.set_state(None)
        await message.answer(str(exc))
        return
    await _after_turn(state, turn, str(dialog.get("key", "")))
    await message.answer(dialog_text(turn), reply_markup=dialog_keyboard(turn))


def _parse_ref(arg: str) -> tuple[int, int] | None:
    try:
        turn, idx = arg.split(".", 1)
        return int(turn), int(idx)
    except ValueError:
        return None


@router.callback_query(InterventionCb.filter(F.action.in_({"refute", "confirm"})))
async def on_hypothesis(
    query: CallbackQuery, callback_data: InterventionCb, ctx: AppContext, state: FSMContext
) -> None:
    ref = _parse_ref(callback_data.arg)
    if ref is None:
        await _answer(query)
        return
    refute = callback_data.action == "refute"
    async with ctx.db.transaction() as s:
        text = await iv.hypothesis_text(
            s, intervention_id=callback_data.intervention_id, turn=ref[0], index=ref[1]
        )
        if text is None:
            await _answer(query, "Не нашёл это предположение.")
            return
        if refute:
            await iv.refute_hypothesis(
                s, ctx, intervention_id=callback_data.intervention_id, text=text
            )
            saved = False
        else:
            res = await iv.confirm_hypothesis(
                s, ctx, intervention_id=callback_data.intervention_id, text=text
            )
            saved = res.ok
    if refute:
        data = await state.get_data()
        dialog = dict(data.get("dialog") or {})
        if dialog.get("intervention_id") == callback_data.intervention_id:
            dialog["refuted"] = list(dict.fromkeys([*dialog.get("refuted", []), text]))
            await state.update_data(dialog=dialog)
        await _answer(query, "Понял, снимаю это предположение.")
        await _reply(
            query,
            f"Понял — «{text}» не так. Это было лишь предположение, больше его не повторю. "
            "Спасибо, что поправил.",
        )
        return
    await _answer(query, "Отметил.")
    await _reply(
        query,
        "Отметил как гипотезу, а не факт"
        + (" — её можно посмотреть и опровергнуть в «🧠 Память»." if saved else "."),
    )


@router.callback_query(InterventionCb.filter(F.action == "opt"))
async def on_option(
    query: CallbackQuery,
    callback_data: InterventionCb,
    ctx: AppContext,
    state: FSMContext,
    update_id: int,
) -> None:
    await _answer(query)
    try:
        async with ctx.db.transaction() as s:
            res = await iv.choose_option(
                s,
                ctx,
                intervention_id=callback_data.intervention_id,
                kind=callback_data.arg,
                op_key=f"{callback_data.intervention_id}:{callback_data.arg}",
            )
            approval = await s.get(PendingApproval, res.approval_id) if res.approval_id else None
            tz = await owner_tz(s)
    except ActionError as exc:
        await _reply(query, str(exc))
        return
    data = await state.get_data()
    if (data.get("dialog") or {}).get("intervention_id") == callback_data.intervention_id:
        await state.set_state(None)
    t = res.task_id
    if res.next == "decompose":
        await _reply(query, res.text)
        await _decompose_reply(
            query,
            ctx,
            state,
            task_id=t,
            mode="finer" if res.kind == "simplify" else None,
            key=f"opt:{callback_data.intervention_id}:{res.kind}",
        )
        return
    if res.next == "postpone":
        markup = InlineKeyboardMarkup(inline_keyboard=[[_postpone(t)]])
        await _reply(query, res.text, markup)
        return
    if res.next == "approval" and approval is not None and isinstance(query.message, Message):
        from procbot.telegram.handlers.approvals import send_approval_card

        await query.message.answer(res.text)
        await send_approval_card(query.message, ctx, [approval], tz=tz)
        return
    if res.next == "short_start":
        await _reply(query, res.text, InlineKeyboardMarkup(inline_keyboard=[[_short_start(t)]]))
        return
    await _reply(query, res.text, InlineKeyboardMarkup(inline_keyboard=[[_start(t), _postpone(t)]]))


@router.callback_query(InterventionCb.filter(F.action == "end"))
async def on_end(
    query: CallbackQuery, callback_data: InterventionCb, ctx: AppContext, state: FSMContext
) -> None:
    await _answer(query)
    async with ctx.db.transaction() as s:
        await iv.record_declined(
            s, ctx, intervention_id=callback_data.intervention_id, note="разговор закончен"
        )
    await state.set_state(None)
    await _reply(
        query,
        "Хорошо, закончим. Отдых и сознательный перенос — нормальные варианты. "
        "Вернуться к задаче можно в любой момент.",
    )


@router.callback_query(InterventionCb.filter(F.action == "outcome"))
async def on_outcome(query: CallbackQuery, callback_data: InterventionCb, ctx: AppContext) -> None:
    async with ctx.db.transaction() as s:
        if callback_data.arg == "no_effect":
            await iv.record_no_effect(
                s, ctx, intervention_id=callback_data.intervention_id, note="кнопка «Не помогло»"
            )
        else:
            await iv.record_declined(s, ctx, intervention_id=callback_data.intervention_id)
    await _answer(query, "Понял, учту.")
    await _reply(
        query,
        "Понял, учту — в следующий раз предложу другой способ. Если хочешь, можно "
        "разобраться подробнее или просто перенести задачу.",
    )


# --------------------------------------------------------------------------- decomposition


async def _decompose_reply(
    query: CallbackQuery,
    ctx: AppContext,
    state: FSMContext,
    *,
    task_id: int,
    mode: dec.Mode | None,
    key: str,
) -> None:
    try:
        result = await dec.decompose(ctx, task_id, mode=mode, op_key=key)
    except ActionError as exc:
        await _reply(query, str(exc))
        return
    if result.error and not result.approval_id:
        await _reply(query, result.error)
        return
    if result.needs_first_step:
        await state.set_state(FirstStepForm.step)
        await state.update_data(task_id=result.task_id)
        await _reply(query, result.note)
        return
    if result.approval_id and isinstance(query.message, Message):
        from procbot.telegram.handlers.approvals import send_approval_card

        async with ctx.db.transaction() as s:
            row = await s.get(PendingApproval, result.approval_id)
            tz = await owner_tz(s)
        if row is not None:
            await send_approval_card(
                query.message,
                ctx,
                [row],
                tz=tz,
                header="Автоматические шаги выключены в настройках — подтверди, пожалуйста:",
            )
        return
    lines = [f"Разбил «{result.title}» на шаги."]
    if result.next_actions:
        lines.append("Ближайшие действия:")
        lines += [f"{n}. {a}" for n, a in enumerate(result.next_actions, 1)]
    if len(result.created) > len(result.next_actions):
        lines.append(f"(всего шагов в плане: {len(result.created)} — все видны в «🪜 Подзадачи»)")
    if result.note:
        lines.append(result.note)
    lines.append("Цель и срок задачи не менялись.")
    markup = InlineKeyboardMarkup(
        inline_keyboard=[
            [_short_start(result.task_id)],
            [_btn("🪜 Все шаги", TaskCb(action="subtasks", task_id=result.task_id).pack())],
        ]
    )
    await _reply(query, "\n".join(lines), markup)


@router.callback_query(InterventionCb.filter(F.action == "split"))
async def on_split(
    query: CallbackQuery, callback_data: InterventionCb, ctx: AppContext, state: FSMContext
) -> None:
    await _answer(query, "Разбиваю на шаги…")
    mode: dec.Mode | None = "finer" if callback_data.intervention_id else None
    await _decompose_reply(
        query,
        ctx,
        state,
        task_id=callback_data.task_id,
        mode=mode,
        key=f"split:{callback_data.intervention_id}:{callback_data.task_id}",
    )


@router.callback_query(InterventionCb.filter(F.action == "step"))
async def on_step(query: CallbackQuery, callback_data: InterventionCb, state: FSMContext) -> None:
    await _answer(query)
    await state.set_state(FirstStepForm.step)
    await state.update_data(task_id=callback_data.task_id)
    await _reply(
        query,
        "Напиши самый первый физический шаг — совсем маленький, на пару минут. "
        "Я добавлю его первым шагом задачи.",
    )


@router.message(FirstStepForm.step, *TEXT_INPUT)
async def on_first_step(
    message: Message, ctx: AppContext, state: FSMContext, update_id: int
) -> None:
    data = await state.get_data()
    task_id = int(data.get("task_id") or 0)
    await state.set_state(None)
    try:
        async with ctx.db.transaction() as s:
            steps = await dec.add_first_step(
                s, ctx, task_id, message.text or "", op_key=f"u{update_id}"
            )
    except ActionError as exc:
        await message.answer(str(exc))
        return
    text = (
        f"Добавил первым шагом: «{steps[0] if steps else message.text}». Начнём с короткого старта?"
    )
    await message.answer(
        text, reply_markup=InlineKeyboardMarkup(inline_keyboard=[[_short_start(task_id)]])
    )


@router.message(StateFilter(DeepDialogForm, FirstStepForm), ~F.text)
async def on_non_text(message: Message) -> None:
    await message.answer("Напиши ответ текстом, пожалуйста.")
