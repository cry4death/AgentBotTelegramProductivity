"""Approval cards: accept / reject per item, accept all / reject all, edit title.

`render_approval_card` is reused by every module that shows proposals. Each
press is idempotent (the service checks the card status), the reply is sent only
after the transaction commits, and after-commit side effects run outside it.
"""

from __future__ import annotations

from collections.abc import Sequence

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from procbot.context import AppContext
from procbot.db.models import PendingApproval
from procbot.domain.actions import ActionType
from procbot.domain.enums import ApprovalStatus
from procbot.services import actions, approvals
from procbot.services import tasks as task_service
from procbot.telegram.callbacks import ApprovalCb
from procbot.telegram.keyboards import Menu
from procbot.telegram.middlewares import mark_processed
from procbot.timeutil import fmt_local

router = Router(name="approvals")

STATUS_ICON = {
    ApprovalStatus.PENDING: "⏳",
    ApprovalStatus.APPLIED: "✅",
    ApprovalStatus.REJECTED: "✖️",
    ApprovalStatus.EXPIRED: "⌛",
    ApprovalStatus.FAILED: "⚠️",
}


class ApprovalEdit(StatesGroup):
    title = State()


# --------------------------------------------------------------------------- rendering


def _all_create(items: Sequence[PendingApproval]) -> bool:
    return all(a.action_type == ActionType.CREATE_TASK for a in items)


def render_approval_card(
    approval: PendingApproval | Sequence[PendingApproval],
    *,
    tz: str | None = None,
    header: str | None = None,
) -> tuple[str, InlineKeyboardMarkup | None]:
    """Text + buttons for one card or a batch (full preview before applying).

    Resolved items show their outcome and lose their buttons; when nothing is
    pending any more the keyboard is None."""
    items = [approval] if isinstance(approval, PendingApproval) else list(approval)
    creating = _all_create(items)
    many = len(items) > 1
    pending = [a for a in items if a.status == ApprovalStatus.PENDING]
    if header is None:
        if creating:
            header = "Похоже, тут есть задачи. Создать их?" if many else "Создать задачу?"
        else:
            header = "Нужно твоё подтверждение:"
    lines = [header, ""]
    for i, a in enumerate(items, 1):
        diff = a.diff or {}
        icon = STATUS_ICON.get(ApprovalStatus(a.status), "•")
        summary = str(diff.get("summary") or a.action_type)
        num = f"{i}. " if many else ""
        status_note = (
            "" if a.status == ApprovalStatus.PENDING else f" — {approvals.STATUS_TEXT[a.status]}"
        )
        if a.status == ApprovalStatus.FAILED and a.result and a.result.get("error"):
            status_note += f" ({a.result['error']})"
        lines.append(f"{icon} {num}{summary}{status_note}")
        for extra in diff.get("lines") or []:
            lines.append(f"    {extra}")
    if pending:
        lines.append("")
        lines.append("Без твоего подтверждения ничего не изменится.")
        exp = min((a.expires_at for a in pending if a.expires_at), default=None)
        if exp is not None and tz:
            lines.append(f"Карточка действует до {fmt_local(exp, tz)}.")
    return "\n".join(lines), _keyboard(items, pending, creating=creating, many=many)


def _keyboard(
    items: Sequence[PendingApproval],
    pending: Sequence[PendingApproval],
    *,
    creating: bool,
    many: bool,
) -> InlineKeyboardMarkup | None:
    if not pending:
        return None
    yes, no = ("Создать", "Не надо") if creating else ("Принять", "Отклонить")
    rows: list[list[InlineKeyboardButton]] = []
    for i, a in enumerate(items, 1):
        if a.status != ApprovalStatus.PENDING:
            continue
        suffix = f" {i}" if many else ""
        row = [
            InlineKeyboardButton(
                text=f"✅ {yes}{suffix}",
                callback_data=ApprovalCb(action="accept", approval_id=a.id).pack(),
            )
        ]
        if a.action_type == ActionType.CREATE_TASK:
            row.append(
                InlineKeyboardButton(
                    text=f"✏️{suffix}",
                    callback_data=ApprovalCb(action="edit", approval_id=a.id).pack(),
                )
            )
        row.append(
            InlineKeyboardButton(
                text=f"✖️ {no}{suffix}",
                callback_data=ApprovalCb(action="reject", approval_id=a.id).pack(),
            )
        )
        rows.append(row)
    if many and len(pending) > 1:
        all_yes, all_no = (
            ("Создать все", "Ничего") if creating else ("Принять все", "Отклонить все")
        )
        first = pending[0].id
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"✅ {all_yes}",
                    callback_data=ApprovalCb(action="accept_all", approval_id=first).pack(),
                ),
                InlineKeyboardButton(
                    text=f"✖️ {all_no}",
                    callback_data=ApprovalCb(action="reject_all", approval_id=first).pack(),
                ),
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def send_approval_card(
    message: Message,
    ctx: AppContext,
    items: Sequence[PendingApproval],
    *,
    tz: str | None,
    header: str | None = None,
) -> int | None:
    """Send a card as a reply (after commit) and remember its message id."""
    if not items:
        return None
    text, kb = render_approval_card(items, tz=tz, header=header)
    sent = await message.answer(text, reply_markup=kb)
    async with ctx.db.transaction() as s:
        await approvals.set_card_message(s, [a.id for a in items], sent.message_id)
    return sent.message_id


async def _edit(query: CallbackQuery, text: str, kb: InlineKeyboardMarkup | None) -> None:
    msg = query.message
    if isinstance(msg, Message):
        try:
            await msg.edit_text(text, reply_markup=kb)
            return
        except TelegramBadRequest as exc:
            if "not modified" in str(exc):
                return  # double press: the message already shows this
    if query.bot is not None:
        await query.bot.send_message(query.from_user.id, text, reply_markup=kb)


# --------------------------------------------------------------------------- callbacks


async def _resolve(
    query: CallbackQuery, callback_data: ApprovalCb, ctx: AppContext, update_id: int
) -> None:
    now = ctx.clock.now()
    action = callback_data.action
    async with ctx.db.transaction() as s:
        row = await approvals.get_approval(s, callback_data.approval_id)
        if row is None:
            await mark_processed(s, update_id, now)
            answer, text, kb, hooks = "Карточка не найдена.", None, None, []
        else:
            if action == "accept":
                res = [await approvals.approve(s, ctx, row.id)]
            elif action == "reject":
                res = [await approvals.reject(s, ctx, row.id)]
            elif action == "accept_all":
                res = await approvals.approve_group(s, ctx, row.id)
            else:
                res = await approvals.reject_group(s, ctx, row.id)
            hooks = [h for r in res for h in r.after_commit]
            answer = res[0].message if len(res) == 1 else approvals.summarize(res)
            group = await approvals.group_of(s, row)
            tz = await task_service.owner_tz(s)
            text, kb = render_approval_card(group, tz=tz)
            await mark_processed(s, update_id, now)
    # committed: now side effects, then the reply
    await actions.run_after_commit(hooks)
    if text is not None:
        await _edit(query, text, kb)
    await query.answer(answer[:200] if answer else None)


@router.callback_query(
    ApprovalCb.filter(F.action.in_({"accept", "reject", "accept_all", "reject_all"}))
)
async def on_resolve(
    query: CallbackQuery, callback_data: ApprovalCb, ctx: AppContext, update_id: int
) -> None:
    await _resolve(query, callback_data, ctx, update_id)


@router.callback_query(ApprovalCb.filter(F.action == "edit"))
async def on_edit(
    query: CallbackQuery, callback_data: ApprovalCb, ctx: AppContext, state: FSMContext
) -> None:
    async with ctx.db.transaction() as s:
        row = await approvals.get_approval(s, callback_data.approval_id)
        ok = row is not None and row.status == ApprovalStatus.PENDING
    if not ok:
        await query.answer("Карточка уже не ожидает подтверждения.")
        return
    await state.set_state(ApprovalEdit.title)
    await state.update_data(approval_id=callback_data.approval_id)
    if isinstance(query.message, Message):
        await query.message.answer("Напиши, как назвать задачу (одной строкой):")
    await query.answer()


@router.message(ApprovalEdit.title, F.text, ~F.text.startswith("/"), ~F.text.in_(set(Menu.ALL)))
async def on_edit_title(
    message: Message, ctx: AppContext, state: FSMContext, update_id: int
) -> None:
    data = await state.get_data()
    approval_id = int(data.get("approval_id") or 0)
    title = " ".join((message.text or "").split())[:500]
    error: str | None = None
    group: list[PendingApproval] = []
    tz: str | None = None
    async with ctx.db.transaction() as s:
        row = await approvals.get_approval(s, approval_id)
        if row is None or not title:
            error = "Не получилось — карточка не найдена."
        else:
            try:
                act = actions.parse_action(row.action_payload)
                act = act.model_copy(update={"task": act.task.model_copy(update={"title": title})})
                diff = await actions.describe_action(s, ctx, act)
                await approvals.update_pending_payload(s, ctx, row.id, act, diff)
                group = await approvals.group_of(s, row)
                tz = await task_service.owner_tz(s)
            except actions.ActionError as exc:
                error = str(exc)
        await mark_processed(s, update_id, ctx.clock.now())
    await state.clear()
    if error:
        await message.answer(error)
        return
    await send_approval_card(message, ctx, group, tz=tz)
