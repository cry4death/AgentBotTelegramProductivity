"""Free-text fallback (must stay the LAST router).

Stage A has no AI, so free text is never executed and never silently turned
into tasks (rule 3, scenario A, acceptance 4). Each line / ';'-part becomes a
CREATE_TASK proposal awaiting approval on ONE card with per-item "Создать" /
"Не надо" and "Создать все" / "Ничего" (question 36 is open: always confirm,
both options offered). Stage B replaces candidate extraction with AI and keeps
this flow (services.tasks.propose_task_candidates).

A reply "да" to a card is an unambiguous consent to everything still pending on
that exact card.
"""

from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from procbot.context import AppContext
from procbot.db.models import PendingApproval
from procbot.domain.actions import TaskDraft
from procbot.domain.enums import Actor, ApprovalStatus
from procbot.services import actions, approvals
from procbot.services import tasks as task_service
from procbot.telegram.handlers.approvals import render_approval_card, send_approval_card
from procbot.telegram.keyboards import Menu
from procbot.telegram.middlewares import mark_processed

router = Router(name="chat")

YES_WORDS = frozenset({"да", "да!", "да.", "ок", "окей", "ok", "yes", "давай", "создавай"})
NOT_YET_TEXT = "Этот раздел появится на следующих этапах."
UNKNOWN_COMMAND = "Не знаю такой команды. Меню — внизу, задачи — /tasks, день — /today."


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
async def on_free_text(message: Message, ctx: AppContext, update_id: int) -> None:
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
    titles = task_service.split_candidates(text)
    if not titles:
        return
    drafts = [TaskDraft(title=t) for t in titles]
    async with ctx.db.transaction() as s:
        cards = await task_service.propose_task_candidates(
            s, ctx, drafts, op_prefix=f"u{update_id}", origin=Actor.SYSTEM
        )
        tz = await task_service.owner_tz(s)
        await mark_processed(s, update_id, ctx.clock.now())
    if not cards:
        await message.answer("Не нашёл, что можно оформить задачей.")
        return
    header = None
    if len(titles) > task_service.MAX_CANDIDATES:
        header = (
            f"Похоже, тут есть задачи (показаны первые {task_service.MAX_CANDIDATES}). Создать их?"
        )
    await send_approval_card(message, ctx, cards, tz=tz, header=header)


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
