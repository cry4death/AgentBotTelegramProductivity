"""Goals UI: «🎯 Проекты/цели» (/goals). Goal list with honest progress and the
14-day alignment line, goal card (outcome, criteria, horizon, roadmap, projects,
open linked tasks), create goal (FSM: title -> outcome -> horizon -> milestones),
edit, mark milestones done, link tasks, projects, and «Критический разбор»
(AI review -> approval cards; without AI a code-only summary).

Thin: parse input -> services.goals / ActionExecutor -> reply after commit.
Everything the owner does here is an explicit command (origin USER); the review
only produces CHANGE_GOAL / CHANGE_MILESTONE approval cards.

GoalCb usage: `goal_id` carries the main entity id (a goal, or a project for the
p* actions), `arg` a second id or a mode.
"""

from __future__ import annotations

from datetime import date, timedelta

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.models import Goal, Milestone, Project
from procbot.domain.actions import ChangeGoal, ChangeMilestone, CreateGoal
from procbot.domain.enums import Actor, EntityStatus
from procbot.services import actions, approvals
from procbot.services import goals as gs
from procbot.services import structure as st
from procbot.services import tasks as ts
from procbot.services.when import parse_when
from procbot.telegram.callbacks import GoalCb, TaskCb
from procbot.telegram.handlers.approvals import send_approval_card
from procbot.telegram.keyboards import Menu
from procbot.telegram.middlewares import mark_processed
from procbot.timeutil import local_date

router = Router(name="goals")

MAX_LIST = 15
MAX_TASKS = 8
TEXT_INPUT = (F.text, ~F.text.startswith("/"), ~F.text.in_(set(Menu.ALL)))

EDIT_FIELDS: dict[str, tuple[str, str]] = {
    "title": ("Название", "Новое название цели:"),
    "outcome": ("Результат", "Какой измеримый результат? Например: «уровень B1 по испанскому»."),
    "criteria": ("Критерии", "По каким критериям поймём, что цель достигнута?"),
    "horizon": ("Горизонт", "К какому сроку? Например: «31.12.2026»."),
    "reported": (
        "Прогресс со слов",
        "Как сам оцениваешь продвижение? Запишу как «со слов» — отдельно от измеренного.",
    ),
}


class GoalForm(StatesGroup):
    title = State()
    outcome = State()
    horizon = State()
    milestones = State()
    edit = State()  # data: goal_id, field
    ms_add = State()  # data: goal_id
    project = State()  # data: goal_id (0 = without goal)


# --------------------------------------------------------------------------- helpers


def _btn(text: str, cb: GoalCb | TaskCb) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=cb.pack())


def _short(text: str, n: int = 32) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


async def _edit(query: CallbackQuery, text: str, kb: InlineKeyboardMarkup | None) -> None:
    msg = query.message
    if isinstance(msg, Message):
        try:
            await msg.edit_text(text, reply_markup=kb)
            return
        except TelegramBadRequest as exc:
            if "not modified" in str(exc):
                return
    if query.bot is not None:
        await query.bot.send_message(query.from_user.id, text, reply_markup=kb)


async def _say(query: CallbackQuery, text: str, kb: InlineKeyboardMarkup | None = None) -> None:
    if isinstance(query.message, Message):
        await query.message.answer(text, reply_markup=kb)
    elif query.bot is not None:
        await query.bot.send_message(query.from_user.id, text, reply_markup=kb)


def _skip(step: str, label: str = "Пропустить") -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[_btn(label, GoalCb(action="skip", arg=step))]])


# --------------------------------------------------------------------------- list


async def render_list(
    session: AsyncSession, ctx: AppContext, *, show_all: bool = False, note: str | None = None
) -> tuple[str, InlineKeyboardMarkup]:
    statuses = None if show_all else ("active", "paused")
    goals = await gs.list_goals(session, statuses=statuses)
    lines: list[str] = []
    if note:
        lines += [note, ""]
    lines.append("🎯 Все цели" if show_all else "🎯 Цели")
    rows: list[list[InlineKeyboardButton]] = []
    if not goals:
        lines += ["", "Целей пока нет. Добавь первую — с измеримым результатом и сроком."]
    for i, g in enumerate(goals[:MAX_LIST], 1):
        p = await gs.goal_progress(session, ctx, g.id)
        status = (
            "" if g.status == "active" else f" ({gs.GOAL_STATUS_LABELS.get(g.status, g.status)})"
        )
        lines.append(f"{i}. {g.title}{status} — {gs.progress_short(p)}")
        rows.append([_btn(f"{i}. {_short(g.title)}", GoalCb(action="view", goal_id=g.id))])
    if goals and not show_all:
        rep = await gs.goal_alignment(session, ctx, ctx.clock.now() - timedelta(days=14))
        lines += ["", "За 14 дней:", *gs.alignment_lines(rep)]
    rows.append(
        [
            _btn("➕ Новая цель", GoalCb(action="new")),
            _btn("📁 Проекты", GoalCb(action="projects")),
        ]
    )
    rows.append([_btn("🧐 Критический разбор", GoalCb(action="review"))])
    rows.append(
        [
            _btn("← Активные", GoalCb(action="list"))
            if show_all
            else _btn("🗂 Все цели", GoalCb(action="list", arg="all"))
        ]
    )
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


@router.message(Command("goals"))
@router.message(F.text == Menu.GOALS)
async def on_goals(message: Message, ctx: AppContext, state: FSMContext) -> None:
    await state.clear()
    async with ctx.db.transaction() as s:
        text, kb = await render_list(s, ctx)
    await message.answer(text, reply_markup=kb)


@router.callback_query(GoalCb.filter(F.action == "list"))
async def on_list(query: CallbackQuery, callback_data: GoalCb, ctx: AppContext) -> None:
    async with ctx.db.transaction() as s:
        text, kb = await render_list(s, ctx, show_all=callback_data.arg == "all")
    await _edit(query, text, kb)
    await query.answer()


# --------------------------------------------------------------------------- card


async def render_card(
    session: AsyncSession, ctx: AppContext, goal: Goal, *, note: str | None = None
) -> tuple[str, InlineKeyboardMarkup]:
    p = await gs.goal_progress(session, ctx, goal.id)
    lines: list[str] = []
    if note:
        lines += [note, ""]
    status = (
        ""
        if goal.status == "active"
        else f" ({gs.GOAL_STATUS_LABELS.get(goal.status, goal.status)})"
    )
    lines.append(f"🎯 {goal.title}{status}")
    lines.append(f"Результат: {goal.measurable_outcome or 'не задан'}")
    lines.append(f"Критерии: {goal.success_criteria or 'не заданы'}")
    if goal.horizon_date is not None:
        left = p.days_left or 0
        tail = f"осталось {left} дн." if left >= 0 else "горизонт прошёл"
        lines.append(f"Горизонт: {goal.horizon_date:%d.%m.%Y} ({tail})")
    else:
        lines.append("Горизонт: не задан")
    lines += gs.progress_lines(p)
    milestones = await gs.goal_milestones(session, goal.id)
    if milestones:
        lines += ["", "Дорожная карта:"]
        for m in milestones:
            key = "🔑 " if m.is_key else ""
            due = f" (до {m.due_date:%d.%m})" if m.due_date else ""
            lines.append(f"{gs.MILESTONE_ICON.get(m.status, '•')} {key}{m.title}{due}")
    projects = await gs.goal_projects(session, goal.id)
    if projects:
        lines.append("")
        lines.append("Проекты: " + ", ".join(pr.title for pr in projects))
    open_tasks = await gs.open_linked_tasks(session, goal.id)
    if open_tasks:
        lines += ["", "Открытые задачи:"]
        lines += [f"• {t.title}" for t in open_tasks[:MAX_TASKS]]
        if len(open_tasks) > MAX_TASKS:
            lines.append(f"…и ещё {len(open_tasks) - MAX_TASKS}")
    return "\n".join(lines), card_keyboard(goal, has_milestones=bool(milestones))


def card_keyboard(goal: Goal, *, has_milestones: bool) -> InlineKeyboardMarkup:
    gid = goal.id
    rows: list[list[InlineKeyboardButton]] = []
    first = [_btn("➕ Этап", GoalCb(action="msadd", goal_id=gid))]
    if has_milestones:
        first.insert(0, _btn("✅ Отметить этап", GoalCb(action="ms", goal_id=gid)))
    rows.append(first)
    rows.append(
        [
            _btn("🔗 Привязать задачу", GoalCb(action="link", goal_id=gid)),
            _btn("📁 Новый проект", GoalCb(action="pnew", goal_id=gid)),
        ]
    )
    rows.append(
        [
            _btn("✏️ Изменить", GoalCb(action="edit", goal_id=gid)),
            _btn("🧐 Разбор", GoalCb(action="review", goal_id=gid)),
        ]
    )
    status_row: list[InlineKeyboardButton] = []
    if goal.status in ("active", "paused"):
        status_row.append(_btn("🏁 Достигнута", GoalCb(action="status", goal_id=gid, arg="done")))
    if goal.status == "active":
        status_row.append(_btn("⏸ Пауза", GoalCb(action="status", goal_id=gid, arg="paused")))
    else:
        status_row.append(
            _btn("▶️ Активировать", GoalCb(action="status", goal_id=gid, arg="active"))
        )
    if goal.status != "archived":
        status_row.append(_btn("📦 Архив", GoalCb(action="status", goal_id=gid, arg="archived")))
    rows.append(status_row)
    rows.append([_btn("← К целям", GoalCb(action="list"))])
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def _card_or_missing(
    session: AsyncSession, ctx: AppContext, goal_id: int, note: str | None = None
) -> tuple[str, InlineKeyboardMarkup | None]:
    goal = await gs.get_goal(session, goal_id)
    if goal is None:
        return "Цель не найдена.", None
    return await render_card(session, ctx, goal, note=note)


@router.callback_query(GoalCb.filter(F.action == "view"))
async def on_view(query: CallbackQuery, callback_data: GoalCb, ctx: AppContext) -> None:
    async with ctx.db.transaction() as s:
        text, kb = await _card_or_missing(s, ctx, callback_data.goal_id)
    await _edit(query, text, kb)
    await query.answer()


# --------------------------------------------------------------------------- create


@router.callback_query(GoalCb.filter(F.action == "new"))
async def on_new(query: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await state.set_state(GoalForm.title)
    await _say(query, "Как сформулируешь цель? Например: «Выучить испанский до B1».")
    await query.answer()


@router.message(Command("goal_new"))
async def on_new_cmd(message: Message, state: FSMContext) -> None:
    await state.clear()
    await state.set_state(GoalForm.title)
    await message.answer("Как сформулируешь цель? Например: «Выучить испанский до B1».")


@router.message(GoalForm.title, *TEXT_INPUT)
async def on_title(message: Message, state: FSMContext) -> None:
    title = " ".join((message.text or "").split())
    if not title:
        await message.answer("Напиши название цели.")
        return
    await state.update_data(title=title[:300])
    await state.set_state(GoalForm.outcome)
    await message.answer(
        "Какой измеримый результат будет означать успех? Например: «сдать DELE B1».",
        reply_markup=_skip("outcome"),
    )


@router.message(GoalForm.outcome, *TEXT_INPUT)
async def on_outcome(message: Message, state: FSMContext) -> None:
    await state.update_data(outcome=(message.text or "").strip()[:2000])
    await _ask_horizon(message, state)


async def _ask_horizon(message: Message, state: FSMContext) -> None:
    await state.set_state(GoalForm.horizon)
    await message.answer(
        "К какому сроку? Например: «31.12.2026».", reply_markup=_skip("horizon", "Без срока")
    )


@router.message(GoalForm.horizon, *TEXT_INPUT)
async def on_horizon(message: Message, ctx: AppContext, state: FSMContext) -> None:
    async with ctx.db.transaction() as s:
        today = local_date(ctx.clock.now(), await gs.owner_tz(s))
    when = parse_when(message.text or "", today=today)
    if when is None or when.day is None or when.at is not None:
        await message.answer("Не понял дату. Напиши так: «31.12.2026».")
        return
    await state.update_data(horizon=when.day.isoformat())
    await _ask_milestones(message, state)


async def _ask_milestones(message: Message, state: FSMContext) -> None:
    await state.set_state(GoalForm.milestones)
    await message.answer(
        "Этапы дорожной карты — каждый с новой строки. Ключевые отметь «!» в начале.",
        reply_markup=_skip("milestones", "Без этапов"),
    )


@router.message(GoalForm.milestones, *TEXT_INPUT)
async def on_milestones(
    message: Message, ctx: AppContext, state: FSMContext, update_id: int
) -> None:
    items = [x.strip(" -•\t") for x in (message.text or "").splitlines() if x.strip(" -•\t")]
    await _create(message, ctx, state, items[:20], update_id)


@router.callback_query(GoalCb.filter(F.action == "skip"))
async def on_skip(
    query: CallbackQuery, callback_data: GoalCb, ctx: AppContext, state: FSMContext, update_id: int
) -> None:
    current = await state.get_state()
    msg = query.message if isinstance(query.message, Message) else None
    if msg is None or current is None:
        await query.answer("Начни заново из меню «Проекты/цели».")
        return
    step = callback_data.arg
    if step == "outcome" and current == GoalForm.outcome.state:
        await _ask_horizon(msg, state)
    elif step == "horizon" and current == GoalForm.horizon.state:
        await _ask_milestones(msg, state)
    elif step == "milestones" and current == GoalForm.milestones.state:
        await _create(msg, ctx, state, [], update_id)
    await query.answer()


async def _create(
    message: Message, ctx: AppContext, state: FSMContext, milestones: list[str], update_id: int
) -> None:
    data = await state.get_data()
    await state.clear()
    horizon = date.fromisoformat(str(data["horizon"])) if data.get("horizon") else None
    action = CreateGoal(
        title=str(data.get("title") or ""),
        measurable_outcome=str(data["outcome"]) if data.get("outcome") else None,
        horizon_date=horizon,
        milestones=milestones,
    )
    async with ctx.db.transaction() as s:
        res = await actions.execute(
            s, ctx, action, origin=Actor.USER, op_key=f"u{update_id}:goal_new"
        )
        if res.status is actions.ExecStatus.APPLIED:
            text, kb = await _card_or_missing(s, ctx, int(res.result["goal_id"]), "Цель добавлена.")
        else:
            text, kb = f"Не получилось: {res.error or 'ошибка'}", None
        await mark_processed(s, update_id, ctx.clock.now())
    await message.answer(text, reply_markup=kb)


# --------------------------------------------------------------------------- edit


@router.callback_query(GoalCb.filter(F.action == "edit"))
async def on_edit(query: CallbackQuery, callback_data: GoalCb) -> None:
    gid = callback_data.goal_id
    keys = list(EDIT_FIELDS)
    rows = [
        [_btn(EDIT_FIELDS[k][0], GoalCb(action="ef", goal_id=gid, arg=k)) for k in keys[i : i + 2]]
        for i in range(0, len(keys), 2)
    ]
    rows.append([_btn("← К цели", GoalCb(action="view", goal_id=gid))])
    await _edit(query, "Что изменить?", InlineKeyboardMarkup(inline_keyboard=rows))
    await query.answer()


@router.callback_query(GoalCb.filter(F.action == "ef"))
async def on_edit_field(query: CallbackQuery, callback_data: GoalCb, state: FSMContext) -> None:
    field = callback_data.arg
    if field not in EDIT_FIELDS:
        await query.answer("Кнопка устарела.")
        return
    await state.clear()
    await state.set_state(GoalForm.edit)
    await state.update_data(goal_id=callback_data.goal_id, field=field)
    await _say(query, EDIT_FIELDS[field][1])
    await query.answer()


@router.message(GoalForm.edit, *TEXT_INPUT)
async def on_edit_text(
    message: Message, ctx: AppContext, state: FSMContext, update_id: int
) -> None:
    data = await state.get_data()
    gid, field = int(data.get("goal_id") or 0), str(data.get("field") or "")
    text_in = (message.text or "").strip()
    async with ctx.db.transaction() as s:
        today = local_date(ctx.clock.now(), await gs.owner_tz(s))
    if field == "horizon":
        when = parse_when(text_in, today=today)
        if when is None or when.day is None or when.at is not None:
            await message.answer("Не понял дату. Напиши так: «31.12.2026».")
            return
        action: ChangeGoal | None = ChangeGoal(goal_id=gid, horizon_date=when.day)
    elif field == "title":
        action = ChangeGoal(goal_id=gid, title=text_in)
    elif field == "outcome":
        action = ChangeGoal(goal_id=gid, measurable_outcome=text_in)
    elif field == "criteria":
        action = ChangeGoal(goal_id=gid, success_criteria=text_in)
    else:
        action = None
    await state.clear()
    async with ctx.db.transaction() as s:
        if action is None:  # reported progress: the owner's own words
            try:
                await gs.update_goal(s, ctx, gid, actor=Actor.USER, progress_note=text_in)
                note = "Записал прогресс «со слов»."
            except gs.GoalError as exc:
                note = str(exc)
        else:
            res = await actions.execute(
                s, ctx, action, origin=Actor.USER, op_key=f"u{update_id}:goal_edit"
            )
            note = (
                "Изменил."
                if res.status is actions.ExecStatus.APPLIED
                else f"Не получилось: {res.error}"
            )
        text, kb = await _card_or_missing(s, ctx, gid, note)
        await mark_processed(s, update_id, ctx.clock.now())
    await message.answer(text, reply_markup=kb)


@router.callback_query(GoalCb.filter(F.action == "status"))
async def on_status(
    query: CallbackQuery, callback_data: GoalCb, ctx: AppContext, update_id: int
) -> None:
    status = callback_data.arg
    if status not in gs.GOAL_STATUSES:
        await query.answer("Кнопка устарела.")
        return
    async with ctx.db.transaction() as s:
        res = await actions.execute(
            s,
            ctx,
            ChangeGoal(goal_id=callback_data.goal_id, status=status),
            origin=Actor.USER,
            op_key=f"u{update_id}:goal_status",
        )
        note = (
            f"Статус: {gs.GOAL_STATUS_LABELS[status]}."
            if res.status is actions.ExecStatus.APPLIED
            else f"Не получилось: {res.error}"
        )
        text, kb = await _card_or_missing(s, ctx, callback_data.goal_id, note)
        await mark_processed(s, update_id, ctx.clock.now())
    await _edit(query, text, kb)
    await query.answer()


# --------------------------------------------------------------------------- milestones


async def render_milestones(
    session: AsyncSession, ctx: AppContext, goal_id: int, note: str | None = None
) -> tuple[str, InlineKeyboardMarkup]:
    goal = await gs.require_goal(session, goal_id)
    lines = ([note, ""] if note else []) + [
        f"Этапы — {goal.title}",
        "Нажми этап, чтобы отметить выполненным; ↩️ — вернуть в работу.",
    ]
    rows: list[list[InlineKeyboardButton]] = []
    for m in await gs.goal_milestones(session, goal_id):
        if m.status == EntityStatus.ARCHIVED:
            continue
        icon = gs.MILESTONE_ICON.get(m.status, "•")
        key = "🔑 " if m.is_key else ""
        done = m.status == EntityStatus.DONE
        rows.append(
            [
                _btn(
                    f"{'↩️' if done else icon} {key}{_short(m.title, 36)}",
                    GoalCb(action="msr" if done else "msd", goal_id=goal_id, arg=str(m.id)),
                )
            ]
        )
    rows.append([_btn("← К цели", GoalCb(action="view", goal_id=goal_id))])
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


@router.callback_query(GoalCb.filter(F.action == "ms"))
async def on_milestones_list(query: CallbackQuery, callback_data: GoalCb, ctx: AppContext) -> None:
    async with ctx.db.transaction() as s:
        try:
            text, kb = await render_milestones(s, ctx, callback_data.goal_id)
        except gs.GoalError as exc:
            text, kb = str(exc), None
    await _edit(query, text, kb)
    await query.answer()


@router.callback_query(GoalCb.filter(F.action.in_({"msd", "msr"})))
async def on_milestone_status(
    query: CallbackQuery, callback_data: GoalCb, ctx: AppContext, update_id: int
) -> None:
    try:
        ms_id = int(callback_data.arg)
    except ValueError:
        await query.answer("Кнопка устарела.")
        return
    async with ctx.db.transaction() as s:
        ms = await s.get(Milestone, ms_id)
        if ms is None:
            note = "Этап не найден."
        else:
            # Explicit target status: a repeated press is a no-op, not a toggle.
            new = "done" if callback_data.action == "msd" else "active"
            res = await actions.execute(
                s,
                ctx,
                ChangeMilestone(milestone_id=ms_id, status=new),
                origin=Actor.USER,
                op_key=f"u{update_id}:ms",
            )
            note = (
                (
                    f"✅ Этап «{ms.title}» выполнен."
                    if new == "done"
                    else f"Этап «{ms.title}» снова в работе."
                )
                if res.status is actions.ExecStatus.APPLIED
                else f"Не получилось: {res.error}"
            )
        text, kb = await render_milestones(s, ctx, callback_data.goal_id, note)
        await mark_processed(s, update_id, ctx.clock.now())
    await _edit(query, text, kb)
    await query.answer(note[:200])


@router.callback_query(GoalCb.filter(F.action == "msadd"))
async def on_milestone_add(query: CallbackQuery, callback_data: GoalCb, state: FSMContext) -> None:
    await state.clear()
    await state.set_state(GoalForm.ms_add)
    await state.update_data(goal_id=callback_data.goal_id)
    await _say(query, "Новые этапы — каждый с новой строки. Ключевые отметь «!» в начале.")
    await query.answer()


@router.message(GoalForm.ms_add, *TEXT_INPUT)
async def on_milestone_text(
    message: Message, ctx: AppContext, state: FSMContext, update_id: int
) -> None:
    data = await state.get_data()
    await state.clear()
    gid = int(data.get("goal_id") or 0)
    items = [x.strip(" -•\t") for x in (message.text or "").splitlines() if x.strip(" -•\t")]
    async with ctx.db.transaction() as s:
        added = 0
        try:
            for raw in items[:20]:
                title, key = gs.split_milestone_title(raw)
                if title:
                    await gs.add_milestone(s, ctx, gid, title=title, is_key=key, actor=Actor.USER)
                    added += 1
            note = f"Добавил этапов: {added}."
        except gs.GoalError as exc:
            note = str(exc)
        text, kb = await _card_or_missing(s, ctx, gid, note)
        await mark_processed(s, update_id, ctx.clock.now())
    await message.answer(text, reply_markup=kb)


# --------------------------------------------------------------------------- link tasks


async def _task_picker(
    session: AsyncSession, action: str, entity_id: int, back: GoalCb, header: str
) -> tuple[str, InlineKeyboardMarkup]:
    tasks = await gs.unlinked_open_tasks(session, limit=12)
    rows = [
        [_btn(_short(t.title, 40), GoalCb(action=action, goal_id=entity_id, arg=str(t.id)))]
        for t in tasks
    ]
    rows.append([_btn("← Назад", back)])
    text = header if tasks else "Нет открытых задач без цели — привязывать нечего."
    return text, InlineKeyboardMarkup(inline_keyboard=rows)


@router.callback_query(GoalCb.filter(F.action == "link"))
async def on_link(query: CallbackQuery, callback_data: GoalCb, ctx: AppContext) -> None:
    gid = callback_data.goal_id
    async with ctx.db.transaction() as s:
        text, kb = await _task_picker(
            s, "lt", gid, GoalCb(action="view", goal_id=gid), "Какую задачу привязать к цели?"
        )
    await _edit(query, text, kb)
    await query.answer()


@router.callback_query(GoalCb.filter(F.action.in_({"lt", "plt"})))
async def on_link_task(
    query: CallbackQuery, callback_data: GoalCb, ctx: AppContext, update_id: int
) -> None:
    try:
        task_id = int(callback_data.arg)
    except ValueError:
        await query.answer("Кнопка устарела.")
        return
    to_project = callback_data.action == "plt"
    async with ctx.db.transaction() as s:
        try:
            if to_project:
                task = await gs.link_task(
                    s, ctx, task_id, actor=Actor.USER, project_id=callback_data.goal_id
                )
            else:
                task = await gs.link_task(
                    s, ctx, task_id, actor=Actor.USER, goal_id=callback_data.goal_id
                )
            note = f"🔗 Привязал «{task.title}»."
        except (gs.GoalError, ts.TaskError) as exc:
            note = str(exc)
        if to_project:
            text, kb = await render_project(s, ctx, callback_data.goal_id, note)
        else:
            text, kb = await _card_or_missing(s, ctx, callback_data.goal_id, note)
        await mark_processed(s, update_id, ctx.clock.now())
    await _edit(query, text, kb)
    await query.answer(note[:200])


# --------------------------------------------------------------------------- projects


async def render_projects(
    session: AsyncSession, ctx: AppContext, note: str | None = None
) -> tuple[str, InlineKeyboardMarkup]:
    projects = await st.list_projects(session)
    lines = ([note, ""] if note else []) + ["📁 Проекты"]
    rows: list[list[InlineKeyboardButton]] = []
    if not projects:
        lines += ["", "Проектов пока нет."]
    for i, pr in enumerate(projects[:MAX_LIST], 1):
        goal = await gs.get_goal(session, pr.goal_id) if pr.goal_id else None
        lines.append(f"{i}. {pr.title}" + (f" — цель «{goal.title}»" if goal else " — без цели"))
        rows.append([_btn(f"{i}. {_short(pr.title)}", GoalCb(action="pv", goal_id=pr.id))])
    rows.append(
        [
            _btn("➕ Новый проект", GoalCb(action="pnew")),
            _btn("← К целям", GoalCb(action="list")),
        ]
    )
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


async def render_project(
    session: AsyncSession, ctx: AppContext, project_id: int, note: str | None = None
) -> tuple[str, InlineKeyboardMarkup | None]:
    pr = await session.get(Project, project_id)
    if pr is None:
        return "Проект не найден.", None
    goal = await gs.get_goal(session, pr.goal_id) if pr.goal_id else None
    lines = ([note, ""] if note else []) + [f"📁 {pr.title}"]
    lines.append(f"Цель: {goal.title}" if goal else "Цель: не привязан")
    if pr.status != "active":
        lines.append(f"Статус: {gs.GOAL_STATUS_LABELS.get(pr.status, pr.status)}")
    milestones = await st.list_milestones(session, project_id=pr.id)
    if milestones:
        lines += ["", "Этапы:"] + [
            f"{gs.MILESTONE_ICON.get(m.status, '•')} {m.title}" for m in milestones
        ]
    open_tasks = await gs.project_open_tasks(session, pr.id)
    if open_tasks:
        lines += ["", "Открытые задачи:"] + [f"• {t.title}" for t in open_tasks[:MAX_TASKS]]
    rows = [
        [
            _btn("🎯 Привязать к цели", GoalCb(action="pg", goal_id=pr.id)),
            _btn("🔗 Привязать задачу", GoalCb(action="plink", goal_id=pr.id)),
        ],
        [
            _btn("📦 В архив", GoalCb(action="parch", goal_id=pr.id)),
            _btn("← К проектам", GoalCb(action="projects")),
        ],
    ]
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


@router.callback_query(GoalCb.filter(F.action == "projects"))
async def on_projects(query: CallbackQuery, ctx: AppContext) -> None:
    async with ctx.db.transaction() as s:
        text, kb = await render_projects(s, ctx)
    await _edit(query, text, kb)
    await query.answer()


@router.callback_query(GoalCb.filter(F.action == "pv"))
async def on_project_view(query: CallbackQuery, callback_data: GoalCb, ctx: AppContext) -> None:
    async with ctx.db.transaction() as s:
        text, kb = await render_project(s, ctx, callback_data.goal_id)
    await _edit(query, text, kb)
    await query.answer()


@router.callback_query(GoalCb.filter(F.action == "pnew"))
async def on_project_new(query: CallbackQuery, callback_data: GoalCb, state: FSMContext) -> None:
    await state.clear()
    await state.set_state(GoalForm.project)
    await state.update_data(goal_id=callback_data.goal_id)
    await _say(query, "Как назвать проект?")
    await query.answer()


@router.message(GoalForm.project, *TEXT_INPUT)
async def on_project_title(
    message: Message, ctx: AppContext, state: FSMContext, update_id: int
) -> None:
    data = await state.get_data()
    await state.clear()
    gid = int(data.get("goal_id") or 0) or None
    async with ctx.db.transaction() as s:
        try:
            if gid is not None:
                await gs.require_goal(s, gid)
            pr = await st.create_project(
                s, ctx, title=message.text or "", actor=Actor.USER, goal_id=gid
            )
            text, kb = await render_project(s, ctx, pr.id, "Проект создан.")
        except (ValueError, gs.GoalError) as exc:
            text, kb = (str(exc) if isinstance(exc, gs.GoalError) else "Пустое название."), None
        await mark_processed(s, update_id, ctx.clock.now())
    await message.answer(text, reply_markup=kb)


@router.callback_query(GoalCb.filter(F.action == "pg"))
async def on_project_goal_pick(
    query: CallbackQuery, callback_data: GoalCb, ctx: AppContext
) -> None:
    pid = callback_data.goal_id
    async with ctx.db.transaction() as s:
        goals = await gs.list_goals(s, statuses=("active", "paused"))
    rows = [
        [_btn(_short(g.title, 40), GoalCb(action="pgs", goal_id=pid, arg=str(g.id)))] for g in goals
    ]
    rows.append([_btn("Без цели", GoalCb(action="pgs", goal_id=pid, arg="0"))])
    rows.append([_btn("← К проекту", GoalCb(action="pv", goal_id=pid))])
    await _edit(query, "К какой цели относится проект?", InlineKeyboardMarkup(inline_keyboard=rows))
    await query.answer()


@router.callback_query(GoalCb.filter(F.action == "pgs"))
async def on_project_goal_set(
    query: CallbackQuery, callback_data: GoalCb, ctx: AppContext, update_id: int
) -> None:
    gid = int(callback_data.arg or 0) or None
    async with ctx.db.transaction() as s:
        try:
            await gs.set_project_goal(s, ctx, callback_data.goal_id, gid, actor=Actor.USER)
            note = "Привязал." if gid else "Отвязал от цели."
        except gs.GoalError as exc:
            note = str(exc)
        text, kb = await render_project(s, ctx, callback_data.goal_id, note)
        await mark_processed(s, update_id, ctx.clock.now())
    await _edit(query, text, kb)
    await query.answer()


@router.callback_query(GoalCb.filter(F.action == "plink"))
async def on_project_link(query: CallbackQuery, callback_data: GoalCb, ctx: AppContext) -> None:
    pid = callback_data.goal_id
    async with ctx.db.transaction() as s:
        text, kb = await _task_picker(
            s, "plt", pid, GoalCb(action="pv", goal_id=pid), "Какую задачу привязать к проекту?"
        )
    await _edit(query, text, kb)
    await query.answer()


@router.callback_query(GoalCb.filter(F.action == "parch"))
async def on_project_archive(
    query: CallbackQuery, callback_data: GoalCb, ctx: AppContext, update_id: int
) -> None:
    async with ctx.db.transaction() as s:
        try:
            await st.set_project_status(
                s, ctx, callback_data.goal_id, EntityStatus.ARCHIVED, actor=Actor.USER
            )
            note = "Проект в архиве."
        except LookupError:
            note = "Проект не найден."
        text, kb = await render_projects(s, ctx, note)
        await mark_processed(s, update_id, ctx.clock.now())
    await _edit(query, text, kb)
    await query.answer()


# --------------------------------------------------------------------------- review


@router.callback_query(GoalCb.filter(F.action == "review"))
async def on_review(
    query: CallbackQuery, callback_data: GoalCb, ctx: AppContext, update_id: int
) -> None:
    await query.answer("Готовлю разбор…")
    goal_id = callback_data.goal_id or None
    try:
        outcome = await gs.run_goal_review(ctx, op_prefix=f"u{update_id}:goalrev", goal_id=goal_id)
    except gs.GoalError as exc:
        await _say(query, str(exc))
        return
    await _say(query, outcome.text)
    if not outcome.approval_ids or not isinstance(query.message, Message):
        return
    async with ctx.db.transaction() as s:
        cards = [a for a in [await approvals.get_approval(s, i) for i in outcome.approval_ids] if a]
        tz = await gs.owner_tz(s)
    await send_approval_card(
        query.message, ctx, cards, tz=tz, header="Предлагаемые изменения целей:"
    )
