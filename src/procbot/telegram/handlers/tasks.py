"""Tasks UI: "➕ Добавить", "📋 Задачи", task cards, completion, postponing,
subtasks, editing. Commands /add and /tasks.

Thin: parse input -> TaskService / ActionExecutor -> reply after commit.
Text the user types in "Добавить" is an explicit command, so the task is created
directly (origin USER). Free text elsewhere goes to handlers.chat (confirmation).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.models import Category, Context, Task, TaskEvent
from procbot.domain.actions import CreateSubtasks, CreateTask, SubtaskDraft, TaskDraft
from procbot.domain.enums import Actor, ObstacleReason, TaskEventType, TaskStatus
from procbot.services import actions
from procbot.services import tasks as ts
from procbot.services.dayview import build_day_view, render_day_view
from procbot.services.when import parse_when
from procbot.telegram.callbacks import SessionCb, TaskCb
from procbot.telegram.keyboards import Menu
from procbot.telegram.middlewares import mark_processed
from procbot.timeutil import local_date, resolve_local, to_local

router = Router(name="tasks")

MAX_LIST = 20
MAX_ADD_LINES = 15

VIEW_LABELS: dict[ts.TaskView, str] = {
    ts.TaskView.TODAY: "Сегодня",
    ts.TaskView.UPCOMING: "Скоро",
    ts.TaskView.BACKLOG: "Бэклог",
    ts.TaskView.SOMEDAY: "Когда-нибудь",
    ts.TaskView.DONE: "Готово",
}

REASON_LABELS: dict[ObstacleReason, str] = {
    ObstacleReason.UNCLEAR: "Непонятно, с чего начать",
    ObstacleReason.TOO_BIG: "Слишком большая",
    ObstacleReason.UNPLEASANT: "Неприятная",
    ObstacleReason.DISTRACTIONS: "Отвлекаюсь",
    ObstacleReason.LOW_ENERGY: "Мало сил",
    ObstacleReason.PRIORITY_CONFLICT: "Есть дела важнее",
    ObstacleReason.EXTERNAL: "Внешние обстоятельства",
    ObstacleReason.OTHER: "Другое",
}

WHEN_HINT = "Например: «сегодня 15:00», «завтра», «пт 10:00», «27.10», «27.10 9:30»."


class TaskForm(StatesGroup):
    title = State()  # "Напиши задачу"
    when = State()  # data: task_id, mode = plan | postpone | due, count
    criteria = State()  # data: task_id
    partial_note = State()  # data: task_id
    subtasks = State()  # data: task_id
    rename = State()  # data: task_id
    estimate = State()  # data: task_id


TEXT_INPUT = (F.text, ~F.text.startswith("/"), ~F.text.in_(set(Menu.ALL)))


# --------------------------------------------------------------------------- helpers


def _btn(text: str, cb: TaskCb | SessionCb) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=cb.pack())


def _short(text: str, n: int = 30) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


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


async def _names(session: AsyncSession, task: Task) -> tuple[str | None, str | None]:
    cat = await session.get(Category, task.category_id) if task.category_id else None
    con = await session.get(Context, task.context_id) if task.context_id else None
    return (cat.name if cat else None), (con.name if con else None)


# --------------------------------------------------------------------------- task card


async def render_card(
    session: AsyncSession, ctx: AppContext, task: Task, *, note: str | None = None
) -> tuple[str, InlineKeyboardMarkup]:
    tz = await ts.owner_tz(session)
    today = local_date(ctx.clock.now(), tz)
    subs = await ts.list_subtasks(session, task.id) if task.parent_id is None else []
    cat, con = await _names(session, task)
    lines: list[str] = []
    if note:
        lines += [note, ""]
    prefix = "🪜 Шаг" if task.parent_id else "📌"
    lines.append(f"{prefix} {task.title}")
    if task.parent_id:
        parent = await session.get(Task, task.parent_id)
        if parent is not None:
            lines.append(f"Задача: {parent.title}")
    lines.append(f"Статус: {ts.STATUS_LABELS.get(task.status, task.status)}")
    when = ts.format_when(task, tz, today)
    if when:
        lines.append(f"Когда: {when}")
    if ts.is_overdue(task, ctx.clock.now(), tz):
        lines.append("⚠️ Срок прошёл")
    if task.description:
        lines.append(task.description)
    if task.estimate_minutes:
        lines.append(f"Оценка: ~{task.estimate_minutes} мин")
    if task.done_criteria:
        lines.append(f"Критерий готовности: {task.done_criteria}")
    lines.append(
        "Закрытие: "
        + ("сверка с критерием" if ts.requires_criteria_check(task) else "одной кнопкой")
    )
    if task.progress_note:
        lines.append(f"Прогресс: {task.progress_note}")
    if subs:
        done = sum(1 for s in subs if s.status == TaskStatus.DONE)
        lines.append(f"Шаги: {done}/{len(subs)} готово")
        nxt = next((s for s in subs if ts.is_open(s)), None)
        if nxt is not None:
            lines.append(f"Ближайший шаг: {nxt.title}")
    if task.postpone_count:
        lines.append(f"Переносов: {task.postpone_count}")
    meta = " · ".join(x for x in (cat and f"Категория: {cat}", con and f"Контекст: {con}") if x)
    if meta:
        lines.append(meta)
    return "\n".join(lines), card_keyboard(task)


def card_keyboard(task: Task) -> InlineKeyboardMarkup:
    tid = task.id
    rows: list[list[InlineKeyboardButton]] = []
    if task.status in ts.CLOSED_STATUSES:
        rows.append([_btn("↩️ Вернуть в работу", TaskCb(action="reopen", task_id=tid))])
        rows.append([_btn("🗑 Удалить", TaskCb(action="delete", task_id=tid))])
    else:
        rows.append(
            [
                _btn("✅ Готово", TaskCb(action="done", task_id=tid)),
                _btn("◐ Частично", TaskCb(action="partial", task_id=tid)),
            ]
        )
        rows.append([_btn("▶️ Начать работу", SessionCb(action="pick", task_id=tid))])
        second = [_btn("📆 Перенести", TaskCb(action="postpone", task_id=tid))]
        if task.parent_id is None:
            second.append(_btn("🪜 Подзадачи", TaskCb(action="subtasks", task_id=tid)))
        rows.append(second)
        rows.append(
            [
                _btn("✏️ Изменить", TaskCb(action="edit", task_id=tid)),
                _btn("🗑 Удалить", TaskCb(action="delete", task_id=tid)),
            ]
        )
        rows.append([_btn("✖️ Отменить задачу", TaskCb(action="cancel", task_id=tid))])
    if task.parent_id is not None:
        rows.append([_btn("← К шагам", TaskCb(action="subtasks", task_id=task.parent_id))])
    else:
        rows.append([_btn("← К списку", TaskCb(action="list", arg=ts.TaskView.TODAY.value))])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def quick_keyboard(task_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                _btn("Сегодня", TaskCb(action="q", task_id=task_id, arg="today")),
                _btn("Завтра", TaskCb(action="q", task_id=task_id, arg="tomorrow")),
                _btn("В бэклог", TaskCb(action="q", task_id=task_id, arg="backlog")),
            ],
            [
                _btn("🕘 Указать время", TaskCb(action="q", task_id=task_id, arg="time")),
                _btn("🎯 Критерий готовности", TaskCb(action="q", task_id=task_id, arg="criteria")),
            ],
            [_btn("Открыть карточку", TaskCb(action="view", task_id=task_id))],
        ]
    )


async def _card_or_missing(
    session: AsyncSession, ctx: AppContext, task_id: int, note: str | None = None
) -> tuple[str, InlineKeyboardMarkup | None]:
    task = await ts.get_task(session, task_id)
    if task is None:
        return "Задача не найдена (возможно, удалена).", None
    return await render_card(session, ctx, task, note=note)


# --------------------------------------------------------------------------- lists


async def render_list(
    session: AsyncSession, ctx: AppContext, view: ts.TaskView
) -> tuple[str, InlineKeyboardMarkup]:
    tz = await ts.owner_tz(session)
    today = local_date(ctx.clock.now(), tz)
    items = await ts.list_tasks(session, ctx, view, limit=MAX_LIST + 1)
    lines = [f"📋 Задачи — {VIEW_LABELS[view]}"]
    rows: list[list[InlineKeyboardButton]] = []
    if not items:
        lines.append("")
        lines.append("Здесь пусто.")
    for i, t in enumerate(items[:MAX_LIST], 1):
        when = ts.format_when(t, tz, today)
        mark = "▶ " if t.status == TaskStatus.IN_PROGRESS else ""
        lines.append(f"{i}. {mark}{t.title}" + (f" · {when}" if when else ""))
        rows.append([_btn(f"{i}. {_short(t.title, 40)}", TaskCb(action="view", task_id=t.id))])
    if len(items) > MAX_LIST:
        lines.append(f"…и ещё. Показаны первые {MAX_LIST}.")
    filters = [
        _btn(("• " if v is view else "") + label, TaskCb(action="list", arg=v.value))
        for v, label in VIEW_LABELS.items()
    ]
    rows.append(filters[:3])
    rows.append(filters[3:])
    rows.append([_btn("➕ Добавить", TaskCb(action="add"))])
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


@router.message(Command("tasks"))
@router.message(F.text == Menu.TASKS)
async def on_tasks(message: Message, ctx: AppContext, state: FSMContext) -> None:
    await state.clear()
    async with ctx.db.transaction() as s:
        text, kb = await render_list(s, ctx, ts.TaskView.TODAY)
    await message.answer(text, reply_markup=kb)


@router.callback_query(TaskCb.filter(F.action == "list"))
async def on_list(query: CallbackQuery, callback_data: TaskCb, ctx: AppContext) -> None:
    try:
        view = ts.TaskView(callback_data.arg or ts.TaskView.TODAY.value)
    except ValueError:
        view = ts.TaskView.TODAY
    async with ctx.db.transaction() as s:
        text, kb = await render_list(s, ctx, view)
    await _edit(query, text, kb)
    await query.answer()


# --------------------------------------------------------------------------- add


ADD_PROMPT = "Напиши задачу. Несколько — каждую с новой строки."


@router.message(Command("add"))
@router.message(F.text == Menu.ADD)
async def on_add(
    message: Message,
    ctx: AppContext,
    state: FSMContext,
    update_id: int,
    command: CommandObject | None = None,
) -> None:
    await state.clear()
    if command is not None and command.args:
        await _create_from_text(message, ctx, command.args, update_id)
        return
    await state.set_state(TaskForm.title)
    await message.answer(ADD_PROMPT)


@router.callback_query(TaskCb.filter(F.action == "add"))
async def on_add_button(query: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(TaskForm.title)
    if isinstance(query.message, Message):
        await query.message.answer(ADD_PROMPT)
    await query.answer()


@router.message(TaskForm.title, *TEXT_INPUT)
async def on_add_text(message: Message, ctx: AppContext, state: FSMContext, update_id: int) -> None:
    await state.clear()
    await _create_from_text(message, ctx, message.text or "", update_id)


async def _create_from_text(message: Message, ctx: AppContext, text: str, update_id: int) -> None:
    titles = ts.split_candidates(text)[:MAX_ADD_LINES]
    if not titles:
        await message.answer("Пустая задача — ничего не добавил.")
        return
    created: list[tuple[int, str]] = []
    errors: list[str] = []
    async with ctx.db.transaction() as s:
        for i, title in enumerate(titles):
            res = await actions.execute(
                s,
                ctx,
                CreateTask(task=TaskDraft(title=title)),
                origin=Actor.USER,
                op_key=f"u{update_id}:add:{i}",
            )
            if res.status is actions.ExecStatus.APPLIED:
                created.append((int(res.result["task_id"]), str(res.result["title"])))
            else:
                errors.append(res.error or title)
        await mark_processed(s, update_id, ctx.clock.now())
    if len(created) == 1:
        tid, title = created[0]
        await message.answer(
            f"Добавил: «{title}». Когда займёшься?", reply_markup=quick_keyboard(tid)
        )
    elif created:
        rows = [
            [_btn(f"{i}. {_short(t, 40)}", TaskCb(action="view", task_id=tid))]
            for i, (tid, t) in enumerate(created, 1)
        ]
        text_out = "Добавил задачи:\n" + "\n".join(
            f"{i}. {t}" for i, (_, t) in enumerate(created, 1)
        )
        await message.answer(text_out, reply_markup=InlineKeyboardMarkup(inline_keyboard=rows))
    if errors:
        await message.answer("Не получилось: " + "; ".join(errors))


@router.callback_query(TaskCb.filter(F.action == "q"))
async def on_quick(
    query: CallbackQuery, callback_data: TaskCb, ctx: AppContext, state: FSMContext
) -> None:
    tid, arg = callback_data.task_id, callback_data.arg
    if arg == "time":
        await _ask_when(query, state, tid, mode="plan")
        return
    if arg == "criteria":
        await state.set_state(TaskForm.criteria)
        await state.update_data(task_id=tid)
        await _reply(query, "Как поймём, что задача готова? Напиши критерий одной фразой.")
        return
    async with ctx.db.transaction() as s:
        try:
            today = await ts.local_today(s, ctx)
            if arg == "today":
                await ts.plan_task(s, ctx, tid, actor=Actor.USER, planned_date=today)
                note = "Запланировал на сегодня."
            elif arg == "tomorrow":
                await ts.plan_task(
                    s, ctx, tid, actor=Actor.USER, planned_date=today + timedelta(days=1)
                )
                note = "Запланировал на завтра."
            else:
                await ts.move_to_backlog(s, ctx, tid, actor=Actor.USER)
                note = "Убрал в бэклог."
        except ts.TaskError as exc:
            note = str(exc)
        text, kb = await _card_or_missing(s, ctx, tid, note)
    await _edit(query, text, kb)
    await query.answer()


async def _reply(query: CallbackQuery, text: str, kb: InlineKeyboardMarkup | None = None) -> None:
    if isinstance(query.message, Message):
        await query.message.answer(text, reply_markup=kb)
    elif query.bot is not None:
        await query.bot.send_message(query.from_user.id, text, reply_markup=kb)
    await query.answer()


# --------------------------------------------------------------------------- view


@router.callback_query(TaskCb.filter(F.action == "view"))
async def on_view(query: CallbackQuery, callback_data: TaskCb, ctx: AppContext) -> None:
    async with ctx.db.transaction() as s:
        text, kb = await _card_or_missing(s, ctx, callback_data.task_id)
    await _edit(query, text, kb)
    await query.answer()


# --------------------------------------------------------------------------- done / partial


def criteria_keyboard(task_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                _btn(
                    "✅ Соответствует полностью", TaskCb(action="crit", task_id=task_id, arg="full")
                )
            ],
            [_btn("◐ Частично", TaskCb(action="crit", task_id=task_id, arg="part"))],
            [_btn("✖️ Нет", TaskCb(action="crit", task_id=task_id, arg="no"))],
        ]
    )


def criteria_question(task: Task) -> str:
    crit = task.done_criteria or "критерий не записан — оцени сам, сделано ли всё задуманное"
    return f"«{task.title}»\nКритерий готовности: {crit}\n\nРезультат соответствует?"


@router.callback_query(TaskCb.filter(F.action == "done"))
async def on_done(
    query: CallbackQuery, callback_data: TaskCb, ctx: AppContext, update_id: int
) -> None:
    tid = callback_data.task_id
    async with ctx.db.transaction() as s:
        task = await ts.get_task(s, tid)
        if task is None:
            text, kb = "Задача не найдена (возможно, удалена).", None
        elif task.status != TaskStatus.DONE and ts.requires_criteria_check(task):
            text, kb = criteria_question(task), criteria_keyboard(tid)
        else:
            already = task.status == TaskStatus.DONE
            await ts.complete_task(s, ctx, tid, actor=Actor.USER)
            note = "Уже было отмечено как готово." if already else "✅ Готово!"
            text, kb = await _after_done(s, ctx, task, callback_data.arg, note)
        await mark_processed(s, update_id, ctx.clock.now())
    await _edit(query, text, kb)
    await query.answer()


async def _after_done(
    s: AsyncSession, ctx: AppContext, task: Task, back: str, note: str
) -> tuple[str, InlineKeyboardMarkup | None]:
    if back == "day":
        view = await build_day_view(s, ctx)
        text, kb = render_day_view(view)
        return f"{note} «{task.title}»\n\n{text}", kb
    if back == "sub" and task.parent_id is not None:
        return await render_subtasks(s, ctx, task.parent_id, note=f"{note} «{task.title}»")
    return await render_card(s, ctx, task, note=note)


@router.callback_query(TaskCb.filter(F.action == "crit"))
async def on_criteria_answer(
    query: CallbackQuery,
    callback_data: TaskCb,
    ctx: AppContext,
    state: FSMContext,
    update_id: int,
) -> None:
    tid, arg = callback_data.task_id, callback_data.arg
    if arg == "part":
        await _ask_partial(query, state, tid)
        return
    async with ctx.db.transaction() as s:
        if arg == "full":
            task = await ts.get_task(s, tid)
            if task is None:
                text, kb = "Задача не найдена.", None
            else:
                await ts.complete_task(s, ctx, tid, actor=Actor.USER, criteria_confirmed=True)
                text, kb = await render_card(s, ctx, task, note="✅ Готово — критерий выполнен.")
        else:
            text, kb = await _card_or_missing(
                s,
                ctx,
                tid,
                "Хорошо, задача остаётся открытой. Можно продолжить работу "
                "или отметить частичный прогресс.",
            )
        await mark_processed(s, update_id, ctx.clock.now())
    await _edit(query, text, kb)
    await query.answer()


async def _ask_partial(query: CallbackQuery, state: FSMContext, tid: int) -> None:
    await state.set_state(TaskForm.partial_note)
    await state.update_data(task_id=tid)
    kb = InlineKeyboardMarkup(
        inline_keyboard=[[_btn("Без заметки", TaskCb(action="partial_save", task_id=tid))]]
    )
    await _reply(
        query,
        "Что уже сделано? Напиши коротко — сохраню как прогресс (задача не закроется).",
        kb,
    )


@router.callback_query(TaskCb.filter(F.action == "partial"))
async def on_partial(query: CallbackQuery, callback_data: TaskCb, state: FSMContext) -> None:
    await _ask_partial(query, state, callback_data.task_id)


@router.callback_query(TaskCb.filter(F.action == "partial_save"))
async def on_partial_save(
    query: CallbackQuery, callback_data: TaskCb, ctx: AppContext, state: FSMContext
) -> None:
    await state.clear()
    text, kb = await _save_partial(ctx, callback_data.task_id, None)
    await _edit(query, text, kb)
    await query.answer()


@router.message(TaskForm.partial_note, *TEXT_INPUT)
async def on_partial_note(
    message: Message, ctx: AppContext, state: FSMContext, update_id: int
) -> None:
    data = await state.get_data()
    await state.clear()
    text, kb = await _save_partial(ctx, int(data.get("task_id") or 0), message.text, update_id)
    await message.answer(text, reply_markup=kb)


async def _save_partial(
    ctx: AppContext, tid: int, note: str | None, update_id: int | None = None
) -> tuple[str, InlineKeyboardMarkup | None]:
    async with ctx.db.transaction() as s:
        try:
            task = await ts.record_progress(s, ctx, tid, actor=Actor.USER, note=note)
            result = await render_card(
                s, ctx, task, note="◐ Сохранил прогресс. Задача в работе, не закрыта."
            )
        except ts.TaskError as exc:
            result = (str(exc), None)
        if update_id is not None:
            await mark_processed(s, update_id, ctx.clock.now())
    return result


# --------------------------------------------------------------------------- postpone


def postpone_keyboard(task: Task) -> InlineKeyboardMarkup:
    tid, n = task.id, task.postpone_count

    def b(text: str, target: str) -> InlineKeyboardButton:
        return _btn(text, TaskCb(action="pp", task_id=tid, arg=f"{target}.{n}"))

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [b("Позже сегодня", "later"), b("Завтра", "tomorrow")],
            [b("📅 Выбрать дату", "date"), b("📥 В бэклог", "backlog")],
            [b("💤 Когда-нибудь", "someday")],
            [_btn("← Назад", TaskCb(action="view", task_id=tid))],
        ]
    )


def later_keyboard(task: Task) -> InlineKeyboardMarkup:
    tid, n = task.id, task.postpone_count

    def b(text: str, target: str) -> InlineKeyboardButton:
        return _btn(text, TaskCb(action="pp", task_id=tid, arg=f"{target}.{n}"))

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [b("+1 ч", "h1"), b("+2 ч", "h2"), b("+3 ч", "h3")],
            [b("Без точного времени", "todaynt"), b("🕘 Указать время", "date")],
            [_btn("← Назад", TaskCb(action="postpone", task_id=tid))],
        ]
    )


def reasons_keyboard(task_id: int) -> InlineKeyboardMarkup:
    rows = [
        [_btn(label, TaskCb(action="why", task_id=task_id, arg=reason.value))]
        for reason, label in REASON_LABELS.items()
    ]
    rows.append([_btn("Пропустить", TaskCb(action="why", task_id=task_id, arg="skip"))])
    return InlineKeyboardMarkup(inline_keyboard=rows)


@router.callback_query(TaskCb.filter(F.action == "postpone"))
async def on_postpone_menu(query: CallbackQuery, callback_data: TaskCb, ctx: AppContext) -> None:
    async with ctx.db.transaction() as s:
        task = await ts.get_task(s, callback_data.task_id)
    if task is None or task.status in ts.CLOSED_STATUSES:
        await query.answer("Задача уже закрыта или удалена.")
        return
    await _edit(query, f"Перенести «{task.title}» — куда?", postpone_keyboard(task))
    await query.answer()


def _round_up(dt: datetime, minutes: int = 5) -> datetime:
    dt = dt.replace(second=0, microsecond=0)
    extra = (-dt.minute) % minutes
    return dt + timedelta(minutes=extra)


@router.callback_query(TaskCb.filter(F.action == "pp"))
async def on_postpone(
    query: CallbackQuery,
    callback_data: TaskCb,
    ctx: AppContext,
    state: FSMContext,
    update_id: int,
) -> None:
    tid = callback_data.task_id
    target, _, token = callback_data.arg.partition(".")
    async with ctx.db.transaction() as s:
        task = await ts.get_task(s, tid)
    if task is None or task.status in ts.CLOSED_STATUSES:
        await query.answer("Задача уже закрыта или удалена.")
        return
    if token != str(task.postpone_count):
        # A stale or double-pressed button: the task was already moved.
        await query.answer("Уже перенесено.")
        return
    if target == "later":
        await _edit(query, f"«{task.title}» — когда сегодня?", later_keyboard(task))
        await query.answer()
        return
    if target == "date":
        await _ask_when(query, state, tid, mode="postpone", count=task.postpone_count)
        return
    now = ctx.clock.now()
    kwargs: dict[str, object] = {}
    async with ctx.db.transaction() as s:
        tz = await ts.owner_tz(s)
        today = local_date(now, tz)
        if target in ("h1", "h2", "h3"):
            kwargs["planned_start_at"] = _round_up(now + timedelta(hours=int(target[1])))
        elif target == "todaynt":
            kwargs["planned_date"] = today
        elif target == "tomorrow":
            if task.planned_start_at is not None:
                local_t = to_local(task.planned_start_at, tz).time()
                kwargs["planned_start_at"] = resolve_local(today + timedelta(days=1), local_t, tz)
            else:
                kwargs["planned_date"] = today + timedelta(days=1)
        elif target == "backlog":
            kwargs["to_backlog"] = True
        elif target == "someday":
            kwargs["to_someday"] = True
        try:
            moved, _ev = await ts.postpone_task(s, ctx, tid, actor=Actor.USER, **kwargs)  # type: ignore[arg-type]
            where = ts.format_plan(moved, tz, today) or (
                "когда-нибудь" if moved.status == TaskStatus.SOMEDAY else "без даты"
            )
            text = (
                f"Перенёс «{moved.title}»: {where}.\n\n"
                "Что помешало? Это по желанию — можно пропустить. "
                "Варианты — только предположения, не диагноз."
            )
            kb: InlineKeyboardMarkup | None = reasons_keyboard(tid)
        except ts.TaskError as exc:
            text, kb = str(exc), None
        await mark_processed(s, update_id, now)
    await _edit(query, text, kb)
    await query.answer()


@router.callback_query(TaskCb.filter(F.action == "why"))
async def on_reason(
    query: CallbackQuery, callback_data: TaskCb, ctx: AppContext, update_id: int
) -> None:
    tid, arg = callback_data.task_id, callback_data.arg
    async with ctx.db.transaction() as s:
        note: str | None = None
        task = await ts.get_task(s, tid)
        if task is not None and arg != "skip":
            try:
                reason = ObstacleReason(arg)
            except ValueError:
                reason = ObstacleReason.OTHER
            last = (
                await s.execute(
                    select(TaskEvent)
                    .where(
                        TaskEvent.task_id == tid,
                        TaskEvent.event_type == TaskEventType.POSTPONED,
                    )
                    .order_by(TaskEvent.id.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if last is None or not await _reason_recorded(s, tid, last.id):
                await ts.report_obstacle(
                    s,
                    ctx,
                    tid,
                    reason=reason,
                    actor=Actor.USER,
                    related_event_id=last.id if last else None,
                )
            note = f"Записал: {REASON_LABELS[reason].lower()}."
            if reason in (ObstacleReason.UNCLEAR, ObstacleReason.TOO_BIG):
                note += " Можно разбить на маленькие шаги — кнопка «Подзадачи»."
        text, kb = await _card_or_missing(s, ctx, tid, note)
        await mark_processed(s, update_id, ctx.clock.now())
    await _edit(query, text, kb)
    await query.answer()


async def _reason_recorded(s: AsyncSession, task_id: int, event_id: int) -> bool:
    rows = (
        await s.execute(
            select(TaskEvent).where(
                TaskEvent.task_id == task_id,
                TaskEvent.event_type == TaskEventType.OBSTACLE_REPORTED,
            )
        )
    ).scalars()
    return any((r.payload or {}).get("event_id") == event_id for r in rows)


# --------------------------------------------------------------------------- date input


async def _ask_when(
    query: CallbackQuery, state: FSMContext, tid: int, *, mode: str, count: int | None = None
) -> None:
    await state.set_state(TaskForm.when)
    await state.update_data(task_id=tid, mode=mode, count=count)
    prompt = {
        "plan": "Когда начать?",
        "postpone": "На когда перенести?",
        "due": "Какой срок? («нет» — убрать срок)",
    }.get(mode, "Когда?")
    await _reply(query, f"{prompt} {WHEN_HINT}")


@router.message(TaskForm.when, *TEXT_INPUT)
async def on_when_text(
    message: Message, ctx: AppContext, state: FSMContext, update_id: int
) -> None:
    data = await state.get_data()
    tid, mode = int(data.get("task_id") or 0), str(data.get("mode") or "plan")
    raw = (message.text or "").strip()
    async with ctx.db.transaction() as s:
        tz = await ts.owner_tz(s)
        now = ctx.clock.now()
        today = local_date(now, tz)
        clear_due = mode == "due" and raw.lower() in {"нет", "убрать", "без срока"}
        parsed = None if clear_due else parse_when(raw, today=today)
        if parsed is None and not clear_due:
            text, kb, done = f"Не понял дату. {WHEN_HINT}", None, False
        else:
            done = True
            try:
                note = await _apply_when(s, ctx, tid, mode, parsed, tz, data.get("count"))
            except ts.TaskError as exc:
                note = str(exc)
            text, kb = await _card_or_missing(s, ctx, tid, note)
        await mark_processed(s, update_id, now)
    if done:
        await state.clear()
    await message.answer(text, reply_markup=kb)


async def _apply_when(
    s: AsyncSession,
    ctx: AppContext,
    tid: int,
    mode: str,
    parsed: object,
    tz: str,
    count: object,
) -> str:
    from procbot.services.when import When

    now = ctx.clock.now()
    today = local_date(now, tz)
    if mode == "due":
        if parsed is None:
            await ts.set_deadline(s, ctx, tid, actor=Actor.USER)
            return "Срок убран."
        assert isinstance(parsed, When)
        if parsed.at is not None:
            await ts.set_deadline(
                s,
                ctx,
                tid,
                actor=Actor.USER,
                due_at=resolve_local(parsed.day or today, parsed.at, tz),
            )
        else:
            await ts.set_deadline(s, ctx, tid, actor=Actor.USER, due_date=parsed.day)
        return "Срок обновлён."
    assert isinstance(parsed, When)
    kwargs: dict[str, object] = {}
    if parsed.at is not None:
        day = parsed.day or today
        start = resolve_local(day, parsed.at, tz)
        if parsed.day is None and start <= now:
            start = resolve_local(today + timedelta(days=1), parsed.at, tz)
        kwargs["planned_start_at"] = start
    else:
        kwargs["planned_date"] = parsed.day
    if mode == "postpone":
        task = await ts.require_task(s, tid)
        if count is not None and int(str(count)) != task.postpone_count:
            return "Уже перенесено."
        moved, _ = await ts.postpone_task(s, ctx, tid, actor=Actor.USER, **kwargs)  # type: ignore[arg-type]
        return f"Перенёс на {ts.format_plan(moved, tz, today)}."
    moved = await ts.plan_task(s, ctx, tid, actor=Actor.USER, **kwargs)  # type: ignore[arg-type]
    return f"Запланировал: {ts.format_plan(moved, tz, today)}."


# --------------------------------------------------------------------------- subtasks


async def render_subtasks(
    session: AsyncSession, ctx: AppContext, parent_id: int, *, note: str | None = None
) -> tuple[str, InlineKeyboardMarkup | None]:
    parent = await ts.get_task(session, parent_id)
    if parent is None:
        return "Задача не найдена (возможно, удалена).", None
    subs = await ts.list_subtasks(session, parent_id)
    lines: list[str] = []
    if note:
        lines += [note, ""]
    lines.append(f"🪜 Шаги: {parent.title}")
    rows: list[list[InlineKeyboardButton]] = []
    if not subs:
        lines.append("Пока нет шагов. Добавь маленькие конкретные действия.")
    for i, sub in enumerate(subs, 1):
        mark = (
            "✅" if sub.status == TaskStatus.DONE else ("▶" if sub.status == "in_progress" else "◻️")
        )
        lines.append(f"{mark} {i}. {sub.title}")
        if ts.is_open(sub):
            rows.append(
                [
                    _btn(
                        f"✅ {i}. {_short(sub.title, 30)}",
                        TaskCb(action="done", task_id=sub.id, arg="sub"),
                    ),
                    _btn("…", TaskCb(action="view", task_id=sub.id)),
                ]
            )
    if parent.status not in ts.CLOSED_STATUSES:
        rows.append([_btn("➕ Добавить шаги", TaskCb(action="subadd", task_id=parent_id))])
    rows.append([_btn("← К задаче", TaskCb(action="view", task_id=parent_id))])
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


@router.callback_query(TaskCb.filter(F.action == "subtasks"))
async def on_subtasks(query: CallbackQuery, callback_data: TaskCb, ctx: AppContext) -> None:
    async with ctx.db.transaction() as s:
        text, kb = await render_subtasks(s, ctx, callback_data.task_id)
    await _edit(query, text, kb)
    await query.answer()


@router.callback_query(TaskCb.filter(F.action == "subadd"))
async def on_subadd(query: CallbackQuery, callback_data: TaskCb, state: FSMContext) -> None:
    await state.set_state(TaskForm.subtasks)
    await state.update_data(task_id=callback_data.task_id)
    await _reply(query, "Напиши шаги — каждый с новой строки.")


@router.message(TaskForm.subtasks, *TEXT_INPUT)
async def on_subtasks_text(
    message: Message, ctx: AppContext, state: FSMContext, update_id: int
) -> None:
    data = await state.get_data()
    await state.clear()
    tid = int(data.get("task_id") or 0)
    titles = ts.split_candidates(message.text or "")[:30]
    async with ctx.db.transaction() as s:
        note: str | None = None
        if titles:
            res = await actions.execute(
                s,
                ctx,
                CreateSubtasks(parent_id=tid, subtasks=[SubtaskDraft(title=t) for t in titles]),
                origin=Actor.USER,
                op_key=f"u{update_id}:sub",
            )
            note = (
                f"Добавил шагов: {len(titles)}."
                if res.status is actions.ExecStatus.APPLIED
                else (res.error or "Не получилось добавить шаги.")
            )
        text, kb = await render_subtasks(s, ctx, tid, note=note)
        await mark_processed(s, update_id, ctx.clock.now())
    await message.answer(text, reply_markup=kb)


# --------------------------------------------------------------------------- edit


def edit_keyboard(task: Task) -> InlineKeyboardMarkup:
    tid = task.id
    verif = "одной кнопкой" if task.verification == "button" else "сверка с критерием"

    def b(text: str, arg: str) -> InlineKeyboardButton:
        return _btn(text, TaskCb(action="ed", task_id=tid, arg=arg))

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [b("Название", "title"), b("Оценка времени", "estimate")],
            [b("Критерий готовности", "criteria")],
            [b(f"Закрытие: {verif} ⇄", "verif")],
            [b("Время начала", "plan"), b("Срок", "due")],
            [_btn("← Назад", TaskCb(action="view", task_id=tid))],
        ]
    )


@router.callback_query(TaskCb.filter(F.action == "edit"))
async def on_edit_menu(query: CallbackQuery, callback_data: TaskCb, ctx: AppContext) -> None:
    async with ctx.db.transaction() as s:
        task = await ts.get_task(s, callback_data.task_id)
    if task is None:
        await query.answer("Задача не найдена.")
        return
    await _edit(query, f"Что изменить в «{task.title}»?", edit_keyboard(task))
    await query.answer()


@router.callback_query(TaskCb.filter(F.action == "ed"))
async def on_edit_field(
    query: CallbackQuery,
    callback_data: TaskCb,
    ctx: AppContext,
    state: FSMContext,
    update_id: int,
) -> None:
    tid, arg = callback_data.task_id, callback_data.arg
    if arg in ("plan", "due"):
        await _ask_when(query, state, tid, mode=arg)
        return
    prompts = {
        "title": (TaskForm.rename, "Новое название:"),
        "criteria": (TaskForm.criteria, "Как поймём, что задача готова? («нет» — убрать критерий)"),
        "estimate": (TaskForm.estimate, "Сколько минут примерно займёт? Напиши число."),
    }
    if arg in prompts:
        st, prompt = prompts[arg]
        await state.set_state(st)
        await state.update_data(task_id=tid)
        await _reply(query, prompt)
        return
    if arg == "verif":
        async with ctx.db.transaction() as s:
            task = await ts.get_task(s, tid)
            if task is None:
                await query.answer("Задача не найдена.")
                return
            new = "criteria" if task.verification == "button" else "button"
            task = await ts.update_task(s, ctx, tid, actor=Actor.USER, verification=new)
            kb = edit_keyboard(task)
            title = task.title
            await mark_processed(s, update_id, ctx.clock.now())
        await _edit(query, f"Что изменить в «{title}»?", kb)
        await query.answer("Способ закрытия изменён.")
        return
    await query.answer()


@router.message(TaskForm.rename, *TEXT_INPUT)
async def on_rename(message: Message, ctx: AppContext, state: FSMContext, update_id: int) -> None:
    await _apply_field(message, ctx, state, update_id, "title", message.text or "")


@router.message(TaskForm.criteria, *TEXT_INPUT)
async def on_criteria_text(
    message: Message, ctx: AppContext, state: FSMContext, update_id: int
) -> None:
    raw = (message.text or "").strip()
    value = None if raw.lower() in {"нет", "убрать"} else raw
    await _apply_field(message, ctx, state, update_id, "done_criteria", value)


@router.message(TaskForm.estimate, *TEXT_INPUT)
async def on_estimate(message: Message, ctx: AppContext, state: FSMContext, update_id: int) -> None:
    raw = (message.text or "").strip().split()[0] if (message.text or "").strip() else ""
    if not raw.isdigit() or not 1 <= int(raw) <= 24 * 60:
        await message.answer("Нужно число минут, например 30.")
        return
    await _apply_field(message, ctx, state, update_id, "estimate_minutes", int(raw))


async def _apply_field(
    message: Message,
    ctx: AppContext,
    state: FSMContext,
    update_id: int,
    field: str,
    value: object,
) -> None:
    data = await state.get_data()
    await state.clear()
    tid = int(data.get("task_id") or 0)
    async with ctx.db.transaction() as s:
        try:
            changes: dict[str, Any] = {field: value}
            await ts.update_task(s, ctx, tid, actor=Actor.USER, **changes)
            note = "Сохранил."
            if field == "done_criteria" and value:
                task = await ts.require_task(s, tid)
                if ts.requires_criteria_check(task):
                    note += " Перед закрытием спрошу, выполнен ли критерий."
        except ts.TaskError as exc:
            note = str(exc)
        text, kb = await _card_or_missing(s, ctx, tid, note)
        await mark_processed(s, update_id, ctx.clock.now())
    await message.answer(text, reply_markup=kb)


# --------------------------------------------------------------------------- delete / cancel


@router.callback_query(TaskCb.filter(F.action == "delete"))
async def on_delete(query: CallbackQuery, callback_data: TaskCb, ctx: AppContext) -> None:
    async with ctx.db.transaction() as s:
        task = await ts.get_task(s, callback_data.task_id)
    if task is None:
        await query.answer("Задача уже удалена.")
        return
    tid = task.id
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                _btn("Да, удалить", TaskCb(action="delok", task_id=tid)),
                _btn("Нет", TaskCb(action="view", task_id=tid)),
            ]
        ]
    )
    await _edit(
        query,
        f"Удалить «{task.title}» вместе с шагами? Это нельзя отменить. "
        "Если задача просто не нужна — можно «Отменить задачу», история сохранится.",
        kb,
    )
    await query.answer()


@router.callback_query(TaskCb.filter(F.action == "delok"))
async def on_delete_ok(
    query: CallbackQuery, callback_data: TaskCb, ctx: AppContext, update_id: int
) -> None:
    async with ctx.db.transaction() as s:
        task = await ts.get_task(s, callback_data.task_id)
        if task is None:
            text = "Задача уже удалена."
        else:
            title = task.title
            await ts.delete_task(s, ctx, task.id, actor=Actor.USER)
            text = f"🗑 Удалил «{title}»."
        await mark_processed(s, update_id, ctx.clock.now())
    kb = InlineKeyboardMarkup(
        inline_keyboard=[[_btn("📋 К задачам", TaskCb(action="list", arg="today"))]]
    )
    await _edit(query, text, kb)
    await query.answer()


@router.callback_query(TaskCb.filter(F.action.in_({"cancel", "reopen"})))
async def on_cancel_reopen(
    query: CallbackQuery, callback_data: TaskCb, ctx: AppContext, update_id: int
) -> None:
    tid = callback_data.task_id
    async with ctx.db.transaction() as s:
        task = await ts.get_task(s, tid)
        if task is None:
            text, kb = "Задача не найдена.", None
        elif callback_data.action == "cancel":
            await ts.cancel_task(s, ctx, tid, actor=Actor.USER)
            text, kb = await render_card(
                s, ctx, task, note="✖️ Задача отменена. Это нормальный исход — история сохранена."
            )
        else:
            await ts.reopen_task(s, ctx, tid, actor=Actor.USER)
            text, kb = await render_card(s, ctx, task, note="↩️ Вернул в работу.")
        await mark_processed(s, update_id, ctx.clock.now())
    await _edit(query, text, kb)
    await query.answer()
