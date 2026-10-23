"""Habits UI: «🔁 Привычки» (/habits). List with today's status and stability,
quick marks (done / minimum / conscious skip / amount, idempotent per day),
habit card with stats, missed-day analysis, create (FSM), edit, pause/archive,
"suggest an increase" (an approval card, never applied by itself).

Thin: parse input -> services.habits / ActionExecutor -> reply after commit.
Creating or editing a habit here is the owner's explicit command (origin USER);
a suggestion from the bot is a CHANGE_HABIT approval card (FR-15).
"""

from __future__ import annotations

from datetime import date, datetime

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.models import Habit
from procbot.domain.actions import ChangeHabit, CreateHabit
from procbot.domain.enums import Actor, EntityStatus
from procbot.services import actions, approvals
from procbot.services import habits as hs
from procbot.telegram.callbacks import HabitCb
from procbot.telegram.handlers.approvals import send_approval_card
from procbot.telegram.keyboards import Menu
from procbot.telegram.middlewares import mark_processed

router = Router(name="habits")

MAX_LIST = 15
TEXT_INPUT = (F.text, ~F.text.startswith("/"), ~F.text.in_(set(Menu.ALL)))
STATUS_CODES = {"d": hs.DONE, "p": hs.PARTIAL, "s": hs.SKIPPED}


class HabitForm(StatesGroup):
    title = State()  # data: mode=new
    days = State()  # data: mode, habit_id
    target = State()
    time = State()
    amount = State()  # data: habit_id, day
    rename = State()  # data: habit_id


# --------------------------------------------------------------------------- helpers


def _btn(text: str, cb: HabitCb | str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=cb if isinstance(cb, str) else cb.pack())


def _short(text: str, n: int = 28) -> str:
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


def _day_arg(raw: str) -> date | None:
    try:
        return datetime.strptime(raw, "%Y%m%d").date()
    except ValueError:
        return None


# --------------------------------------------------------------------------- list


async def render_list(
    session: AsyncSession, ctx: AppContext, *, inactive: bool = False, note: str | None = None
) -> tuple[str, InlineKeyboardMarkup]:
    today = await hs.local_today(session, ctx)
    statuses = ("paused", "archived") if inactive else ("active",)
    habits = await hs.list_habits(session, statuses=statuses)
    due = {h.habit_id: h for h in await hs.due_habits(session, ctx, today)}
    lines: list[str] = []
    if note:
        lines += [note, ""]
    lines.append(
        "🔁 Привычки на паузе и в архиве" if inactive else f"🔁 Привычки — сегодня {today:%d.%m}"
    )
    rows: list[list[InlineKeyboardButton]] = []
    if not habits:
        lines += ["", "Здесь пусто." if inactive else "Привычек пока нет — добавь первую."]
    for i, h in enumerate(habits[:MAX_LIST], 1):
        st = await hs.habit_stability(session, ctx, h.id, days=14)
        occ = await hs.get_occurrence(session, h.id, today)
        status = occ.status if occ is not None else None
        if inactive:
            mark = "⏸" if h.status == EntityStatus.PAUSED else "📦"
            today_txt = ""
        elif status is not None:
            mark, today_txt = hs.STATUS_ICON[status], f"сегодня: {hs.STATUS_LABEL[status]}"
        elif h.id in due:
            mark, today_txt = "○", "сегодня по графику"
        else:
            mark, today_txt = "–", "сегодня не по графику"
        reg = f"14 дн.: {st.kept}/{st.counted} ({hs.pct(st.regularity)})" if st.counted else ""
        tail = " · ".join(x for x in (today_txt, reg) if x)
        lines.append(f"{i}. {mark} {h.title}" + (f" — {tail}" if tail else ""))
        row = [_btn(f"{i}. {_short(h.title)}", HabitCb(action="view", habit_id=h.id))]
        if not inactive and status is None:
            row.append(_btn("✅", hs.mark_cb(h.id, hs.DONE, today, "l")))
            row.append(_btn("◐", hs.mark_cb(h.id, hs.PARTIAL, today, "l")))
        rows.append(row)
    if inactive:
        rows.append([_btn("← К привычкам", HabitCb(action="list"))])
    else:
        rows.append(
            [
                _btn("➕ Новая привычка", HabitCb(action="new")),
                _btn("🗂 Пауза и архив", HabitCb(action="inactive")),
            ]
        )
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


@router.message(Command("habits"))
@router.message(F.text == Menu.HABITS)
async def on_habits(message: Message, ctx: AppContext, state: FSMContext) -> None:
    await state.clear()
    async with ctx.db.transaction() as s:
        text, kb = await render_list(s, ctx)
    await message.answer(text, reply_markup=kb)


@router.callback_query(HabitCb.filter(F.action.in_({"list", "inactive"})))
async def on_list(query: CallbackQuery, callback_data: HabitCb, ctx: AppContext) -> None:
    async with ctx.db.transaction() as s:
        text, kb = await render_list(s, ctx, inactive=callback_data.action == "inactive")
    await _edit(query, text, kb)
    await query.answer()


# --------------------------------------------------------------------------- card


async def render_card(
    session: AsyncSession, ctx: AppContext, habit: Habit, *, note: str | None = None
) -> tuple[str, InlineKeyboardMarkup]:
    today = await hs.local_today(session, ctx)
    sched = hs.habit_schedule(habit)
    lines: list[str] = []
    if note:
        lines += [note, ""]
    lines.append(f"🔁 {habit.title}")
    when = f" · время: {habit.preferred_time:%H:%M}" if habit.preferred_time else ""
    lines.append(f"График: {hs.schedule_text(sched)}{when}")
    lines.append(f"Объём: {hs.target_text(habit.target)}")
    if habit.status == EntityStatus.PAUSED:
        lines.append("Статус: на паузе (дни паузы не считаются пропусками)")
    elif habit.status == EntityStatus.ARCHIVED:
        lines.append("Статус: в архиве")
    occ = await hs.get_occurrence(session, habit.id, today)
    if habit.status == EntityStatus.ACTIVE:
        if occ is not None:
            amount = f" ({hs.fmt_num(occ.amount)})" if occ.amount is not None else ""
            lines.append(
                f"Сегодня: {hs.STATUS_ICON[occ.status]} {hs.STATUS_LABEL[occ.status]}{amount}"
            )
        else:
            lines.append("Сегодня: не отмечено")
    due = [d for d in await hs.due_habits(session, ctx, today) if d.habit_id == habit.id]
    if sched.flexible and due:
        lines.append(f"На этой неделе: {due[0].week_kept}/{due[0].week_target}")
    lines.append("")
    lines.append("Устойчивость (единичный пропуск ничего не обнуляет):")
    for days in hs.DEFAULT_WINDOWS:
        st = await hs.habit_stability(session, ctx, habit.id, days=days)
        lines.append(f"• {hs.stability_text(st)}")
    st28 = await hs.habit_stability(session, ctx, habit.id, days=28)
    ret = hs.return_text(st28)
    if ret:
        lines.append(f"• {ret}")
    return "\n".join(lines), card_keyboard(habit, today, marked=occ is not None)


def card_keyboard(habit: Habit, today: date, *, marked: bool) -> InlineKeyboardMarkup:
    hid = habit.id
    rows: list[list[InlineKeyboardButton]] = []
    if habit.status == EntityStatus.ARCHIVED:
        rows.append([_btn("▶️ Вернуть из архива", HabitCb(action="st", habit_id=hid, arg="active"))])
        rows.append([_btn("← К списку", HabitCb(action="list"))])
        return InlineKeyboardMarkup(inline_keyboard=rows)
    if habit.status == EntityStatus.ACTIVE:
        rows.append(
            [
                _btn("✅ Сделал", hs.mark_cb(hid, hs.DONE, today, "c")),
                _btn("◐ Минимум", hs.mark_cb(hid, hs.PARTIAL, today, "c")),
            ]
        )
        rows.append(
            [
                _btn("✍️ Сколько сделал", HabitCb(action="amt", habit_id=hid)),
                _btn("⏭ Пропускаю осознанно", hs.mark_cb(hid, hs.SKIPPED, today, "c")),
            ]
        )
    rows.append(
        [
            _btn("🔍 Разбор пропусков", HabitCb(action="miss", habit_id=hid)),
            _btn("📈 Предложить шаг", HabitCb(action="inc", habit_id=hid)),
        ]
    )
    pause = (
        _btn("▶️ Возобновить", HabitCb(action="st", habit_id=hid, arg="active"))
        if habit.status == EntityStatus.PAUSED
        else _btn("⏸ Пауза", HabitCb(action="st", habit_id=hid, arg="paused"))
    )
    rows.append([_btn("✏️ Изменить", HabitCb(action="edit", habit_id=hid)), pause])
    rows.append(
        [
            _btn("📦 В архив", HabitCb(action="st", habit_id=hid, arg="archived")),
            _btn("← К списку", HabitCb(action="list")),
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def _card_or_missing(
    session: AsyncSession, ctx: AppContext, habit_id: int, note: str | None = None
) -> tuple[str, InlineKeyboardMarkup | None]:
    habit = await hs.get_habit(session, habit_id)
    if habit is None:
        return "Привычка не найдена.", None
    return await render_card(session, ctx, habit, note=note)


@router.callback_query(HabitCb.filter(F.action == "view"))
async def on_view(query: CallbackQuery, callback_data: HabitCb, ctx: AppContext) -> None:
    async with ctx.db.transaction() as s:
        text, kb = await _card_or_missing(s, ctx, callback_data.habit_id)
    await _edit(query, text, kb)
    await query.answer()


# --------------------------------------------------------------------------- marks


@router.callback_query(HabitCb.filter(F.action == "m"))
async def on_mark(
    query: CallbackQuery, callback_data: HabitCb, ctx: AppContext, update_id: int
) -> None:
    code, _, rest = callback_data.arg.partition(".")
    raw_day, _, back = rest.partition(".")
    status = STATUS_CODES.get(code)
    day = _day_arg(raw_day)
    hid = callback_data.habit_id
    if status is None or day is None:
        await query.answer("Кнопка устарела.")
        return
    text: str | None = None
    kb: InlineKeyboardMarkup | None = None
    async with ctx.db.transaction() as s:
        try:
            res = await hs.mark(s, ctx, hid, status, day=day, actor=Actor.USER)
            label = hs.STATUS_LABEL[res.occurrence.status]
            note = f"Отмечено: {label}." if res.changed else f"Уже отмечено: {label}."
        except hs.HabitError as exc:
            note = str(exc)
        if back == "l":
            text, kb = await render_list(s, ctx, note=note)
        elif back == "c":
            text, kb = await _card_or_missing(s, ctx, hid, note)
        await mark_processed(s, update_id, ctx.clock.now())
    if text is not None:
        await _edit(query, text, kb)
    await query.answer(note[:200])


@router.callback_query(HabitCb.filter(F.action == "amt"))
async def on_amount_ask(
    query: CallbackQuery, callback_data: HabitCb, ctx: AppContext, state: FSMContext
) -> None:
    async with ctx.db.transaction() as s:
        habit = await hs.get_habit(s, callback_data.habit_id)
        today = await hs.local_today(s, ctx)
    if habit is None:
        await query.answer("Привычка не найдена.")
        return
    await state.set_state(HabitForm.amount)
    await state.update_data(habit_id=habit.id, day=today.isoformat())
    unit = (habit.target or {}).get("unit") or ""
    await _say(query, f"Сколько сделал{f' ({unit})' if unit else ''}? Напиши число.")
    await query.answer()


@router.message(HabitForm.amount, *TEXT_INPUT)
async def on_amount(message: Message, ctx: AppContext, state: FSMContext, update_id: int) -> None:
    data = await state.get_data()
    raw = (message.text or "").replace(",", ".").split()
    try:
        amount = float(raw[0]) if raw else 0.0
    except ValueError:
        amount = 0.0
    if amount <= 0:
        await message.answer("Нужно положительное число, например 5.")
        return
    await state.clear()
    hid = int(data.get("habit_id") or 0)
    day = date.fromisoformat(str(data.get("day"))) if data.get("day") else None
    async with ctx.db.transaction() as s:
        try:
            res = await hs.mark_amount(s, ctx, hid, amount, day=day)
            note = f"Записал {hs.fmt_num(amount)}: {hs.STATUS_LABEL[res.occurrence.status]}."
        except hs.HabitError as exc:
            note = str(exc)
        text, kb = await _card_or_missing(s, ctx, hid, note)
        await mark_processed(s, update_id, ctx.clock.now())
    await message.answer(text, reply_markup=kb)


# --------------------------------------------------------------------------- misses


async def render_misses(
    session: AsyncSession, ctx: AppContext, habit_id: int, *, note: str | None = None
) -> tuple[str, InlineKeyboardMarkup]:
    habit = await hs.require_habit(session, habit_id)
    a = await hs.missed_analysis(session, ctx, habit_id, days=28)
    lines: list[str] = []
    if note:
        lines += [note, ""]
    lines += [f"🔍 Пропуски — {habit.title}", "", hs.analysis_text(a)]
    rows: list[list[InlineKeyboardButton]] = []
    if a.unexplained:
        lines += ["", "Что случилось в эти дни? Можно не отвечать — это только для анализа."]
        for d in a.unexplained[:7]:
            rows.append(
                [
                    _btn(
                        f"{d:%d.%m} ({hs.WEEKDAY_SHORT[d.weekday()]})",
                        HabitCb(action="why", habit_id=habit_id, arg=f"{d:%Y%m%d}"),
                    )
                ]
            )
    rows.append([_btn("← К привычке", HabitCb(action="view", habit_id=habit_id))])
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


@router.callback_query(HabitCb.filter(F.action == "miss"))
async def on_misses(query: CallbackQuery, callback_data: HabitCb, ctx: AppContext) -> None:
    async with ctx.db.transaction() as s:
        try:
            text, kb = await render_misses(s, ctx, callback_data.habit_id)
        except hs.HabitError as exc:
            text, kb = str(exc), None
    await _edit(query, text, kb)
    await query.answer()


@router.callback_query(HabitCb.filter(F.action == "why"))
async def on_why(query: CallbackQuery, callback_data: HabitCb) -> None:
    day = _day_arg(callback_data.arg)
    if day is None:
        await query.answer("Кнопка устарела.")
        return
    hid = callback_data.habit_id
    rows = [
        [_btn(label, HabitCb(action="rsn", habit_id=hid, arg=f"{day:%Y%m%d}.{code}"))]
        for code, label in hs.MISS_REASONS.items()
    ]
    rows.append(
        [
            _btn(
                "⏭ Это был осознанный пропуск",
                HabitCb(action="rsn", habit_id=hid, arg=f"{day:%Y%m%d}.skip"),
            )
        ]
    )
    rows.append(
        [
            _btn(
                "✅ На самом деле сделал",
                HabitCb(action="rsn", habit_id=hid, arg=f"{day:%Y%m%d}.done"),
            )
        ]
    )
    rows.append([_btn("← Назад", HabitCb(action="miss", habit_id=hid))])
    await _edit(
        query,
        f"{day:%d.%m} — что помешало? Выбери ближе всего подходящее.",
        InlineKeyboardMarkup(inline_keyboard=rows),
    )
    await query.answer()


@router.callback_query(HabitCb.filter(F.action == "rsn"))
async def on_reason(
    query: CallbackQuery, callback_data: HabitCb, ctx: AppContext, update_id: int
) -> None:
    raw_day, _, code = callback_data.arg.partition(".")
    day = _day_arg(raw_day)
    hid = callback_data.habit_id
    if day is None:
        await query.answer("Кнопка устарела.")
        return
    async with ctx.db.transaction() as s:
        try:
            if code == "skip":
                await hs.mark(s, ctx, hid, hs.SKIPPED, day=day)
                note = f"{day:%d.%m}: отметил как осознанный пропуск."
            elif code == "done":
                await hs.mark(s, ctx, hid, hs.DONE, day=day)
                note = f"{day:%d.%m}: отметил как сделано (задним числом)."
            else:
                await hs.record_miss(s, ctx, hid, day, code)
                note = f"{day:%d.%m}: записал причину «{hs.MISS_REASONS.get(code, code)}»."
            text, kb = await render_misses(s, ctx, hid, note=note)
        except hs.HabitError as exc:
            note = str(exc)
            text, kb = note, None
        await mark_processed(s, update_id, ctx.clock.now())
    await _edit(query, text, kb)
    await query.answer(note[:200])


# --------------------------------------------------------------------------- increase


@router.callback_query(HabitCb.filter(F.action == "inc"))
async def on_increase(
    query: CallbackQuery, callback_data: HabitCb, ctx: AppContext, update_id: int
) -> None:
    hid = callback_data.habit_id
    cards = []
    async with ctx.db.transaction() as s:
        try:
            sug = await hs.suggest_increase(s, ctx, hid)
        except hs.HabitError as exc:
            sug, text = None, str(exc)
        else:
            text = "Для этой привычки нечего увеличивать: у неё нет объёма в числах."
        if sug is not None:
            if await hs.has_pending_change(s, hid):
                text = "Предложение по этой привычке уже ждёт твоего решения."
            else:
                res = await hs.propose_change(
                    s, ctx, sug.action, origin=Actor.SYSTEM, op_key=f"u{update_id}:habit_inc:{hid}"
                )
                if res.approval_id is not None:
                    row = await approvals.get_approval(s, res.approval_id)
                    cards = [row] if row is not None else []
                text = (
                    f"Устойчивость: {hs.stability_text(sug.stability)}.\n"
                    "Это только предложение — объём меняется лишь после твоего «Принять»."
                )
                if not sug.stability.enough_data:
                    text += "\nДанных пока мало — возможно, рано увеличивать."
        tz = await hs.owner_tz(s)
        await mark_processed(s, update_id, ctx.clock.now())
    await _say(query, text)
    if cards and isinstance(query.message, Message):
        await send_approval_card(query.message, ctx, cards, tz=tz)
    await query.answer()


# --------------------------------------------------------------------------- status


@router.callback_query(HabitCb.filter(F.action == "st"))
async def on_status(
    query: CallbackQuery, callback_data: HabitCb, ctx: AppContext, update_id: int
) -> None:
    try:
        status = EntityStatus(callback_data.arg)
    except ValueError:
        await query.answer("Кнопка устарела.")
        return
    notes = {
        EntityStatus.PAUSED: "На паузе. Дни паузы не считаются пропусками.",
        EntityStatus.ACTIVE: "Привычка снова активна.",
        EntityStatus.ARCHIVED: "Убрал в архив. История сохранена.",
    }
    async with ctx.db.transaction() as s:
        try:
            await hs.set_status(s, ctx, callback_data.habit_id, status, actor=Actor.USER)
            note = notes.get(status, "Готово.")
        except hs.HabitError as exc:
            note = str(exc)
        text, kb = await _card_or_missing(s, ctx, callback_data.habit_id, note)
        await mark_processed(s, update_id, ctx.clock.now())
    await _edit(query, text, kb)
    await query.answer()


# --------------------------------------------------------------------------- create / edit


def _schedule_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [_btn("Каждый день", HabitCb(action="sch", arg="daily"))],
            [_btn("По дням недели", HabitCb(action="sch", arg="weekly"))],
            [_btn("N раз в неделю (дни свободные)", HabitCb(action="sch", arg="tpw"))],
        ]
    )


def _skip_kb(arg: str, label: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[_btn(label, HabitCb(action="skip", arg=arg))]])


TARGET_PROMPT = (
    "Какой объём? Например: «10 мин, минимум 2» или «20 отжиманий, минимум 5». "
    "Минимум — это вариант на тяжёлый день, он тоже засчитывается."
)
TIME_PROMPT = "Во сколько удобно? Напиши время ЧЧ:ММ, например 07:30."


@router.callback_query(HabitCb.filter(F.action == "new"))
async def on_new(query: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await state.set_state(HabitForm.title)
    await state.update_data(mode="new")
    await _say(query, "Как назовём привычку? Например: «Зарядка» или «Испанский».")
    await query.answer()


@router.message(Command("habit_new"))
async def on_new_cmd(message: Message, state: FSMContext) -> None:
    await state.clear()
    await state.set_state(HabitForm.title)
    await state.update_data(mode="new")
    await message.answer("Как назовём привычку? Например: «Зарядка» или «Испанский».")


@router.message(HabitForm.title, *TEXT_INPUT)
async def on_new_title(message: Message, state: FSMContext) -> None:
    title = " ".join((message.text or "").split())
    if not title:
        await message.answer("Напиши название.")
        return
    await state.update_data(title=title[:300])
    await state.set_state(None)
    await message.answer("Как часто?", reply_markup=_schedule_keyboard())


@router.callback_query(HabitCb.filter(F.action == "sch"))
async def on_schedule_type(
    query: CallbackQuery, callback_data: HabitCb, ctx: AppContext, state: FSMContext, update_id: int
) -> None:
    data = await state.get_data()
    if not data.get("mode"):
        await query.answer("Начни заново из меню «Привычки».")
        return
    if callback_data.arg == "weekly":
        await state.set_state(HabitForm.days)
        await _say(query, "По каким дням? Например: «пн ср пт» или «пн-пт».")
    elif callback_data.arg == "tpw":
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [_btn(str(n), HabitCb(action="tpw", arg=str(n))) for n in range(1, 4)],
                [_btn(str(n), HabitCb(action="tpw", arg=str(n))) for n in range(4, 7)],
            ]
        )
        await _say(query, "Сколько раз в неделю?", kb)
    else:
        await _after_schedule(query, ctx, state, {"type": hs.DAILY}, update_id)
    await query.answer()


@router.callback_query(HabitCb.filter(F.action == "tpw"))
async def on_times(
    query: CallbackQuery, callback_data: HabitCb, ctx: AppContext, state: FSMContext, update_id: int
) -> None:
    if not (await state.get_data()).get("mode"):
        await query.answer("Начни заново из меню «Привычки».")
        return
    await _after_schedule(
        query, ctx, state, {"type": hs.TIMES_PER_WEEK, "n": int(callback_data.arg or 3)}, update_id
    )
    await query.answer()


@router.message(HabitForm.days, *TEXT_INPUT)
async def on_days(message: Message, ctx: AppContext, state: FSMContext, update_id: int) -> None:
    days = hs.parse_weekdays(message.text or "")
    if not days:
        await message.answer("Не понял дни. Напиши так: «пн ср пт» или «пн-пт».")
        return
    await _after_schedule(message, ctx, state, {"type": hs.WEEKLY, "days": sorted(days)}, update_id)


async def _after_schedule(
    src: CallbackQuery | Message,
    ctx: AppContext,
    state: FSMContext,
    schedule: dict[str, object],
    update_id: int,
) -> None:
    data = await state.get_data()
    if data.get("mode") == "edit":
        await _apply_edit(
            src,
            ctx,
            state,
            ChangeHabit(habit_id=int(data["habit_id"]), schedule=schedule),
            update_id,
        )
        return
    await state.update_data(schedule=schedule)
    await state.set_state(HabitForm.target)
    await _send(src, TARGET_PROMPT, _skip_kb("target", "Без объёма"))


@router.message(HabitForm.target, *TEXT_INPUT)
async def on_target(message: Message, ctx: AppContext, state: FSMContext, update_id: int) -> None:
    target = hs.parse_target(message.text or "")
    if target is None:
        await message.answer("Не нашёл число. Например: «10 мин, минимум 2».")
        return
    await _after_target(message, ctx, state, target, update_id)


async def _after_target(
    src: CallbackQuery | Message,
    ctx: AppContext,
    state: FSMContext,
    target: dict[str, object],
    update_id: int,
) -> None:
    data = await state.get_data()
    if data.get("mode") == "edit":
        await _apply_edit(
            src, ctx, state, ChangeHabit(habit_id=int(data["habit_id"]), target=target), update_id
        )
        return
    await state.update_data(target=target)
    await state.set_state(HabitForm.time)
    await _send(src, TIME_PROMPT, _skip_kb("time", "Без конкретного времени"))


@router.message(HabitForm.time, *TEXT_INPUT)
async def on_time(message: Message, ctx: AppContext, state: FSMContext, update_id: int) -> None:
    try:
        at = hs.parse_hhmm(message.text or "")
    except hs.HabitError as exc:
        await message.answer(str(exc))
        return
    await _after_time(message, ctx, state, at.strftime("%H:%M") if at else "", update_id)


@router.callback_query(HabitCb.filter(F.action == "skip"))
async def on_skip(
    query: CallbackQuery, callback_data: HabitCb, ctx: AppContext, state: FSMContext, update_id: int
) -> None:
    if not (await state.get_data()).get("mode"):
        await query.answer("Начни заново из меню «Привычки».")
        return
    if callback_data.arg == "target":
        await _after_target(query, ctx, state, {}, update_id)
    else:
        await _after_time(query, ctx, state, "", update_id)
    await query.answer()


async def _after_time(
    src: CallbackQuery | Message, ctx: AppContext, state: FSMContext, at: str, update_id: int
) -> None:
    data = await state.get_data()
    if data.get("mode") == "edit":
        await _apply_edit(
            src,
            ctx,
            state,
            ChangeHabit(habit_id=int(data["habit_id"]), preferred_time=at),
            update_id,
        )
        return
    await state.clear()
    action = CreateHabit(
        title=str(data.get("title") or ""),
        schedule=dict(data.get("schedule") or {"type": hs.DAILY}),
        target=dict(data.get("target") or {}),
        preferred_time=at or None,
    )
    async with ctx.db.transaction() as s:
        res = await actions.execute(
            s, ctx, action, origin=Actor.USER, op_key=f"u{update_id}:habit_new"
        )
        if res.status is actions.ExecStatus.APPLIED:
            text, kb = await _card_or_missing(
                s, ctx, int(res.result["habit_id"]), "Привычка добавлена."
            )
        else:
            text, kb = f"Не получилось: {res.error or 'ошибка'}", None
        await mark_processed(s, update_id, ctx.clock.now())
    await _send(src, text, kb)


async def _apply_edit(
    src: CallbackQuery | Message,
    ctx: AppContext,
    state: FSMContext,
    action: ChangeHabit,
    update_id: int,
) -> None:
    """The owner's own edit: an explicit command, applied directly (origin USER)."""
    await state.clear()
    async with ctx.db.transaction() as s:
        res = await actions.execute(
            s, ctx, action, origin=Actor.USER, op_key=f"u{update_id}:habit_edit"
        )
        note = (
            "Изменил."
            if res.status is actions.ExecStatus.APPLIED
            else f"Не получилось: {res.error}"
        )
        text, kb = await _card_or_missing(s, ctx, action.habit_id, note)
        await mark_processed(s, update_id, ctx.clock.now())
    await _send(src, text, kb)


async def _send(
    src: CallbackQuery | Message, text: str, kb: InlineKeyboardMarkup | None = None
) -> None:
    if isinstance(src, Message):
        await src.answer(text, reply_markup=kb)
    else:
        await _say(src, text, kb)


@router.callback_query(HabitCb.filter(F.action == "edit"))
async def on_edit(query: CallbackQuery, callback_data: HabitCb) -> None:
    hid = callback_data.habit_id
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                _btn("Название", HabitCb(action="ef", habit_id=hid, arg="title")),
                _btn("График", HabitCb(action="ef", habit_id=hid, arg="schedule")),
            ],
            [
                _btn("Объём", HabitCb(action="ef", habit_id=hid, arg="target")),
                _btn("Время", HabitCb(action="ef", habit_id=hid, arg="time")),
            ],
            [_btn("← К привычке", HabitCb(action="view", habit_id=hid))],
        ]
    )
    await _edit(query, "Что изменить?", kb)
    await query.answer()


@router.callback_query(HabitCb.filter(F.action == "ef"))
async def on_edit_field(query: CallbackQuery, callback_data: HabitCb, state: FSMContext) -> None:
    hid, field = callback_data.habit_id, callback_data.arg
    await state.clear()
    await state.update_data(mode="edit", habit_id=hid)
    if field == "title":
        await state.set_state(HabitForm.rename)
        await _say(query, "Новое название:")
    elif field == "schedule":
        await _say(query, "Как часто?", _schedule_keyboard())
    elif field == "target":
        await state.set_state(HabitForm.target)
        await _say(query, TARGET_PROMPT, _skip_kb("target", "Убрать объём"))
    else:
        await state.set_state(HabitForm.time)
        await _say(query, TIME_PROMPT, _skip_kb("time", "Убрать время"))
    await query.answer()


@router.message(HabitForm.rename, *TEXT_INPUT)
async def on_rename(message: Message, ctx: AppContext, state: FSMContext, update_id: int) -> None:
    data = await state.get_data()
    await state.clear()
    hid = int(data.get("habit_id") or 0)
    async with ctx.db.transaction() as s:
        try:
            await hs.update_habit(s, ctx, hid, actor=Actor.USER, title=message.text or "")
            note = "Переименовал."
        except hs.HabitError as exc:
            note = str(exc)
        text, kb = await _card_or_missing(s, ctx, hid, note)
        await mark_processed(s, update_id, ctx.clock.now())
    await message.answer(text, reply_markup=kb)
