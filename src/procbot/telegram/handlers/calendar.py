"""/calendar: connect Google Calendar, pick calendars, status, sync now, disconnect.

Thin handler over services.calendar / services.calendar_oauth. Also offers
"create this block in the calendar" for a plan item or a work session; that only
creates an approval card (the insert happens after the owner accepts the card).
"""

from __future__ import annotations

import logging
from datetime import timedelta

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from procbot.context import AppContext
from procbot.db.models import PlanItem, Task, WorkSession
from procbot.domain.enums import Actor
from procbot.services import approvals
from procbot.services import calendar as cal
from procbot.services import calendar_oauth as oauth
from procbot.services.actions import ExecStatus
from procbot.services.calendar_google import CalendarApiError
from procbot.services.tasks import owner_tz
from procbot.telegram.callbacks import CalendarCb
from procbot.telegram.handlers.approvals import send_approval_card
from procbot.telegram.middlewares import mark_processed
from procbot.timeutil import fmt_local

log = logging.getLogger(__name__)

router = Router(name="calendar")

DOCS_HINT = "docs/deploy.md, раздел «Google Calendar»"
NOT_CONFIGURED = (
    "📅 Google Calendar не настроен на сервере.\n\n"
    "Чтобы подключить календарь, владелец сервера должен создать OAuth-клиент в Google "
    "Cloud Console и вписать в .env:\n"
    "• GOOGLE_CLIENT_ID\n"
    "• GOOGLE_CLIENT_SECRET\n"
    "• GOOGLE_REDIRECT_URI — https://<ваш-домен>/oauth/google/callback, либо пусто "
    "(тогда подключение через копирование адреса из браузера)\n"
    "и перезапустить бота. Подробно: " + DOCS_HINT + ".\n\n"
    "Без календаря бот работает по локальному расписанию."
)
ROLE_RU = {
    "owner": "владелец",
    "writer": "запись",
    "reader": "чтение",
    "freeBusyReader": "занятость",
}


class CalendarConnect(StatesGroup):
    paste = State()


def _btn(text: str, action: str, arg: str = "") -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=CalendarCb(action=action, arg=arg).pack())


def offer_button(
    *, plan_item_id: int | None = None, session_id: int | None = None
) -> InlineKeyboardButton:
    """Button other modules can put under a proposed plan item / work session.

    Pressing it shows an approval card; nothing is created before the owner accepts."""
    arg = f"pi{plan_item_id}" if plan_item_id is not None else f"ws{session_id}"
    return _btn("📅 В календарь", "offer", arg)


async def _show(target: Message, text: str, kb: InlineKeyboardMarkup | None, *, edit: bool) -> None:
    if edit:
        try:
            await target.edit_text(text, reply_markup=kb)
            return
        except TelegramBadRequest as exc:
            if "not modified" in str(exc):
                return
    await target.answer(text, reply_markup=kb)


def _cq_message(query: CallbackQuery) -> Message | None:
    return query.message if isinstance(query.message, Message) else None


# --------------------------------------------------------------------------- views


def _scope_text(scopes: list[str]) -> str:
    parts: list[str] = []
    if cal.can_read_events(scopes):
        parts.append("чтение событий")
    roles = cal.writable_roles(scopes)
    if "writer" in roles:
        parts.append("создание событий")
    elif "owner" in roles:
        parts.append("создание событий в своих календарях")
    return ", ".join(parts) or "—"


async def status_view(ctx: AppContext) -> tuple[str, InlineKeyboardMarkup | None]:
    async with ctx.db.transaction() as s:
        st = await cal.connection_status(s, ctx)
        tz = await owner_tz(s)
    if not st.configured:
        return NOT_CONFIGURED, None
    lines = ["📅 Google Calendar"]
    if st.connected:
        lines.append("Состояние: подключён")
    elif st.needs_reauth:
        lines.append("Состояние: нужно подключить заново (Google отозвал доступ)")
    else:
        lines.append("Состояние: не подключён")
        lines.append("Без календаря бот планирует по локальному расписанию.")
        return "\n".join(lines), InlineKeyboardMarkup(
            inline_keyboard=[[_btn("🔗 Подключить", "connect")]]
        )
    if st.read_calendar_ids:
        names = ", ".join(st.name_of(c) for c in st.read_calendar_ids)
        lines.append(f"Читаю занятость из: {names}")
    else:
        lines.append("Календари для чтения не выбраны — бот пока не видит встречи.")
    if st.write_calendar_id:
        lines.append(f"Создаю события (после подтверждения) в: {st.name_of(st.write_calendar_id)}")
    elif st.write_declined:
        lines.append("Создание событий: выключено")
    else:
        lines.append("Календарь для создания событий: не выбран")
    lines.append(f"Права: {_scope_text(st.scopes)}")
    if st.last_sync_at is not None:
        lines.append(f"Последняя синхронизация: {fmt_local(st.last_sync_at, tz)}")
    elif st.read_calendar_ids:
        lines.append("Синхронизации ещё не было.")
    if st.stale:
        lines.append(cal.STALE_NOTE)
    if st.last_error:
        lines.append(f"Последняя ошибка: {st.last_error}")
    rows = [
        [_btn("📖 Читать календари", "read"), _btn("✍️ Куда создавать", "write")],
        [_btn("🔄 Синхронизировать", "sync"), _btn("🔌 Отключить", "disc")],
    ]
    if st.needs_reauth:
        rows.insert(0, [_btn("🔗 Подключить заново", "connect")])
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


async def read_picker(ctx: AppContext) -> tuple[str, InlineKeyboardMarkup]:
    async with ctx.db.transaction() as s:
        st = await cal.connection_status(s, ctx)
    lines = [
        "📖 Из каких календарей учитывать занятость?",
        "Отметь нужные (можно несколько). Заранее ничего не выбрано.",
    ]
    rows: list[list[InlineKeyboardButton]] = []
    for i, c in enumerate(st.calendars):
        mark = "✅" if c.id in st.read_calendar_ids else "⬜"
        suffix = " (основной)" if c.primary else ""
        rows.append([_btn(f"{mark} {c.name}{suffix}"[:60], "rd", str(i))])
    if not st.calendars:
        lines.append("Список календарей пуст — обнови его.")
    rows.append([_btn("🔄 Обновить список", "rlist"), _btn("Готово", "rdone")])
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


async def write_picker(ctx: AppContext) -> tuple[str, InlineKeyboardMarkup]:
    async with ctx.db.transaction() as s:
        st = await cal.connection_status(s, ctx)
    writable = cal.writable_calendars(st)
    lines = [
        "✍️ В какой календарь создавать рабочие блоки?",
        "Каждое событие создаётся только после отдельного подтверждения; существующие "
        "события бот не меняет и не удаляет.",
    ]
    rows: list[list[InlineKeyboardButton]] = []
    for c in writable:
        i = st.calendars.index(c)
        mark = "🔘" if c.id == st.write_calendar_id else "⚪"
        rows.append([_btn(f"{mark} {c.name}"[:60], "wr", str(i))])
    if not writable:
        lines.append("Нет календарей, в которые выданы права записи.")
    mark = "🔘" if st.write_declined and not st.write_calendar_id else "⚪"
    rows.append([_btn(f"{mark} Не создавать события", "wr", "none")])
    rows.append([_btn("← Назад", "menu")])
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


# --------------------------------------------------------------------------- command


@router.message(Command("calendar"))
async def cmd_calendar(message: Message, ctx: AppContext, state: FSMContext) -> None:
    if await state.get_state() == CalendarConnect.paste.state:
        await state.clear()
    text, kb = await status_view(ctx)
    await message.answer(text, reply_markup=kb)


async def _start_connect(msg: Message, ctx: AppContext, state: FSMContext) -> None:
    if cal.get_service(ctx) is None:
        await _show(msg, NOT_CONFIGURED, None, edit=True)
        return
    start = await oauth.start_authorization(ctx)
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔐 Войти через Google", url=start.url)],
            [_btn("Отмена", "cancel")],
        ]
    )
    minutes = ctx.config.calendar.oauth_state_ttl_minutes
    if start.mode == oauth.RedirectMode.WEB:
        text = (
            "Открой ссылку, войди в Google и разреши доступ к календарю. После этого "
            f"вернись сюда — я напишу. Ссылка действует {minutes} мин."
        )
    else:
        await state.set_state(CalendarConnect.paste)
        text = (
            "1. Открой ссылку, войди в Google и разреши доступ к календарю.\n"
            f"2. Браузер перейдёт на адрес {start.redirect_uri}… и покажет ошибку "
            "«не удаётся открыть страницу» — это нормально.\n"
            "3. Скопируй адрес из адресной строки целиком и пришли его сюда одним "
            "сообщением. Я сразу удалю это сообщение из чата.\n"
            f"Ссылка действует {minutes} мин."
        )
    await _show(msg, text, kb, edit=False)


@router.message(StateFilter(CalendarConnect.paste), F.text, ~F.text.startswith("/"))
async def on_paste(
    message: Message, ctx: AppContext, state: FSMContext, update_id: int | None = None
) -> None:
    # The URL contains a one-time code: keep it out of the chat and the history.
    if update_id is not None:
        ctx.registry.setdefault("redacted_update_ids", set()).add(update_id)
    await ctx.messenger.delete_message(message.chat.id, message.message_id)
    code, st_param, error = oauth.parse_pasted(message.text or "")
    if error:
        await oauth.cancel_on_error(ctx, st_param)
        await state.clear()
        await message.answer("Google вернул отказ — календарь не подключён. /calendar")
        return
    if code is None:
        await message.answer(
            "Не вижу в сообщении кода от Google. Пришли адрес из адресной строки целиком "
            "(он содержит code=…), или нажми «Отмена»."
        )
        return
    await state.clear()
    res = await oauth.complete_authorization(ctx, code=code, state=st_param, require_state=False)
    if not res.ok:
        await message.answer(res.message)
        return
    text, kb = await read_picker(ctx)
    await message.answer(res.message + "\n\n" + text, reply_markup=kb)


# --------------------------------------------------------------------------- buttons


@router.callback_query(CalendarCb.filter())
async def on_calendar_cb(
    query: CallbackQuery,
    callback_data: CalendarCb,
    ctx: AppContext,
    state: FSMContext,
    update_id: int,
) -> None:
    msg = _cq_message(query)
    if msg is None:
        await query.answer()
        return
    action, arg = callback_data.action, callback_data.arg
    if action == "offer":
        await _offer(query, msg, ctx, arg, update_id)
        return
    if cal.get_service(ctx) is None:
        await query.answer()
        await _show(msg, NOT_CONFIGURED, None, edit=True)
        return
    if action == "connect":
        await query.answer()
        await _start_connect(msg, ctx, state)
    elif action == "cancel":
        await state.clear()
        await query.answer("Отменено")
        text, kb = await status_view(ctx)
        await _show(msg, text, kb, edit=True)
    elif action == "read":
        await query.answer()
        text, kb = await read_picker(ctx)
        await _show(msg, text, kb, edit=True)
    elif action == "rlist":
        try:
            await cal.refresh_calendar_list(ctx)
            await query.answer("Список обновлён")
        except (CalendarApiError, cal.CalendarNotConnected):
            await query.answer("Не удалось получить список календарей")
        text, kb = await read_picker(ctx)
        await _show(msg, text, kb, edit=True)
    elif action == "rd":
        await _toggle_read(query, msg, ctx, arg)
    elif action == "rdone":
        await _read_done(query, msg, ctx)
    elif action == "write":
        await query.answer()
        text, kb = await write_picker(ctx)
        await _show(msg, text, kb, edit=True)
    elif action == "wr":
        await _choose_write(query, msg, ctx, arg)
    elif action == "sync":
        await query.answer("Синхронизирую…")
        await _sync_and_show(msg, ctx)
    elif action == "disc":
        await query.answer()
        kb = InlineKeyboardMarkup(
            inline_keyboard=[[_btn("Да, отключить", "disc_yes"), _btn("Нет", "menu")]]
        )
        await _show(
            msg,
            "Отключить Google Calendar? Токены доступа и кэш событий будут удалены, "
            "доступ бота в Google отозван. События в календаре не изменятся.",
            kb,
            edit=True,
        )
    elif action == "disc_yes":
        revoked = await cal.disconnect(ctx)
        await query.answer("Отключено")
        note = (
            ""
            if revoked
            else "\n(Отозвать доступ в Google не удалось — проверь https://myaccount.google.com/permissions)"
        )
        text, kb = await status_view(ctx)
        await _show(
            msg, "Календарь отключён, токены удалены." + note + "\n\n" + text, kb, edit=True
        )
    else:  # "menu" and unknown
        await query.answer()
        text, kb = await status_view(ctx)
        await _show(msg, text, kb, edit=True)


async def _calendar_at(ctx: AppContext, arg: str) -> str | None:
    if not arg.isdigit():
        return None
    async with ctx.db.transaction() as s:
        cals = await cal.available_calendars(s)
    i = int(arg)
    return cals[i].id if 0 <= i < len(cals) else None


async def _toggle_read(query: CallbackQuery, msg: Message, ctx: AppContext, arg: str) -> None:
    cid = await _calendar_at(ctx, arg)
    result: bool | None = None
    if cid is not None:
        async with ctx.db.transaction() as s:
            result = await cal.toggle_read_calendar(s, ctx, cid)
    await query.answer(
        "Календарь не найден — обнови список"
        if result is None
        else ("Выбран" if result else "Снят")
    )
    text, kb = await read_picker(ctx)
    await _show(msg, text, kb, edit=True)


async def _read_done(query: CallbackQuery, msg: Message, ctx: AppContext) -> None:
    async with ctx.db.transaction() as s:
        st = await cal.connection_status(s, ctx)
    if not st.write_calendar_id and not st.write_declined:
        await query.answer()
        text, kb = await write_picker(ctx)
        await _show(msg, text, kb, edit=True)
        return
    await query.answer("Синхронизирую…")
    await _sync_and_show(msg, ctx)


async def _choose_write(query: CallbackQuery, msg: Message, ctx: AppContext, arg: str) -> None:
    cid = None if arg == "none" else await _calendar_at(ctx, arg)
    ok = False
    if arg == "none" or cid is not None:
        async with ctx.db.transaction() as s:
            ok = await cal.set_write_calendar(s, ctx, cid)
    if not ok:
        await query.answer("В этот календарь нельзя создавать события")
        text, kb = await write_picker(ctx)
        await _show(msg, text, kb, edit=True)
        return
    await query.answer("Сохранено")
    await _sync_and_show(msg, ctx)


async def _sync_and_show(msg: Message, ctx: AppContext) -> None:
    report = await cal.sync_calendars(ctx)
    if report.skipped == "no_calendars":
        note = "Календари для чтения не выбраны — синхронизировать нечего."
    elif report.skipped == "busy":
        note = "Синхронизация уже идёт."
    elif report.skipped:
        note = "Синхронизация не выполнена: календарь не подключён."
    elif report.errors:
        note = "⚠️ Часть календарей не загрузилась; использую сохранённые данные."
    else:
        note = f"🔄 Синхронизировано: календарей {report.calendars}, событий {report.events}."
    text, kb = await status_view(ctx)
    await _show(msg, note + "\n\n" + text, kb, edit=True)


# --------------------------------------------------------------------------- offer a block


async def _offer(
    query: CallbackQuery, msg: Message, ctx: AppContext, arg: str, update_id: int
) -> None:
    """Create an approval card "create this block in the calendar"."""
    error: str | None = None
    row = None
    tz: str | None = None
    async with ctx.db.transaction() as s:
        title, start, end, task_id, plan_item_id = None, None, None, None, None
        if arg.startswith("pi") and arg[2:].isdigit():
            item = await s.get(PlanItem, int(arg[2:]))
            if item is not None:
                title, start, end = item.title, item.start_at, item.end_at
                task_id, plan_item_id = item.task_id, item.id
        elif arg.startswith("ws") and arg[2:].isdigit():
            ws = await s.get(WorkSession, int(arg[2:]))
            minutes = None if ws is None else (ws.planned_minutes or ws.work_minutes)
            if ws is not None and ws.planned_start_at is not None and minutes:
                task = await s.get(Task, ws.task_id) if ws.task_id else None
                title = task.title if task is not None else "Рабочая сессия"
                start = ws.planned_start_at
                end = start + timedelta(minutes=minutes)
                task_id = ws.task_id
        if title is None or start is None or end is None:
            error = "У этого блока нет точного времени — создать событие нельзя."
        else:
            res = await cal.propose_calendar_block(
                s,
                ctx,
                title=title,
                start_at=start,
                end_at=end,
                task_id=task_id,
                plan_item_id=plan_item_id,
                origin=Actor.SYSTEM,
                op_key=f"u{update_id}:calendar:offer",
            )
            if res.status == ExecStatus.PENDING_APPROVAL and res.approval_id is not None:
                row = await approvals.get_approval(s, res.approval_id)
                tz = await owner_tz(s)
            else:
                error = res.error or "Не получилось предложить событие."
        await mark_processed(s, update_id, ctx.clock.now())
    if error is not None or row is None:
        await query.answer((error or "Не получилось")[:200], show_alert=True)
        return
    await query.answer()
    await send_approval_card(msg, ctx, [row], tz=tz, header="Создать событие в календаре?")
