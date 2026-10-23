"""«🧠 Память» (FR-12, spec section 7): view, confirm, refute, correct, delete,
full wipe and export.

Every record shows where it came from: «вы сказали / измерено / предположение
ИИ / подтверждено / опровергнуто», plus its primary sources. Changes are applied
in one transaction (summaries built from the record are invalidated in the same
transaction) and only then answered. Repeated presses are no-ops.
"""

from __future__ import annotations

from aiogram import F, Router
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from procbot.context import AppContext
from procbot.db.models import MemoryItem
from procbot.domain.enums import MemoryKind, MemoryOrigin, MemoryStatus
from procbot.services import memory, privacy
from procbot.services.profile import get_or_create_profile, get_profile
from procbot.telegram.callbacks import MemoryCb, SettingsCb
from procbot.telegram.keyboards import Menu
from procbot.telegram.middlewares import mark_processed
from procbot.timeutil import fmt_local

router = Router(name="memory")

LIST_LIMIT = 15
REFUTED = "refuted"


class MemoryForm(StatesGroup):
    edit = State()  # new text for a record (correct)
    fix = State()  # "how it really is" after refuting a hypothesis
    wipe_all = State()  # typed confirmation phrase for the full wipe


def _b(text: str, action: str, item_id: int = 0, arg: str = "") -> InlineKeyboardButton:
    return InlineKeyboardButton(
        text=text, callback_data=MemoryCb(action=action, item_id=item_id, arg=arg).pack()
    )


def _kb(*rows: list[InlineKeyboardButton]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[r for r in rows if r])


async def _show(target: Message, text: str, kb: InlineKeyboardMarkup | None, *, edit: bool) -> None:
    if edit:
        try:
            await target.edit_text(text, reply_markup=kb)
            return
        except TelegramBadRequest:
            pass
    await target.answer(text, reply_markup=kb)


def _short(text: str, n: int = 70) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def origin_label(item: MemoryItem) -> str:
    return memory.ORIGIN_LABELS.get(item.origin, item.origin)


# --------------------------------------------------------------------------- views


async def home_view(ctx: AppContext) -> tuple[str, InlineKeyboardMarkup]:
    async with ctx.db.transaction() as s:
        c = await memory.counts(s)
        flags = await memory.memory_flags(s)
        summaries = len(await memory.valid_summaries(s))
    by_kind, by_origin = c["kind"], c["origin"]
    lines = ["🧠 Память", ""]
    if not flags.enabled:
        lines.append(
            "Долговременная память выключена (⚙️ Настройки → Память): новое не запоминается."
        )
    total = sum(by_kind.values())
    if total == 0:
        lines.append("Пока ничего не запомнено.")
    else:
        lines.append(f"Действующих записей: {total}")
        for kind in MemoryKind:
            if by_kind.get(kind):
                lines.append(f"• {memory.KIND_LABELS_PLURAL[kind]}: {by_kind[kind]}")
        origins = ", ".join(
            f"{memory.ORIGIN_LABELS[o]} — {by_origin[o]}" for o in MemoryOrigin if by_origin.get(o)
        )
        if origins:
            lines.append(f"Происхождение: {origins}")
    refuted = c["status"].get(MemoryStatus.REFUTED, 0)
    if refuted:
        lines.append(f"Опровергнуто (не используется): {refuted}")
    lines.append(f"Сводок для контекста ИИ: {summaries}")
    kb = _kb(
        [
            _b("Факты", "list", arg=MemoryKind.FACT),
            _b("Предпочтения", "list", arg=MemoryKind.PREFERENCE),
        ],
        [
            _b("Закономерности", "list", arg=MemoryKind.PATTERN),
            _b("Гипотезы", "list", arg=MemoryKind.HYPOTHESIS),
        ],
        [_b("🙅 Опровергнутые", "list", arg=REFUTED), _b("👤 Профиль", "profile")],
        [_b("📤 Экспорт данных", "export"), _b("🧹 Очистить…", "wipe")],
    )
    return "\n".join(lines), kb


async def list_view(ctx: AppContext, arg: str) -> tuple[str, InlineKeyboardMarkup]:
    async with ctx.db.transaction() as s:
        if arg == REFUTED:
            items = await memory.list_items(
                s, statuses=[MemoryStatus.REFUTED], limit=LIST_LIMIT + 1
            )
            title = "🙅 Опровергнутые (в контекст ИИ не попадают)"
        else:
            kind = MemoryKind(arg)
            items = await memory.list_items(s, kinds=[kind], limit=LIST_LIMIT + 1)
            title = f"🧠 {memory.KIND_LABELS_PLURAL[kind]}"
    more = len(items) > LIST_LIMIT
    items = items[:LIST_LIMIT]
    lines = [title, ""]
    if not items:
        lines.append("Здесь пока пусто.")
    for it in items:
        lines.append(f"#{it.id} {_short(it.content)} — {origin_label(it)}")
    if more:
        lines.append(f"Показаны последние {LIST_LIMIT}.")
    buttons = [_b(f"#{it.id}", "view", it.id) for it in items]
    rows = [buttons[i : i + 5] for i in range(0, len(buttons), 5)]
    rows.append([_b("« Назад", "home")])
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


async def item_view(ctx: AppContext, item_id: int) -> tuple[str, InlineKeyboardMarkup]:
    async with ctx.db.transaction() as s:
        item = await memory.get_item(s, item_id)
        if item is None or item.status == MemoryStatus.DELETED:
            return "Запись не найдена (возможно, уже удалена).", _kb([_b("« Назад", "home")])
        p = await get_profile(s)
        tz = p.timezone if p else None
        sources = [
            await memory.describe_source(s, x, tz) for x in await memory.item_sources(s, item.id)
        ]
    lines = [
        f"{memory.KIND_LABELS.get(item.kind, item.kind)} #{item.id}",
        "",
        item.content,
        "",
        f"Происхождение: {origin_label(item)}",
        f"Статус: {memory.STATUS_LABELS.get(item.status, item.status)}",
    ]
    if item.confidence is not None:
        lines.append(f"Уверенность: {round(item.confidence * 100)}%")
    if item.verification_hint:
        lines.append(f"Как проверить: {item.verification_hint}")
    lines.append(
        f"Записано: {fmt_local(item.created_at, tz)}, изменено: {fmt_local(item.updated_at, tz)}"
    )
    lines.append("")
    if sources:
        lines.append("Основания:")
        lines += [f"• {x}" for x in sources[:10]]
    else:
        lines.append("Основания: не привязаны.")
    first: list[InlineKeyboardButton] = []
    if not (item.status == MemoryStatus.ACTIVE and item.origin == MemoryOrigin.USER_CONFIRMED):
        first.append(_b("✅ Подтвердить", "confirm", item.id))
    if item.status != MemoryStatus.REFUTED:
        first.append(_b("🙅 Опровергнуть", "refute", item.id))
    kb = _kb(
        first,
        [_b("✏️ Исправить", "edit", item.id), _b("🗑 Удалить", "del", item.id)],
        [
            _b(
                "« К списку",
                "list",
                arg=REFUTED if item.status == MemoryStatus.REFUTED else item.kind,
            )
        ],
    )
    return "\n".join(lines), kb


async def profile_view(ctx: AppContext) -> tuple[str, InlineKeyboardMarkup]:
    from procbot.telegram.handlers.settings import overview_text

    async with ctx.db.transaction() as s:
        p = await get_profile(s)
        if p is None:
            p, _ = await get_or_create_profile(
                s, telegram_user_id=ctx.owner_id, now=ctx.clock.now()
            )
    text = "👤 Профиль\n\n" + overview_text(ctx, p)
    if p.goals_note:
        text += f"\n\nЦели из знакомства: {p.goals_note}"
    kb = _kb(
        [
            InlineKeyboardButton(
                text="⚙️ Изменить в настройках", callback_data=SettingsCb(action="home").pack()
            )
        ],
        [_b("« Назад", "home")],
    )
    return text, kb


def wipe_view(ctx: AppContext) -> tuple[str, InlineKeyboardMarkup]:
    text = (
        "🧹 Очистка данных\n\n"
        "• «Память» — все записи памяти и сводки для ИИ.\n"
        "• «Память и история» — плюс история диалогов (тексты сообщений стираются).\n"
        "• «Всё, включая задачи» — плюс задачи, планы, сессии, цели и привычки. "
        "Настройки профиля и ключи остаются.\n\n" + privacy.retention_notice(ctx)
    )
    kb = _kb(
        [_b("🧠 Память", "wipe_ask", arg="mem")],
        [_b("🧠 Память и история диалогов", "wipe_ask", arg="hist")],
        [_b("⚠️ Всё, включая задачи", "wipe_ask", arg="all")],
        [_b("« Назад", "home")],
    )
    return text, kb


# --------------------------------------------------------------------------- entry


@router.message(Command("memory"))
@router.message(F.text == Menu.MEMORY)
async def on_memory(message: Message, ctx: AppContext, state: FSMContext) -> None:
    await state.clear()
    text, kb = await home_view(ctx)
    await message.answer(text, reply_markup=kb)


@router.message(Command("export"))
async def on_export_cmd(message: Message, ctx: AppContext) -> None:
    await _send_export(message, ctx)


async def _send_export(message: Message, ctx: AppContext) -> None:
    name, data = await privacy.export_json(ctx)
    await message.answer_document(
        BufferedInputFile(data, filename=name),
        caption=(
            "📤 Экспорт: профиль, задачи, планы, сессии, память и история диалогов (JSON). "
            "Ключей ИИ и токенов в нём нет."
        ),
    )


# --------------------------------------------------------------------------- callbacks


@router.callback_query(MemoryCb.filter())
async def on_memory_cb(
    query: CallbackQuery,
    callback_data: MemoryCb,
    ctx: AppContext,
    state: FSMContext,
    update_id: int,
) -> None:
    msg = query.message if isinstance(query.message, Message) else None
    action, item_id, arg = callback_data.action, callback_data.item_id, callback_data.arg
    if msg is None:
        await query.answer()
        return
    if action == "home":
        await state.clear()
        await query.answer()
        text, kb = await home_view(ctx)
        await _show(msg, text, kb, edit=True)
        return
    if action == "list":
        await query.answer()
        if arg != REFUTED and arg not in {k.value for k in MemoryKind}:
            return
        text, kb = await list_view(ctx, arg)
        await _show(msg, text, kb, edit=True)
        return
    if action == "view":
        await query.answer()
        text, kb = await item_view(ctx, item_id)
        await _show(msg, text, kb, edit=True)
        return
    if action == "profile":
        await query.answer()
        text, kb = await profile_view(ctx)
        await _show(msg, text, kb, edit=True)
        return
    if action in ("confirm", "refute"):
        await _confirm_or_refute(
            query, msg, ctx, action, item_id, from_report=arg == "wk", update_id=update_id
        )
        return
    if action == "edit":
        await state.set_state(MemoryForm.edit)
        await state.update_data(memory_item_id=item_id)
        await query.answer()
        await msg.answer("Напиши, как должно быть записано. /cancel — отмена.")
        return
    if action == "fix":
        await state.set_state(MemoryForm.fix)
        await state.update_data(memory_item_id=item_id)
        await query.answer()
        await msg.answer("Как на самом деле? Напиши своими словами. /cancel — отмена.")
        return
    if action == "del":
        await query.answer()
        await _show(
            msg,
            "Удалить запись? Текст будет стёрт, сводки с ней — удалены.\n\n"
            + privacy.retention_notice(ctx),
            _kb([_b("🗑 Да, удалить", "delok", item_id), _b("Отмена", "view", item_id)]),
            edit=True,
        )
        return
    if action == "delok":
        async with ctx.db.transaction() as s:
            deleted = await memory.delete_item(s, ctx, item_id)
            await mark_processed(s, update_id, ctx.clock.now())
        await query.answer("Удалено" if deleted else "Уже удалено")
        text, kb = await home_view(ctx)
        await _show(msg, ("🗑 Запись удалена.\n\n" if deleted else "") + text, kb, edit=True)
        return
    if action == "wipe":
        await query.answer()
        text, kb = wipe_view(ctx)
        await _show(msg, text, kb, edit=True)
        return
    if action == "wipe_ask":
        await query.answer()
        if arg == "all":
            await state.set_state(MemoryForm.wipe_all)
            await msg.answer(
                "⚠️ Будут удалены задачи, планы, сессии, цели, привычки, память и история "
                "диалогов. Это нельзя отменить.\n"
                f"Чтобы подтвердить, напиши: {privacy.WIPE_ALL_PHRASE}\n"
                "Любой другой ответ — отмена."
            )
            return
        what = "всю память и сводки" if arg == "mem" else "всю память, сводки и историю диалогов"
        await _show(
            msg,
            f"Удалить {what}? Это нельзя отменить.\n\n" + privacy.retention_notice(ctx),
            _kb([_b("🧹 Да, удалить", "wipe_ok", arg=arg), _b("Отмена", "home")]),
            edit=True,
        )
        return
    if action == "wipe_ok":
        if arg not in ("mem", "hist"):
            await query.answer()
            return
        async with ctx.db.transaction() as s:
            res = await memory.wipe_all(s, ctx, include_history=arg == "hist")
            await mark_processed(s, update_id, ctx.clock.now())
        await query.answer("Очищено")
        extra = f", сообщений истории: {res.messages}" if arg == "hist" else ""
        await _show(
            msg,
            f"🧹 Готово. Удалено записей памяти: {res.items}, сводок: {res.summaries}{extra}.\n\n"
            + privacy.retention_notice(ctx),
            _kb([_b("« К памяти", "home")]),
            edit=True,
        )
        return
    if action == "export":
        await query.answer("Готовлю файл…")
        await _send_export(msg, ctx)
        return
    await query.answer()


async def _confirm_or_refute(
    query: CallbackQuery,
    msg: Message,
    ctx: AppContext,
    action: str,
    item_id: int,
    *,
    from_report: bool,
    update_id: int,
) -> None:
    try:
        async with ctx.db.transaction() as s:
            if action == "confirm":
                res = await memory.confirm(s, ctx, item_id)
            else:
                res = await memory.refute(s, ctx, item_id)
            await mark_processed(s, update_id, ctx.clock.now())
    except memory.MemoryServiceError as exc:
        await query.answer(str(exc)[:200], show_alert=True)
        return
    if action == "confirm":
        await query.answer("Подтверждено" if res.changed else "Уже подтверждено")
    else:
        await query.answer("Опровергнуто" if res.changed else "Уже опровергнуто")
    if from_report:
        # scenario H: keep the report message, answer below it
        text = (
            f"🙅 Понял: «{_short(res.item.content, 120)}» — опровергнуто. Больше не использую это "
            "в разговорах и отчётах; сводки памяти пересоберутся без него."
        )
        await msg.answer(
            text,
            reply_markup=_kb(
                [_b("✏️ Написать, как на самом деле", "fix", item_id)], [_b("🧠 Память", "home")]
            ),
        )
        return
    text, kb = await item_view(ctx, item_id)
    await _show(msg, text, kb, edit=True)


# --------------------------------------------------------------------------- text input


@router.message(StateFilter(MemoryForm), F.text)
async def on_memory_text(
    message: Message, ctx: AppContext, state: FSMContext, update_id: int
) -> None:
    text = (message.text or "").strip()
    current = await state.get_state()
    data = await state.get_data()
    if text in Menu.ALL or (text.startswith("/") and current != MemoryForm.wipe_all.state):
        await state.clear()
        raise SkipHandler()
    await state.clear()
    if current == MemoryForm.wipe_all.state:
        if text != privacy.WIPE_ALL_PHRASE:
            await message.answer("Отменено, ничего не удалено.")
            return
        async with ctx.db.transaction() as s:
            counts = await privacy.wipe_everything(s, ctx)
            await mark_processed(s, update_id, ctx.clock.now())
        await message.answer(
            f"🧹 Всё удалено: задач {counts['tasks']}, записей памяти {counts['memory_items']}, "
            f"сообщений истории {counts['messages']}.\n\n" + privacy.retention_notice(ctx)
        )
        return
    item_id = int(data.get("memory_item_id") or 0)
    try:
        async with ctx.db.transaction() as s:
            if current == MemoryForm.edit.state:
                res = await memory.correct(s, ctx, item_id, text)
                reply = (
                    "✏️ Исправлено и отмечено как подтверждённое вами."
                    if res.changed
                    else "Без изменений."
                )
                item_ref = res.item.id
            else:
                refuted = await memory.require_item(s, item_id)
                kind = (
                    MemoryKind.PATTERN
                    if refuted.kind in (MemoryKind.HYPOTHESIS, MemoryKind.PATTERN)
                    else MemoryKind(refuted.kind)
                )
                sources = [memory.SourceRef(memory.SourceType.MEMORY_ITEM, refuted.id)]
                created = await memory.add_fact(
                    s,
                    ctx,
                    text,
                    kind=kind,
                    sources=[*sources, memory.SourceRef(memory.SourceType.UPDATE, update_id)],
                )
                if created is None:
                    reply = (
                        "Память выключена — поправка не сохранена (опровержение осталось в силе)."
                    )
                    item_ref = 0
                else:
                    reply = "Записал как ваше утверждение («вы сказали»). Спасибо за поправку!"
                    item_ref = created.id
            await mark_processed(s, update_id, ctx.clock.now())
    except memory.MemoryServiceError as exc:
        await message.answer(str(exc))
        return
    kb = _kb([_b("Открыть запись", "view", item_ref)]) if item_ref else None
    await message.answer(reply, reply_markup=kb)
