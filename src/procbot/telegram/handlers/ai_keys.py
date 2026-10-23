"""/ai: AI keys, models per tier, AI spend (FR-14, architecture "Ключи").

Thin handler over procbot.ai.gateway. Keys are shown only as ****last4.
Adding a key via chat works only when ai.yaml `features.allow_keys_via_chat`
is true (open question); the message with the key is deleted immediately and
the key is never put into FSM data, logs or replies.
"""

from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from procbot.ai.cost import fmt_usd, month_spend_by_credential, owner_tz, spend_report
from procbot.ai.credentials import CredentialState
from procbot.ai.gateway import AIGateway, get_gateway
from procbot.context import AppContext
from procbot.domain.enums import CredentialStatus
from procbot.telegram.callbacks import AIKeyCb
from procbot.timeutil import fmt_local

log = logging.getLogger(__name__)

router = Router(name="ai_keys")

CHAT_PROVIDERS: dict[str, str] = {
    "openai": "OpenAI",
    "openai_compatible": "OpenAI-совместимый (по base_url)",
    "anthropic": "Anthropic",
    "gemini": "Google Gemini",
}

STATUS_RU = {
    CredentialStatus.ACTIVE: "активен",
    CredentialStatus.COOLING_DOWN: "остывает",
    CredentialStatus.DISABLED: "отключён",
}

NO_GATEWAY = "ИИ-шлюз не запущен: бот работает без ИИ (задачи, сессии и напоминания доступны)."
ENV_ONLY = (
    "Добавление ключей через чат выключено (features.allow_keys_via_chat в config/ai.yaml).\n"
    "Ключи добавляются на сервере: впишите ключ в .env (например OPENAI_API_KEY=...), "
    "укажите эту переменную в config/ai.yaml в разделе credentials (api_key_env) и "
    "перезапустите бота. Ключ сохранится в базе в зашифрованном виде."
)


class AddKey(StatesGroup):
    base_url = State()
    key = State()


def _btn(text: str, action: str, credential_id: int = 0, arg: str = "") -> InlineKeyboardButton:
    return InlineKeyboardButton(
        text=text, callback_data=AIKeyCb(action=action, credential_id=credential_id, arg=arg).pack()
    )


def _menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [_btn("🔑 Ключи", "keys"), _btn("🧩 Модели", "models")],
            [_btn("💸 Расходы ИИ", "spend", arg="month"), _btn("➕ Добавить ключ", "add")],
        ]
    )


def _back_kb(*extra: list[InlineKeyboardButton]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[*extra, [_btn("← Назад", "menu")]])


async def _show(target: Message, text: str, kb: InlineKeyboardMarkup | None, *, edit: bool) -> None:
    if edit:
        try:
            await target.edit_text(text, reply_markup=kb)
            return
        except TelegramBadRequest:
            pass
    await target.answer(text, reply_markup=kb)


def _cq_message(query: CallbackQuery) -> Message | None:
    return query.message if isinstance(query.message, Message) else None


async def _tz(ctx: AppContext) -> str | None:
    async with ctx.db.transaction() as s:
        return await owner_tz(s)


# --------------------------------------------------------------------------- views


async def overview_text(ctx: AppContext, gw: AIGateway | None) -> str:
    if gw is None:
        return "🤖 ИИ\n" + NO_GATEWAY
    lines = ["🤖 ИИ"]
    if not gw.configured:
        lines.append("Состояние: не настроен — " + (gw.config_error or "нет config/ai.yaml"))
        lines.append("Бот работает без ИИ: задачи, сессии и напоминания доступны.")
    elif gw.no_ai_mode and gw.no_ai_since is not None:
        since = fmt_local(gw.no_ai_since, await _tz(ctx))
        lines.append(f"Состояние: режим без ИИ с {since}, жду восстановления провайдеров")
    else:
        lines.append("Состояние: работает")
    states = list(gw.credentials.states.values())
    usable = [s for s in states if s.available(ctx.clock.now())]
    lines.append(f"Ключей: {len(states)}, доступно сейчас: {len(usable)}")
    day = await spend_report(ctx, "day")
    month = await spend_report(ctx, "month")
    mark = " *" if month.estimated else ""
    lines.append(
        f"Расходы: сегодня {fmt_usd(day.total_usd)}, месяц {fmt_usd(month.total_usd)}{mark}"
    )
    b = gw.config.budget
    if b.monthly_hard_usd is None and b.monthly_soft_usd is None:
        lines.append("Бюджет: не задан (идёт только учёт)")
    else:
        lines.append(
            "Бюджет: мягкий "
            + (fmt_usd(b.monthly_soft_usd) if b.monthly_soft_usd is not None else "—")
            + ", жёсткий "
            + (fmt_usd(b.monthly_hard_usd) if b.monthly_hard_usd is not None else "—")
        )
    if month.estimated:
        lines.append("* для части моделей цены не заданы — расход оценочный")
    return "\n".join(lines)


def _status_line(st: CredentialState, tz: str | None, now_spend: float) -> str:
    status = STATUS_RU.get(st.status, st.status.value)
    if st.status == CredentialStatus.COOLING_DOWN and st.cooldown_until is not None:
        status += f" до {fmt_local(st.cooldown_until, tz, '%H:%M')}"
    key = (
        f"****{st.key_last4}" if st.key_last4 else ("без ключа" if st.has_key else "ключ не задан")
    )
    src = " (чат)" if st.source == "chat" else ""
    return f"• {st.label}{src} — {st.provider}, {key}, {status}, месяц {fmt_usd(now_spend)}"


async def keys_view(ctx: AppContext, gw: AIGateway) -> tuple[str, InlineKeyboardMarkup]:
    await gw.credentials.refresh()
    tz = await _tz(ctx)
    spend = await month_spend_by_credential(ctx)
    states = sorted(gw.credentials.states.values(), key=lambda s: s.label)
    if not states:
        text = "🔑 Ключи ИИ\nКлючей нет. " + (
            "" if gw.config.features.allow_keys_via_chat else ENV_ONLY
        )
        return text, _back_kb()
    lines = ["🔑 Ключи ИИ"]
    rows: list[list[InlineKeyboardButton]] = []
    for st in states:
        lines.append(_status_line(st, tz, spend.get(st.label, 0.0)))
        if st.last_error and st.status != CredentialStatus.ACTIVE:
            lines.append(f"   последняя ошибка: {st.last_error}")
        rows.append([_btn(f"{st.label}", "key", st.id)])
    return "\n".join(lines), _back_kb(*rows)


def key_kb(st: CredentialState) -> InlineKeyboardMarkup:
    toggle = (
        _btn("✅ Включить", "enable", st.id)
        if st.status == CredentialStatus.DISABLED
        else _btn("⛔ Отключить", "disable", st.id)
    )
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [_btn("🧪 Проверить", "test", st.id), toggle],
            [_btn("← К ключам", "keys")],
        ]
    )


async def key_text(ctx: AppContext, gw: AIGateway, st: CredentialState) -> str:
    tz = await _tz(ctx)
    spend = (await month_spend_by_credential(ctx)).get(st.label, 0.0)
    lines = [_status_line(st, tz, spend).lstrip("• ")]
    if st.base_url:
        lines.append(f"base_url: {st.base_url}")
    if st.spend_limit_usd is not None:
        lines.append(f"Лимит ключа: {fmt_usd(st.spend_limit_usd)} в месяц")
    if st.allowed_data is not None:
        lines.append("Можно отправлять: " + ", ".join(sorted(st.allowed_data)))
    models = [a for a, m in gw.config.models.items() if st.label in gw.credentials.pool_for(m)]
    lines.append("Модели: " + (", ".join(models) if models else "нет (не указан в ai.yaml)"))
    if st.last_error:
        lines.append(f"Последняя ошибка: {st.last_error}")
    return "\n".join(lines)


def models_text(gw: AIGateway) -> str:
    lines = ["🧩 Модели по уровням"]
    names = {"fast": "быстрый", "standard": "стандартный", "strong": "сильный", "stt": "речь"}
    for tier, aliases in gw.tier_table().items():
        lines.append(f"{tier} ({names.get(tier, tier)}):")
        if not aliases:
            lines.append("   — не настроено")
        for alias in aliases:
            m = gw.config.models[alias]
            creds = ", ".join(m.credentials)
            extra = "аудио" if m.audio_input else f"структура: {m.structured_output.value}"
            lines.append(f"   {alias}: {m.model} [{creds}] — {extra}")
    return "\n".join(lines)


def _breakdown(title: str, data: dict[str, float]) -> list[str]:
    if not data:
        return []
    items = sorted(data.items(), key=lambda kv: -kv[1])
    return [title + ":"] + [f"   {k}: {fmt_usd(v)}" for k, v in items]


async def spend_text(ctx: AppContext, period: str) -> str:
    rep = await spend_report(ctx, "day" if period == "day" else "month")
    head = "сегодня" if period == "day" else "в этом месяце"
    lines = [
        f"💸 Расходы ИИ {head}: {fmt_usd(rep.total_usd)} (вызовов: {rep.calls}, попыток: {rep.attempts})"
    ]
    lines += _breakdown("По провайдерам", rep.by_provider)
    lines += _breakdown("По ключам", rep.by_credential)
    lines += _breakdown("По моделям", rep.by_model)
    lines += _breakdown("По задачам", rep.by_task)
    if rep.estimated:
        lines.append("Часть цен или токенов неизвестна — сумма оценочная.")
    return "\n".join(lines)


def _spend_kb() -> InlineKeyboardMarkup:
    return _back_kb([_btn("За день", "spend", arg="day"), _btn("За месяц", "spend", arg="month")])


# --------------------------------------------------------------------------- handlers


@router.message(Command("ai"))
async def on_ai(message: Message, ctx: AppContext, state: FSMContext) -> None:
    await state.clear()
    gw = get_gateway(ctx)
    await message.answer(await overview_text(ctx, gw), reply_markup=_menu_kb() if gw else None)


@router.callback_query(AIKeyCb.filter())
async def on_ai_cb(
    query: CallbackQuery, callback_data: AIKeyCb, ctx: AppContext, state: FSMContext
) -> None:
    msg = _cq_message(query)
    gw = get_gateway(ctx)
    if msg is None:
        await query.answer()
        return
    if gw is None:
        await query.answer()
        await _show(msg, NO_GATEWAY, None, edit=True)
        return
    action = callback_data.action
    if action == "menu":
        await state.clear()
        await query.answer()
        await _show(msg, await overview_text(ctx, gw), _menu_kb(), edit=True)
    elif action == "keys":
        await query.answer()
        text, kb = await keys_view(ctx, gw)
        await _show(msg, text, kb, edit=True)
    elif action in ("key", "test", "enable", "disable"):
        await _key_action(query, msg, gw, ctx, action, callback_data.credential_id)
    elif action == "models":
        await query.answer()
        await _show(msg, models_text(gw), _back_kb(), edit=True)
    elif action == "spend":
        await query.answer()
        await _show(msg, await spend_text(ctx, callback_data.arg), _spend_kb(), edit=True)
    elif action == "add":
        await query.answer()
        if not gw.config.features.allow_keys_via_chat:
            await _show(msg, ENV_ONLY, _back_kb(), edit=True)
            return
        rows = [[_btn(name, "prov", arg=code)] for code, name in CHAT_PROVIDERS.items()]
        await _show(msg, "Какой провайдер?", _back_kb(*rows), edit=True)
    elif action == "prov":
        await query.answer()
        if not gw.config.features.allow_keys_via_chat or callback_data.arg not in CHAT_PROVIDERS:
            await _show(msg, ENV_ONLY, _back_kb(), edit=True)
            return
        await state.update_data(ai_provider=callback_data.arg, ai_base_url=None)
        if callback_data.arg == "openai_compatible":
            await state.set_state(AddKey.base_url)
            await _show(
                msg,
                "Пришли base_url API, например https://openrouter.ai/api/v1 "
                "или http://localhost:11434/v1",
                _back_kb(),
                edit=True,
            )
        else:
            await state.set_state(AddKey.key)
            await _show(msg, _paste_key_text(), _back_kb(), edit=True)
    else:
        await query.answer()


def _paste_key_text() -> str:
    return (
        "Пришли ключ одним сообщением. Я сразу удалю сообщение из чата, сохраню ключ "
        "в зашифрованном виде и сделаю пробный запрос."
    )


async def _key_action(
    query: CallbackQuery, msg: Message, gw: AIGateway, ctx: AppContext, action: str, cid: int
) -> None:
    st = gw.credential_by_id(cid)
    if st is None:
        await query.answer("Ключ не найден")
        return
    if action == "test":
        await query.answer("Проверяю…")
        ok, info, models = await gw.test_credential(cid)
        st = gw.credential_by_id(cid) or st
        text = await key_text(ctx, gw, st)
        text += "\n\n" + ("✅ " if ok else "❌ ") + info
        if models:
            shown = ", ".join(models[:15]) + (" …" if len(models) > 15 else "")
            text += "\nДоступные модели: " + shown
        await _show(msg, text, key_kb(st), edit=True)
        return
    if action in ("enable", "disable"):
        new = await gw.credentials.set_enabled(cid, action == "enable")
        await query.answer("Включён" if action == "enable" else "Отключён")
        st = new or st
    else:
        await query.answer()
    await _show(msg, await key_text(ctx, gw, st), key_kb(st), edit=True)


@router.message(StateFilter(AddKey.base_url), F.text)
async def on_base_url(message: Message, ctx: AppContext, state: FSMContext) -> None:
    url = (message.text or "").strip()
    if not url.startswith(("https://", "http://")):
        await message.answer(
            "Это не похоже на адрес. Пришли base_url, начиная с https:// или http://"
        )
        return
    await state.update_data(ai_base_url=url.rstrip("/"))
    await state.set_state(AddKey.key)
    await message.answer(_paste_key_text())


@router.message(StateFilter(AddKey.key), F.text)
async def on_key(
    message: Message, ctx: AppContext, state: FSMContext, update_id: int | None = None
) -> None:
    # Delete the key from the chat first, before anything else can fail, and keep
    # it out of the conversation history (DedupMiddleware logs incoming text).
    if update_id is not None:
        ctx.registry.setdefault("redacted_update_ids", set()).add(update_id)
    await ctx.messenger.delete_message(message.chat.id, message.message_id)
    key = (message.text or "").strip()
    data = await state.get_data()
    await state.clear()
    gw = get_gateway(ctx)
    provider = data.get("ai_provider")
    if gw is None or not gw.config.features.allow_keys_via_chat or provider not in CHAT_PROVIDERS:
        await message.answer(ENV_ONLY)
        return
    if len(key) < 8 or " " in key:
        await message.answer("Ключ не похож на настоящий, сообщение удалено. Попробуй ещё раз: /ai")
        return
    st = await gw.credentials.add_key(
        provider=provider, api_key=key, base_url=data.get("ai_base_url")
    )
    del key
    ok, info, models = await gw.test_credential(st.id)
    st = gw.credential_by_id(st.id) or st
    text = f"Ключ сохранён как «{st.label}» (****{st.key_last4}). Сообщение с ключом удалено.\n"
    text += ("✅ " if ok else "❌ ") + info
    if models:
        text += "\nДоступные модели: " + ", ".join(models[:15]) + (" …" if len(models) > 15 else "")
    in_use = [a for a, m in gw.config.models.items() if st.label in gw.credentials.pool_for(m)]
    if not in_use:
        text += (
            "\nЭтот ключ пока не используется ни одной моделью: укажите модели этого провайдера "
            "в config/ai.yaml."
        )
    await message.answer(text, reply_markup=key_kb(st))
