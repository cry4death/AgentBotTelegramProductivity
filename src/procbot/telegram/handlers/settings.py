"""Settings (FR-01, FR-07, FR-08): view and edit everything in owner_profile.

- Unset values are shown as "не задано" (open questions 2, 3, 5: nothing invented).
- Every change goes through services.profile.update_profile (audited, actor=user).
  The bot never changes morning/evening time by itself (FR-03): only here.
- Buttons carry the target value (not "toggle"), so repeated presses are no-ops.
- Personality changes touch only the profile row (acceptance 21).
"""

from __future__ import annotations

from typing import Any

from aiogram import F, Router
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from procbot.context import AppContext
from procbot.db.models import OwnerProfile
from procbot.services.preferences import (
    AUTONOMY_TOGGLES,
    KNOB_MAX,
    KNOB_MIN,
    KNOBS,
    MEMORY_TOGGLES,
    NOT_SET,
    PRESETS,
    QUICK_TIMEZONES,
    WEEKDAYS_SHORT,
    flag,
    format_days,
    format_window,
    format_windows,
    from_compact,
    hhmm,
    is_valid_timezone,
    parse_hhmm,
    parse_pause_until,
    parse_timezone,
    parse_weekly,
    parse_window,
    pause_for,
    utc_offset_label,
)
from procbot.services.profile import get_or_create_profile, get_profile, update_profile
from procbot.services.profile_hooks import after_profile_change
from procbot.telegram.callbacks import SettingsCb
from procbot.telegram.keyboards import Menu
from procbot.timeutil import fmt_local

router = Router(name="settings")

LEAD_PICKS = (5, 10, 15, 30)
DELAY_PICKS = (5, 10, 15, 30)
POMODORO_WORK_PICKS = (15, 25, 45, 50)
POMODORO_BREAK_PICKS = (5, 10, 15)
SHORT_START_PICKS = (2, 5, 10)
SLEEP_PICKS = ("2300-0700", "0000-0800", "0100-0900")
QUIET_PICKS = ("2200-0800", "2300-0700", "2300-0900")
TIME_PICKS_MORNING = ("0700", "0800", "0900", "1000")
TIME_PICKS_EVENING = ("2000", "2100", "2200", "2300")
WEEKLY_TIME_PICKS = ("1000", "1800", "2000")
PERSONA_NOTE = "Характер меняет только тон ответов: задачи, история и память не затрагиваются."


class SettingsForm(StatesGroup):
    timezone = State()
    sleep = State()
    quiet = State()
    morning = State()
    evening = State()
    weekly = State()
    pause = State()


PROMPTS: dict[str, str] = {
    "timezone": "Напиши часовой пояс: город (Москва, Berlin) или IANA-имя (Europe/Moscow).",
    "sleep": "Напиши интервал сна ЧЧ:ММ-ЧЧ:ММ, например 23:30-07:30.",
    "quiet": "Напиши интервал тишины ЧЧ:ММ-ЧЧ:ММ, при желании с днями: «23:00-09:00 выходные».",
    "morning": "Напиши время утреннего плана ЧЧ:ММ.",
    "evening": "Напиши время вечернего разбора ЧЧ:ММ.",
    "weekly": "Напиши день и время недельного обзора, например «вс 18:00».",
    "pause": "До какого момента пауза? Формат ДД.ММ, ДД.ММ ЧЧ:ММ или ЧЧ:ММ.",
}
_FORM: dict[str, State] = {
    "timezone": SettingsForm.timezone,
    "sleep": SettingsForm.sleep,
    "quiet": SettingsForm.quiet,
    "morning": SettingsForm.morning,
    "evening": SettingsForm.evening,
    "weekly": SettingsForm.weekly,
    "pause": SettingsForm.pause,
}


# --------------------------------------------------------------------------- helpers


def _b(text: str, action: str, arg: str = "") -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=SettingsCb(action=action, arg=arg).pack())


def _kb(*rows: list[InlineKeyboardButton]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[r for r in rows if r])


def _back() -> list[InlineKeyboardButton]:
    return [_b("« Назад", "home")]


def _hm(code: str) -> str:
    return f"{code[:2]}:{code[2:]}"


def _range_label(code: str) -> str:
    a, _, b = code.partition("-")
    return f"{_hm(a)}–{_hm(b)}"


def _window_from_code(code: str) -> dict[str, Any] | None:
    """'2300-0800' or '2300-0800-56' (days) -> window dict."""
    parts = code.split("-")
    if len(parts) not in (2, 3):
        return None
    try:
        window = parse_window(f"{_hm(parts[0])}-{_hm(parts[1])}")
    except ValueError:
        return None
    if len(parts) == 3 and parts[2]:
        window["days"] = sorted({int(c) for c in parts[2] if c.isdigit() and int(c) < 7})
    return window


def _window_code(raw: dict[str, Any]) -> str:
    start = str(raw.get("start", "")).replace(":", "")
    end = str(raw.get("end", "")).replace(":", "")
    days = raw.get("days")
    return f"{start}-{end}" + ("" if days is None else "-" + "".join(str(d) for d in days))


def _minutes(value: int | None, unit: str = "мин") -> str:
    return NOT_SET if value is None else f"{value} {unit}"


def _effective(profile_value: int | None, config_value: int | None) -> str:
    if profile_value is not None:
        return f"за {profile_value} мин" if profile_value else "выкл"
    if config_value is not None:
        return f"{config_value} мин (по умолчанию из app.yaml)"
    return "выкл"


async def _load(ctx: AppContext, chat_id: int | None = None) -> OwnerProfile:
    async with ctx.db.transaction() as s:
        if chat_id is not None:
            profile, _ = await get_or_create_profile(
                s, telegram_user_id=ctx.owner_id, now=ctx.clock.now(), chat_id=chat_id
            )
        else:
            profile = await get_profile(s)
            if profile is None:
                profile, _ = await get_or_create_profile(
                    s, telegram_user_id=ctx.owner_id, now=ctx.clock.now()
                )
    return profile


async def save(ctx: AppContext, **changes: Any) -> OwnerProfile:
    """Persist profile changes in one audited transaction (reply only after it)."""
    async with ctx.db.transaction() as s:
        if await get_profile(s) is None:
            await get_or_create_profile(s, telegram_user_id=ctx.owner_id, now=ctx.clock.now())
        profile = await update_profile(s, now=ctx.clock.now(), **changes)
        await after_profile_change(s, ctx, changes.keys())
        return profile


async def _show(target: Message, text: str, kb: InlineKeyboardMarkup | None, *, edit: bool) -> None:
    if edit:
        try:
            await target.edit_text(text, reply_markup=kb)
            return
        except TelegramBadRequest:
            pass  # "message is not modified" or too old to edit: send a new one
    await target.answer(text, reply_markup=kb)


# --------------------------------------------------------------------------- views


def overview_text(ctx: AppContext, p: OwnerProfile) -> str:
    now = ctx.clock.now()
    tz = p.timezone
    routine = p.routine or {}
    weekdays = routine.get("weekdays")
    sleep = routine.get("sleep")
    pause = (
        f"до {fmt_local(p.pause_until, tz)}"
        if p.pause_until is not None and p.pause_until > now
        else "нет"
    )
    weekly = (
        f"{WEEKDAYS_SHORT[p.weekly_review_dow]} {hhmm(p.weekly_review_time)}"
        if p.weekly_review_dow is not None and p.weekly_review_time is not None
        else NOT_SET
    )
    rc = ctx.config.reminders
    lines = [
        "⚙️ Настройки",
        f"Часовой пояс: {tz or NOT_SET + ' (пока используется UTC)'}",
        f"Будни: {format_days(weekdays) if weekdays is not None else NOT_SET}",
        f"Сон: {format_window(sleep) if sleep else NOT_SET}",
        f"Часы тишины: {format_windows(p.quiet_hours)}",
        f"Утренний план: {hhmm(p.morning_time)} · Вечерний разбор: {hhmm(p.evening_time)}",
        f"Недельный обзор: {weekly}",
        f"Уведомления: {'вкл' if p.notifications_enabled else 'выкл'} · Пауза: {pause}",
        (
            f"Характер: {PRESETS.get(p.style_preset, p.style_preset)} · "
            f"строгость {p.strictness}/4 · юмор {p.humor}/4 · "
            f"подробность {p.verbosity}/4 · мат: {'можно' if p.profanity_allowed else 'нет'}"
        ),
        (
            f"Напоминание перед задачей: {_effective(p.pre_task_lead_min, rc.pre_task_lead_minutes)}"
            f" · Проверка старта: {_effective(p.start_delay_min, rc.start_delay_after_minutes)}"
        ),
        (
            f"Помодоро: {_minutes(p.pomodoro_work_min)} / перерыв {_minutes(p.pomodoro_break_min)}"
            f" · Короткий старт: {_minutes(p.short_start_min)}"
        ),
        (
            f"Память: {'вкл' if flag(p.memory_settings, 'enabled') else 'выкл'} · "
            f"Автономность: {sum(flag(p.autonomy_settings, k) for k in AUTONOMY_TOGGLES)}"
            f"/{len(AUTONOMY_TOGGLES)} разрешено"
        ),
    ]
    if p.goals_note:
        lines.append(f"Цели (заметка): {p.goals_note[:200]}")
    return "\n".join(lines)


def overview_kb(p: OwnerProfile) -> InlineKeyboardMarkup:
    notif_to = "off" if p.notifications_enabled else "on"
    return _kb(
        [_b("🕒 Часовой пояс", "tz"), _b("📆 Будни", "wd")],
        [_b("😴 Сон", "sleep"), _b("🤫 Часы тишины", "quiet")],
        [_b("🌅 Утро", "morning"), _b("🌙 Вечер", "evening"), _b("🗓 Неделя", "weekly")],
        [
            _b(
                "🔕 Выключить уведомления" if p.notifications_enabled else "🔔 Включить",
                "notif",
                notif_to,
            ),
            _b("⏸ Пауза", "pause"),
        ],
        [_b("🎭 Характер", "persona"), _b("⏰ Напоминания", "remind")],
        [_b("🍅 Рабочие сессии", "work"), _b("🧠 Память и автономность", "mem")],
    )


def view(ctx: AppContext, p: OwnerProfile, section: str) -> tuple[str, InlineKeyboardMarkup]:
    now = ctx.clock.now()
    if section == "tz":
        labels = dict(QUICK_TIMEZONES)
        btns = [
            _b(
                f"{'• ' if p.timezone == z else ''}{labels[z]} ({utc_offset_label(z, now)})",
                "tz_set",
                z,
            )
            for z, _ in QUICK_TIMEZONES
        ]
        rows = [btns[i : i + 2] for i in range(0, len(btns), 2)]
        return (
            f"🕒 Часовой пояс: {p.timezone or NOT_SET}",
            _kb(*rows, [_b("✏️ Ввести вручную", "input", "timezone")], _back()),
        )
    if section == "wd":
        days = set((p.routine or {}).get("weekdays") or [])
        btns = [
            _b(
                f"{'✅' if d in days else '⬜'} {WEEKDAYS_SHORT[d]}",
                "wd_t",
                f"{d}.{0 if d in days else 1}",
            )
            for d in range(7)
        ]
        current = (
            format_days(sorted(days)) if (p.routine or {}).get("weekdays") is not None else NOT_SET
        )
        return (
            f"📆 Рабочие дни (будни): {current}\nОстальные дни считаются выходными.",
            _kb(btns[:4], btns[4:], [_b("Сбросить", "wd_clr")], _back()),
        )
    if section == "sleep":
        sleep = (p.routine or {}).get("sleep")
        return (
            f"😴 Примерный сон: {format_window(sleep) if sleep else NOT_SET}",
            _kb(
                [_b(_range_label(c), "sleep_set", c) for c in SLEEP_PICKS],
                [_b("✏️ Ввести вручную", "input", "sleep"), _b("Не задано", "sleep_set", "off")],
                _back(),
            ),
        )
    if section == "quiet":
        windows = list(p.quiet_hours or [])
        dels = [[_b(f"🗑 {format_window(w)}", "q_del", _window_code(w))] for w in windows]
        return (
            "🤫 Часы тишины (в это время бот не пишет первым; важность задачи их не отменяет):\n"
            + (format_windows(windows) if windows else NOT_SET),
            _kb(
                *dels,
                [_b(f"+ {_range_label(c)}", "q_add", c) for c in QUIET_PICKS],
                [_b("✏️ Добавить свой", "input", "quiet"), _b("Очистить", "q_clr")],
                _back(),
            ),
        )
    if section in ("morning", "evening"):
        picks = TIME_PICKS_MORNING if section == "morning" else TIME_PICKS_EVENING
        current = p.morning_time if section == "morning" else p.evening_time
        title = "🌅 Утренний план" if section == "morning" else "🌙 Вечерний разбор"
        action = "mo_set" if section == "morning" else "ev_set"
        return (
            f"{title}: {hhmm(current)}\nНе задано — бот не пишет первым в это время.",
            _kb(
                [_b(_hm(c), action, c) for c in picks],
                [_b("✏️ Ввести вручную", "input", section), _b("Не задано", action, "off")],
                _back(),
            ),
        )
    if section == "weekly":
        cur = (
            f"{WEEKDAYS_SHORT[p.weekly_review_dow]} {hhmm(p.weekly_review_time)}"
            if p.weekly_review_dow is not None
            else NOT_SET
        )
        days = [
            _b(f"{'• ' if p.weekly_review_dow == d else ''}{WEEKDAYS_SHORT[d]}", "wk_d", str(d))
            for d in range(7)
        ]
        times = [_b(_hm(c), "wk_t", c) for c in WEEKLY_TIME_PICKS]
        return (
            f"🗓 Недельный обзор: {cur}\nВыбери день, затем время.",
            _kb(
                days[:4],
                days[4:],
                times if p.weekly_review_dow is not None else [],
                [_b("✏️ Ввести вручную", "input", "weekly"), _b("Не задано", "wk_off")],
                _back(),
            ),
        )
    if section == "pause":
        active = p.pause_until is not None and p.pause_until > now
        status = f"до {fmt_local(p.pause_until, p.timezone)}" if active else "нет"
        return (
            f"⏸ Ручная пауза: {status}\nНа время паузы бот не пишет первым; "
            "отложенное потом не придёт лавиной.",
            _kb(
                [_b("1 ч", "pz", "1h"), _b("3 ч", "pz", "3h"), _b("До завтра", "pz", "tomorrow")],
                [_b("📅 До даты", "input", "pause"), _b("Снять паузу", "pz", "clear")],
                _back(),
            ),
        )
    if section == "persona":
        presets = [
            _b(f"{'• ' if p.style_preset == k else ''}{label}", "ps", k)
            for k, label in PRESETS.items()
        ]
        knob_rows = []
        for key, label in KNOBS.items():
            value = getattr(p, key)
            knob_rows.append(
                [_b(label, "noop")]
                + [
                    _b(f"•{v}" if v == value else str(v), "kn", f"{key}.{v}")
                    for v in range(KNOB_MIN, KNOB_MAX + 1)
                ]
            )
        prof = p.profanity_allowed
        return (
            "🎭 Характер\n"
            f"Стиль: {PRESETS.get(p.style_preset, p.style_preset)}\n"
            f"Строгость {p.strictness}/4 · Юмор {p.humor}/4 · Подробность {p.verbosity}/4 · "
            f"Мат: {'можно' if prof else 'нет'}\n{PERSONA_NOTE}",
            _kb(
                presets[:2],
                presets[2:],
                *knob_rows,
                [_b("Мат: запретить" if prof else "Мат: разрешить", "prof", "0" if prof else "1")],
                _back(),
            ),
        )
    if section == "remind":
        rc = ctx.config.reminders
        return (
            "⏰ Напоминания\n"
            f"Перед задачей: {_effective(p.pre_task_lead_min, rc.pre_task_lead_minutes)}\n"
            f"Проверка старта (если не начал): "
            f"{_effective(p.start_delay_min, rc.start_delay_after_minutes)}\n"
            "Всего не больше 4 напоминаний за любые 60 минут.",
            _kb(
                [_b("Перед задачей:", "noop")],
                [_b(f"{m} мин", "rl", str(m)) for m in LEAD_PICKS] + [_b("выкл", "rl", "off")],
                [_b("Проверка старта через:", "noop")],
                [_b(f"{m} мин", "rd", str(m)) for m in DELAY_PICKS] + [_b("выкл", "rd", "off")],
                _back(),
            ),
        )
    if section == "work":
        return (
            "🍅 Рабочие сессии (не задано — спрошу при первом запуске)\n"
            f"Помодоро: работа {_minutes(p.pomodoro_work_min)}, "
            f"перерыв {_minutes(p.pomodoro_break_min)}\n"
            f"Короткий старт: {_minutes(p.short_start_min)}",
            _kb(
                [_b("Помодоро, работа:", "noop")],
                [_b(str(m), "pw", str(m)) for m in POMODORO_WORK_PICKS] + [_b("—", "pw", "off")],
                [_b("Перерыв:", "noop")],
                [_b(str(m), "pb", str(m)) for m in POMODORO_BREAK_PICKS] + [_b("—", "pb", "off")],
                [_b("Короткий старт:", "noop")],
                [_b(str(m), "ss", str(m)) for m in SHORT_START_PICKS] + [_b("—", "ss", "off")],
                _back(),
            ),
        )
    if section == "mem":
        mem_rows = [
            [
                _b(
                    f"{'✅' if flag(p.memory_settings, k) else '⬜'} {label}",
                    "mt",
                    f"{k}.{0 if flag(p.memory_settings, k) else 1}",
                )
            ]
            for k, label in MEMORY_TOGGLES.items()
        ]
        auto_rows = [
            [
                _b(
                    f"{'✅' if flag(p.autonomy_settings, k) else '⬜'} {label}",
                    "at",
                    f"{k}.{0 if flag(p.autonomy_settings, k) else 1}",
                )
            ]
            for k, label in AUTONOMY_TOGGLES.items()
        ]
        return (
            "🧠 Память и автономность\n"
            "Память: что бот запоминает о тебе (просмотр и удаление — в «🧠 Память»).\n"
            "Автономность: что бот может делать без отдельного подтверждения. "
            "Новые задачи, переносы, календарь, привычки и цели всегда требуют согласия.",
            _kb(*mem_rows, [_b("— без подтверждения разрешено —", "noop")], *auto_rows, _back()),
        )
    return overview_text(ctx, p), overview_kb(p)


# --------------------------------------------------------------------------- entry points


@router.message(Command("settings"))
@router.message(F.text == Menu.SETTINGS)
async def cmd_settings(message: Message, ctx: AppContext, state: FSMContext) -> None:
    await state.clear()
    p = await _load(ctx, message.chat.id)
    await message.answer(overview_text(ctx, p), reply_markup=overview_kb(p))


SECTIONS = {
    "tz",
    "wd",
    "sleep",
    "quiet",
    "morning",
    "evening",
    "weekly",
    "pause",
    "persona",
    "remind",
    "work",
    "mem",
}


def _int_or_none(arg: str, allowed: tuple[int, ...]) -> tuple[bool, int | None]:
    if arg == "off":
        return True, None
    try:
        v = int(arg)
    except ValueError:
        return False, None
    return (v in allowed), v


async def apply_action(ctx: AppContext, p: OwnerProfile, action: str, arg: str) -> str | None:
    """Apply one settings button. Returns the section to re-render, or None if invalid."""
    now = ctx.clock.now()
    if action == "tz_set":
        if not is_valid_timezone(arg):
            return None
        await save(ctx, timezone=arg)
        return "tz"
    if action == "wd_t":
        day_s, _, on = arg.partition(".")
        if not day_s.isdigit() or int(day_s) > 6:
            return None
        days = set((p.routine or {}).get("weekdays") or [])
        (days.add if on == "1" else days.discard)(int(day_s))
        await save(ctx, routine={**(p.routine or {}), "weekdays": sorted(days)})
        return "wd"
    if action == "wd_clr":
        routine = dict(p.routine or {})
        routine.pop("weekdays", None)
        await save(ctx, routine=routine)
        return "wd"
    if action == "sleep_set":
        routine = dict(p.routine or {})
        if arg == "off":
            routine.pop("sleep", None)
        else:
            w = _window_from_code(arg)
            if w is None:
                return None
            routine["sleep"] = {"start": w["start"], "end": w["end"]}
        await save(ctx, routine=routine)
        return "sleep"
    if action == "q_add":
        w = _window_from_code(arg)
        if w is None:
            return None
        windows = list(p.quiet_hours or [])
        if w not in windows:
            windows.append(w)
            await save(ctx, quiet_hours=windows)
        return "quiet"
    if action == "q_del":
        windows = [w for w in (p.quiet_hours or []) if _window_code(w) != arg]
        if len(windows) != len(p.quiet_hours or []):
            await save(ctx, quiet_hours=windows)
        return "quiet"
    if action == "q_clr":
        if p.quiet_hours:
            await save(ctx, quiet_hours=[])
        return "quiet"
    if action in ("mo_set", "ev_set"):
        field = "morning_time" if action == "mo_set" else "evening_time"
        t = None if arg == "off" else from_compact(arg)
        if arg != "off" and t is None:
            return None
        await save(ctx, **{field: t})
        return "morning" if action == "mo_set" else "evening"
    if action == "wk_d":
        if not arg.isdigit() or int(arg) > 6:
            return None
        await save(ctx, weekly_review_dow=int(arg))
        return "weekly"
    if action == "wk_t":
        t = from_compact(arg)
        if t is None or p.weekly_review_dow is None:
            return None
        await save(ctx, weekly_review_time=t)
        return "weekly"
    if action == "wk_off":
        await save(ctx, weekly_review_dow=None, weekly_review_time=None)
        return "weekly"
    if action == "notif":
        await save(ctx, notifications_enabled=(arg == "on"))
        return "home"
    if action == "pz":
        if arg == "clear":
            await save(ctx, pause_until=None)
        else:
            try:
                until = pause_for(arg, now, p.timezone, p.morning_time)
            except ValueError:
                return None
            await save(ctx, pause_until=until)
        return "pause"
    if action == "ps":
        if arg not in PRESETS:
            return None
        await save(ctx, style_preset=arg)
        return "persona"
    if action == "kn":
        key, _, value = arg.partition(".")
        if key not in KNOBS or not value.isdigit() or not KNOB_MIN <= int(value) <= KNOB_MAX:
            return None
        await save(ctx, **{key: int(value)})
        return "persona"
    if action == "prof":
        await save(ctx, profanity_allowed=(arg == "1"))
        return "persona"
    if action in ("rl", "rd"):
        picks = LEAD_PICKS if action == "rl" else DELAY_PICKS
        ok, v = _int_or_none(arg, picks)
        if not ok:
            return None
        if v is None:
            v = 0  # 0 = explicitly off for the owner (None would fall back to app.yaml)
        await save(ctx, **{"pre_task_lead_min" if action == "rl" else "start_delay_min": v})
        return "remind"
    if action in ("pw", "pb", "ss"):
        field, picks = {
            "pw": ("pomodoro_work_min", POMODORO_WORK_PICKS),
            "pb": ("pomodoro_break_min", POMODORO_BREAK_PICKS),
            "ss": ("short_start_min", SHORT_START_PICKS),
        }[action]
        ok, v = _int_or_none(arg, picks)
        if not ok:
            return None
        await save(ctx, **{field: v})
        return "work"
    if action in ("mt", "at"):
        key, _, on = arg.partition(".")
        toggles = MEMORY_TOGGLES if action == "mt" else AUTONOMY_TOGGLES
        if key not in toggles:
            return None
        attr = "memory_settings" if action == "mt" else "autonomy_settings"
        await save(ctx, **{attr: {**(getattr(p, attr) or {}), key: on == "1"}})
        return "mem"
    return None


@router.callback_query(SettingsCb.filter())
async def on_settings(
    cq: CallbackQuery, callback_data: SettingsCb, ctx: AppContext, state: FSMContext
) -> None:
    action, arg = callback_data.action, callback_data.arg
    msg = cq.message if isinstance(cq.message, Message) else None
    if action == "noop":
        await cq.answer()
        return
    if action == "input":
        if arg not in _FORM:
            await cq.answer()
            return
        await state.set_state(_FORM[arg])
        await cq.answer()
        if msg is not None:
            await msg.answer(PROMPTS[arg])
        return
    p = await _load(ctx)
    if action == "home" or action in SECTIONS:
        section = action
    else:
        result = await apply_action(ctx, p, action, arg)
        if result is None:
            await cq.answer("Неизвестное значение")
            return
        section = result
        p = await _load(ctx)
    await cq.answer("Сохранено" if action not in SECTIONS and action != "home" else None)
    if msg is not None:
        text, kb = view(ctx, p, section)
        await _show(msg, text, kb, edit=True)


# --------------------------------------------------------------------------- text input


@router.message(StateFilter(SettingsForm), F.text)
async def on_settings_text(message: Message, ctx: AppContext, state: FSMContext) -> None:
    text = (message.text or "").strip()
    current = await state.get_state()
    field = next((k for k, v in _FORM.items() if v.state == current), None)
    if field is None or text in Menu.ALL or text.startswith("/"):
        await state.clear()
        raise SkipHandler()
    p = await _load(ctx)
    now = ctx.clock.now()
    section = field
    try:
        if field == "timezone":
            match = parse_timezone(text, now)
            if match.zone is None:
                if match.candidates:
                    await message.answer(
                        "Подходит несколько поясов — выбери:",
                        reply_markup=_kb(
                            *[
                                [_b(f"{z} ({utc_offset_label(z, now)})", "tz_set", z)]
                                for z in match.candidates
                            ],
                            _back(),
                        ),
                    )
                    await state.clear()
                else:
                    await message.answer("Не узнаю такой часовой пояс. Пример: Europe/Moscow.")
                return
            await save(ctx, timezone=match.zone)
            section = "tz"
        elif field == "sleep":
            w = parse_window(text)
            await save(
                ctx, routine={**(p.routine or {}), "sleep": {"start": w["start"], "end": w["end"]}}
            )
        elif field == "quiet":
            w = parse_window(text)
            windows = list(p.quiet_hours or [])
            if w not in windows:
                windows.append(w)
            await save(ctx, quiet_hours=windows)
        elif field in ("morning", "evening"):
            t = parse_hhmm(text)
            if t is None:
                raise ValueError("bad time")
            await save(ctx, **{f"{field}_time": t})
        elif field == "weekly":
            dow, t = parse_weekly(text)
            await save(ctx, weekly_review_dow=dow, weekly_review_time=t)
        elif field == "pause":
            until = parse_pause_until(text, now, p.timezone, p.morning_time)
            await save(ctx, pause_until=until)
    except ValueError:
        await message.answer(f"Не понял. {PROMPTS[field]}")
        return
    await state.clear()
    p = await _load(ctx)
    text_out, kb = view(ctx, p, "tz" if section == "timezone" else section)
    await message.answer("Сохранено.\n" + text_out, reply_markup=kb)


__all__ = ["SettingsForm", "apply_action", "overview_text", "router", "save", "view"]
