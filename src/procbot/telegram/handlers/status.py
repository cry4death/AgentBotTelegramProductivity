"""/status for the owner: version, uptime, DB, pending reminders, last backup,
AI and calendar state (architecture "Наблюдаемость")."""

from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from sqlalchemy import func, select

from procbot.context import AppContext
from procbot.db.models import ReminderSchedule
from procbot.domain.enums import ReminderState
from procbot.http_server import app_version, db_healthy, uptime_seconds
from procbot.services.backup import last_backup
from procbot.services.profile import get_profile
from procbot.timeutil import fmt_local

router = Router(name="status")

PENDING_STATES = (ReminderState.PENDING, ReminderState.PROCESSING, ReminderState.RESCHEDULED)


def format_uptime(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    s = int(seconds)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, _ = divmod(s, 60)
    if d:
        return f"{d} д {h} ч"
    if h:
        return f"{h} ч {m} мин"
    return f"{m} мин"


async def status_text(ctx: AppContext) -> str:
    db_ok = await db_healthy(ctx.db)
    pending: int | None = None
    tz: str | None = None
    backup_line = "ещё не было"
    if db_ok:
        async with ctx.db.transaction() as s:
            pending = (
                await s.execute(
                    select(func.count(ReminderSchedule.id)).where(
                        ReminderSchedule.state.in_([str(x) for x in PENDING_STATES])
                    )
                )
            ).scalar_one()
            profile = await get_profile(s)
            tz = profile.timezone if profile else None
        last = await last_backup(ctx.db)
        if last is not None:
            mark = "проверен" if last.status == "verified" else last.status
            backup_line = f"{fmt_local(last.created_at, tz)} ({mark})"
    if ctx.config.backup.interval_hours is None:
        backup_line += " · авто-бэкап выключен"
    ai = await _ai_line(ctx)
    calendar = "не подключён" if ctx.calendar is None else "подключён"
    return "\n".join(
        [
            "🩺 Состояние",
            f"Версия: {app_version()}",
            f"Работает: {format_uptime(uptime_seconds(ctx))}",
            f"База данных: {'ок' if db_ok else 'ОШИБКА'}",
            f"Запланированных напоминаний: {pending if pending is not None else '—'}",
            f"Последний бэкап: {backup_line}",
            f"ИИ: {ai}",
            f"Google Calendar: {calendar}",
        ]
    )


@router.message(Command("status"))
async def cmd_status(message: Message, ctx: AppContext) -> None:
    await message.answer(await status_text(ctx))


async def _ai_line(ctx: AppContext) -> str:
    from procbot.ai.gateway import get_gateway

    gw = get_gateway(ctx)
    if gw is None:
        return "ИИ не настроен — работаю в режиме без ИИ"
    try:
        states = await gw.credentials.refresh()
    except Exception:
        return "состояние ключей недоступно"
    now = ctx.clock.now()
    if not states:
        return "ключи не настроены — работаю в режиме без ИИ"
    usable = sum(1 for st in states.values() if st.available(now))
    mode = " · режим без ИИ (жду восстановления)" if gw.no_ai_mode else ""
    return f"ключей {len(states)}, доступно {usable}{mode} (подробно: /ai)"
