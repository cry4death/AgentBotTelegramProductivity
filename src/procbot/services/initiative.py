"""Initiative messages (FR-07): daily seeders and stage-A template renderers.

Seeders plan the NEXT occurrence of the owner's morning plan, evening review and
weekly review as ``reminder_schedule`` rows (DST-safe via
``timeutil.next_local_occurrence``, acceptance 23; dedup key per local date).
If a time is not set (open question 2) nothing is planned - the settings screen
shows "не задано". Missed occurrences (downtime) are handled by the normal
scheduler path: re-evaluated, stale ones folded into one digest.

Renderers work without AI (stage A); stage B may register richer ones for the
same categories. Every renderer re-checks relevance and returns None when the
message became pointless (e.g. the task already started).
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import date, datetime, timedelta

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.models import (
    Intervention,
    OwnerProfile,
    ReminderSchedule,
    Task,
    TaskEvent,
    WorkSession,
)
from procbot.domain.enums import (
    MessageCategory,
    ObstacleReason,
    ReminderState,
    SessionStatus,
    TaskEventType,
    TaskStatus,
    TimeSource,
)
from procbot.services.jobs import PeriodicJob, register_job
from procbot.services.outbound import Outgoing, register_renderer
from procbot.services.profile import get_profile
from procbot.services.reminders import (
    ACTIVE_TASK_STATUSES,
    LIVE_STATES,
    payload_dt,
    render_payload_text,
    start_delay_key,
    task_reminder_relevant,
    upsert_reminder,
)
from procbot.telegram.callbacks import DayCb, ReminderCb, SessionCb, TaskCb
from procbot.timeutil import fmt_local, local_date, local_day_bounds, next_local_occurrence

log = logging.getLogger(__name__)

SEED_JOB_NAME = "initiative.seed"
SEED_INTERVAL_SECONDS = 300.0

_SEEDED = (
    (MessageCategory.MORNING_PLAN, "morning"),
    (MessageCategory.EVENING_REVIEW, "evening"),
    (MessageCategory.WEEKLY_REVIEW, "weekly"),
)

# Quick obstacle buttons after a start delay (FR-04): hypotheses, not diagnoses.
REASON_LABELS: dict[ObstacleReason, str] = {
    ObstacleReason.UNCLEAR: "🤔 Не понимаю, с чего начать",
    ObstacleReason.TOO_BIG: "🐘 Слишком большая",
    ObstacleReason.UNPLEASANT: "😖 Неприятно",
    ObstacleReason.DISTRACTIONS: "📱 Отвлекаюсь",
    ObstacleReason.LOW_ENERGY: "🔋 Нет сил",
    ObstacleReason.PRIORITY_CONFLICT: "⚖️ Другое важнее",
    ObstacleReason.EXTERNAL: "🚪 Внешние дела",
    ObstacleReason.OTHER: "🤷 Другое",
}


def _btn(text: str, data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=data)


def _short(title: str, n: int = 40) -> str:
    return title if len(title) <= n else title[: n - 1] + "…"


# --------------------------------------------------------------------------- seeders


def _seed_spec(
    profile: OwnerProfile, category: MessageCategory
) -> tuple[object | None, int | None]:
    if category == MessageCategory.MORNING_PLAN:
        return profile.morning_time, None
    if category == MessageCategory.EVENING_REVIEW:
        return profile.evening_time, None
    if profile.weekly_review_dow is None:
        return None, None
    return profile.weekly_review_time, profile.weekly_review_dow


async def seed_initiative_in(session: AsyncSession, ctx: AppContext) -> int:
    """Plan the next morning / evening / weekly occurrence. Idempotent. Returns how
    many reminders are now planned ahead (0..3)."""
    now = ctx.clock.now()
    profile = await get_profile(session)
    planned = 0
    for category, prefix in _SEEDED:
        future = (
            (
                await session.execute(
                    select(ReminderSchedule).where(
                        ReminderSchedule.category == category,
                        ReminderSchedule.state.in_([str(s) for s in LIVE_STATES]),
                        ReminderSchedule.due_at > now,
                    )
                )
            )
            .scalars()
            .all()
        )
        t, dow = _seed_spec(profile, category) if profile is not None else (None, None)
        if profile is None or t is None:
            for r in future:  # time was unset: nothing is planned (open question 2)
                r.state = ReminderState.CANCELLED
                r.last_reason = "time not set"
                r.updated_at = now
            continue
        tz = profile.timezone
        due = next_local_occurrence(now, t, tz, weekday=dow)  # type: ignore[arg-type]
        day = local_date(due, tz)
        key = f"{prefix}:{day.isoformat()}"
        for r in future:
            if r.dedup_key != key:  # the time changed: the old occurrence goes
                r.state = ReminderState.CANCELLED
                r.last_reason = "time changed"
                r.updated_at = now
        await upsert_reminder(
            session,
            now=now,
            dedup_key=key,
            category=category,
            due_at=due,
            payload={"local_date": day.isoformat()},
        )
        planned += 1
    return planned


async def seed_initiative_reminders(ctx: AppContext) -> int:
    """Call after the owner changed morning/evening/weekly times or the timezone
    (the periodic job does it too, every few minutes)."""
    async with ctx.db.transaction() as s:
        return await seed_initiative_in(s, ctx)


async def _seed_job(ctx: AppContext) -> None:
    await seed_initiative_reminders(ctx)


# --------------------------------------------------------------------------- day helpers


async def _day_tasks(
    session: AsyncSession, ctx: AppContext, day: date, tz: str | None
) -> list[Task]:
    """Open tasks for the local day (planned today, then deadline today, then overdue)."""
    try:
        from procbot.services.tasks import today_buckets

        buckets = await today_buckets(session, ctx, day=day)
        return [*buckets.planned, *buckets.due_today, *buckets.overdue]
    except (ImportError, AttributeError, NotImplementedError):  # pragma: no cover
        start, end = local_day_bounds(day, tz)
        rows = (
            (
                await session.execute(
                    select(Task).where(Task.status.in_([str(s) for s in ACTIVE_TASK_STATUSES]))
                )
            )
            .scalars()
            .all()
        )
        return [
            t
            for t in rows
            if t.planned_date == day
            or (t.planned_start_at is not None and start <= t.planned_start_at < end)
        ]


def _payload_day(r: ReminderSchedule, now: datetime, tz: str | None) -> date:
    raw = (r.payload or {}).get("local_date")
    if raw:
        try:
            return date.fromisoformat(str(raw))
        except ValueError:
            pass
    return local_date(now, tz)


def _task_line(t: Task, tz: str | None) -> str:
    when = fmt_local(t.planned_start_at, tz, "%H:%M") + " — " if t.planned_start_at else ""
    return f"• {when}{t.title}"


# --------------------------------------------------------------------------- renderers


async def render_morning(ctx: AppContext, r: ReminderSchedule) -> Outgoing | None:
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        profile = await get_profile(s)
        tz = profile.timezone if profile else None
        day = _payload_day(r, now, tz)
        tasks = await _day_tasks(s, ctx, day, tz)
        lines = [_task_line(t, tz) for t in tasks[:10]]
        more = len(tasks) - len(lines)
    date_s = day.strftime("%d.%m")
    if lines:
        text = f"Доброе утро! На {date_s} в плане:\n" + "\n".join(lines)
        if more > 0:
            text += f"\n…и ещё {more}"
        text += "\n\nС чего начнём? Можно сразу запустить работу или открыть день и поправить план."
    else:
        text = (
            f"Доброе утро! На {date_s} пока ничего не запланировано. "
            "Добавить дела можно кнопкой «➕ Добавить», а план — в «📅 Мой день»."
        )
    kb = [
        [_btn(f"▶️ {_short(t.title, 30)}", SessionCb(action="pick", task_id=t.id).pack())]
        for t in tasks[:3]
    ]
    kb.append([_btn("📅 Мой день", DayCb(action="show").pack())])
    return Outgoing(text=text, reply_markup=InlineKeyboardMarkup(inline_keyboard=kb))


async def render_evening(ctx: AppContext, r: ReminderSchedule) -> Outgoing | None:
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        profile = await get_profile(s)
        tz = profile.timezone if profile else None
        day = _payload_day(r, now, tz)
        start, end = local_day_bounds(day, tz)
        done = list(
            (
                await s.execute(
                    select(Task).where(
                        Task.status == TaskStatus.DONE,
                        Task.completed_at >= start,
                        Task.completed_at < end,
                    )
                )
            ).scalars()
        )
        started_ids = {
            tid
            for tid in (
                await s.execute(
                    select(WorkSession.task_id).where(
                        WorkSession.task_id.is_not(None),
                        WorkSession.status.in_(
                            [SessionStatus.ACTIVE, SessionStatus.PAUSED, SessionStatus.FINISHED]
                        ),
                        WorkSession.started_at >= start,
                        WorkSession.started_at < end,
                    )
                )
            ).scalars()
            if tid is not None
        }
        done_ids = {t.id for t in done}
        started = [
            t for tid in sorted(started_ids - done_ids) if (t := await s.get(Task, tid)) is not None
        ]
        planned = await _day_tasks(s, ctx, day, tz)
        not_started = [
            t
            for t in planned
            if t.id not in done_ids and t.id not in started_ids and t.status == TaskStatus.TODO
        ]

    def section(title: str, items: list[Task]) -> str:
        head = f"{title}: {len(items)}"
        body = "\n".join(f"  • {t.title}" for t in items[:5])
        return head + ("\n" + body if body else "")

    text = (
        f"Вечерний разбор за {day.strftime('%d.%m')}.\n"
        + section("✅ Сделано", done)
        + "\n"
        + section("🔸 Начато, но не закончено", started)
        + "\n"
        + section("⏳ Не начато", not_started)
        + "\n\nБыли заметные отклонения от плана? Это не оценка — просто чтобы точнее "
        "планировать дальше."
    )
    kb = [
        [
            _btn(
                "👍 Всё нормально", ReminderCb(action="evening", reminder_id=r.id, arg="ok").pack()
            ),
            _btn(
                "Были отклонения",
                ReminderCb(action="evening", reminder_id=r.id, arg="deviation").pack(),
            ),
        ],
        [_btn("📅 План на завтра", DayCb(action="show").pack())],
    ]
    return Outgoing(text=text, reply_markup=InlineKeyboardMarkup(inline_keyboard=kb))


async def render_weekly(ctx: AppContext, r: ReminderSchedule) -> Outgoing | None:
    """Minimal counters; full weekly analytics (six directions, FR-11) is stage C."""
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        profile = await get_profile(s)
        tz = profile.timezone if profile else None
        last_day = _payload_day(r, now, tz)
        start = local_day_bounds(last_day - timedelta(days=6), tz)[0]
        end = local_day_bounds(last_day, tz)[1]
        done = (
            await s.execute(
                select(func.count(Task.id)).where(
                    Task.status == TaskStatus.DONE,
                    Task.completed_at >= start,
                    Task.completed_at < end,
                )
            )
        ).scalar_one()
        sessions = (
            await s.execute(
                select(WorkSession.time_source, func.count(WorkSession.id))
                .where(
                    WorkSession.status == SessionStatus.FINISHED,
                    WorkSession.started_at >= start,
                    WorkSession.started_at < end,
                )
                .group_by(WorkSession.time_source)
            )
        ).all()
        postponed = (
            await s.execute(
                select(func.count(TaskEvent.id)).where(
                    TaskEvent.event_type.in_([TaskEventType.POSTPONED, TaskEventType.RESCHEDULED]),
                    TaskEvent.created_at >= start,
                    TaskEvent.created_at < end,
                )
            )
        ).scalar_one()
        reasons = Counter(
            (
                await s.execute(
                    select(Intervention.reason_category).where(
                        Intervention.reason_category.is_not(None),
                        Intervention.created_at >= start,
                        Intervention.created_at < end,
                    )
                )
            ).scalars()
        )
    by_source = {str(src): int(n) for src, n in sessions}
    exact = by_source.get(TimeSource.EXACT, 0)
    other = sum(by_source.values()) - exact
    sess_line = f"Рабочих сессий: {exact} с точным таймером"
    if other:
        sess_line += f", ещё {other} отмечено задним числом (время приблизительное)"
    labels = {k.value: v for k, v in REASON_LABELS.items()}
    if reasons:
        reason_line = "Причины задержек, которые ты отмечал: " + ", ".join(
            f"{labels.get(k or '', k)} — {n}" for k, n in reasons.most_common(4)
        )
    else:
        reason_line = "Причины задержек: отметок не было."
    text = (
        f"Итоги недели {(last_day - timedelta(days=6)).strftime('%d.%m')}–"
        f"{last_day.strftime('%d.%m')}.\n"
        f"✅ Завершено задач: {done}\n"
        f"⏱ {sess_line}\n"
        f"📆 Переносов: {postponed}\n"
        f"🧩 {reason_line}\n\n"
        "Это пока базовые счётчики. Полная недельная аналитика — задержки старта, "
        "эффективность советов, продвижение целей и гипотезы — появится в следующих "
        "версиях; сейчас этих данных в отчёте нет."
    )
    kb = [[_btn("📅 Мой день", DayCb(action="show").pack())]]
    return Outgoing(text=text, reply_markup=InlineKeyboardMarkup(inline_keyboard=kb))


async def _task_for(
    s: AsyncSession, r: ReminderSchedule, now: datetime
) -> tuple[Task, datetime | None] | None:
    task_id = r.task_id or (r.payload or {}).get("task_id")
    if task_id is None:
        return None
    task = await s.get(Task, int(task_id))
    planned = payload_dt(r.payload or {}, "planned_start_at")
    if not await task_reminder_relevant(s, task, planned, now):
        return None
    assert task is not None
    return task, planned


async def render_pre_task(ctx: AppContext, r: ReminderSchedule) -> Outgoing | None:
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        found = await _task_for(s, r, now)
        if found is None:
            return None
        task, planned = found
        snoozed = bool((r.payload or {}).get("snoozed_from"))
        if planned is not None and not snoozed and planned <= now:
            # the start-delay question (if configured) takes over once it is due
            delay = (
                await s.execute(
                    select(ReminderSchedule.id).where(
                        ReminderSchedule.dedup_key == start_delay_key(task.id, planned),
                        ReminderSchedule.due_at <= now,
                    )
                )
            ).first()
            if delay is not None:
                return None
        profile = await get_profile(s)
        tz = profile.timezone if profile else None
    at = fmt_local(planned, tz, "%H:%M") if planned else ""
    if planned is not None and planned > now:
        minutes = max(1, round((planned - now).total_seconds() / 60))
        text = f"Через {minutes} мин, в {at} — «{task.title}»."
    elif planned is not None:
        text = f"Пора начинать «{task.title}» (по плану в {at})."
    else:
        text = f"Напоминаю про «{task.title}»."
    if task.estimate_minutes:
        text += f"\nОценка: ≈{task.estimate_minutes} мин."
    kb = [
        [
            _btn("▶️ Начать сейчас", SessionCb(action="pick", task_id=task.id).pack()),
            _btn("⏰ Позже", ReminderCb(action="later", reminder_id=r.id, task_id=task.id).pack()),
        ]
    ]
    return Outgoing(text=text, reply_markup=InlineKeyboardMarkup(inline_keyboard=kb))


def start_delay_keyboard(reminder_id: int, task_id: int) -> InlineKeyboardMarkup:
    def reason(rsn: ObstacleReason) -> InlineKeyboardButton:
        return _btn(
            REASON_LABELS[rsn],
            ReminderCb(
                action="reason", reminder_id=reminder_id, task_id=task_id, arg=rsn.value
            ).pack(),
        )

    R = ObstacleReason
    rows = [
        [_btn("▶️ Начинаю", SessionCb(action="pick", task_id=task_id).pack())],
        [reason(R.UNCLEAR)],
        [reason(R.TOO_BIG), reason(R.UNPLEASANT)],
        [reason(R.DISTRACTIONS), reason(R.LOW_ENERGY)],
        [reason(R.PRIORITY_CONFLICT), reason(R.EXTERNAL)],
        [reason(R.OTHER)],
        [
            _btn("📆 Перенести", TaskCb(action="postpone", task_id=task_id).pack()),
            _btn(
                "☕ Отдохнуть",
                ReminderCb(action="rest", reminder_id=reminder_id, task_id=task_id).pack(),
            ),
        ],
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def render_start_delay(ctx: AppContext, r: ReminderSchedule) -> Outgoing | None:
    """Scenario C, step 3: a SHORT question with quick buttons first (acceptance 9)."""
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        found = await _task_for(s, r, now)
        if found is None:
            return None
        task, planned = found
        profile = await get_profile(s)
        tz = profile.timezone if profile else None
    at = fmt_local(planned, tz, "%H:%M") if planned else None
    head = f"«{task.title}» было запланировано на {at}" if at else f"«{task.title}»"
    text = (
        f"{head}, старт пока не отмечен.\n"
        "Если уже работаешь — нажми «Начинаю». Если что-то мешает, выбери, что ближе:"
    )
    return Outgoing(text=text, reply_markup=start_delay_keyboard(r.id, task.id))


def register(ctx: AppContext) -> None:
    """Register initiative renderers and the seeder job."""
    register_renderer(ctx, MessageCategory.MORNING_PLAN, render_morning)
    register_renderer(ctx, MessageCategory.EVENING_REVIEW, render_evening)
    register_renderer(ctx, MessageCategory.WEEKLY_REVIEW, render_weekly)
    register_renderer(ctx, MessageCategory.PRE_TASK, render_pre_task)
    register_renderer(ctx, MessageCategory.START_DELAY, render_start_delay)
    # REPLAN: hook only - a payload text renderer until stage B registers its own.
    register_renderer(ctx, MessageCategory.REPLAN, render_payload_text)
    register_job(
        ctx,
        PeriodicJob(
            name=SEED_JOB_NAME,
            interval_seconds=SEED_INTERVAL_SECONDS,
            run=_seed_job,
            run_on_start=True,
        ),
    )
