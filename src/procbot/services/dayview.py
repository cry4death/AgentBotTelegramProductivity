"""Deterministic "Мой день" (stage A, no AI): overdue, planned today (by time),
due today, missed plans, backlog size; the nearest step of each task (FR-05).

Kept in a service so the morning-plan renderer (stage B) and the scheduler can
reuse it. Text is plain (no parse mode).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.models import Task
from procbot.services import tasks as task_service
from procbot.telegram.callbacks import DayCb, SessionCb, TaskCb
from procbot.timeutil import fmt_local

WEEKDAYS = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")
MAX_BUTTON_ROWS = 10
STAGE_A_NOTE = "План на утро с учётом календаря и приоритетов от ИИ появится на этапе B."


@dataclass
class DayView:
    buckets: task_service.TodayBuckets
    next_steps: dict[int, Task] = field(default_factory=dict)

    @property
    def day(self) -> date:
        return self.buckets.day

    @property
    def tz(self) -> str:
        return self.buckets.tz

    @property
    def is_empty(self) -> bool:
        return not self.buckets.all()


async def build_day_view(
    session: AsyncSession, ctx: AppContext, *, day: date | None = None
) -> DayView:
    buckets = await task_service.today_buckets(session, ctx, day=day)
    steps: dict[int, Task] = {}
    for t in buckets.all():
        sub = await task_service.next_open_subtask(session, t.id)
        if sub is not None:
            steps[t.id] = sub
    return DayView(buckets=buckets, next_steps=steps)


def _short(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _line(view: DayView, t: Task, *, show_time: bool) -> list[str]:
    tz, today = view.tz, view.day
    prefix = ""
    if show_time and t.planned_start_at is not None:
        prefix = fmt_local(t.planned_start_at, tz, "%H:%M") + " "
    mark = "▶ " if t.status == "in_progress" else ""
    extra: list[str] = []
    due = task_service.format_due(t, tz, today)
    if due:
        extra.append(due)
    if t.estimate_minutes:
        extra.append(f"~{t.estimate_minutes} мин")
    if t.postpone_count:
        extra.append(f"переносов: {t.postpone_count}")
    tail = f" ({', '.join(extra)})" if extra else ""
    out = [f"• {prefix}{mark}{t.title}{tail}"]
    step = view.next_steps.get(t.id)
    if step is not None:
        out.append(f"   ↳ следующий шаг: {step.title}")
    return out


def render_day_text(view: DayView, *, note: str | None = STAGE_A_NOTE) -> str:
    b = view.buckets
    lines = [f"📅 Мой день — {WEEKDAYS[b.day.weekday()]}, {b.day.strftime('%d.%m')}"]
    sections = (
        ("⚠️ Просрочено", b.overdue, False),
        ("🕘 Сегодня по плану", b.planned, True),
        ("📌 Срок сегодня", b.due_today, True),
        ("↩️ Не сделано в запланированный день", b.missed, False),
    )
    for header, items, show_time in sections:
        if not items:
            continue
        lines.append("")
        lines.append(header)
        for t in items:
            lines.extend(_line(view, t, show_time=show_time))
    if view.is_empty:
        lines.append("")
        lines.append("На сегодня ничего не запланировано.")
    if b.backlog_count:
        lines.append("")
        lines.append(f"В бэклоге {b.backlog_count} без даты — можно взять оттуда что-то посильное.")
    if note:
        lines.append("")
        lines.append(note)
    return "\n".join(lines)


def render_day_keyboard(view: DayView) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for t in view.buckets.all()[:MAX_BUTTON_ROWS]:
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"▶️ {_short(t.title, 28)}",
                    callback_data=SessionCb(action="pick", task_id=t.id).pack(),
                ),
                InlineKeyboardButton(
                    text="✅", callback_data=TaskCb(action="done", task_id=t.id).pack()
                ),
                InlineKeyboardButton(
                    text="…", callback_data=TaskCb(action="view", task_id=t.id).pack()
                ),
            ]
        )
    nav = [InlineKeyboardButton(text="🔄 Обновить", callback_data=DayCb(action="show").pack())]
    if view.buckets.backlog_count:
        nav.append(
            InlineKeyboardButton(
                text="📥 Бэклог", callback_data=TaskCb(action="list", arg="backlog").pack()
            )
        )
    rows.append(nav)
    return InlineKeyboardMarkup(inline_keyboard=rows)


def render_day_view(
    view: DayView, *, note: str | None = STAGE_A_NOTE
) -> tuple[str, InlineKeyboardMarkup]:
    return render_day_text(view, note=note), render_day_keyboard(view)
