"""Reports (FR-11): evening review and weekly review.

Numbers come from services.analytics (computed by code); this module only words
them (templates are kept apart from the computation: the exact report forms are
open question 9). The weekly review may ask the ``weekly_analysis`` AI task for
a comment, hypotheses and proposals - the model receives ONLY precomputed
numbers, and a validator rejects numbers in its text that the code did not
compute. Without AI (AIUnavailable / switched off) the report is built from
templates only.

- AI hypotheses are saved through the ActionExecutor (SAVE_HYPOTHESIS, AUTO,
  stored as ``ai_inferred``); proposals that would change data become pending
  approvals (never applied silently).
- Recurring reasons measured by code are added to memory as ``system_measured``
  patterns and the report says so (transparent learning, no settings changes).
- The owner can dispute a conclusion (scenario H): hypothesis refuted / memory
  corrected, see handlers.reports and handlers.memory.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Literal

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.ai import AITaskSpec, AIUnavailable, DataBlock, Prompt, Tier, register_task
from procbot.ai import prompting as P
from procbot.ai.gateway import feature_enabled, run_task
from procbot.context import AppContext
from procbot.db.models import ConversationMessage, MemoryItem, ReminderSchedule, Task, TaskEvent
from procbot.domain.actions import ChangeTaskStatus, RescheduleTask, SaveHypothesis
from procbot.domain.enums import (
    Actor,
    MemoryKind,
    MemoryOrigin,
    MessageCategory,
    ObstacleReason,
    TaskEventType,
    TaskStatus,
)
from procbot.services import actions, analytics, memory
from procbot.services.analytics import (
    INSUFFICIENT,
    QUALITY_LABELS,
    REASON_LABELS,
    Analytics,
    DaySummary,
    Direction,
    Metric,
    Period,
    Quality,
)
from procbot.services.outbound import Outgoing, register_renderer
from procbot.telegram.callbacks import DayCb, MemoryCb, ReportCb, TaskCb
from procbot.timeutil import local_date, local_day_bounds

log = logging.getLogger(__name__)

WEEKLY_TASK = "weekly_analysis"
TELEGRAM_LIMIT = 4000
MAX_PERIOD_DAYS = 92
MAX_DISPUTE_BUTTONS = 5

# --------------------------------------------------------------------------- AI task


class WeeklyHypothesisOut(BaseModel):
    content: str = Field(min_length=3, max_length=500)
    verification_hint: str = Field(min_length=3, max_length=500)  # concrete way to test it
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class WeeklyProposalOut(BaseModel):
    text: str = Field(min_length=3, max_length=300)
    kind: Literal["advice", "reschedule_task", "defer_task"] = "advice"
    task_id: int | None = None
    planned_date: date | None = None


class WeeklyAnalysisOut(BaseModel):
    summary: str = Field(min_length=1, max_length=2000)
    hypotheses: list[WeeklyHypothesisOut] = Field(default_factory=list, max_length=5)
    proposals: list[WeeklyProposalOut] = Field(default_factory=list, max_length=5)


WEEKLY_SYSTEM = (
    "Ты пишешь короткий комментарий к недельной аналитике персонального помощника против "
    "прокрастинации. Все цифры уже посчитаны кодом и переданы в данных: НЕ считай ничего "
    "сам, не выводи новых чисел, не пересчитывай проценты — упоминай только числа из данных. "
    "Время «retro/approximate» — приблизительное, никогда не называй его точным. "
    "Связь между помощью и результатом не доказывает причинность — не утверждай обратного. "
    "Гипотезы — предположения о закономерностях, не диагнозы: у каждой укажи конкретный "
    "способ проверить её на следующей неделе. Не ставь психологических диагнозов и не "
    "обвиняй. Предложения на новую неделю: kind=advice — просто совет; reschedule_task / "
    "defer_task — только для задач из списка open_tasks (task_id оттуда), они пойдут "
    "пользователю на подтверждение."
)

WEEKLY_ANALYSIS = register_task(
    AITaskSpec(
        name=WEEKLY_TASK,
        tier=Tier.STRONG,
        output=WeeklyAnalysisOut,
        max_output_tokens=1500,
        data_categories=frozenset({P.ANALYTICS, P.MEMORY, P.TASKS}),
        system=WEEKLY_SYSTEM,
        description="Недельный разбор: текст, гипотезы и предложения по готовым цифрам",
    )
)

_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _numbers(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _NUM.findall(text):
        v = raw.replace(",", ".")
        if "." in v:
            v = v.rstrip("0").rstrip(".") or "0"
        out.add(v)
    return out


def make_validator(payload: dict[str, Any], open_task_ids: set[int], after: date):
    """Business checks of a weekly_analysis answer (beyond the schema)."""
    allowed = _numbers(json.dumps(payload, ensure_ascii=False)) | {str(i) for i in range(11)}

    def validate(out: BaseModel) -> list[str]:
        assert isinstance(out, WeeklyAnalysisOut)
        errors: list[str] = []
        invented = sorted(_numbers(out.summary) - allowed)
        if invented:
            errors.append(
                "summary содержит числа, которых нет в данных (цифры считает код, не ты): "
                + ", ".join(invented[:10])
            )
        for i, pr in enumerate(out.proposals):
            if pr.kind == "advice":
                continue
            if pr.task_id is None or pr.task_id not in open_task_ids:
                errors.append(f"proposals[{i}].task_id должен быть из open_tasks")
            if pr.kind == "reschedule_task" and (
                pr.planned_date is None or pr.planned_date <= after
            ):
                errors.append(f"proposals[{i}].planned_date должен быть после {after.isoformat()}")
        return errors

    return validate


# --------------------------------------------------------------------------- templates


def fmt_metric(m: Metric) -> str:
    value = INSUFFICIENT if m.value is None else m.value
    unit = f" {m.unit}" if m.unit and not isinstance(value, str) else ""
    extra = [QUALITY_LABELS[m.quality]]
    if m.note:
        extra.append(m.note)
    return f"• {m.label}: {value}{unit} ({'; '.join(extra)})"


def render_hypotheses(d: Direction, *, limit: int = MAX_DISPUTE_BUTTONS) -> list[str]:
    lines = [d.title]
    hyps: list[dict[str, Any]] = d.data.get("hypotheses", [])
    new_ids = set(d.data.get("new_ids", []))
    if not hyps:
        lines.append(f"• {INSUFFICIENT}: гипотез пока нет.")
    for i, h in enumerate(hyps[:limit], 1):
        mark = "🆕 " if h["id"] in new_ids else ""
        check = f" Как проверить: {h['how_to_check']}" if h.get("how_to_check") else ""
        origin = memory.ORIGIN_LABELS.get(h.get("origin") or "", "предположение ИИ")
        lines.append(f"• Г{i}. {mark}{h['content']} ({origin}).{check}")
    if len(hyps) > limit:
        lines.append(f"…и ещё {len(hyps) - limit} — в «🧠 Память».")
    lines += [f"ℹ️ {n}" for n in d.notes]
    return lines


def render_direction(d: Direction, *, max_details: int = 4) -> list[str]:
    if d.key == "hypotheses":
        return render_hypotheses(d)
    lines = [d.title]
    if not d.has_data:
        lines.append(f"• {INSUFFICIENT}")
    else:
        lines += [fmt_metric(m) for m in d.metrics]
        lines += [f"• {x}" for x in d.details[:max_details]]
    lines += [f"ℹ️ {n}" for n in d.notes]
    return lines


def template_proposals(a: Analytics) -> list[str]:
    """Code-made suggestions for the next week (used with or without AI)."""
    out: list[str] = []
    sd = a.directions["start_delay"].data
    if sd.get("unconfirmed_n"):
        out.append(
            "Отмечай начало кнопкой «▶️ Начать» — тогда задержка старта будет измерена точно."
        )
    tp = a.directions["tasks_plans"].data
    if tp.get("approved_plans", 0) == 0:
        out.append("Попробуй утверждать план дня — будет с чем сравнить выполнение.")
    if a.directions["procrastination"].data.get("tasks_postponed_repeatedly"):
        out.append(
            "Для задач, которые переносились несколько раз, стоит выбрать первый маленький шаг."
        )
    return out


@dataclass
class WeeklyAI:
    status: str  # ok / unavailable / disabled
    summary: str | None = None
    advice: list[str] = field(default_factory=list)
    call_id: int | None = None


def render_weekly_text(
    a: Analytics,
    ai: WeeklyAI,
    *,
    measured: list[str],
    pending_proposals: int,
    title: str | None = None,
) -> str:
    head = title or f"📊 Итоги недели {a.period.label}"
    for max_details in (4, 2, 0):
        parts = [
            head,
            "Цифры посчитаны кодом по твоим отметкам; в скобках — качество источника.",
            "",
        ]
        for d in a.ordered():
            parts += render_direction(d, max_details=max_details)
            parts.append("")
        if ai.status == "ok" and ai.summary:
            parts += [
                "🤖 Комментарий ИИ (по готовым цифрам, сам ничего не считал):",
                ai.summary,
                "",
            ]
        elif ai.status == "unavailable":
            parts += [
                "ИИ сейчас недоступен — отчёт собран по шаблону, без комментария и новых гипотез.",
                "",
            ]
        elif ai.status == "disabled":
            parts += ["Комментарий ИИ к отчёту выключен — отчёт собран по шаблону.", ""]
        proposals = [*ai.advice, *template_proposals(a)]
        if proposals or pending_proposals:
            parts.append("🗓 Предложения на следующую неделю:")
            parts += [f"• {x}" for x in dict.fromkeys(proposals)]
            if pending_proposals:
                parts.append(
                    f"• Изменений, которые ждут твоего подтверждения: {pending_proposals} — "
                    "кнопка «📝 Предложения». Без подтверждения ничего не изменится."
                )
            parts.append("")
        if measured:
            parts.append(
                "🧠 Добавлено в память как измеренное: "
                + "; ".join(f"«{m}»" for m in measured)
                + ". Посмотреть и оспорить — «🧠 Память»."
            )
        parts.append("Не согласен с выводом? Нажми «Не согласен» — память будет исправлена.")
        text = "\n".join(parts).strip()
        if len(text) <= TELEGRAM_LIMIT:
            return text
    return text[: TELEGRAM_LIMIT - 1] + "…"


def weekly_keyboard(
    hypothesis_ids: list[int], *, group_key: str | None, pending_proposals: int
) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    pair: list[InlineKeyboardButton] = []
    for i, item_id in enumerate(hypothesis_ids[:MAX_DISPUTE_BUTTONS], 1):
        pair.append(
            InlineKeyboardButton(
                text=f"🙅 Не согласен с Г{i}",
                callback_data=MemoryCb(action="refute", item_id=item_id, arg="wk").pack(),
            )
        )
        if len(pair) == 2:
            rows.append(pair)
            pair = []
    if pair:
        rows.append(pair)
    rows.append(
        [
            InlineKeyboardButton(
                text="💬 Не согласен с выводом", callback_data=ReportCb(action="dispute").pack()
            )
        ]
    )
    if pending_proposals and group_key:
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"📝 Предложения ({pending_proposals})",
                    callback_data=ReportCb(action="props", arg=group_key).pack(),
                )
            ]
        )
    rows.append(
        [InlineKeyboardButton(text="🧠 Память", callback_data=MemoryCb(action="home").pack())]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


# --------------------------------------------------------------------------- weekly


@dataclass
class WeeklyReport:
    text: str
    markup: InlineKeyboardMarkup
    analytics: Analytics
    ai: WeeklyAI
    hypothesis_ids: list[int]
    new_hypothesis_ids: list[int]
    approval_ids: list[int]
    group_key: str | None
    measured: list[str]


async def _open_tasks(s: AsyncSession, limit: int = 30) -> list[dict[str, Any]]:
    rows = (
        (
            await s.execute(
                select(Task)
                .where(
                    Task.status.in_([TaskStatus.TODO, TaskStatus.IN_PROGRESS]),
                    Task.parent_id.is_(None),
                )
                .order_by(Task.id)
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return [{"id": t.id, "title": t.title} for t in rows]


async def record_measured_patterns(s: AsyncSession, ctx: AppContext, a: Analytics) -> list[str]:
    """Recurring reasons -> memory patterns marked as measured. Returns the texts
    of NEW items (the report names them, nothing changes silently)."""
    created: list[str] = []
    for code, n in analytics.recurring_reasons(a).items():
        label = REASON_LABELS.get(code, code)
        content = f"Часто отмечаемая причина задержек: «{label}»"
        sources = [
            memory.SourceRef(memory.SourceType.INTERVENTION, i)
            for i in a.evidence.get(f"reason:{code}", [])
        ] + [
            memory.SourceRef(memory.SourceType.TASK_EVENT, i)
            for i in a.evidence.get(f"reason_ev:{code}", [])
        ]
        item, new = await memory.add_measured_pattern(
            s,
            ctx,
            content,
            sources=sources,
            verification_hint=(
                f"Отмечено {n} раз за {a.period.label}. Проверить: повторится ли эта причина "
                "в отметках следующих недель."
            ),
        )
        if item is not None and new:
            created.append(content)
    return created


def _proposal_action(pr: WeeklyProposalOut) -> Any | None:
    if pr.kind == "reschedule_task" and pr.task_id is not None and pr.planned_date is not None:
        return RescheduleTask(
            task_id=pr.task_id, planned_date=pr.planned_date, reason=pr.text[:200]
        )
    if pr.kind == "defer_task" and pr.task_id is not None:
        return ChangeTaskStatus(task_id=pr.task_id, status="someday", reason=pr.text[:200])
    return None


async def build_weekly(
    ctx: AppContext,
    *,
    first_day: date,
    last_day: date,
    op_prefix: str,
    use_ai: bool = True,
    title: str | None = None,
) -> WeeklyReport:
    """Compute, (optionally) comment with AI, persist hypotheses / proposals /
    measured patterns, render. Must run outside a transaction."""
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        tz = await analytics.owner_tz(s)
        period = Period.of(first_day, last_day, tz)
        a = await analytics.compute(s, ctx, period)
        open_tasks = await _open_tasks(s)
        existing = [
            {"id": h.id, "content": h.content} for h in await memory.active_hypotheses(s, limit=10)
        ]
        today = local_date(now, tz)
    payload = analytics.ai_payload(a)
    ai = WeeklyAI(status="disabled")
    out: WeeklyAnalysisOut | None = None
    if use_ai and feature_enabled(ctx, "weekly_analysis"):
        prompt = Prompt(
            instruction=(
                "Напиши 3–6 предложений комментария к недельной аналитике (только по данным), "
                "до 3 гипотез о закономерностях с конкретным способом проверки и до 3 "
                "предложений на следующую неделю. Не повторяй уже сохранённые гипотезы."
            ),
            data=[
                DataBlock(P.ANALYTICS, "weekly_numbers", payload),
                DataBlock(P.TASKS, "open_tasks", open_tasks),
                DataBlock(P.MEMORY, "saved_hypotheses", existing),
            ],
        )
        validator = make_validator(payload, {t["id"] for t in open_tasks}, max(today, last_day))
        try:
            res = await run_task(
                ctx, WEEKLY_TASK, prompt, initiator="services.reports", validator=validator
            )
            out = res.as_(WeeklyAnalysisOut)
            ai = WeeklyAI(status="ok", summary=out.summary.strip(), call_id=res.call_id)
        except AIUnavailable as exc:
            log.info("weekly analysis without AI: %s", exc)
            ai = WeeklyAI(status="unavailable")
    group_key = f"{op_prefix}:wk"
    new_ids: list[int] = []
    approval_ids: list[int] = []
    async with ctx.db.transaction() as s:
        measured = await record_measured_patterns(s, ctx, a)
        if out is not None:
            for i, h in enumerate(out.hypotheses):
                r = await actions.execute(
                    s,
                    ctx,
                    SaveHypothesis(
                        content=h.content,
                        verification_hint=h.verification_hint,
                        confidence=h.confidence,
                    ),
                    origin=Actor.AI,
                    op_key=f"{op_prefix}:wk:h{i}",
                    ai_call_id=ai.call_id,
                )
                item_id = r.result.get("memory_item_id")
                if r.status == actions.ExecStatus.APPLIED and item_id is not None:
                    new_ids.append(int(item_id))
            for i, pr in enumerate(out.proposals):
                action = _proposal_action(pr)
                if action is None:
                    ai.advice.append(pr.text)
                    continue
                r = await actions.execute(
                    s,
                    ctx,
                    action,
                    origin=Actor.AI,
                    op_key=f"{op_prefix}:wk:p{i}",
                    group_key=group_key,
                    ai_call_id=ai.call_id,
                )
                if r.status == actions.ExecStatus.PENDING_APPROVAL and r.approval_id is not None:
                    approval_ids.append(r.approval_id)
                elif not r.ok:
                    ai.advice.append(pr.text)  # cannot be applied: shown as advice only
        hyps = await memory.active_hypotheses(s, limit=10)
        a.directions["hypotheses"] = analytics.hypotheses_direction(hyps, new_ids=new_ids)
        hypothesis_ids = [h.id for h in hyps]
    pending = len(approval_ids)
    text = render_weekly_text(a, ai, measured=measured, pending_proposals=pending, title=title)
    markup = weekly_keyboard(
        hypothesis_ids, group_key=group_key if pending else None, pending_proposals=pending
    )
    return WeeklyReport(
        text=text,
        markup=markup,
        analytics=a,
        ai=ai,
        hypothesis_ids=hypothesis_ids,
        new_hypothesis_ids=new_ids,
        approval_ids=approval_ids,
        group_key=group_key if pending else None,
        measured=measured,
    )


# --------------------------------------------------------------------------- evening


def _short(title: str, n: int = 28) -> str:
    return title if len(title) <= n else title[: n - 1] + "…"


def _titles(tasks: list[Task], n: int = 5) -> list[str]:
    lines = [f"  • {t.title}" for t in tasks[:n]]
    if len(tasks) > n:
        lines.append(f"  …и ещё {len(tasks) - n}")
    return lines


def render_evening(summary: DaySummary) -> tuple[str, InlineKeyboardMarkup]:
    lines = [f"🌙 Итоги дня {summary.day.strftime('%d.%m')}"]
    lines.append(f"✅ Сделано: {len(summary.done)}")
    lines += _titles(summary.done)
    if summary.in_progress:
        lines.append(f"🔸 Начато или частично (это ещё не «сделано»): {len(summary.in_progress)}")
        lines += _titles(summary.in_progress, 3)
    if summary.not_started:
        lines.append(f"⏳ Не начато из запланированного: {len(summary.not_started)}")
        lines += _titles(summary.not_started, 3)
    if summary.postponed_today:
        lines.append(f"📆 Перенесено: {len(summary.postponed_today)}")
    work: list[str] = []
    if summary.work_exact_min:
        work.append(f"{summary.work_exact_min} мин по таймеру ({QUALITY_LABELS[Quality.EXACT]})")
    if summary.work_retro_min:
        work.append(
            f"{summary.work_retro_min} мин задним числом "
            f"({QUALITY_LABELS[Quality.RETRO]}, приблизительно)"
        )
    lines.append("⏱ Работа: " + ("; ".join(work) if work else "отметок нет"))
    exact = [
        o.minutes for o in summary.delays if o.quality == Quality.EXACT and o.minutes is not None
    ]
    retro = [
        o.minutes for o in summary.delays if o.quality == Quality.RETRO and o.minutes is not None
    ]
    unconf = [o for o in summary.delays if o.started is None]
    starts: list[str] = []
    if exact:
        starts.append(
            f"задержка {analytics.fmt_delay(statistics.median(exact))} "
            f"({QUALITY_LABELS[Quality.EXACT]}, стартов: {len(exact)})"
        )
    if retro:
        starts.append(
            f"по отметкам задним числом {analytics.fmt_delay(statistics.median(retro))} "
            f"({QUALITY_LABELS[Quality.RETRO]}, приблизительно)"
        )
    if unconf:
        starts.append(
            f"без подтверждённого начала: {len(unconf)} ({QUALITY_LABELS[Quality.UNKNOWN]})"
        )
    if starts:
        lines.append("🚦 Старт: " + "; ".join(starts))
    rows: list[list[InlineKeyboardButton]] = []
    deviations = summary.deviations
    lines.append("")
    if deviations:
        lines.append("Заметные отклонения: " + "; ".join(deviations) + ".")
        if summary.not_started:
            lines.append("Что помешало? Выбери задачу — это чтобы точнее планировать, а не оценка.")
        for t in summary.not_started[:3]:
            rows.append(
                [
                    InlineKeyboardButton(
                        text=f"🤔 {_short(t.title)}",
                        callback_data=ReportCb(action="why", arg=str(t.id)).pack(),
                    ),
                    InlineKeyboardButton(
                        text="📆 Перенести",
                        callback_data=TaskCb(action="postpone", task_id=t.id).pack(),
                    ),
                ]
            )
    else:
        lines.append("Заметных отклонений от плана нет — вопросов не будет. Хорошего вечера!")
    rows.append(
        [
            InlineKeyboardButton(
                text="📅 План на завтра", callback_data=DayCb(action="show").pack()
            ),
            InlineKeyboardButton(text="📊 Неделя", callback_data=ReportCb(action="week").pack()),
        ]
    )
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


async def build_evening(
    ctx: AppContext, day: date | None = None
) -> tuple[str, InlineKeyboardMarkup]:
    async with ctx.db.transaction() as s:
        tz = await analytics.owner_tz(s)
        d = day or local_date(ctx.clock.now(), tz)
        summary = await analytics.day_summary(s, ctx, d)
    return render_evening(summary)


async def record_evening_reason(
    s: AsyncSession, ctx: AppContext, task_id: int, reason: str
) -> bool:
    """The owner's answer "what got in the way" for a planned task. Recorded as
    an obstacle on the task (a hypothesis, not a diagnosis). Repeated presses of
    the same answer the same day are no-ops. Returns True if recorded now."""
    from procbot.services import tasks as task_service

    code = ObstacleReason(reason).value
    task = await task_service.require_task(s, task_id)
    tz = await analytics.owner_tz(s)
    start, end = local_day_bounds(local_date(ctx.clock.now(), tz), tz)
    events = (
        (
            await s.execute(
                select(TaskEvent).where(
                    TaskEvent.task_id == task.id,
                    TaskEvent.event_type == TaskEventType.OBSTACLE_REPORTED,
                    TaskEvent.created_at >= start,
                    TaskEvent.created_at < end,
                )
            )
        )
        .scalars()
        .all()
    )
    if any((e.payload or {}).get("category") == code for e in events):
        return False
    await task_service.report_obstacle(
        s, ctx, task.id, reason=code, actor=Actor.USER, note="вечерний разбор"
    )
    return True


async def record_dispute(
    s: AsyncSession, ctx: AppContext, text: str, *, update_id: int | None = None
) -> MemoryItem | None:
    """Scenario H: the owner disagrees with a conclusion. Stored as their own
    statement (origin user_said) linked to the message; None if memory is off."""
    sources: list[memory.SourceRef] = []
    if update_id is not None:
        msg_id = (
            await s.execute(
                select(ConversationMessage.id).where(
                    ConversationMessage.update_id == update_id,
                    ConversationMessage.direction == "in",
                )
            )
        ).scalar_one_or_none()
        if msg_id is not None:
            sources.append(memory.SourceRef(memory.SourceType.MESSAGE, int(msg_id)))
    return await memory.add_fact(
        s,
        ctx,
        f"Поправка к выводам отчёта: {text.strip()}",
        kind=MemoryKind.FACT,
        origin=MemoryOrigin.USER_SAID,
        sources=sources,
        actor=Actor.USER,
    )


# --------------------------------------------------------------------------- periods


_DATE = r"(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?"
_PERIOD = re.compile(rf"^\s*{_DATE}\s*(?:-|–|—|по|до)\s*{_DATE}\s*$", re.IGNORECASE)


def _mk_date(d: str, m: str, y: str | None, today: date) -> date:
    if y:
        year = int(y) + (2000 if len(y) == 2 else 0)
        return date(year, int(m), int(d))
    candidate = date(today.year, int(m), int(d))
    return candidate if candidate <= today else date(today.year - 1, int(m), int(d))


def parse_period(text: str, today: date) -> tuple[date, date]:
    """'01.10-15.10', '1.10.2026 – 15.10.2026', '01.10 по 15.10'. Raises ValueError."""
    m = _PERIOD.match(text)
    if m is None:
        raise ValueError("bad period")
    first = _mk_date(m.group(1), m.group(2), m.group(3), today)
    last = _mk_date(m.group(4), m.group(5), m.group(6), today)
    if last < first:
        first, last = last, first
    if last > today:
        raise ValueError("future")
    if (last - first).days + 1 > MAX_PERIOD_DAYS:
        raise ValueError("too long")
    return first, last


# --------------------------------------------------------------------------- renderers


def _payload_day(r: ReminderSchedule, fallback: date) -> date:
    raw = (r.payload or {}).get("local_date")
    if raw:
        try:
            return date.fromisoformat(str(raw))
        except ValueError:
            pass
    return fallback


async def render_evening_review(ctx: AppContext, r: ReminderSchedule) -> Outgoing | None:
    async with ctx.db.transaction() as s:
        tz = await analytics.owner_tz(s)
    day = _payload_day(r, local_date(ctx.clock.now(), tz))
    text, kb = await build_evening(ctx, day)
    return Outgoing(text=text, reply_markup=kb)


async def render_weekly_review(ctx: AppContext, r: ReminderSchedule) -> Outgoing | None:
    async with ctx.db.transaction() as s:
        tz = await analytics.owner_tz(s)
    last_day = _payload_day(r, local_date(ctx.clock.now(), tz))
    rep = await build_weekly(
        ctx, first_day=last_day - timedelta(days=6), last_day=last_day, op_prefix=f"rm{r.id}"
    )
    return Outgoing(text=rep.text, reply_markup=rep.markup)


def register(ctx: AppContext) -> None:
    """Replaces the stage-A evening/weekly renderers (wiring registers this
    module after services.initiative)."""
    memory.register(ctx)  # SAVE_HYPOTHESIS applier (idempotent)
    register_renderer(ctx, MessageCategory.EVENING_REVIEW, render_evening_review)
    register_renderer(ctx, MessageCategory.WEEKLY_REVIEW, render_weekly_review)
