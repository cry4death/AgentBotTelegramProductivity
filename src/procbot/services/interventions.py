"""ProcrastinationAnalysisService + InterventionService (FR-04, scenario C,
acceptance 9/10). Extends `services.interventions_basic` (its API is re-exported
unchanged).

Flow after a start delay:
1. the START_DELAY card asks a SHORT question with reason buttons (template, no
   AI) — this module's renderer wraps the stage-A one and points the buttons here;
2. a picked reason (the owner said it: origin user_said) -> `suggest_after_reason`:
   a concrete next action tailored by `suggest_intervention` (standard tier),
   template fallback without AI. Recorded as an Intervention (kind, reason,
   origin, used_ai) + an obstacle event on the task;
3. repeated reason / repeated postponement / "unclear" -> the deeper dialog is
   OFFERED; it starts automatically only when the owner configured
   `deep_dialog_after_postpones` (open question 4: None = never automatic). The
   "Разобраться подробнее" button is always there;
4. the deep dialog (`procrastination_dialog`, strong tier) is a capped
   multi-turn exchange over the task's history (postponements, reasons, past
   interventions and their outcomes). It proposes options — simplify, split,
   reschedule, defer long, change approach, drop, rest — and the decision is
   always the owner's: rescheduling goes through the date picker (the owner's
   own choice), deferring / dropping become pending approval cards.

Wording rules: causes are hypotheses ("возможно", "похоже"), never diagnoses or
accusations; answers that state a psychological cause as fact are rejected by the
validator (repair path) and replaced by templates. The owner can refute a
hypothesis ("Это не так"); the refutation is recorded (InterventionOutcome) and
the hypothesis is not repeated. No loops of blame: the dialog is capped, and
rest / conscious postponement are valid outcomes.

Outcomes (started / progressed / completed) are recorded by services.sessions
when a session linked to the intervention runs; "declined" and "no_effect" are
recorded here (button / settle job).
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Literal

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.ai import prompting as P
from procbot.ai.errors import AIUnavailable
from procbot.ai.gateway import run_task
from procbot.ai.prompting import DataBlock, Prompt
from procbot.ai.registry import AITaskSpec, register_task
from procbot.ai.types import Tier
from procbot.context import AppContext
from procbot.db.models import (
    Intervention,
    InterventionOutcome,
    ReminderSchedule,
    Task,
    TaskEvent,
    WorkSession,
)
from procbot.domain.actions import ActionType, ChangeTaskStatus, ChooseIntervention, SaveHypothesis
from procbot.domain.enums import (
    Actor,
    InterventionKind,
    InterventionResult,
    MemoryOrigin,
    MessageCategory,
    ObstacleReason,
    TaskEventType,
    TimeSource,
)
from procbot.services import actions
from procbot.services import tasks as ts
from procbot.services.actions import ActionError, ApplyOutcome
from procbot.services.audit import audit
from procbot.services.interventions_basic import (
    find_recent_intervention,
    record_intervention_outcome,
    record_quick_intervention,
)
from procbot.services.jobs import PeriodicJob, register_job
from procbot.services.outbound import Outgoing, register_renderer
from procbot.services.preferences import flag
from procbot.services.profile import get_profile
from procbot.telegram.callbacks import InterventionCb

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- constants

HISTORY_DAYS = 30
MAX_DIALOG_TURNS = 3  # clarifying questions; then only options (no endless loops)
SETTLE_AFTER = timedelta(hours=12)  # UNCONFIRMED: no session since -> no_effect
SETTLE_JOB = "interventions.settle"
SETTLE_INTERVAL_SECONDS = 3600.0

QUICK_KINDS: frozenset[str] = frozenset(
    {
        InterventionKind.TINY_STEP,
        InterventionKind.SHORT_START,
        InterventionKind.SPLIT,
        InterventionKind.SIMPLIFY,
        InterventionKind.RESCHEDULE,
        InterventionKind.REMOVE_DISTRACTIONS,
        InterventionKind.REST,
        InterventionKind.CHANGE_APPROACH,
    }
)
DialogKind = Literal[
    "simplify",
    "split",
    "reschedule",
    "defer_long",
    "change_approach",
    "drop",
    "rest",
    "short_start",
]
DIALOG_KINDS: tuple[str, ...] = (
    "simplify",
    "split",
    "reschedule",
    "defer_long",
    "change_approach",
    "drop",
    "rest",
    "short_start",
)
OPTION_LABELS: dict[str, str] = {
    "simplify": "🪶 Упростить",
    "split": "✂️ Разбить на шаги",
    "reschedule": "📆 Перенести",
    "defer_long": "🗄 Отложить надолго",
    "change_approach": "🔀 Поменять подход",
    "drop": "🗑 Отказаться от задачи",
    "rest": "☕ Сейчас отдохнуть",
    "short_start": "⏱ Короткий старт",
}

# Statements of a cause as fact, accusations, diagnoses (acceptance 10).
_BANNED: tuple[tuple[str, str], ...] = (
    (r"\bлен(ь|ив|иш|ишь|ится)", "не называй человека ленивым"),
    (r"симул", "не говори о симуляции"),
    (r"оправдани|отговорк|отмазк", "не называй причины оправданиями/отговорками"),
    (r"диагноз|сдвг|\bадхд\b|депресс|расстройств", "без диагнозов"),
    (r"на самом деле ты", "не утверждай, что знаешь «на самом деле»"),
    (r"ты просто не хочешь", "не приписывай нежелание как факт"),
    (r"причина (в том|точно)", "причину формулируй как предположение"),
    (r"ты (просто )?(боишься|избегаешь)", "не утверждай страх/избегание как факт"),
    (r"\bсам виноват|\bтвоя вина", "без обвинений"),
)
_BANNED_RE = [(re.compile(p, re.IGNORECASE), why) for p, why in _BANNED]


def tone_errors(*texts: str | None) -> list[str]:
    """Validator part shared by both AI tasks (and used by tests)."""
    joined = "\n".join(t for t in texts if t)
    return [f"формулировка недопустима: {why}" for rx, why in _BANNED_RE if rx.search(joined)]


# --------------------------------------------------------------------------- AI schemas


class InterventionSuggestion(BaseModel):
    kind: Literal[
        "tiny_step",
        "short_start",
        "split",
        "simplify",
        "reschedule",
        "remove_distractions",
        "rest",
        "change_approach",
    ]
    next_action: str = Field(description="Одно конкретное ближайшее действие")
    message: str = Field(description="Короткое сообщение человеку; причины — как предположение")


class HypothesisOut(BaseModel):
    text: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    how_to_check: str | None = None


class DialogOptionOut(BaseModel):
    kind: DialogKind
    details: str = ""


class DialogTurnOutput(BaseModel):
    message: str
    hypotheses: list[HypothesisOut] = Field(default_factory=list, max_length=3)
    question: str | None = None
    options: list[DialogOptionOut] = Field(default_factory=list, max_length=6)


_SYSTEM = (
    "Ты помогаешь человеку сдвинуться с места, когда он откладывает задачу. Ты не "
    "психотерапевт и не ставишь диагнозов. Причины — только гипотезы: «возможно», "
    "«похоже», с вопросом или способом проверить; никогда не утверждай их как факт и "
    "не обвиняй. Отдых и сознательный перенос — нормальные исходы. Не требуй "
    "раскрывать личное. Ты ничего не меняешь сам — только предлагаешь."
)

SUGGEST_INTERVENTION = register_task(
    AITaskSpec(
        name="suggest_intervention",
        tier=Tier.STANDARD,
        output=InterventionSuggestion,
        max_output_tokens=600,
        context_budget_tokens=4000,
        data_categories=frozenset({P.TASKS, P.PROFILE, P.MEMORY, P.ANALYTICS}),
        system=_SYSTEM,
        description="Конкретное ближайшее действие после названной причины задержки",
    )
)

PROCRASTINATION_DIALOG = register_task(
    AITaskSpec(
        name="procrastination_dialog",
        tier=Tier.STRONG,
        output=DialogTurnOutput,
        max_output_tokens=1200,
        context_budget_tokens=7000,
        data_categories=frozenset(
            {P.TASKS, P.PROFILE, P.MEMORY, P.ANALYTICS, P.HISTORY, P.MESSAGE}
        ),
        system=_SYSTEM,
        description="Углублённый разбор повторных переносов",
    )
)

# --------------------------------------------------------------------------- templates

TEMPLATES: dict[ObstacleReason, tuple[InterventionKind, str]] = {
    ObstacleReason.UNCLEAR: (
        InterventionKind.TINY_STEP,
        "Похоже, пока непонятно, с какого конца взяться — это нормально. Назови самый "
        "первый физический шаг, совсем маленький, на пару минут: открыть файл, записать "
        "три пункта, найти нужную ссылку. А потом можно запустить короткий старт.",
    ),
    ObstacleReason.TOO_BIG: (
        InterventionKind.SPLIT,
        "Возможно, задача просто слишком крупная, чтобы браться за неё целиком. Давай "
        "разобьём её на шаги и начнём с самого маленького.",
    ),
    ObstacleReason.UNPLEASANT: (
        InterventionKind.SHORT_START,
        "Бывает, что задача просто неприятная. Иногда помогает договориться с собой "
        "только начать — коротко, без обязательства закончить.",
    ),
    ObstacleReason.DISTRACTIONS: (
        InterventionKind.REMOVE_DISTRACTIONS,
        "Если что-то отвлекает — попробуй убрать телефон подальше, закрыть лишние вкладки "
        "и включить таймер. Во время сессии я не буду писать с проверками.",
    ),
    ObstacleReason.LOW_ENERGY: (
        InterventionKind.REST,
        "Если сил мало, это стоит учитывать, а не преодолевать любой ценой. Можно "
        "отдохнуть, сделать совсем короткий старт или сознательно перенести задачу.",
    ),
    ObstacleReason.PRIORITY_CONFLICT: (
        InterventionKind.RESCHEDULE,
        "Если сейчас важнее другое — лучше сознательно перенести эту задачу, чтобы она не "
        "висела. Можно открыть день и пересмотреть порядок.",
    ),
    ObstacleReason.EXTERNAL: (
        InterventionKind.RESCHEDULE,
        "Внешние дела — понятная причина. Перенести задачу на другое время?",
    ),
    ObstacleReason.OTHER: (
        InterventionKind.TINY_STEP,
        "Понял. Если хочешь, разберёмся подробнее, что мешает, — или выбери, что сделать сейчас.",
    ),
}
REASON_TEXT: dict[str, str] = {
    ObstacleReason.UNCLEAR: "непонятно, с чего начать",
    ObstacleReason.TOO_BIG: "задача слишком большая",
    ObstacleReason.UNPLEASANT: "задача неприятная",
    ObstacleReason.DISTRACTIONS: "отвлечения",
    ObstacleReason.LOW_ENERGY: "мало сил",
    ObstacleReason.PRIORITY_CONFLICT: "другое важнее",
    ObstacleReason.EXTERNAL: "внешние дела",
    ObstacleReason.OTHER: "другое",
}


# --------------------------------------------------------------------------- results


@dataclass
class DeepSignal:
    repeated_reason: bool = False
    repeated_postpone: bool = False
    unclear: bool = False
    postpone_count: int = 0
    previous_same_reason: int = 0
    auto: bool = False  # owner-configured threshold reached

    @property
    def offer(self) -> bool:
        return self.repeated_reason or self.repeated_postpone or self.unclear

    def hint(self) -> str | None:
        if self.repeated_reason:
            return (
                f"Эта причина у задачи уже звучала ({self.previous_same_reason + 1}-й раз). "
                "Если хочешь, разберёмся подробнее — без спешки и оценок."
            )
        if self.repeated_postpone:
            return (
                f"Задачу переносили уже {self.postpone_count} раз. Можно разобраться "
                "подробнее и решить, что с ней делать: упростить, отложить или отказаться."
            )
        if self.unclear:
            return "Если причина не до конца ясна, можно разобраться подробнее."
        return None


@dataclass
class QuickSuggestion:
    intervention_id: int
    task_id: int
    task_title: str
    reason: ObstacleReason
    kind: InterventionKind
    text: str
    next_action: str | None
    used_ai: bool
    signal: DeepSignal
    repeated: bool = False  # same press again: nothing new was recorded


@dataclass
class DialogOption:
    kind: str
    label: str
    details: str = ""


@dataclass
class DialogTurn:
    intervention_id: int
    task_id: int
    message: str
    question: str | None
    hypotheses: list[str]
    options: list[DialogOption]
    used_ai: bool
    turn: int
    final: bool
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptionResult:
    intervention_id: int
    task_id: int
    kind: str
    text: str
    next: Literal["decompose", "postpone", "approval", "short_start", "none"] = "none"
    approval_id: int | None = None


# --------------------------------------------------------------------------- history


async def deep_signal(
    session: AsyncSession, ctx: AppContext, task: Task, reason: ObstacleReason | None
) -> DeepSignal:
    since = ctx.clock.now() - timedelta(days=HISTORY_DAYS)
    rows = list(
        (
            await session.execute(
                select(TaskEvent.reason).where(
                    TaskEvent.task_id == task.id,
                    TaskEvent.event_type == TaskEventType.OBSTACLE_REPORTED,
                    TaskEvent.created_at >= since,
                )
            )
        ).scalars()
    )
    same = sum(1 for r in rows if reason is not None and r == reason.value)
    threshold = ctx.config.deep_dialog_after_postpones
    postpones = task.postpone_count or 0
    return DeepSignal(
        repeated_reason=same >= 1,
        repeated_postpone=postpones >= 2,
        unclear=reason in (ObstacleReason.UNCLEAR, ObstacleReason.OTHER),
        postpone_count=postpones,
        previous_same_reason=same,
        auto=threshold is not None and postpones >= threshold,
    )


async def task_history(session: AsyncSession, ctx: AppContext, task: Task) -> dict[str, Any]:
    """Facts for the AI (code-computed; no interpretation)."""
    tz = await ts.owner_tz(session)
    since = ctx.clock.now() - timedelta(days=HISTORY_DAYS)
    events = list(
        (
            await session.execute(
                select(TaskEvent)
                .where(
                    TaskEvent.task_id == task.id,
                    TaskEvent.event_type.in_(
                        [
                            TaskEventType.POSTPONED,
                            TaskEventType.RESCHEDULED,
                            TaskEventType.OBSTACLE_REPORTED,
                            TaskEventType.STARTED,
                            TaskEventType.PROGRESSED,
                        ]
                    ),
                    TaskEvent.created_at >= since,
                )
                .order_by(TaskEvent.created_at)
            )
        ).scalars()
    )
    ivs = list(
        (
            await session.execute(
                select(Intervention)
                .where(Intervention.task_id == task.id, Intervention.created_at >= since)
                .order_by(Intervention.created_at)
            )
        ).scalars()
    )
    past: list[dict[str, Any]] = []
    for iv in ivs[-10:]:
        outcomes = list(
            (
                await session.execute(
                    select(InterventionOutcome.result, InterventionOutcome.note).where(
                        InterventionOutcome.intervention_id == iv.id
                    )
                )
            ).all()
        )
        past.append(
            {
                "kind": iv.kind,
                "reason": iv.reason_category,
                "reason_origin": iv.reason_origin,
                "depth": iv.depth,
                "outcomes": [o[0] for o in outcomes] or ["нет данных"],
                "refuted": [
                    str(o[1]) for o in outcomes if o[1] and str(o[1]).startswith("Опровергнуто")
                ],
            }
        )
    return {
        "task": {
            "title": task.title,
            "status": task.status,
            "deadline": ts.format_due(task, tz) or None,
            "planned": ts.format_plan(task, tz) or None,
            "postpone_count": task.postpone_count or 0,
            "has_steps": bool(await ts.next_open_subtask(session, task.id)),
            "done_criteria": task.done_criteria,
        },
        "events": [
            {
                "type": e.event_type,
                "reason": e.reason,
                "at": ts.format_moment(e.created_at, tz),
            }
            for e in events[-20:]
        ],
        "past_interventions": past,
    }


async def _find_quick(
    session: AsyncSession, *, task_id: int, reminder_id: int | None, reason: ObstacleReason
) -> Intervention | None:
    stmt = select(Intervention).where(
        Intervention.task_id == task_id,
        Intervention.depth == "quick",
        Intervention.reason_category == reason.value,
    )
    stmt = stmt.where(
        Intervention.reminder_id == reminder_id
        if reminder_id is not None
        else Intervention.reminder_id.is_(None)
    )
    return (
        await session.execute(stmt.order_by(Intervention.id.desc()).limit(1))
    ).scalar_one_or_none()


async def _choose_allowed(session: AsyncSession) -> bool:
    profile = await get_profile(session)
    return flag(profile.autonomy_settings if profile else None, "choose_interventions")


# --------------------------------------------------------------------------- quick


async def suggest_after_reason(
    ctx: AppContext,
    *,
    task_id: int,
    reason: ObstacleReason,
    reminder_id: int | None = None,
) -> QuickSuggestion | None:
    """The owner picked a reason: suggest one concrete next action (AI, template
    fallback). A repeated press for the same reminder/reason is a no-op that
    returns the stored suggestion. None = the task no longer exists."""
    async with ctx.db.transaction() as s:
        task = await s.get(Task, task_id)
        if task is None:
            return None
        if reminder_id is not None and await s.get(ReminderSchedule, reminder_id) is None:
            reminder_id = None
        sig = await deep_signal(s, ctx, task, reason)
        existing = await _find_quick(s, task_id=task.id, reminder_id=reminder_id, reason=reason)
        if existing is not None and (reminder_id is not None or _recent(ctx, existing)):
            return QuickSuggestion(
                existing.id,
                task.id,
                task.title,
                reason,
                InterventionKind(existing.kind),
                existing.suggestion or TEMPLATES[reason][1],
                None,
                existing.used_ai,
                sig,
                repeated=True,
            )
        history = await task_history(s, ctx, task)
        ai_allowed = await _choose_allowed(s)
        title = task.title
    kind, text = TEMPLATES[reason]
    next_action: str | None = None
    used_ai = False
    if ai_allowed:
        try:
            out = await _ai_suggestion(ctx, history, reason)
            kind, text, next_action, used_ai = (
                InterventionKind(out.kind),
                out.message.strip()[:900],
                out.next_action.strip()[:300],
                True,
            )
        except AIUnavailable:
            pass
    async with ctx.db.transaction() as s:
        again = await _find_quick(s, task_id=task_id, reminder_id=reminder_id, reason=reason)
        if again is not None and (reminder_id is not None or _recent(ctx, again)):
            return QuickSuggestion(
                again.id,
                task_id,
                title,
                reason,
                InterventionKind(again.kind),
                again.suggestion or text,
                None,
                again.used_ai,
                sig,
                repeated=True,
            )
        suggestion = f"{text}\n\nБлижайший шаг: {next_action}" if next_action else text
        row = Intervention(
            task_id=task_id,
            reminder_id=reminder_id,
            kind=str(kind),
            reason_category=reason.value,
            reason_origin=MemoryOrigin.USER_SAID.value,
            depth="quick",
            suggestion=suggestion,
            used_ai=used_ai,
            created_at=ctx.clock.now(),
        )
        s.add(row)
        await s.flush()
        await ts.report_obstacle(s, ctx, task_id, reason=reason, actor=Actor.USER)
        return QuickSuggestion(
            row.id, task_id, title, reason, kind, suggestion, next_action, used_ai, sig
        )


def _recent(ctx: AppContext, iv: Intervention) -> bool:
    return ctx.clock.now() - iv.created_at <= timedelta(minutes=5)


def _suggestion_validator(output: BaseModel) -> list[str]:
    if not isinstance(output, InterventionSuggestion):
        return ["неожиданный формат ответа"]
    errors = tone_errors(output.message, output.next_action)
    if not output.next_action.strip():
        errors.append("next_action пустой — нужен один конкретный шаг")
    return errors


async def _ai_suggestion(
    ctx: AppContext, history: dict[str, Any], reason: ObstacleReason
) -> InterventionSuggestion:
    from procbot.ai.context import ContextBuilder, Extra
    from procbot.ai.persona import load_persona

    data = {**history, "reason_user_said": REASON_TEXT.get(reason, reason.value)}
    blocks = await ContextBuilder(ctx, task="suggest_intervention").build(
        profile=True,
        history=0,
        tasks=0,
        memory=True,
        extra=[Extra(DataBlock(P.TASKS, "Задача и история переносов", data), 98, None)],
    )
    instruction = (
        "Человек не начал задачу вовремя и назвал причину (reason_user_said). Предложи ОДНО "
        "конкретное ближайшее действие на несколько минут (next_action) и короткое "
        "сообщение (message, 1-3 предложения). Учитывай, какие способы раньше помогали "
        "или нет (past_interventions). Причину формулируй как предположение, без оценок и "
        "давления. Если сил мало — отдых или перенос тоже нормальный вариант."
    )
    result = await run_task(
        ctx,
        "suggest_intervention",
        Prompt(instruction=instruction, data=blocks, persona=await load_persona(ctx)),
        initiator="services.interventions",
        validator=_suggestion_validator,
    )
    return result.as_(InterventionSuggestion)


# --------------------------------------------------------------------------- deep dialog


def _dialog_validator(turn: int, refuted: list[str]) -> Any:
    final = turn >= MAX_DIALOG_TURNS

    def validate(output: BaseModel) -> list[str]:
        if not isinstance(output, DialogTurnOutput):
            return ["неожиданный формат ответа"]
        errors = tone_errors(
            output.message,
            output.question,
            *(h.text for h in output.hypotheses),
            *(o.details for o in output.options),
        )
        if final and output.question:
            errors.append("вопросов больше не задавай: предложи варианты (options)")
        if final and len(output.options) < 2:
            errors.append("предложи хотя бы два варианта в options")
        if not output.question and not output.options:
            errors.append("нужен либо уточняющий вопрос, либо варианты")
        for h in output.hypotheses:
            if any(_same(h.text, r) for r in refuted):
                errors.append(f"гипотеза «{h.text}» уже опровергнута пользователем")
        return errors

    return validate


def _same(a: str, b: str) -> bool:
    na, nb = " ".join(a.lower().split()), " ".join(b.lower().split())
    return na == nb or (len(nb) > 12 and nb in na)


def _template_turn(history: dict[str, Any], reason: str | None) -> DialogTurnOutput:
    """Deep dialog without AI: code-computed facts + the standard options."""
    t = history["task"]
    reasons = [e["reason"] for e in history["events"] if e["type"] == "obstacle_reported"]
    lines = [f"Давай посмотрим на «{t['title']}» спокойно."]
    if t["postpone_count"]:
        lines.append(f"Задачу переносили {t['postpone_count']} раз.")
    if reasons:
        named = ", ".join(REASON_TEXT.get(str(r), str(r)) for r in dict.fromkeys(reasons))
        lines.append(f"Причины, которые ты отмечал: {named}.")
    tried = [p["kind"] for p in history["past_interventions"]]
    if tried:
        lines.append("Уже пробовали: " + ", ".join(dict.fromkeys(tried)) + ".")
    lines.append(
        "Это не оценка — просто чтобы решить, что с задачей делать дальше. Выбери, что "
        "ближе, решение за тобой:"
    )
    kinds = ["split", "simplify", "reschedule", "defer_long", "change_approach", "drop"]
    if reason == ObstacleReason.LOW_ENERGY:
        kinds.insert(0, "rest")
    return DialogTurnOutput(
        message=" ".join(lines),
        options=[DialogOptionOut(kind=k) for k in kinds],  # type: ignore[arg-type]
    )


async def _ai_turn(
    ctx: AppContext, history: dict[str, Any], state: dict[str, Any], user_text: str | None
) -> DialogTurnOutput:
    from procbot.ai.context import ContextBuilder, Extra
    from procbot.ai.persona import load_persona

    turn = int(state.get("turn", 0))
    data = {
        **history,
        "reason_user_said": REASON_TEXT.get(state.get("reason") or ""),
        "dialog_so_far": state.get("transcript", []),
        "refuted_hypotheses": state.get("refuted", []),
        "turn": turn,
        "max_turns": MAX_DIALOG_TURNS,
    }
    blocks = await ContextBuilder(ctx, task="procrastination_dialog").build(
        message=user_text,
        message_label="Ответ пользователя",
        profile=True,
        history=0,
        tasks=0,
        memory=True,
        extra=[Extra(DataBlock(P.TASKS, "Задача, история переносов и разбор", data), 98, None)],
    )
    instruction = (
        "Это углублённый разбор задачи, которую человек много раз откладывает. "
        "Проанализируй историю (events, past_interventions) и ответы (dialog_so_far). "
        "В message — коротко и бережно, что видно из данных; возможные причины — только "
        "как гипотезы в hypotheses (с тем, как проверить), никогда как факт. "
        "Опровергнутые гипотезы (refuted_hypotheses) не повторяй. "
        f"Если turn < {MAX_DIALOG_TURNS} и неясно главное — задай ОДИН короткий вопрос "
        "(question). Иначе question = null и предложи 2-4 варианта (options): упростить, "
        "разбить, перенести, отложить надолго, поменять подход, отказаться, отдохнуть. "
        "Решение принимает человек."
    )
    result = await run_task(
        ctx,
        "procrastination_dialog",
        Prompt(instruction=instruction, data=blocks, persona=await load_persona(ctx)),
        initiator="services.interventions",
        validator=_dialog_validator(turn, list(state.get("refuted", []))),
    )
    return result.as_(DialogTurnOutput)


def _options(out: DialogTurnOutput) -> list[DialogOption]:
    seen: set[str] = set()
    opts: list[DialogOption] = []
    for o in out.options:
        if o.kind in seen:
            continue
        seen.add(o.kind)
        opts.append(DialogOption(o.kind, OPTION_LABELS[o.kind], o.details.strip()[:300]))
    if opts and "rest" not in seen:  # rest / a pause is always a valid outcome
        opts.append(DialogOption("rest", OPTION_LABELS["rest"]))
    return opts


async def _run_turn(
    ctx: AppContext, intervention_id: int, state: dict[str, Any], user_text: str | None
) -> DialogTurn:
    async with ctx.db.transaction() as s:
        iv = await s.get(Intervention, intervention_id)
        task = await s.get(Task, iv.task_id) if iv is not None and iv.task_id else None
        if iv is None or task is None:
            raise ActionError("Задача больше не найдена.")
        history = await task_history(s, ctx, task)
        ai_allowed = await _choose_allowed(s)
        task_id = task.id
    turn = int(state.get("turn", 0))
    used_ai = False
    out: DialogTurnOutput | None = None
    past_refuted = [
        str(n).removeprefix("Опровергнуто пользователем: ")
        for p in history["past_interventions"]
        for n in p["refuted"]
    ]
    ai_state = {**state, "refuted": list(dict.fromkeys([*state.get("refuted", []), *past_refuted]))}
    if ai_allowed:
        try:
            out = await _ai_turn(ctx, history, ai_state, user_text)
            used_ai = True
        except AIUnavailable:
            out = None
    if out is None:
        out = _template_turn(history, state.get("reason"))
    final = turn >= MAX_DIALOG_TURNS or not out.question
    question = None if final else (out.question or "").strip()[:400]
    hypotheses = [h.text.strip()[:300] for h in out.hypotheses if h.text.strip()][:3]
    options = _options(out) if final or out.options else []
    transcript = list(state.get("transcript", []))
    if user_text:
        transcript.append({"user": user_text[:1000]})
    transcript.append({"bot": out.message[:1000], "question": question})
    new_state = {
        **state,
        "intervention_id": intervention_id,
        "task_id": task_id,
        "turn": turn + 1,
        "transcript": transcript[-12:],
        "hypotheses": hypotheses,
    }
    async with ctx.db.transaction() as s:
        iv = await s.get(Intervention, intervention_id)
        if iv is not None:
            iv.used_ai = iv.used_ai or used_ai
            iv.suggestion = out.message[:2000]
            if hypotheses:
                iv.reason_text = ("Гипотезы ИИ (не установлено): " + "; ".join(hypotheses))[:2000]
                if iv.reason_origin is None:
                    iv.reason_origin = MemoryOrigin.AI_INFERRED.value
        if hypotheses:
            await actions.claim_operation(
                s, ctx, _hyp_key(intervention_id, turn + 1), {"items": hypotheses}
            )
    return DialogTurn(
        intervention_id=intervention_id,
        task_id=task_id,
        message=out.message.strip()[:1500],
        question=question,
        hypotheses=hypotheses,
        options=options,
        used_ai=used_ai,
        turn=turn + 1,
        final=final,
        state=new_state,
    )


def _hyp_key(intervention_id: int, turn: int) -> str:
    return f"hyps:{intervention_id}:{turn}"


async def hypothesis_text(
    session: AsyncSession, *, intervention_id: int, turn: int, index: int
) -> str | None:
    """The hypothesis shown at `turn` / position `index` (for "Это не так")."""
    row = await actions.get_operation(session, _hyp_key(intervention_id, turn))
    items = list((row.result or {}).get("items", [])) if row is not None else []
    return str(items[index]) if 0 <= index < len(items) else None


async def start_deep_dialog(
    ctx: AppContext,
    *,
    task_id: int,
    reason: ObstacleReason | None = None,
    reminder_id: int | None = None,
    op_key: str,
) -> DialogTurn:
    """Open the deep dialog (the owner pressed "Разобраться подробнее", or the
    configured threshold was reached). Idempotent by op_key."""
    async with ctx.db.transaction() as s:
        task = await s.get(Task, task_id)
        if task is None:
            raise ActionError("Задача больше не найдена.")
        if reason is None:
            last = (
                await s.execute(
                    select(Intervention.reason_category)
                    .where(
                        Intervention.task_id == task.id, Intervention.reason_category.is_not(None)
                    )
                    .order_by(Intervention.id.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            reason = ObstacleReason(last) if last in set(ObstacleReason) else None
        if reminder_id is not None and await s.get(ReminderSchedule, reminder_id) is None:
            reminder_id = None
        done = await actions.get_operation(s, f"deep:{op_key}")
        if done is not None and (done.result or {}).get("intervention_id"):
            iv_id = int((done.result or {})["intervention_id"])
        else:
            row = Intervention(
                task_id=task.id,
                reminder_id=reminder_id,
                kind=InterventionKind.DEEP_DIALOG.value,
                reason_category=reason.value if reason else None,
                reason_origin=MemoryOrigin.USER_SAID.value if reason else None,
                depth="deep",
                used_ai=False,
                created_at=ctx.clock.now(),
            )
            s.add(row)
            await s.flush()
            iv_id = row.id
            await actions.claim_operation(s, ctx, f"deep:{op_key}", {"intervention_id": iv_id})
    state = {"reason": reason.value if reason else None, "turn": 0, "refuted": []}
    return await _run_turn(ctx, iv_id, state, None)


async def continue_deep_dialog(
    ctx: AppContext, state: dict[str, Any], user_text: str
) -> DialogTurn:
    return await _run_turn(ctx, int(state["intervention_id"]), state, user_text.strip()[:1000])


async def refute_hypothesis(
    session: AsyncSession, ctx: AppContext, *, intervention_id: int, text: str
) -> InterventionOutcome:
    """The owner says a hypothesis is wrong (acceptance 10). Recorded; the
    dialog never repeats it; nothing about it goes to memory."""
    row = await record_intervention_outcome(
        session,
        intervention_id=intervention_id,
        result=InterventionResult.DECLINED,
        now=ctx.clock.now(),
        note=f"Опровергнуто пользователем: {text}"[:1000],
        time_source=TimeSource.EXACT,
    )
    await audit(
        session,
        now=ctx.clock.now(),
        actor=Actor.USER,
        action="hypothesis.refuted",
        entity="intervention",
        entity_id=intervention_id,
        details={"text": text[:500]},
    )
    return row


async def confirm_hypothesis(
    session: AsyncSession, ctx: AppContext, *, intervention_id: int, text: str
) -> actions.ExecResult:
    """The owner says "похоже на правду": proposed to memory as a hypothesis
    (never a fact) through the executor (the memory module owns the tables)."""
    return await actions.execute(
        session,
        ctx,
        SaveHypothesis(
            content=text[:1000],
            verification_hint="пользователь согласился в разборе переносов; проверять дальше",
        ),
        origin=Actor.AI,
        op_key=f"hyp:{intervention_id}:{hashlib.sha1(text.encode()).hexdigest()[:12]}",
    )


async def choose_option(
    session: AsyncSession, ctx: AppContext, *, intervention_id: int, kind: str, op_key: str
) -> OptionResult:
    """The owner picked an option of the deep dialog. Recorded as a new
    intervention (so a following session links to it). Deferring / dropping
    become approval cards; rescheduling opens the owner's own date choice."""
    if kind not in OPTION_LABELS:
        raise ActionError("Неизвестный вариант.")
    deep = await session.get(Intervention, intervention_id)
    task = await session.get(Task, deep.task_id) if deep is not None and deep.task_id else None
    if deep is None or task is None:
        raise ActionError("Задача больше не найдена.")
    iv_kind = (
        InterventionKind(kind) if kind in set(InterventionKind) else InterventionKind.TINY_STEP
    )
    done = await actions.get_operation(session, f"opt:{op_key}")
    if done is None:
        session.add(
            Intervention(
                task_id=task.id,
                reminder_id=deep.reminder_id,
                kind=iv_kind.value,
                reason_category=deep.reason_category,
                reason_origin=deep.reason_origin,
                depth="deep",
                suggestion=OPTION_LABELS[kind],
                used_ai=deep.used_ai,
                created_at=ctx.clock.now(),
            )
        )
        await session.flush()
        await actions.claim_operation(session, ctx, f"opt:{op_key}", {"kind": kind})
    if kind in ("split", "simplify"):
        return OptionResult(intervention_id, task.id, kind, "Давай разобьём на шаги.", "decompose")
    if kind == "reschedule":
        return OptionResult(
            intervention_id,
            task.id,
            kind,
            "Сознательный перенос — нормальный выбор. На когда перенести?",
            "postpone",
        )
    if kind in ("defer_long", "drop"):
        action = ChangeTaskStatus(
            task_id=task.id,
            status="someday" if kind == "defer_long" else "cancelled",
            reason="решение после разбора переносов",
        )
        res = await actions.execute(
            session,
            ctx,
            action,
            origin=Actor.AI,
            op_key=f"opt:{op_key}:status",
        )
        text = (
            "Отложить задачу надолго — тоже решение. Подтверди, пожалуйста:"
            if kind == "defer_long"
            else "Отказаться от задачи — допустимый исход, если она больше не нужна. "
            "Подтверди, пожалуйста:"
        )
        return OptionResult(intervention_id, task.id, kind, text, "approval", res.approval_id)
    if kind == "short_start":
        return OptionResult(
            intervention_id,
            task.id,
            kind,
            "Начнём совсем коротко, без обязательств.",
            "short_start",
        )
    if kind == "rest":
        return OptionResult(
            intervention_id,
            task.id,
            kind,
            "Отдых — нормальный выбор. Вернёмся к задаче, когда будут силы.",
        )
    return OptionResult(
        intervention_id,
        task.id,
        kind,
        "Попробуй подойти иначе: поменяй место, время или формат работы — и начни с "
        "короткого старта.",
        "short_start",
    )


async def record_declined(
    session: AsyncSession, ctx: AppContext, *, intervention_id: int, note: str | None = None
) -> InterventionOutcome | None:
    iv = await session.get(Intervention, intervention_id)
    if iv is None:
        return None
    return await record_intervention_outcome(
        session,
        intervention_id=iv.id,
        result=InterventionResult.DECLINED,
        now=ctx.clock.now(),
        note=note,
    )


async def record_no_effect(
    session: AsyncSession, ctx: AppContext, *, intervention_id: int, note: str | None = None
) -> InterventionOutcome | None:
    iv = await session.get(Intervention, intervention_id)
    if iv is None:
        return None
    return await record_intervention_outcome(
        session,
        intervention_id=iv.id,
        result=InterventionResult.NO_EFFECT,
        now=ctx.clock.now(),
        note=note,
    )


async def settle_outcomes(ctx: AppContext) -> int:
    """Periodic: interventions older than SETTLE_AFTER with no outcome and no
    work session on the task since -> no_effect (system-measured)."""
    now = ctx.clock.now()
    n = 0
    async with ctx.db.transaction() as s:
        rows = list(
            (
                await s.execute(
                    select(Intervention).where(
                        Intervention.created_at <= now - SETTLE_AFTER,
                        Intervention.created_at >= now - timedelta(days=HISTORY_DAYS),
                        Intervention.task_id.is_not(None),
                    )
                )
            ).scalars()
        )
        for iv in rows:
            has = (
                await s.execute(
                    select(InterventionOutcome.id)
                    .where(InterventionOutcome.intervention_id == iv.id)
                    .limit(1)
                )
            ).first()
            if has is not None:
                continue
            worked = (
                await s.execute(
                    select(WorkSession.id)
                    .where(
                        WorkSession.task_id == iv.task_id,
                        WorkSession.started_at.is_not(None),
                        WorkSession.started_at >= iv.created_at,
                    )
                    .limit(1)
                )
            ).first()
            if worked is not None:
                continue
            await record_intervention_outcome(
                s,
                intervention_id=iv.id,
                result=InterventionResult.NO_EFFECT,
                now=now,
                note="нет рабочей сессии после совета",
                time_source=TimeSource.UNKNOWN,
            )
            n += 1
    return n


async def _settle_job(ctx: AppContext) -> None:
    await settle_outcomes(ctx)


# --------------------------------------------------------------------------- start-delay card


async def render_start_delay(ctx: AppContext, r: ReminderSchedule) -> Outgoing | None:
    """The stage-A short question (acceptance 9: quick buttons first), with the
    reason buttons pointing to this module and "Разобраться подробнее" always
    available."""
    from procbot.services import initiative

    out = await initiative.render_start_delay(ctx, r)
    if out is None or out.reply_markup is None:
        return out
    rows: list[list[InlineKeyboardButton]] = []
    task_id = 0
    for row in out.reply_markup.inline_keyboard:
        new_row: list[InlineKeyboardButton] = []
        for b in row:
            data = b.callback_data or ""
            if data.startswith("rm:reason:"):
                _, _, rid, tid, arg = data.split(":", 4)
                task_id = int(tid)
                data = InterventionCb(
                    action="reason", task_id=int(tid), reminder_id=int(rid), arg=arg
                ).pack()
            new_row.append(InlineKeyboardButton(text=b.text, callback_data=data))
        rows.append(new_row)
    text = out.text
    if task_id:
        rows.insert(
            len(rows) - 1,
            [
                InlineKeyboardButton(
                    text="🔍 Разобраться подробнее",
                    callback_data=InterventionCb(
                        action="deep", task_id=task_id, reminder_id=r.id
                    ).pack(),
                )
            ],
        )
        async with ctx.db.transaction() as s:
            task = await s.get(Task, task_id)
            if task is not None:
                sig = await deep_signal(s, ctx, task, None)
                if sig.repeated_postpone:
                    text += (
                        f"\n\nЗадачу переносили уже {sig.postpone_count} раз — если хочешь, "
                        "можно разобраться подробнее."
                    )
    return Outgoing(text=text, reply_markup=InlineKeyboardMarkup(inline_keyboard=rows))


# --------------------------------------------------------------------------- executor


async def _apply_choose(
    session: AsyncSession, ctx: AppContext, action: ChooseIntervention, actor: Actor
) -> ApplyOutcome:
    try:
        kind = InterventionKind(action.kind)
    except ValueError:
        raise ActionError("Неизвестный способ помощи.") from None
    if tone_errors(action.suggestion):
        raise ActionError("Формулировка совета недопустима.")
    if action.task_id is not None and await session.get(Task, action.task_id) is None:
        raise ActionError("Задача не найдена.")
    row = Intervention(
        task_id=action.task_id,
        kind=kind.value,
        depth="quick",
        suggestion=action.suggestion[:2000],
        used_ai=actor == Actor.AI,
        created_at=ctx.clock.now(),
    )
    session.add(row)
    await session.flush()
    return ApplyOutcome({"intervention_id": row.id, "kind": kind.value})


async def _describe_choose(
    session: AsyncSession, ctx: AppContext, action: ChooseIntervention
) -> dict[str, Any]:
    return {"summary": "Способ помощи", "lines": [action.suggestion[:300]]}


def register(ctx: AppContext) -> None:
    """START_DELAY renderer (wraps stage A; must run after services.initiative),
    the CHOOSE_INTERVENTION applier and the outcome settle job."""
    register_renderer(ctx, MessageCategory.START_DELAY, render_start_delay)
    actions.register_applier(
        ctx, ActionType.CHOOSE_INTERVENTION, _apply_choose, describe=_describe_choose
    )
    register_job(
        ctx,
        PeriodicJob(name=SETTLE_JOB, interval_seconds=SETTLE_INTERVAL_SECONDS, run=_settle_job),
    )


__all__ = [
    "MAX_DIALOG_TURNS",
    "OPTION_LABELS",
    "PROCRASTINATION_DIALOG",
    "SUGGEST_INTERVENTION",
    "TEMPLATES",
    "DeepSignal",
    "DialogOption",
    "DialogTurn",
    "DialogTurnOutput",
    "InterventionSuggestion",
    "OptionResult",
    "QuickSuggestion",
    "choose_option",
    "confirm_hypothesis",
    "continue_deep_dialog",
    "deep_signal",
    "find_recent_intervention",
    "hypothesis_text",
    "record_declined",
    "record_intervention_outcome",
    "record_no_effect",
    "record_quick_intervention",
    "refute_hypothesis",
    "register",
    "render_start_delay",
    "settle_outcomes",
    "start_deep_dialog",
    "suggest_after_reason",
    "task_history",
    "tone_errors",
]
