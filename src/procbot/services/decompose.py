"""Adaptive decomposition (FR-05, acceptance 8).

- A large task gets an overall plan of steps, but the UI shows the next 1-3
  actions only (`Decomposition.next_actions`).
- Stuck (obstacle "unclear" / "too big", or the owner says so) -> a finer, clearer
  breakdown of the current step that starts with a very short start.
- Progressing confidently (several steps done recently, no obstacles) -> coarser
  steps are proposed.

Steps are applied through the ActionExecutor as CREATE_SUBTASKS / REPLACE_SUBTASKS
(AUTO in FR-15): only subtasks change — the parent's goal, scope and deadline
are never touched (services.tasks enforces it). If the owner switched the
"manage_subtasks" autonomy toggle off, the same action becomes a pending
approval card instead.

Without AI: nothing is guessed — the owner is asked to type the very first
physical step, which is then added as the first subtask (`add_first_step`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Literal

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
from procbot.db.models import Task, TaskEvent
from procbot.domain.actions import CreateSubtasks, ReorderTasks, ReplaceSubtasks, SubtaskDraft
from procbot.domain.enums import Actor, ObstacleReason, TaskEventType, TaskStatus
from procbot.services import actions, approvals
from procbot.services import tasks as ts
from procbot.services.preferences import flag
from procbot.services.profile import get_profile

log = logging.getLogger(__name__)

Mode = Literal["plan", "finer", "coarser"]

MAX_STEPS = 12
NEXT_ACTIONS = 3
RECENT = timedelta(days=3)
CONFIDENT_DONE_STEPS = 2  # done steps in RECENT without obstacles -> coarser
STUCK_REASONS = frozenset({ObstacleReason.UNCLEAR, ObstacleReason.TOO_BIG})


class StepOut(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    estimate_minutes: int | None = Field(default=None, ge=1, le=480)
    done_criteria: str | None = None


class DecomposeOutput(BaseModel):
    steps: list[StepOut] = Field(min_length=1, max_length=MAX_STEPS)
    first_tiny_step: str | None = Field(
        default=None, description="Очень короткий старт на 2-5 минут"
    )
    overall_estimate_minutes: int | None = Field(default=None, ge=1, le=10_000)
    note: str = ""


ESTIMATE_AND_DECOMPOSE = register_task(
    AITaskSpec(
        name="estimate_and_decompose",
        tier=Tier.STANDARD,
        output=DecomposeOutput,
        max_output_tokens=1200,
        context_budget_tokens=5000,
        data_categories=frozenset({P.TASKS, P.PROFILE, P.MEMORY, P.MESSAGE}),
        system=(
            "Ты помогаешь разбить задачу на конкретные физические шаги. Шаги — только "
            "внутри исходной задачи: не меняй её цель, объём и срок, не добавляй новых "
            "обязательств. Каждый шаг начинается с глагола и понятен без пояснений."
        ),
        description="Оценка длительности, подзадачи, «очень маленький шаг»",
    )
)

_INSTRUCTIONS: dict[str, str] = {
    "plan": (
        "Составь общий план шагов для задачи (до 8 шагов, по порядку выполнения) с "
        "оценкой минут на каждый. Первый шаг — простой и быстрый. В first_tiny_step "
        "предложи старт на 2-5 минут."
    ),
    "finer": (
        "Человек застрял на текущем шаге (current_step, или на всей задаче, если шагов "
        "нет). Разбей именно его на 2-5 более мелких и понятных шагов; первый — очень "
        "короткий старт на 2-5 минут (estimate_minutes не больше 5). Остальные шаги "
        "задачи не трогай — их добавит код."
    ),
    "coarser": (
        "Человек уверенно продвигается. Объедини оставшиеся открытые шаги (open_steps) в "
        "меньшее число более крупных шагов — шагов должно стать меньше, чем сейчас."
    ),
}


@dataclass
class Decomposition:
    task_id: int
    title: str
    mode: Mode
    used_ai: bool = False
    applied: bool = False
    approval_id: int | None = None  # autonomy off: a card waits for consent
    created: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    tiny_step: str | None = None
    note: str = ""
    needs_first_step: bool = False  # no AI: ask the owner to type the first step
    error: str | None = None


# --------------------------------------------------------------------------- signals


async def _recent_obstacles(session: AsyncSession, task_id: int, since: Any) -> list[str]:
    rows = (
        await session.execute(
            select(TaskEvent.reason).where(
                TaskEvent.task_id == task_id,
                TaskEvent.event_type == TaskEventType.OBSTACLE_REPORTED,
                TaskEvent.created_at >= since,
            )
        )
    ).scalars()
    return [r for r in rows if r]


async def detect_mode(session: AsyncSession, ctx: AppContext, task: Task) -> Mode:
    """plan (no steps yet) / finer (stuck recently) / coarser (confident progress)."""
    subs = await ts.list_subtasks(session, task.id)
    open_subs = [s for s in subs if ts.is_open(s)]
    if not open_subs:
        return "plan"
    since = ctx.clock.now() - RECENT
    if any(r in STUCK_REASONS for r in await _recent_obstacles(session, task.id, since)):
        return "finer"
    done_recent = [
        s
        for s in subs
        if s.status == TaskStatus.DONE and s.completed_at is not None and s.completed_at >= since
    ]
    if len(done_recent) >= CONFIDENT_DONE_STEPS and len(open_subs) >= 3:
        return "coarser"
    return "finer"


# --------------------------------------------------------------------------- core


async def _target(session: AsyncSession, task_id: int) -> tuple[Task, Task | None]:
    """(parent, focus step). A subtask is decomposed within its parent."""
    task = await ts.require_task(session, task_id)
    if task.parent_id is not None:
        parent = await ts.require_task(session, task.parent_id)
        return parent, task
    return task, await ts.next_open_subtask(session, task.id)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def decompose_validator(
    mode: Mode, parent_title: str, open_count: int
) -> Any:  # Callable[[BaseModel], list[str]]
    def validate(output: BaseModel) -> list[str]:
        if not isinstance(output, DecomposeOutput):
            return ["неожиданный формат ответа"]
        errors: list[str] = []
        seen: set[str] = set()
        for n, st in enumerate(output.steps, 1):
            key = _norm(st.title)
            if key in seen:
                errors.append(f"шаг {n}: повторяется")
            seen.add(key)
            if key == _norm(parent_title):
                errors.append(f"шаг {n}: повторяет саму задачу — нужен конкретный шаг")
        if mode == "finer":
            first = output.steps[0]
            if first.estimate_minutes is not None and first.estimate_minutes > 10:
                errors.append("первый шаг должен быть очень коротким стартом (до 5-10 минут)")
        if mode == "coarser" and open_count and len(output.steps) >= open_count:
            errors.append(f"шагов должно стать меньше, чем сейчас ({open_count})")
        return errors

    return validate


async def _ai_steps(
    ctx: AppContext,
    data: dict[str, Any],
    mode: Mode,
    parent_title: str,
    open_count: int,
    user_note: str | None,
) -> tuple[DecomposeOutput, int]:
    from procbot.ai.context import ContextBuilder, Extra
    from procbot.ai.persona import load_persona

    blocks = await ContextBuilder(ctx, task="estimate_and_decompose").build(
        message=user_note,
        message_label="Что говорит пользователь",
        profile=True,
        history=0,
        tasks=0,
        memory=True,
        extra=[Extra(DataBlock(P.TASKS, "Задача и её шаги", data), 98, None)],
    )
    result = await run_task(
        ctx,
        "estimate_and_decompose",
        Prompt(instruction=_INSTRUCTIONS[mode], data=blocks, persona=await load_persona(ctx)),
        initiator="services.decompose",
        validator=decompose_validator(mode, parent_title, open_count),
    )
    return result.as_(DecomposeOutput), result.call_id


def _drafts(steps: list[StepOut]) -> list[SubtaskDraft]:
    return [
        SubtaskDraft(
            title=s.title.strip()[:500],
            estimate_minutes=s.estimate_minutes,
            done_criteria=(s.done_criteria or None) and s.done_criteria.strip()[:1000],
        )
        for s in steps[:30]
    ]


async def _next_actions(session: AsyncSession, parent_id: int) -> list[str]:
    subs = [s for s in await ts.list_subtasks(session, parent_id) if ts.is_open(s)]
    return [s.title for s in subs[:NEXT_ACTIONS]]


async def decompose(
    ctx: AppContext,
    task_id: int,
    *,
    mode: Mode | None = None,
    user_note: str | None = None,
    op_key: str,
) -> Decomposition:
    """Propose and (AUTO) apply steps for a task. Network work runs between two
    short transactions. Idempotent by `op_key`."""
    async with ctx.db.transaction() as s:
        parent, focus = await _target(s, task_id)
        if parent.status in ts.CLOSED_STATUSES:
            return Decomposition(parent.id, parent.title, "plan", error="Задача уже закрыта.")
        mode = mode or await detect_mode(s, ctx, parent)
        done = await actions.get_operation(s, f"decompose:{op_key}")
        if done is not None:
            res = done.result or {}
            return Decomposition(
                parent.id,
                parent.title,
                mode,
                used_ai=bool(res.get("used_ai")),
                applied=bool(res.get("applied")),
                approval_id=res.get("approval_id"),
                created=list(res.get("created", [])),
                next_actions=await _next_actions(s, parent.id),
                tiny_step=res.get("tiny_step"),
                note=str(res.get("note") or ""),
            )
        subs = await ts.list_subtasks(s, parent.id)
        open_subs = [x for x in subs if ts.is_open(x)]
        if mode == "coarser" and len(open_subs) < 2:
            mode = "finer" if open_subs else "plan"
        if mode == "finer" and focus is None and open_subs:
            focus = open_subs[0]
        tz = await ts.owner_tz(s)
        data: dict[str, Any] = {
            "mode": mode,
            "task": {
                "title": parent.title,
                "description": parent.description,
                "done_criteria": parent.done_criteria,
                "deadline": ts.format_due(parent, tz) or None,
                "estimate_minutes": parent.estimate_minutes,
                "postpone_count": parent.postpone_count,
            },
            "done_steps": [x.title for x in subs if x.status == TaskStatus.DONE][-10:],
            "open_steps": [
                {"title": x.title, "estimate_minutes": x.estimate_minutes} for x in open_subs
            ],
            "current_step": focus.title if focus is not None else None,
            "recent_obstacles": await _recent_obstacles(
                s, parent.id, ctx.clock.now() - timedelta(days=14)
            ),
        }
        profile = await get_profile(s)
        auto = flag(profile.autonomy_settings if profile else None, "manage_subtasks")
        parent_id, parent_title, focus_id = parent.id, parent.title, focus.id if focus else None
        rest = [
            SubtaskDraft(
                title=x.title, estimate_minutes=x.estimate_minutes, done_criteria=x.done_criteria
            )
            for x in open_subs
            if x.id != focus_id
        ]
        open_count = len(open_subs)
    try:
        out, call_id = await _ai_steps(ctx, data, mode, parent_title, open_count, user_note)
    except AIUnavailable:
        return Decomposition(
            parent_id,
            parent_title,
            mode,
            needs_first_step=True,
            note=(
                "Сейчас не могу сам предложить шаги. Напиши самый первый физический шаг — "
                "совсем маленький, на пару минут (открыть файл, найти ссылку, записать три "
                "пункта), — я добавлю его первым шагом."
            ),
        )
    steps = _drafts(out.steps)
    tiny = (out.first_tiny_step or "").strip()[:300] or None
    action: Any
    if mode == "finer":
        if tiny and _norm(tiny) != _norm(steps[0].title):
            steps.insert(0, SubtaskDraft(title=tiny, estimate_minutes=5))
        new_list = [*steps, *rest]
        action = (
            ReplaceSubtasks(parent_id=parent_id, subtasks=new_list[:30])
            if open_count
            else CreateSubtasks(parent_id=parent_id, subtasks=new_list[:30])
        )
    elif mode == "coarser" or open_count:
        action = ReplaceSubtasks(parent_id=parent_id, subtasks=steps)
    else:
        action = CreateSubtasks(parent_id=parent_id, subtasks=steps)
    async with ctx.db.transaction() as s:
        key = f"decompose:{op_key}"
        result = Decomposition(
            parent_id,
            parent_title,
            mode,
            used_ai=True,
            tiny_step=tiny,
            note=out.note.strip()[:500],
        )
        if auto:
            res = await actions.execute(
                s, ctx, action, origin=Actor.AI, op_key=key, ai_call_id=call_id
            )
            result.applied = res.status == actions.ExecStatus.APPLIED
            if not res.ok:
                result.error = res.error or "Не удалось добавить шаги."
        else:
            diff = await actions.describe_action(s, ctx, action)
            row = await approvals.propose(
                s, ctx, action, origin=Actor.AI, op_key=key, diff=diff, ai_call_id=call_id
            )
            result.approval_id = row.id
        result.created = [d.title for d in action.subtasks]
        result.next_actions = await _next_actions(s, parent_id)
        if not result.applied:
            result.next_actions = [d.title for d in action.subtasks[:NEXT_ACTIONS]]
        await actions.claim_operation(
            s,
            ctx,
            f"decompose:{op_key}",
            {
                "used_ai": True,
                "applied": result.applied,
                "approval_id": result.approval_id,
                "created": result.created,
                "tiny_step": tiny,
                "note": result.note,
            },
        )
    return result


async def add_first_step(
    session: AsyncSession, ctx: AppContext, task_id: int, title: str, *, op_key: str
) -> list[str]:
    """The owner typed the first physical step (fallback without AI): it becomes
    the FIRST open subtask (explicit user input = origin USER). Returns the next
    actions."""
    parent, _ = await _target(session, task_id)
    title = " ".join(title.split())[:500]
    if not title:
        raise actions.ActionError("Шаг пустой.")
    res = await actions.execute(
        session,
        ctx,
        CreateSubtasks(parent_id=parent.id, subtasks=[SubtaskDraft(title=title)]),
        origin=Actor.USER,
        op_key=f"{op_key}:first",
    )
    if not res.ok:
        raise actions.ActionError(res.error or "Не удалось добавить шаг.")
    created = [int(x) for x in res.result.get("created", [])]
    open_ids = [s.id for s in await ts.list_subtasks(session, parent.id) if ts.is_open(s)]
    if created and len(open_ids) > 1 and open_ids[0] != created[0]:
        order = [created[0], *[i for i in open_ids if i != created[0]]]
        await actions.execute(
            session,
            ctx,
            ReorderTasks(task_ids=order),
            origin=Actor.USER,
            op_key=f"{op_key}:first-order",
        )
    return await _next_actions(session, parent.id)


__all__ = [
    "ESTIMATE_AND_DECOMPOSE",
    "DecomposeOutput",
    "Decomposition",
    "StepOut",
    "add_first_step",
    "decompose",
    "decompose_validator",
    "detect_mode",
]
