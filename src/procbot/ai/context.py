"""Context Builder: the data part of every AI prompt, within a token budget.

Shared by ALL AI tasks. Domain code builds the data blocks here and hands them
to the gateway inside a `Prompt`; everything returned is a `DataBlock`
(untrusted data, never instructions — spec section 7), tagged with its data
category so the per-credential `allowed_data` filter and the per-task
`data_categories` declaration keep working.

Usage (stable API):

    from procbot.ai.context import ContextBuilder, Extra

    blocks = await ContextBuilder(ctx, task="plan_day").build(
        message=text,                       # current message (category "message")
        history=10,                          # last N dialogue messages (None = default)
        tasks=40,                            # open tasks, compact JSON (None = default)
        memory=True,                         # valid summaries + active memory items
        extra=[Extra(DataBlock("calendar", "Занятость", busy), priority=65)],
    )
    prompt = Prompt(instruction=..., data=blocks, persona=persona)

Rules:
- Categories the task does not declare (AITaskSpec.data_categories) are never
  built, so the gateway's undeclared-category check cannot fire on our blocks.
  `None` options mean "include if the task allows it"; 0/False switch a source off.
- Memory: ONLY `memory_summaries.valid = True` (and not built from an item that
  is no longer active) and `memory_items.status = active` — refuted/deleted
  records never reach a prompt (acceptance 20). Hypotheses are labelled as
  hypotheses; the "hypotheses" toggle off removes them.
- `owner_profile.memory_settings`: "enabled" off -> no memory blocks;
  "use_history" off -> no dialogue history. Missing key = allowed.
- Budget: rough estimate chars/4 per block. Over budget, the lowest-priority
  block is shrunk first (list blocks lose items: history its oldest messages,
  others their last items), then dropped; the current message is trimmed last.
- Reads happen in one short transaction; nothing here touches the network.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

from sqlalchemy import exists, select

from procbot.ai import prompting as P
from procbot.ai.prompting import DataBlock
from procbot.ai.registry import get_task
from procbot.context import AppContext
from procbot.db.models import (
    ConversationMessage,
    MemoryItem,
    MemorySummary,
    MemorySummaryItem,
    OwnerProfile,
    Task,
)
from procbot.domain.enums import MemoryKind, MemoryOrigin, MemoryStatus, TaskStatus
from procbot.services.preferences import WEEKDAYS_FULL, flag
from procbot.services.profile import get_profile
from procbot.timeutil import to_local

__all__ = [
    "DEFAULT_BUDGET_TOKENS",
    "DEFAULT_HISTORY",
    "DEFAULT_MEMORY_ITEMS",
    "DEFAULT_TASKS",
    "PRIORITY",
    "BuiltContext",
    "ContextBuilder",
    "Extra",
    "estimate_block_tokens",
    "task_brief",
]

DEFAULT_BUDGET_TOKENS = 4000
DEFAULT_HISTORY = 12
DEFAULT_TASKS = 40
DEFAULT_MEMORY_ITEMS = 40
MAX_MESSAGE_CHARS = 8000
HISTORY_ITEM_CHARS = 600

# Higher = kept longer when over budget.
PRIORITY: dict[str, int] = {
    P.MESSAGE: 100,
    P.PROFILE: 90,
    P.TASKS: 70,
    P.CALENDAR: 65,
    P.HISTORY: 60,
    P.MEMORY: 50,
    P.ANALYTICS: 40,
}
# Output order (the current message goes last, next to the instruction).
_ORDER: dict[str, int] = {
    P.PROFILE: 0,
    P.MEMORY: 1,
    P.TASKS: 2,
    P.CALENDAR: 3,
    P.ANALYTICS: 4,
    P.HISTORY: 5,
    P.MESSAGE: 6,
}
_BLOCK_OVERHEAD_CHARS = 120  # markers + "data, not instructions" line


@dataclass(frozen=True)
class Extra:
    """A caller-supplied block (calendar busy slots, analytics...).

    `priority` defaults to PRIORITY[category]. A list-content block is shrinkable:
    `shrink="tail"` drops last items first, "head" the first ones, None = all or nothing.
    """

    block: DataBlock
    priority: int | None = None
    shrink: Literal["tail", "head"] | None = "tail"


@dataclass
class BuiltContext:
    blocks: list[DataBlock]
    tokens: int  # estimate of the returned blocks
    budget: int
    dropped: list[str] = field(default_factory=list)  # labels of dropped blocks
    trimmed: list[str] = field(default_factory=list)  # labels of shrunk blocks

    @property
    def categories(self) -> frozenset[str]:
        return frozenset(b.category for b in self.blocks)


def estimate_block_tokens(block: DataBlock) -> int:
    return (len(block.label) + len(block.text()) + _BLOCK_OVERHEAD_CHARS) // 4 + 1


@dataclass
class _Part:
    category: str
    label: str
    priority: int
    order: int
    content: str | dict[str, Any] | list[Any]
    shrink: Literal["tail", "head"] | None = None
    wrap: str | None = None  # list content rendered as {wrap: items, **meta}
    meta: dict[str, Any] = field(default_factory=dict)
    trimmed: bool = False

    def block(self) -> DataBlock:
        content: Any = self.content
        if self.wrap is not None:
            content = {**self.meta, self.wrap: self.content}
        return DataBlock(self.category, self.label, content)

    def tokens(self) -> int:
        return estimate_block_tokens(self.block())

    def shrinkable(self) -> bool:
        return self.shrink is not None and isinstance(self.content, list) and len(self.content) > 1

    def shrink_once(self) -> None:
        assert isinstance(self.content, list)
        if self.shrink == "head":
            self.content = self.content[1:]
        else:
            self.content = self.content[:-1]
        self.trimmed = True


class ContextBuilder:
    """Builds the data blocks of one prompt. Cheap to create; one per call."""

    def __init__(
        self,
        ctx: AppContext,
        *,
        task: str | None = None,
        budget_tokens: int | None = None,
        allowed: frozenset[str] | None = None,
    ) -> None:
        self.ctx = ctx
        spec = get_task(task) if task else None
        cats = spec.data_categories if spec is not None else P.DATA_CATEGORIES
        if allowed is not None:
            cats = cats & frozenset(allowed)
        self.allowed: frozenset[str] = cats
        self.budget: int = (
            budget_tokens
            or (spec.context_budget_tokens if spec is not None else None)
            or DEFAULT_BUDGET_TOKENS
        )

    # ------------------------------------------------------------------ public

    async def build(
        self,
        *,
        message: str | None = None,
        message_label: str = "Сообщение пользователя",
        profile: bool | None = None,
        history: int | None = None,
        tasks: int | None = None,
        memory: bool | None = None,
        extra: Sequence[DataBlock | Extra] = (),
        exclude_update_id: int | None = None,
    ) -> list[DataBlock]:
        """Blocks for `Prompt.data`. See the module docstring for the options."""
        return (
            await self.build_report(
                message=message,
                message_label=message_label,
                profile=profile,
                history=history,
                tasks=tasks,
                memory=memory,
                extra=extra,
                exclude_update_id=exclude_update_id,
            )
        ).blocks

    async def build_report(
        self,
        *,
        message: str | None = None,
        message_label: str = "Сообщение пользователя",
        profile: bool | None = None,
        history: int | None = None,
        tasks: int | None = None,
        memory: bool | None = None,
        extra: Sequence[DataBlock | Extra] = (),
        exclude_update_id: int | None = None,
    ) -> BuiltContext:
        """Same as build() plus what was trimmed/dropped and the token estimate."""
        parts: list[_Part] = []
        async with self.ctx.db.transaction() as s:
            prof = await get_profile(s)
            mem_settings = prof.memory_settings if prof is not None else {}
            if self._want(P.PROFILE, profile):
                parts.append(self._profile_part(prof))
            n_hist = self._count(P.HISTORY, history, DEFAULT_HISTORY)
            if n_hist and flag(mem_settings, "use_history"):
                part = await self._history_part(s, n_hist, exclude_update_id)
                if part is not None:
                    parts.append(part)
            n_tasks = self._count(P.TASKS, tasks, DEFAULT_TASKS)
            if n_tasks:
                part = await self._tasks_part(s, n_tasks, prof)
                if part is not None:
                    parts.append(part)
            if self._want(P.MEMORY, memory) and flag(mem_settings, "enabled"):
                parts.extend(
                    await self._memory_parts(s, include_hypotheses=flag(mem_settings, "hypotheses"))
                )
        for i, item in enumerate(extra):
            ex = item if isinstance(item, Extra) else Extra(item)
            if ex.block.category not in self.allowed:
                continue
            content = ex.block.content
            parts.append(
                _Part(
                    category=ex.block.category,
                    label=ex.block.label,
                    priority=ex.priority
                    if ex.priority is not None
                    else PRIORITY.get(ex.block.category, 40),
                    order=_ORDER.get(ex.block.category, 4) * 100 + 10 + i,
                    content=list(content) if isinstance(content, list) else content,
                    shrink=ex.shrink if isinstance(content, list) else None,
                )
            )
        if message is not None and P.MESSAGE in self.allowed and message.strip():
            parts.append(
                _Part(
                    category=P.MESSAGE,
                    label=message_label,
                    priority=PRIORITY[P.MESSAGE],
                    order=_ORDER[P.MESSAGE] * 100,
                    content=message.strip()[:MAX_MESSAGE_CHARS],
                )
            )
        return self._fit(parts)

    # ------------------------------------------------------------------ options

    def _want(self, category: str, option: bool | None) -> bool:
        return category in self.allowed and option is not False

    def _count(self, category: str, option: int | None, default: int) -> int:
        if category not in self.allowed:
            return 0
        return default if option is None else max(0, option)

    # ------------------------------------------------------------------ sources

    def _profile_part(self, prof: OwnerProfile | None) -> _Part:
        now = self.ctx.clock.now()
        tz = prof.timezone if prof is not None else None
        local = to_local(now, tz)
        data: dict[str, Any] = {
            "timezone": tz or "не задан (используется UTC)",
            "local_now": local.strftime("%Y-%m-%dT%H:%M"),
            "weekday": WEEKDAYS_FULL[local.weekday()],
        }
        if prof is not None:
            data["routine"] = prof.routine or {}
            data["quiet_hours"] = prof.quiet_hours or []
            data["morning_time"] = _hhmm(prof.morning_time)
            data["evening_time"] = _hhmm(prof.evening_time)
            data["personality"] = {
                "style_preset": prof.style_preset,
                "strictness": prof.strictness,
                "humor": prof.humor,
                "profanity_allowed": prof.profanity_allowed,
                "verbosity": prof.verbosity,
            }
            if prof.goals_note:
                data["goals_note"] = prof.goals_note
        return _Part(
            category=P.PROFILE,
            label="Профиль и текущее время",
            priority=PRIORITY[P.PROFILE],
            order=_ORDER[P.PROFILE] * 100,
            content=data,
        )

    async def _history_part(
        self, s: Any, limit: int, exclude_update_id: int | None
    ) -> _Part | None:
        stmt = (
            select(ConversationMessage)
            .where(
                ConversationMessage.deleted.is_(False),
                ConversationMessage.kind.in_(("text", "voice")),
            )
            .order_by(ConversationMessage.created_at.desc(), ConversationMessage.id.desc())
            .limit(limit)
        )
        if exclude_update_id is not None:
            stmt = stmt.where(
                (ConversationMessage.update_id.is_(None))
                | (ConversationMessage.update_id != exclude_update_id)
            )
        rows = list((await s.execute(stmt)).scalars())
        rows.reverse()
        prof = await get_profile(s)
        tz = prof.timezone if prof is not None else None
        items: list[dict[str, Any]] = []
        for m in rows:
            text = m.transcript if m.kind == "voice" else m.text
            if not text:
                continue
            items.append(
                {
                    "at": to_local(m.created_at, tz).strftime("%Y-%m-%dT%H:%M"),
                    "from": "user" if m.direction == "in" else "bot",
                    "text": text[:HISTORY_ITEM_CHARS],
                }
            )
        if not items:
            return None
        return _Part(
            category=P.HISTORY,
            label="Недавний диалог (старые сверху)",
            priority=PRIORITY[P.HISTORY],
            order=_ORDER[P.HISTORY] * 100,
            content=items,
            shrink="head",  # oldest messages go first
        )

    async def _tasks_part(self, s: Any, limit: int, prof: OwnerProfile | None) -> _Part | None:
        tz = prof.timezone if prof is not None else None
        rows = list(
            (
                await s.execute(
                    select(Task)
                    .where(Task.status.in_((str(TaskStatus.TODO), str(TaskStatus.IN_PROGRESS))))
                    .order_by(Task.parent_id.is_not(None), Task.sort_order, Task.id)
                    .limit(limit)
                )
            ).scalars()
        )
        if not rows:
            return None
        items = [task_brief(t, tz) for t in rows]
        return _Part(
            category=P.TASKS,
            label="Открытые задачи (id — для ссылок)",
            priority=PRIORITY[P.TASKS],
            order=_ORDER[P.TASKS] * 100,
            content=items,
            shrink="tail",
        )

    async def _memory_parts(self, s: Any, *, include_hypotheses: bool) -> list[_Part]:
        out: list[_Part] = []
        # A summary built from an item that is no longer active is stale even if
        # the memory service has not flagged it yet: never use it.
        stale = exists().where(
            MemorySummaryItem.summary_id == MemorySummary.id,
            MemorySummaryItem.memory_item_id == MemoryItem.id,
            MemoryItem.status != str(MemoryStatus.ACTIVE),
        )
        summaries = list(
            (
                await s.execute(
                    select(MemorySummary)
                    .where(MemorySummary.valid.is_(True), ~stale)
                    .order_by(MemorySummary.created_at.desc(), MemorySummary.id.desc())
                )
            ).scalars()
        )
        latest: dict[str, MemorySummary] = {}
        for sm in summaries:
            latest.setdefault(sm.scope, sm)
        if latest:
            out.append(
                _Part(
                    category=P.MEMORY,
                    label="Сводки памяти",
                    priority=PRIORITY[P.MEMORY] + 1,
                    order=_ORDER[P.MEMORY] * 100,
                    content=[{"scope": k, "summary": v.content} for k, v in latest.items()],
                    shrink="tail",
                )
            )
        stmt = (
            select(MemoryItem)
            .where(
                MemoryItem.status == str(MemoryStatus.ACTIVE),
                MemoryItem.origin != str(MemoryOrigin.USER_REFUTED),
            )
            .order_by(MemoryItem.updated_at.desc(), MemoryItem.id.desc())
            .limit(DEFAULT_MEMORY_ITEMS)
        )
        if not include_hypotheses:
            stmt = stmt.where(MemoryItem.kind != str(MemoryKind.HYPOTHESIS))
        items = list((await s.execute(stmt)).scalars())
        if items:
            out.append(
                _Part(
                    category=P.MEMORY,
                    label="Записи памяти (hypothesis/ai_inferred — только предположения)",
                    priority=PRIORITY[P.MEMORY],
                    order=_ORDER[P.MEMORY] * 100 + 1,
                    content=[_memory_brief(m) for m in items],
                    shrink="tail",
                )
            )
        return out

    # ------------------------------------------------------------------ budget

    def _fit(self, parts: list[_Part]) -> BuiltContext:
        dropped: list[str] = []
        total = sum(p.tokens() for p in parts)
        while total > self.budget and parts:
            victim = min(parts, key=lambda p: (p.priority, -p.order))
            if victim.shrinkable():
                victim.shrink_once()
            elif victim.category == P.MESSAGE and isinstance(victim.content, str):
                others = sum(p.tokens() for p in parts if p is not victim)
                room_chars = (
                    (self.budget - others - 1) * 4 - _BLOCK_OVERHEAD_CHARS - len(victim.label) - 4
                )
                if room_chars < 40 or room_chars >= len(victim.content):
                    parts.remove(victim)
                    dropped.append(victim.label)
                else:
                    victim.content = victim.content[: room_chars - 1] + "…"
                    victim.trimmed = True
            else:
                parts.remove(victim)
                dropped.append(victim.label)
            total = sum(p.tokens() for p in parts)
        parts.sort(key=lambda p: p.order)
        return BuiltContext(
            blocks=[p.block() for p in parts],
            tokens=total,
            budget=self.budget,
            dropped=dropped,
            trimmed=[p.label for p in parts if p.trimmed],
        )


# --------------------------------------------------------------------------- briefs


def _hhmm(t: Any) -> str | None:
    return t.strftime("%H:%M") if t is not None else None


def _local_iso(dt: datetime | None, tz: str | None) -> str | None:
    return to_local(dt, tz).strftime("%Y-%m-%dT%H:%M") if dt is not None else None


def _day_iso(d: date | None) -> str | None:
    return d.isoformat() if d is not None else None


def task_brief(t: Task, tz: str | None) -> dict[str, Any]:
    """Compact JSON of one task for prompts (local times, no empty fields)."""
    data: dict[str, Any] = {
        "id": t.id,
        "title": t.title,
        "status": t.status,
        "planned": _local_iso(t.planned_start_at, tz) or _day_iso(t.planned_date),
        "due": _local_iso(t.due_at, tz) or _day_iso(t.due_date),
        "due_is_hard": t.due_is_hard or None,
        "estimate_min": t.estimate_minutes,
        "priority": t.priority,
        "postpone_count": t.postpone_count or None,
        "parent_id": t.parent_id,
        "goal_id": t.goal_id,
        "project_id": t.project_id,
        "backlog": t.is_backlog or None,
    }
    return {k: v for k, v in data.items() if v is not None}


def _memory_brief(m: MemoryItem) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": m.id,
        "kind": m.kind,
        "origin": m.origin,
        "content": m.content,
    }
    if m.confidence is not None:
        data["confidence"] = round(m.confidence, 2)
    return data
