"""MemoryService (FR-12): long-term memory with provenance.

Every ``memory_items`` row has a ``kind`` (fact / preference / pattern /
hypothesis), an ``origin`` (user_said / system_measured / ai_inferred /
user_confirmed / user_refuted), an optional confidence and, for hypotheses, a
concrete way to check it (``verification_hint``). ``memory_sources`` links an
item to the primary evidence (messages, task events, sessions, interventions...)
so important conclusions stay traceable.

Compact summaries for model context (``memory_summaries``) are rebuilt by the
``memory_summarize`` AI task from ACTIVE items only; ``memory_summary_items``
records every item that went into a summary. Contract with the context builder
(acceptance 20): it reads only ``MemorySummary.valid = True`` and
``MemoryItem.status = 'active'``. This module guarantees those flags:

- refute / correct / confirm: every summary built from the item becomes
  ``valid = False`` in the SAME transaction (rebuilt later without it);
- delete: the item content is scrubbed and every summary built from it is
  removed in the same transaction;
- a rebuild whose input changed while the model was answering is discarded;
- AIUnavailable: invalid summaries stay invalid (not used) instead of stale.

Refuted or deleted statements are never re-added automatically (hypotheses from
the AI and measured patterns are checked against them). Memory toggles live in
``owner_profile.memory_settings`` (services.preferences.MEMORY_TOGGLES).
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.ai import AITaskSpec, AIUnavailable, DataBlock, Prompt, Tier, register_task
from procbot.ai import prompting as P
from procbot.ai.gateway import feature_enabled, run_task
from procbot.context import AppContext
from procbot.db.models import (
    AuditLog,
    ConversationMessage,
    Intervention,
    InterventionOutcome,
    MemoryItem,
    MemorySource,
    MemorySummary,
    MemorySummaryItem,
    PlanRevision,
    Task,
    TaskEvent,
    WorkSession,
)
from procbot.domain.actions import ActionType, SaveHypothesis
from procbot.domain.enums import Actor, MemoryKind, MemoryOrigin, MemoryStatus
from procbot.services.actions import ActionError, ApplyOutcome, register_applier
from procbot.services.audit import audit
from procbot.services.jobs import PeriodicJob, register_job
from procbot.services.preferences import flag
from procbot.services.profile import get_profile
from procbot.timeutil import fmt_local

log = logging.getLogger(__name__)

MAX_CONTENT = 2000
MAX_HINT = 1000
REBUILD_JOB = "memory.rebuild"
REBUILD_INTERVAL_SECONDS = 900.0
SUMMARY_TASK = "memory_summarize"
DELETED_AUDIT_ACTION = "memory.deleted"

# --------------------------------------------------------------------------- labels

KIND_LABELS: dict[str, str] = {
    MemoryKind.FACT: "Факт",
    MemoryKind.PREFERENCE: "Предпочтение",
    MemoryKind.PATTERN: "Закономерность",
    MemoryKind.HYPOTHESIS: "Гипотеза",
}
KIND_LABELS_PLURAL: dict[str, str] = {
    MemoryKind.FACT: "Факты",
    MemoryKind.PREFERENCE: "Предпочтения",
    MemoryKind.PATTERN: "Закономерности",
    MemoryKind.HYPOTHESIS: "Гипотезы",
}
ORIGIN_LABELS: dict[str, str] = {
    MemoryOrigin.USER_SAID: "вы сказали",
    MemoryOrigin.SYSTEM_MEASURED: "измерено",
    MemoryOrigin.AI_INFERRED: "предположение ИИ",
    MemoryOrigin.USER_CONFIRMED: "подтверждено",
    MemoryOrigin.USER_REFUTED: "опровергнуто",
}
STATUS_LABELS: dict[str, str] = {
    MemoryStatus.ACTIVE: "действует",
    MemoryStatus.REFUTED: "опровергнуто — не используется",
    MemoryStatus.DELETED: "удалено",
}

# --------------------------------------------------------------------------- scopes

SCOPE_PROFILE = "profile"
SCOPE_PATTERNS = "patterns"
SCOPE_KINDS: dict[str, tuple[str, ...]] = {
    SCOPE_PROFILE: (MemoryKind.FACT.value, MemoryKind.PREFERENCE.value),
    SCOPE_PATTERNS: (MemoryKind.PATTERN.value, MemoryKind.HYPOTHESIS.value),
}
SCOPES: tuple[str, ...] = tuple(SCOPE_KINDS)


def scope_of(kind: str) -> str:
    for scope, kinds in SCOPE_KINDS.items():
        if kind in kinds:
            return scope
    return SCOPE_PROFILE


class SourceType(StrEnum):
    MESSAGE = "message"  # conversation_messages
    TASK = "task"
    TASK_EVENT = "task_event"
    SESSION = "session"  # work_sessions
    INTERVENTION = "intervention"
    INTERVENTION_OUTCOME = "intervention_outcome"
    PLAN_REVISION = "plan_revision"
    OBSERVATION = "behavior_observation"
    MEMORY_ITEM = "memory_item"  # e.g. a correction of a refuted hypothesis


@dataclass(frozen=True)
class SourceRef:
    type: str
    id: int


class MemoryServiceError(ActionError):
    """Invalid memory operation; `str(exc)` is a short Russian message."""


@dataclass(frozen=True)
class ChangeResult:
    item: MemoryItem
    changed: bool  # False = repeated press / nothing to do


# --------------------------------------------------------------------------- helpers

_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _SPACES.sub(" ", text).strip().strip(".!?…;:,").strip().lower()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def _clean(text: str | None, limit: int, what: str = "Текст") -> str:
    value = _SPACES.sub(" ", (text or "")).strip()
    if not value:
        raise MemoryServiceError(f"{what} не может быть пустым.")
    if len(value) > limit:
        raise MemoryServiceError(f"{what} слишком длинный (больше {limit} символов).")
    return value


@dataclass(frozen=True)
class MemoryFlags:
    enabled: bool
    hypotheses: bool
    use_history: bool


async def memory_flags(s: AsyncSession) -> MemoryFlags:
    p = await get_profile(s)
    settings = p.memory_settings if p is not None else None
    return MemoryFlags(
        enabled=flag(settings, "enabled"),
        hypotheses=flag(settings, "hypotheses"),
        use_history=flag(settings, "use_history"),
    )


async def get_item(s: AsyncSession, item_id: int) -> MemoryItem | None:
    return await s.get(MemoryItem, item_id)


async def require_item(s: AsyncSession, item_id: int) -> MemoryItem:
    item = await s.get(MemoryItem, item_id)
    if item is None or item.status == MemoryStatus.DELETED:
        raise MemoryServiceError("Запись памяти не найдена (возможно, уже удалена).")
    return item


async def _audit(
    s: AsyncSession,
    ctx: AppContext,
    action: str,
    actor: Actor | str,
    item_id: int | None,
    details: dict[str, Any] | None = None,
) -> None:
    # Never put memory content into the audit log (it outlives deletions).
    await audit(
        s,
        now=ctx.clock.now(),
        actor=actor,
        action=action,
        entity="memory_item",
        entity_id=item_id,
        details=details or {},
    )


async def add_sources(
    s: AsyncSession, ctx: AppContext, item: MemoryItem, sources: Iterable[SourceRef]
) -> int:
    """Link primary evidence to an item (deduplicated). Returns how many were added."""
    existing = {
        (t, i)
        for t, i in (
            await s.execute(
                select(MemorySource.source_type, MemorySource.source_id).where(
                    MemorySource.memory_item_id == item.id
                )
            )
        ).all()
    }
    added = 0
    for ref in sources:
        key = (str(ref.type), int(ref.id))
        if key in existing:
            continue
        existing.add(key)
        s.add(
            MemorySource(
                memory_item_id=item.id,
                source_type=key[0],
                source_id=key[1],
                created_at=ctx.clock.now(),
            )
        )
        added += 1
    if added:
        await s.flush()
    return added


async def _find_same(s: AsyncSession, content: str, statuses: Sequence[str]) -> list[MemoryItem]:
    norm = normalize(content)
    rows = (
        (await s.execute(select(MemoryItem).where(MemoryItem.status.in_(list(statuses)))))
        .scalars()
        .all()
    )
    return [r for r in rows if normalize(r.content) == norm]


async def is_blocked(s: AsyncSession, content: str) -> bool:
    """True if the user refuted or deleted this statement before: automatic
    sources (AI, measurements) must not bring it back."""
    if await _find_same(s, content, [MemoryStatus.REFUTED]):
        return True
    digest = content_hash(content)
    rows = (
        (
            await s.execute(
                select(AuditLog.details).where(
                    AuditLog.action == DELETED_AUDIT_ACTION, AuditLog.entity == "memory_item"
                )
            )
        )
        .scalars()
        .all()
    )
    return any((d or {}).get("content_sha256") == digest for d in rows)


# --------------------------------------------------------------------------- adding


async def add_item(
    s: AsyncSession,
    ctx: AppContext,
    *,
    kind: MemoryKind | str,
    origin: MemoryOrigin | str,
    content: str,
    confidence: float | None = None,
    verification_hint: str | None = None,
    sources: Iterable[SourceRef] = (),
    actor: Actor | str = Actor.SYSTEM,
) -> MemoryItem:
    """Low-level insert (no toggles, no dedup). Prefer add_fact / add_hypothesis."""
    kind = MemoryKind(kind)
    origin = MemoryOrigin(origin)
    text = _clean(content, MAX_CONTENT)
    hint = _clean(verification_hint, MAX_HINT, "Способ проверки") if verification_hint else None
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise MemoryServiceError("Уверенность должна быть от 0 до 1.")
    now = ctx.clock.now()
    item = MemoryItem(
        kind=str(kind),
        origin=str(origin),
        status=MemoryStatus.ACTIVE,
        content=text,
        confidence=confidence,
        verification_hint=hint,
        created_at=now,
        updated_at=now,
    )
    s.add(item)
    await s.flush()
    srcs = list(sources)
    await add_sources(s, ctx, item, srcs)
    await _audit(
        s,
        ctx,
        "memory.added",
        actor,
        item.id,
        {"kind": str(kind), "origin": str(origin), "sources": len(srcs)},
    )
    return item


async def add_fact(
    s: AsyncSession,
    ctx: AppContext,
    content: str,
    *,
    kind: MemoryKind | str = MemoryKind.FACT,
    origin: MemoryOrigin | str = MemoryOrigin.USER_SAID,
    sources: Iterable[SourceRef] = (),
    confidence: float | None = None,
    actor: Actor | str = Actor.USER,
) -> MemoryItem | None:
    """A fact / preference / pattern the user said (or the system measured).

    Returns None when long-term memory is switched off. A repeat of an active
    statement only adds its new sources."""
    kind = MemoryKind(kind)
    origin = MemoryOrigin(origin)
    if kind == MemoryKind.HYPOTHESIS:
        raise MemoryServiceError("Гипотезы сохраняются через add_hypothesis.")
    if origin in (MemoryOrigin.AI_INFERRED, MemoryOrigin.USER_REFUTED):
        raise MemoryServiceError("Недопустимое происхождение для факта.")
    if not (await memory_flags(s)).enabled:
        return None
    same = await _find_same(s, content, [MemoryStatus.ACTIVE])
    if same:
        await add_sources(s, ctx, same[0], sources)
        return same[0]
    return await add_item(
        s,
        ctx,
        kind=kind,
        origin=origin,
        content=content,
        confidence=confidence,
        sources=sources,
        actor=actor,
    )


async def add_measured_pattern(
    s: AsyncSession,
    ctx: AppContext,
    content: str,
    *,
    sources: Iterable[SourceRef] = (),
    confidence: float | None = None,
    verification_hint: str | None = None,
) -> tuple[MemoryItem | None, bool]:
    """Gradual, transparent profile learning: a pattern measured by code from the
    owner's data, stored as ``system_measured`` (never a silent settings change).
    Returns (item, created). Refuted/deleted statements are not re-added."""
    if not (await memory_flags(s)).enabled or await is_blocked(s, content):
        return None, False
    same = await _find_same(s, content, [MemoryStatus.ACTIVE])
    if same:
        await add_sources(s, ctx, same[0], sources)
        return same[0], False
    item = await add_item(
        s,
        ctx,
        kind=MemoryKind.PATTERN,
        origin=MemoryOrigin.SYSTEM_MEASURED,
        content=content,
        confidence=confidence,
        verification_hint=verification_hint,
        sources=sources,
        actor=Actor.SYSTEM,
    )
    return item, True


async def add_hypothesis(
    s: AsyncSession,
    ctx: AppContext,
    content: str,
    *,
    verification_hint: str | None = None,
    confidence: float | None = None,
    sources: Iterable[SourceRef] = (),
    actor: Actor | str = Actor.AI,
) -> MemoryItem | None:
    """An AI hypothesis about a pattern: stored as ``ai_inferred`` (never a fact).

    None when memory or hypotheses are switched off, or when the user already
    refuted/deleted the same statement. A repeat returns the active item."""
    flags = await memory_flags(s)
    if not flags.enabled or not flags.hypotheses or await is_blocked(s, content):
        return None
    same = await _find_same(s, content, [MemoryStatus.ACTIVE])
    if same:
        await add_sources(s, ctx, same[0], sources)
        return same[0]
    return await add_item(
        s,
        ctx,
        kind=MemoryKind.HYPOTHESIS,
        origin=MemoryOrigin.AI_INFERRED,
        content=content,
        confidence=confidence,
        verification_hint=verification_hint,
        sources=sources,
        actor=actor,
    )


# --------------------------------------------------------------------------- summaries: invalidation


async def summaries_built_from(s: AsyncSession, item_ids: Sequence[int]) -> list[int]:
    if not item_ids:
        return []
    return list(
        (
            await s.execute(
                select(MemorySummaryItem.summary_id)
                .where(MemorySummaryItem.memory_item_id.in_(list(item_ids)))
                .distinct()
            )
        )
        .scalars()
        .all()
    )


async def invalidate_summaries_for(
    s: AsyncSession, item_ids: Sequence[int], *, drop: bool = False
) -> int:
    """Mark every summary built from these items invalid (drop=True removes
    them, used on deletion so the deleted text does not linger). Same transaction
    as the item change, so the context builder never sees a stale summary."""
    ids = await summaries_built_from(s, item_ids)
    if not ids:
        return 0
    if drop:
        await s.execute(delete(MemorySummaryItem).where(MemorySummaryItem.summary_id.in_(ids)))
        await s.execute(delete(MemorySummary).where(MemorySummary.id.in_(ids)))
    else:
        await s.execute(update(MemorySummary).where(MemorySummary.id.in_(ids)).values(valid=False))
    return len(ids)


# --------------------------------------------------------------------------- user corrections


async def confirm(
    s: AsyncSession, ctx: AppContext, item_id: int, *, actor: Actor = Actor.USER
) -> ChangeResult:
    """The user confirms the statement (also brings back a refuted one)."""
    item = await require_item(s, item_id)
    if item.status == MemoryStatus.ACTIVE and item.origin == MemoryOrigin.USER_CONFIRMED:
        return ChangeResult(item, False)
    item.status = MemoryStatus.ACTIVE
    item.origin = MemoryOrigin.USER_CONFIRMED
    item.updated_at = ctx.clock.now()
    await s.flush()
    await invalidate_summaries_for(s, [item.id])
    await _audit(s, ctx, "memory.confirmed", actor, item.id, {"kind": item.kind})
    return ChangeResult(await require_item(s, item_id), True)


async def refute(
    s: AsyncSession, ctx: AppContext, item_id: int, *, actor: Actor = Actor.USER
) -> ChangeResult:
    """The user says it is wrong: status refuted, origin user_refuted. The item
    leaves the model context at once (summaries built from it are invalidated)."""
    item = await require_item(s, item_id)
    if item.status == MemoryStatus.REFUTED:
        return ChangeResult(item, False)
    item.status = MemoryStatus.REFUTED
    item.origin = MemoryOrigin.USER_REFUTED
    item.updated_at = ctx.clock.now()
    await s.flush()
    await invalidate_summaries_for(s, [item.id])
    await _audit(s, ctx, "memory.refuted", actor, item.id, {"kind": item.kind})
    return ChangeResult(await require_item(s, item_id), True)


async def correct(
    s: AsyncSession,
    ctx: AppContext,
    item_id: int,
    new_content: str,
    *,
    actor: Actor = Actor.USER,
) -> ChangeResult:
    """The user rewrites the statement; it becomes ``user_confirmed`` and active."""
    item = await require_item(s, item_id)
    text = _clean(new_content, MAX_CONTENT)
    if (
        text == item.content
        and item.origin == MemoryOrigin.USER_CONFIRMED
        and item.status == MemoryStatus.ACTIVE
    ):
        return ChangeResult(item, False)
    item.content = text
    item.origin = MemoryOrigin.USER_CONFIRMED
    item.status = MemoryStatus.ACTIVE
    item.updated_at = ctx.clock.now()
    await s.flush()
    await invalidate_summaries_for(s, [item.id])
    await _audit(s, ctx, "memory.corrected", actor, item.id, {"kind": item.kind})
    return ChangeResult(await require_item(s, item_id), True)


async def delete_item(
    s: AsyncSession, ctx: AppContext, item_id: int, *, actor: Actor = Actor.USER
) -> bool:
    """Delete one record: status deleted, content scrubbed, sources unlinked and
    every summary built from it removed - all in one transaction. Only a hash of
    the text is kept (in the audit log) so the AI cannot silently re-add it.
    False if it was already deleted."""
    item = await s.get(MemoryItem, item_id)
    if item is None or item.status == MemoryStatus.DELETED:
        return False
    digest = content_hash(item.content)
    await invalidate_summaries_for(s, [item.id], drop=True)
    await s.execute(delete(MemorySource).where(MemorySource.memory_item_id == item_id))
    item = await s.get(MemoryItem, item_id)
    assert item is not None
    item.status = MemoryStatus.DELETED
    item.content = ""
    item.verification_hint = None
    item.confidence = None
    item.updated_at = ctx.clock.now()
    await s.flush()
    await _audit(
        s, ctx, DELETED_AUDIT_ACTION, actor, item_id, {"kind": item.kind, "content_sha256": digest}
    )
    return True


@dataclass(frozen=True)
class WipeResult:
    items: int
    summaries: int
    messages: int = 0


async def wipe_all(
    s: AsyncSession,
    ctx: AppContext,
    *,
    include_history: bool = False,
    actor: Actor = Actor.USER,
) -> WipeResult:
    """Full clear: every memory item, source and summary is removed; with
    ``include_history`` the conversation history is scrubbed too."""
    n_items = (await s.execute(select(func.count(MemoryItem.id)))).scalar_one()
    n_summaries = (await s.execute(select(func.count(MemorySummary.id)))).scalar_one()
    await s.execute(delete(MemorySummaryItem))
    await s.execute(delete(MemorySummary))
    await s.execute(delete(MemorySource))
    await s.execute(delete(MemoryItem))
    # old per-item audit rows (kinds, hashes) go too: nothing of memory remains
    await s.execute(update(AuditLog).where(AuditLog.entity == "memory_item").values(details={}))
    messages = 0
    if include_history:
        from procbot.services.privacy import scrub_history  # privacy imports this module

        messages = await scrub_history(s, ctx)
    await audit(
        s,
        now=ctx.clock.now(),
        actor=actor,
        action="memory.wiped",
        entity="memory",
        details={"items": n_items, "summaries": n_summaries, "history": include_history},
    )
    return WipeResult(items=int(n_items), summaries=int(n_summaries), messages=messages)


# --------------------------------------------------------------------------- queries


async def list_items(
    s: AsyncSession,
    *,
    kinds: Sequence[str] | None = None,
    origins: Sequence[str] | None = None,
    statuses: Sequence[str] = (MemoryStatus.ACTIVE,),
    limit: int = 50,
) -> list[MemoryItem]:
    stmt = select(MemoryItem).where(MemoryItem.status.in_([str(x) for x in statuses]))
    if kinds:
        stmt = stmt.where(MemoryItem.kind.in_([str(k) for k in kinds]))
    if origins:
        stmt = stmt.where(MemoryItem.origin.in_([str(o) for o in origins]))
    stmt = stmt.order_by(MemoryItem.updated_at.desc(), MemoryItem.id.desc()).limit(limit)
    return list((await s.execute(stmt)).scalars().all())


async def active_hypotheses(s: AsyncSession, *, limit: int = 20) -> list[MemoryItem]:
    return await list_items(s, kinds=[MemoryKind.HYPOTHESIS], limit=limit)


async def counts(s: AsyncSession) -> dict[str, dict[str, int]]:
    """{'kind': {...active by kind}, 'origin': {...active by origin}, 'status': {...}}."""
    out: dict[str, dict[str, int]] = {"kind": {}, "origin": {}, "status": {}}
    for kind, n in (
        await s.execute(
            select(MemoryItem.kind, func.count(MemoryItem.id))
            .where(MemoryItem.status == MemoryStatus.ACTIVE)
            .group_by(MemoryItem.kind)
        )
    ).all():
        out["kind"][str(kind)] = int(n)
    for origin, n in (
        await s.execute(
            select(MemoryItem.origin, func.count(MemoryItem.id))
            .where(MemoryItem.status == MemoryStatus.ACTIVE)
            .group_by(MemoryItem.origin)
        )
    ).all():
        out["origin"][str(origin)] = int(n)
    for status, n in (
        await s.execute(
            select(MemoryItem.status, func.count(MemoryItem.id)).group_by(MemoryItem.status)
        )
    ).all():
        out["status"][str(status)] = int(n)
    return out


async def item_sources(s: AsyncSession, item_id: int) -> list[MemorySource]:
    return list(
        (
            await s.execute(
                select(MemorySource)
                .where(MemorySource.memory_item_id == item_id)
                .order_by(MemorySource.id)
            )
        )
        .scalars()
        .all()
    )


def _short(text: str | None, n: int = 60) -> str:
    t = _SPACES.sub(" ", text or "").strip()
    return t if len(t) <= n else t[: n - 1] + "…"


async def describe_source(s: AsyncSession, src: MemorySource, tz: str | None) -> str:
    """One human line per piece of evidence (Russian)."""
    t, sid = src.source_type, src.source_id
    if t == SourceType.MESSAGE:
        m = await s.get(ConversationMessage, sid)
        if m is None or m.deleted:
            return "сообщение (удалено из истории)"
        return f"ваше сообщение {fmt_local(m.created_at, tz)}: «{_short(m.text or m.transcript)}»"
    if t == SourceType.TASK:
        task = await s.get(Task, sid)
        return f"задача «{_short(task.title)}»" if task else "задача (удалена)"
    if t == SourceType.TASK_EVENT:
        ev = await s.get(TaskEvent, sid)
        if ev is None:
            return "событие задачи (удалено)"
        task = await s.get(Task, ev.task_id)
        title = f" «{_short(task.title, 40)}»" if task else ""
        return f"событие задачи{title}: {ev.event_type} {fmt_local(ev.created_at, tz)}"
    if t == SourceType.SESSION:
        ws = await s.get(WorkSession, sid)
        if ws is None:
            return "рабочая сессия (удалена)"
        from procbot.services.sessions import SOURCE_LABELS

        src_label = SOURCE_LABELS.get(ws.time_source, ws.time_source)
        return f"рабочая сессия {fmt_local(ws.started_at, tz)} ({src_label})"
    if t == SourceType.INTERVENTION:
        iv = await s.get(Intervention, sid)
        if iv is None:
            return "способ помощи (удалён)"
        reason = f", причина: {iv.reason_category}" if iv.reason_category else ""
        return f"предложенная помощь «{iv.kind}» {fmt_local(iv.created_at, tz)}{reason}"
    if t == SourceType.INTERVENTION_OUTCOME:
        out = await s.get(InterventionOutcome, sid)
        return f"итог помощи: {out.result}" if out else "итог помощи (удалён)"
    if t == SourceType.PLAN_REVISION:
        rev = await s.get(PlanRevision, sid)
        return f"версия плана {rev.version}" if rev else "версия плана (удалена)"
    if t == SourceType.MEMORY_ITEM:
        other = await s.get(MemoryItem, sid)
        if other is None or other.status == MemoryStatus.DELETED:
            return "запись памяти (удалена)"
        return f"запись памяти #{other.id} ({ORIGIN_LABELS.get(other.origin, other.origin)})"
    return f"{t} #{sid}"


async def valid_summaries(s: AsyncSession) -> list[MemorySummary]:
    """What a model may see from summaries (the context builder contract)."""
    return list(
        (
            await s.execute(
                select(MemorySummary)
                .where(MemorySummary.valid.is_(True))
                .order_by(MemorySummary.scope, MemorySummary.id)
            )
        )
        .scalars()
        .all()
    )


# --------------------------------------------------------------------------- summaries: rebuild


class MemorySummaryOut(BaseModel):
    """Answer of `memory_summarize`."""

    summary: str = Field(min_length=1, max_length=4000)
    used_item_ids: list[int] = Field(default_factory=list)


SUMMARY_SYSTEM = (
    "Ты собираешь краткую сводку долговременной памяти для контекста ассистента. "
    "Используй ТОЛЬКО переданные записи, ничего не добавляй и не обобщай сверх них. "
    "Сохраняй происхождение: «предположение ИИ» называй предположением, а не фактом; "
    "«измерено» — как наблюдение по данным; «подтверждено»/«вы сказали» — как слова "
    "пользователя. Не ставь диагнозов и не делай психологических выводов."
)

MEMORY_SUMMARIZE = register_task(
    AITaskSpec(
        name=SUMMARY_TASK,
        tier=Tier.FAST,
        output=MemorySummaryOut,
        max_output_tokens=900,
        data_categories=frozenset({P.MEMORY}),
        system=SUMMARY_SYSTEM,
        description="Пересборка сводок памяти",
    )
)

SCOPE_TITLES = {
    SCOPE_PROFILE: "факты и предпочтения пользователя",
    SCOPE_PATTERNS: "закономерности и гипотезы о поведении",
}


@dataclass(frozen=True)
class RebuildResult:
    scope: str
    status: str  # rebuilt / up_to_date / empty / unavailable / discarded / disabled
    summary_id: int | None = None


async def _scope_items(s: AsyncSession, scope: str) -> list[MemoryItem]:
    return list(
        (
            await s.execute(
                select(MemoryItem)
                .where(
                    MemoryItem.status == MemoryStatus.ACTIVE,
                    MemoryItem.kind.in_(list(SCOPE_KINDS[scope])),
                )
                .order_by(MemoryItem.id)
            )
        )
        .scalars()
        .all()
    )


async def _linked_items(s: AsyncSession, summary_id: int) -> set[int]:
    return set(
        (
            await s.execute(
                select(MemorySummaryItem.memory_item_id).where(
                    MemorySummaryItem.summary_id == summary_id
                )
            )
        )
        .scalars()
        .all()
    )


async def needs_rebuild(s: AsyncSession, scope: str) -> bool:
    items = await _scope_items(s, scope)
    valid = (
        (
            await s.execute(
                select(MemorySummary).where(
                    MemorySummary.scope == scope, MemorySummary.valid.is_(True)
                )
            )
        )
        .scalars()
        .all()
    )
    ids = {i.id for i in items}
    if not ids:
        return bool(valid)
    if len(valid) != 1:
        return True
    return await _linked_items(s, valid[0].id) != ids


async def _drop_scope(s: AsyncSession, scope: str) -> None:
    ids = list(
        (await s.execute(select(MemorySummary.id).where(MemorySummary.scope == scope)))
        .scalars()
        .all()
    )
    if ids:
        await s.execute(delete(MemorySummaryItem).where(MemorySummaryItem.summary_id.in_(ids)))
        await s.execute(delete(MemorySummary).where(MemorySummary.id.in_(ids)))


def _item_payload(item: MemoryItem) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": item.id,
        "kind": KIND_LABELS.get(item.kind, item.kind),
        "origin": ORIGIN_LABELS.get(item.origin, item.origin),
        "content": item.content,
    }
    if item.confidence is not None:
        out["confidence"] = round(item.confidence, 2)
    if item.verification_hint:
        out["how_to_check"] = item.verification_hint
    return out


async def rebuild_scope(ctx: AppContext, scope: str, *, force: bool = False) -> RebuildResult:
    """Rebuild one scope's summary from ACTIVE items only. Must run outside a
    transaction (calls the AI)."""
    if scope not in SCOPE_KINDS:
        raise ValueError(f"unknown memory scope {scope!r}")
    async with ctx.db.transaction() as s:
        if not (await memory_flags(s)).enabled:
            return RebuildResult(scope, "disabled")
        items = await _scope_items(s, scope)
        if not items:
            await _drop_scope(s, scope)
            return RebuildResult(scope, "empty")
        if not force and not await needs_rebuild(s, scope):
            return RebuildResult(scope, "up_to_date")
        snapshot = {i.id: i.updated_at for i in items}
        payload = [_item_payload(i) for i in items]
    ids = set(snapshot)

    def validate(out: BaseModel) -> list[str]:
        assert isinstance(out, MemorySummaryOut)
        unknown = sorted(set(out.used_item_ids) - ids)
        return [f"used_item_ids содержит неизвестные id: {unknown}"] if unknown else []

    prompt = Prompt(
        instruction=(
            f"Составь сводку ({SCOPE_TITLES[scope]}) из записей памяти: 2–8 коротких "
            "предложений на русском. В used_item_ids перечисли id использованных записей."
        ),
        data=[DataBlock(P.MEMORY, f"memory:{scope}", payload)],
    )
    try:
        result = await run_task(
            ctx, SUMMARY_TASK, prompt, initiator="services.memory", validator=validate
        )
    except AIUnavailable as exc:
        log.info("memory summary for %s not rebuilt: %s", scope, exc)
        return RebuildResult(scope, "unavailable")
    out = result.as_(MemorySummaryOut)
    async with ctx.db.transaction() as s:
        # The input must be unchanged: an item refuted/deleted/edited while the
        # model was answering must not get into a valid summary (acceptance 20).
        for item_id, stamp in snapshot.items():
            item = await s.get(MemoryItem, item_id)
            if item is None or item.status != MemoryStatus.ACTIVE or item.updated_at != stamp:
                return RebuildResult(scope, "discarded")
        if not (await memory_flags(s)).enabled:
            return RebuildResult(scope, "disabled")
        await _drop_scope(s, scope)
        summary = MemorySummary(
            scope=scope, content=out.summary.strip(), valid=True, created_at=ctx.clock.now()
        )
        s.add(summary)
        await s.flush()
        # Link EVERY input item, not only the ones the model claims it used: any
        # of them may have leaked into the text, so any of them invalidates it.
        for item_id in sorted(snapshot):
            s.add(MemorySummaryItem(summary_id=summary.id, memory_item_id=item_id))
        await s.flush()
        await audit(
            s,
            now=ctx.clock.now(),
            actor=Actor.SYSTEM,
            action="memory.summary_rebuilt",
            entity="memory_summary",
            entity_id=summary.id,
            details={"scope": scope, "items": len(snapshot), "ai_call_id": result.call_id},
        )
        return RebuildResult(scope, "rebuilt", summary.id)


async def rebuild_summaries(ctx: AppContext, *, force: bool = False) -> list[RebuildResult]:
    return [await rebuild_scope(ctx, scope, force=force) for scope in SCOPES]


async def rebuild_job(ctx: AppContext) -> None:
    """Periodic rebuild (feature switch ai_features.memory_rebuild)."""
    if not feature_enabled(ctx, "memory_rebuild"):
        return
    await rebuild_summaries(ctx)


# --------------------------------------------------------------------------- action applier


async def _apply_save_hypothesis(
    s: AsyncSession, ctx: AppContext, action: SaveHypothesis, actor: Actor
) -> ApplyOutcome:
    item = await add_hypothesis(
        s,
        ctx,
        action.content,
        verification_hint=action.verification_hint,
        confidence=action.confidence,
        actor=actor,
    )
    if item is None:
        raise MemoryServiceError(
            "Гипотеза не сохранена: память или гипотезы выключены, либо вы её уже "
            "опровергли или удалили."
        )
    return ApplyOutcome(result={"memory_item_id": item.id})


async def _describe_save_hypothesis(
    s: AsyncSession, ctx: AppContext, action: SaveHypothesis
) -> dict[str, Any]:
    lines = [f"Как проверить: {action.verification_hint}"] if action.verification_hint else []
    return {"summary": f"Запомнить гипотезу: {_short(action.content, 120)}", "lines": lines}


def register(ctx: AppContext) -> None:
    register_applier(
        ctx, ActionType.SAVE_HYPOTHESIS, _apply_save_hypothesis, describe=_describe_save_hypothesis
    )
    register_job(
        ctx,
        PeriodicJob(name=REBUILD_JOB, interval_seconds=REBUILD_INTERVAL_SECONDS, run=rebuild_job),
    )
