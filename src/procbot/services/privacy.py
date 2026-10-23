"""Privacy (spec section 7, FR-12): export, history deletion, full wipe.

- ``build_export`` returns everything personal as JSON-ready data (profile,
  tasks, plans, sessions, interventions, memory, conversation history...). It
  never contains secrets: encrypted AI keys, OAuth tokens and secret-looking
  settings are left out by construction and by a final check.
- ``scrub_history`` marks conversation messages deleted and scrubs their text and
  transcript (and the text of bot messages kept for the 4/60 limit).
- ``wipe_everything`` = memory + summaries + history + tasks and every derived
  record; profile/settings, AI keys and the calendar connection stay.
- Deleted data still exists in backups made BEFORE the deletion until their
  retention ends (open question 8: retention not set = backups are not removed
  automatically). Data already processed by an external AI provider cannot be
  erased by us instantly. ``retention_notice`` says this honestly in the UI.

This module writes to tables owned by other services only to delete/scrub them
on the owner's explicit request (the privacy right overrides ownership).
"""

from __future__ import annotations

import json
from datetime import date, datetime, time
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.base import Base
from procbot.db.models import (
    AICredential,
    AuditLog,
    BehaviorObservation,
    CalendarConnection,
    CalendarEventCache,
    CalendarEventLink,
    Category,
    Context,
    ConversationMessage,
    DailyPlan,
    Goal,
    Habit,
    HabitOccurrence,
    Intervention,
    InterventionOutcome,
    MemoryItem,
    MemorySource,
    MemorySummary,
    MemorySummaryItem,
    Milestone,
    OutboundMessage,
    OwnerProfile,
    PendingApproval,
    PlanItem,
    PlanRevision,
    Project,
    ReminderSchedule,
    SessionCorrection,
    SessionPause,
    Setting,
    Task,
    TaskEvent,
    WorkSession,
)
from procbot.domain.enums import Actor, MemoryStatus
from procbot.services import memory
from procbot.services.audit import audit

EXPORT_VERSION = 1
WIPE_ALL_PHRASE = "УДАЛИТЬ ВСЁ"
SCRUBBED_OUTBOUND = "[удалено]"

# Column names that must never leave the database in an export.
_SECRET_COLUMNS = frozenset({"encrypted_key", "encrypted_token", "key_last4", "token"})
_SECRET_MARKERS = ("token", "secret", "password", "api_key", "apikey", "encrypted", "credential")


# --------------------------------------------------------------------------- notices


def retention_notice(ctx: AppContext) -> str:
    """Honest text about what deletion does and does not reach."""
    days = ctx.config.backup.retention_days
    if days is None:
        backups = (
            "Резервные копии: удалённые данные остаются в уже созданных зашифрованных "
            "копиях базы. Срок хранения копий пока не утверждён, поэтому старые копии "
            "автоматически не удаляются — их можно удалить вручную на сервере."
        )
    else:
        backups = (
            "Резервные копии: удалённые данные остаются в уже созданных зашифрованных "
            f"копиях, пока они не будут удалены по сроку хранения ({days} дн.)."
        )
    return (
        f"{backups}\n"
        "Внешние ИИ-провайдеры: то, что уже было отправлено модели, хранится по правилам "
        "провайдера — мгновенно стереть это у них бот не может."
    )


# --------------------------------------------------------------------------- history


async def scrub_history(s: AsyncSession, ctx: AppContext) -> int:
    """Delete the conversation history: rows are marked deleted and their
    text/transcript scrubbed. Bot messages kept in ``outbound_messages`` (needed
    for the reminder limit) lose their text and buttons too. Returns how many
    conversation rows were scrubbed."""
    n = (
        await s.execute(
            select(func.count(ConversationMessage.id)).where(
                (ConversationMessage.deleted.is_(False))
                | ConversationMessage.text.is_not(None)
                | ConversationMessage.transcript.is_not(None)
            )
        )
    ).scalar_one()
    await s.execute(update(ConversationMessage).values(deleted=True, text=None, transcript=None))
    await s.execute(update(OutboundMessage).values(text=SCRUBBED_OUTBOUND, reply_markup=None))
    # Memory items lose their links to the scrubbed messages' content anyway;
    # the link rows themselves stay (they hold only ids).
    await audit(
        s,
        now=ctx.clock.now(),
        actor=Actor.USER,
        action="privacy.history_deleted",
        entity="conversation",
        details={"messages": int(n)},
    )
    return int(n)


# --------------------------------------------------------------------------- full wipe


async def wipe_everything(s: AsyncSession, ctx: AppContext) -> dict[str, int]:
    """Memory, summaries, history AND tasks, plans, sessions, interventions,
    goals, habits, pending approvals and the calendar event cache. Profile,
    settings, AI keys and the calendar connection itself stay."""
    counts: dict[str, int] = {}
    res = await memory.wipe_all(s, ctx, include_history=True)
    counts["memory_items"] = res.items
    counts["messages"] = res.messages
    counts["tasks"] = int((await s.execute(select(func.count(Task.id)))).scalar_one())
    # Children first; FK cascades cover the rest.
    for model in (
        InterventionOutcome,
        SessionPause,
        SessionCorrection,
        BehaviorObservation,
        CalendarEventLink,
        PendingApproval,
        PlanItem,
        PlanRevision,
        DailyPlan,
        HabitOccurrence,
    ):
        await s.execute(delete(model))
    await s.execute(update(WorkSession).values(intervention_id=None))
    await s.execute(delete(WorkSession))
    await s.execute(delete(Intervention))
    # reminders tied to deleted entities (initiative ones stay; the seeder keeps them)
    await s.execute(
        delete(ReminderSchedule).where(
            ReminderSchedule.task_id.is_not(None)
            | ReminderSchedule.session_id.is_not(None)
            | ReminderSchedule.plan_id.is_not(None)
        )
    )
    await s.execute(delete(TaskEvent))
    await s.execute(update(Task).values(parent_id=None))
    for model in (Task, Habit, Milestone, Project, Goal, Category, Context, CalendarEventCache):
        await s.execute(delete(model))
    await s.execute(update(OwnerProfile).values(goals_note=None))
    # The audit log keeps what happened, not the content (task snapshots etc.).
    await s.execute(update(AuditLog).where(AuditLog.action != "privacy.wiped").values(details={}))
    await audit(
        s,
        now=ctx.clock.now(),
        actor=Actor.USER,
        action="privacy.wiped",
        entity="all",
        details=counts,
    )
    return counts


# --------------------------------------------------------------------------- export


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    return value


def _row(obj: Base, *, exclude: frozenset[str] = frozenset()) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in obj.__table__.columns:  # type: ignore[attr-defined]
        name = col.key
        if name in exclude or name in _SECRET_COLUMNS or name.startswith("encrypted"):
            continue
        out[name] = _jsonable(getattr(obj, name))
    return out


async def _all(s: AsyncSession, model: type[Base], *order: Any) -> list[Any]:
    stmt = select(model)
    if order:
        stmt = stmt.order_by(*order)
    return list((await s.execute(stmt)).scalars().all())


def _secret_key(key: str) -> bool:
    k = key.lower()
    return any(m in k for m in _SECRET_MARKERS)


async def build_export(s: AsyncSession, ctx: AppContext) -> dict[str, Any]:
    """Everything personal, JSON-ready. Sections: meta, profile, settings, tasks,
    task_events, goals, projects, milestones, habits, plans, sessions,
    interventions, observations, memory, history, integrations."""
    profile = await s.get(OwnerProfile, 1)
    settings = {
        row.key: row.value
        for row in await _all(s, Setting, Setting.key)
        if not _secret_key(row.key)
    }
    revisions = await _all(s, PlanRevision, PlanRevision.id)
    items = await _all(s, PlanItem, PlanItem.id)
    plans = []
    for plan in await _all(s, DailyPlan, DailyPlan.local_date):
        revs = []
        for rev in (r for r in revisions if r.plan_id == plan.id):
            rd = _row(rev)
            rd["items"] = [_row(i) for i in items if i.revision_id == rev.id]
            revs.append(rd)
        pd = _row(plan)
        pd["revisions"] = revs
        plans.append(pd)
    pauses = await _all(s, SessionPause, SessionPause.id)
    corrections = await _all(s, SessionCorrection, SessionCorrection.id)
    sessions = []
    for ws in await _all(s, WorkSession, WorkSession.id):
        wd = _row(ws)
        wd["pauses"] = [_row(p) for p in pauses if p.session_id == ws.id]
        wd["corrections"] = [_row(c) for c in corrections if c.session_id == ws.id]
        sessions.append(wd)
    outcomes = await _all(s, InterventionOutcome, InterventionOutcome.id)
    interventions = []
    for iv in await _all(s, Intervention, Intervention.id):
        d = _row(iv)
        d["outcomes"] = [_row(o) for o in outcomes if o.intervention_id == iv.id]
        interventions.append(d)
    occurrences = await _all(s, HabitOccurrence, HabitOccurrence.local_date)
    habits = []
    for h in await _all(s, Habit, Habit.id):
        d = _row(h)
        d["occurrences"] = [_row(o) for o in occurrences if o.habit_id == h.id]
        habits.append(d)
    sources = await _all(s, MemorySource, MemorySource.id)
    mem_items = []
    for it in await _all(s, MemoryItem, MemoryItem.id):
        if it.status == MemoryStatus.DELETED:
            continue
        d = _row(it)
        d["origin_label"] = memory.ORIGIN_LABELS.get(it.origin, it.origin)
        d["sources"] = [
            {"type": x.source_type, "id": x.source_id} for x in sources if x.memory_item_id == it.id
        ]
        mem_items.append(d)
    links = await _all(s, MemorySummaryItem)
    summaries = []
    for sm in await _all(s, MemorySummary, MemorySummary.id):
        d = _row(sm)
        d["built_from"] = sorted(x.memory_item_id for x in links if x.summary_id == sm.id)
        summaries.append(d)
    history = [
        _row(m, exclude=frozenset({"deleted"}))
        for m in await _all(s, ConversationMessage, ConversationMessage.created_at)
        if not m.deleted
    ]
    calendars = [
        {
            "provider": c.provider,
            "status": c.status,
            "read_calendar_ids": c.read_calendar_ids,
            "write_calendar_id": c.write_calendar_id,
            "last_sync_at": _jsonable(c.last_sync_at),
        }
        for c in await _all(s, CalendarConnection, CalendarConnection.id)
    ]
    ai_keys = [
        {"label": k.label, "provider": k.provider, "status": k.status}
        for k in await _all(s, AICredential, AICredential.id)
    ]
    data: dict[str, Any] = {
        "meta": {
            "format": "procbot-export",
            "version": EXPORT_VERSION,
            "exported_at": _jsonable(ctx.clock.now()),
            "timezone": profile.timezone if profile else None,
            "note": (
                "Экспорт не содержит ключей ИИ, токенов календаря и других секретов. "
                + retention_notice(ctx)
            ),
        },
        "profile": _row(profile) if profile else None,
        "settings": settings,
        "tasks": [_row(t) for t in await _all(s, Task, Task.id)],
        "task_events": [_row(e) for e in await _all(s, TaskEvent, TaskEvent.id)],
        "goals": [_row(g) for g in await _all(s, Goal, Goal.id)],
        "projects": [_row(p) for p in await _all(s, Project, Project.id)],
        "milestones": [_row(m) for m in await _all(s, Milestone, Milestone.id)],
        "habits": habits,
        "plans": plans,
        "sessions": sessions,
        "interventions": interventions,
        "observations": [
            _row(o) for o in await _all(s, BehaviorObservation, BehaviorObservation.id)
        ],
        "memory": {"items": mem_items, "summaries": summaries},
        "history": history,
        "integrations": {"calendar": calendars, "ai_keys": ai_keys},
    }
    _assert_no_secrets(data)
    return data


def _assert_no_secrets(node: Any, path: str = "") -> None:
    """Defence in depth: no key that looks like a secret anywhere in the export."""
    if isinstance(node, dict):
        for k, v in node.items():
            key = str(k)
            if key in _SECRET_COLUMNS or key.startswith("encrypted"):
                raise AssertionError(f"secret-looking field in export: {path}/{key}")
            _assert_no_secrets(v, f"{path}/{key}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _assert_no_secrets(v, f"{path}[{i}]")


def export_bytes(data: dict[str, Any]) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=1, default=str).encode("utf-8")


async def export_json(ctx: AppContext) -> tuple[str, bytes]:
    """(filename, bytes) for the owner's export document."""
    async with ctx.db.transaction() as s:
        data = await build_export(s, ctx)
        await audit(
            s,
            now=ctx.clock.now(),
            actor=Actor.USER,
            action="privacy.exported",
            entity="all",
            details={"sections": sorted(data)},
        )
    name = f"procbot-export-{ctx.clock.now().strftime('%Y%m%d-%H%M')}.json"
    return name, export_bytes(data)
