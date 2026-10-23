"""Physical data model.

Follows the logical schema of spec section 5 plus the architecture doc additions.
Conventions:
- every timestamp is an aware UTC datetime (UTCDateTime); local wall-clock rules are
  stored as LocalTime + the owner's IANA timezone and resolved per date;
- journals (task_events, audit_log, session_corrections) are append-only;
- enums are stored as strings (see procbot.domain.enums);
- every table has exactly one owning service that writes to it.
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from procbot.db.base import Base, UTCDateTime

# --------------------------------------------------------------------------- profile


class OwnerProfile(Base):
    """The single owner. Exactly one row (id=1) once onboarding started."""

    __tablename__ = "owner_profile"

    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_user_id: Mapped[int] = mapped_column(Integer, unique=True)
    chat_id: Mapped[int | None]
    timezone: Mapped[str | None] = mapped_column(String(64))  # IANA, chosen at onboarding
    onboarding_step: Mapped[str | None] = mapped_column(String(32))  # None = finished
    # Daily rhythm (open question 2: no defaults are invented; None = not set).
    morning_time: Mapped[time | None]
    evening_time: Mapped[time | None]
    weekly_review_dow: Mapped[int | None]  # 0=Mon..6=Sun
    weekly_review_time: Mapped[time | None]
    # list of {"start":"23:00","end":"08:00","days":[0..6] | null}
    quiet_hours: Mapped[list[Any]] = mapped_column(JSON, default=list)
    # {"weekdays":[0,1,2,3,4], "sleep": {"start":"..","end":".."}, "notes": "..."}
    routine: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    pause_until: Mapped[datetime | None] = mapped_column(UTCDateTime)  # manual pause
    notifications_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    # Personality (FR-08), each knob independent.
    style_preset: Mapped[str] = mapped_column(String(32), default="neutral")
    strictness: Mapped[int] = mapped_column(Integer, default=2)  # 0..4
    humor: Mapped[int] = mapped_column(Integer, default=1)  # 0..4
    profanity_allowed: Mapped[bool] = mapped_column(Boolean, default=False)
    verbosity: Mapped[int] = mapped_column(Integer, default=2)  # 0..4
    # Work session defaults (open question 5: asked on first use, None = ask).
    pomodoro_work_min: Mapped[int | None]
    pomodoro_break_min: Mapped[int | None]
    short_start_min: Mapped[int | None]
    # Reminder timing (open question 3). None = fall back to app.yaml, which is
    # None by default = that reminder type is off until the user sets it.
    pre_task_lead_min: Mapped[int | None]
    start_delay_min: Mapped[int | None]
    # Memory / autonomy toggles, see services.profile for keys.
    memory_settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    autonomy_settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    goals_note: Mapped[str | None] = mapped_column(Text)  # free-text base goals from onboarding
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]


class Setting(Base):
    """Key-value runtime settings and overrides (e.g. AI routing overrides)."""

    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[Any] = mapped_column(JSON)
    updated_at: Mapped[datetime]


# --------------------------------------------------------------------------- structure


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    created_at: Mapped[datetime]


class Context(Base):
    """Execution context: 'at computer', 'outside', 'phone', ..."""

    __tablename__ = "contexts"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    created_at: Mapped[datetime]


class Goal(Base):
    __tablename__ = "goals"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(300))
    description: Mapped[str | None] = mapped_column(Text)
    measurable_outcome: Mapped[str | None] = mapped_column(Text)
    success_criteria: Mapped[str | None] = mapped_column(Text)
    horizon_date: Mapped[date | None]
    status: Mapped[str] = mapped_column(String(16), default="active")
    progress_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    goal_id: Mapped[int | None] = mapped_column(ForeignKey("goals.id", ondelete="SET NULL"))
    title: Mapped[str] = mapped_column(String(300))
    description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="active")
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]


class Milestone(Base):
    __tablename__ = "milestones"

    id: Mapped[int] = mapped_column(primary_key=True)
    goal_id: Mapped[int | None] = mapped_column(ForeignKey("goals.id", ondelete="CASCADE"))
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String(300))
    due_date: Mapped[date | None]
    status: Mapped[str] = mapped_column(String(16), default="active")
    is_key: Mapped[bool] = mapped_column(Boolean, default=False)  # key milestone (FR-15)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]


class Task(Base):
    __tablename__ = "tasks"
    __table_args__ = (
        Index("ix_tasks_status_planned", "status", "planned_start_at"),
        Index("ix_tasks_parent", "parent_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    description: Mapped[str | None] = mapped_column(Text)
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    goal_id: Mapped[int | None] = mapped_column(ForeignKey("goals.id", ondelete="SET NULL"))
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id", ondelete="SET NULL"))
    milestone_id: Mapped[int | None] = mapped_column(
        ForeignKey("milestones.id", ondelete="SET NULL")
    )
    category_id: Mapped[int | None] = mapped_column(
        ForeignKey("categories.id", ondelete="SET NULL")
    )
    context_id: Mapped[int | None] = mapped_column(ForeignKey("contexts.id", ondelete="SET NULL"))
    status: Mapped[str] = mapped_column(String(16), default="todo")
    # Deadline: either an exact moment or a whole local day.
    due_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    due_date: Mapped[date | None]
    due_is_hard: Mapped[bool] = mapped_column(Boolean, default=False)  # obligatory deadline
    planned_start_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    planned_date: Mapped[date | None]  # planned for a day without exact time
    estimate_minutes: Mapped[int | None]
    estimate_source: Mapped[str | None] = mapped_column(String(16))
    priority: Mapped[int | None]  # ordering hint, no fixed scale
    importance: Mapped[int | None]  # open question 4: no fixed scale yet
    done_criteria: Mapped[str | None] = mapped_column(Text)  # definition of done
    # "button": close by a tap; "criteria": compare with done_criteria before closing.
    verification: Mapped[str] = mapped_column(String(16), default="button")
    is_backlog: Mapped[bool] = mapped_column(Boolean, default=False)
    recurrence: Mapped[dict[str, Any] | None] = mapped_column(JSON)  # recurring chores
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    postpone_count: Mapped[int] = mapped_column(Integer, default=0)
    # Per-task reminder timing set by user or AdjustReminders (AUTO in FR-15):
    # {"pre_task_lead_min": int|None, "start_delay_min": int|None}
    reminder_overrides: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # approved = the user confirmed this commitment (FR-15 rescheduling needs consent)
    approved: Mapped[bool] = mapped_column(Boolean, default=True)
    created_by: Mapped[str] = mapped_column(String(16), default="user")  # Actor
    progress_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    subtasks: Mapped[list[Task]] = relationship(
        "Task", back_populates="parent", cascade="all, delete-orphan", passive_deletes=True
    )
    parent: Mapped[Task | None] = relationship("Task", remote_side=[id], back_populates="subtasks")


class TaskEvent(Base):
    """Append-only journal of task changes, postponements, completions, reasons."""

    __tablename__ = "task_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    event_type: Mapped[str] = mapped_column(String(32))
    actor: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime]


# --------------------------------------------------------------------------- habits


class Habit(Base):
    __tablename__ = "habits"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(300))
    description: Mapped[str | None] = mapped_column(Text)
    goal_id: Mapped[int | None] = mapped_column(ForeignKey("goals.id", ondelete="SET NULL"))
    # {"type":"daily"} | {"type":"weekly","days":[0,2,4]} | {"type":"times_per_week","n":3}
    schedule: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    preferred_time: Mapped[time | None]
    # flexible target, e.g. {"unit":"min","amount":10,"min_amount":2}
    target: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="active")
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]


class HabitOccurrence(Base):
    __tablename__ = "habit_occurrences"
    __table_args__ = (UniqueConstraint("habit_id", "local_date", name="uq_habit_occ_day"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    habit_id: Mapped[int] = mapped_column(ForeignKey("habits.id", ondelete="CASCADE"))
    local_date: Mapped[date]
    status: Mapped[str] = mapped_column(String(16))
    amount: Mapped[float | None] = mapped_column(Float)
    note: Mapped[str | None] = mapped_column(Text)
    time_source: Mapped[str] = mapped_column(String(16), default="exact")
    created_at: Mapped[datetime]


# --------------------------------------------------------------------------- plans


class DailyPlan(Base):
    __tablename__ = "daily_plans"

    id: Mapped[int] = mapped_column(primary_key=True)
    local_date: Mapped[date] = mapped_column(unique=True)
    status: Mapped[str] = mapped_column(String(16), default="proposed")
    current_revision_id: Mapped[int | None]
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]


class PlanRevision(Base):
    __tablename__ = "plan_revisions"
    __table_args__ = (UniqueConstraint("plan_id", "version", name="uq_plan_rev_version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    plan_id: Mapped[int] = mapped_column(ForeignKey("daily_plans.id", ondelete="CASCADE"))
    version: Mapped[int]
    status: Mapped[str] = mapped_column(String(16), default="proposed")
    created_by: Mapped[str] = mapped_column(String(16))  # Actor
    explanation: Mapped[str | None] = mapped_column(Text)
    assumptions: Mapped[list[Any]] = mapped_column(JSON, default=list)
    calendar_stale: Mapped[bool] = mapped_column(Boolean, default=False)
    used_ai: Mapped[bool] = mapped_column(Boolean, default=False)
    change_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime]
    approved_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class PlanItem(Base):
    __tablename__ = "plan_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    revision_id: Mapped[int] = mapped_column(
        ForeignKey("plan_revisions.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(16))
    task_id: Mapped[int | None] = mapped_column(ForeignKey("tasks.id", ondelete="SET NULL"))
    habit_id: Mapped[int | None] = mapped_column(ForeignKey("habits.id", ondelete="SET NULL"))
    title: Mapped[str] = mapped_column(String(500))
    start_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    end_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    estimate_is_approximate: Mapped[bool] = mapped_column(Boolean, default=True)
    rationale: Mapped[str | None] = mapped_column(Text)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    outcome: Mapped[str | None] = mapped_column(String(16))  # done/partial/skipped/moved


# --------------------------------------------------------------------------- work


class WorkSession(Base):
    __tablename__ = "work_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int | None] = mapped_column(
        ForeignKey("tasks.id", ondelete="SET NULL"), index=True
    )
    mode: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="unconfirmed")
    planned_start_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    ended_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    time_source: Mapped[str] = mapped_column(String(16), default="exact")
    confidence: Mapped[str | None] = mapped_column(String(16))  # high/medium/low
    outcome: Mapped[str | None] = mapped_column(String(16))
    result_note: Mapped[str | None] = mapped_column(Text)
    evidence: Mapped[str | None] = mapped_column(Text)  # text/link the user gave
    work_minutes: Mapped[int | None]  # pomodoro period
    break_minutes: Mapped[int | None]
    planned_minutes: Mapped[int | None]  # short start / free timer target
    intervention_id: Mapped[int | None] = mapped_column(
        ForeignKey("interventions.id", ondelete="SET NULL", use_alter=True)
    )
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]

    pauses: Mapped[list[SessionPause]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="SessionPause.started_at"
    )


class SessionPause(Base):
    __tablename__ = "session_pauses"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("work_sessions.id", ondelete="CASCADE"))
    started_at: Mapped[datetime]
    ended_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    time_source: Mapped[str] = mapped_column(String(16), default="exact")

    session: Mapped[WorkSession] = relationship(back_populates="pauses")


class SessionCorrection(Base):
    """Append-only journal of corrections to session times (FR-06)."""

    __tablename__ = "session_corrections"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("work_sessions.id", ondelete="CASCADE"))
    field: Mapped[str] = mapped_column(String(32))
    old_value: Mapped[str | None] = mapped_column(Text)
    new_value: Mapped[str | None] = mapped_column(Text)
    actor: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime]


# --------------------------------------------------------------------------- reminders


class ReminderSchedule(Base):
    __tablename__ = "reminder_schedule"
    __table_args__ = (Index("ix_reminder_due", "state", "due_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    category: Mapped[str] = mapped_column(String(32))
    due_at: Mapped[datetime]
    # After this moment the reminder is stale: fold into a digest instead of sending.
    stale_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    dedup_key: Mapped[str] = mapped_column(String(200), unique=True)
    state: Mapped[str] = mapped_column(String(16), default="pending")
    task_id: Mapped[int | None] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("work_sessions.id", ondelete="CASCADE")
    )
    plan_id: Mapped[int | None] = mapped_column(ForeignKey("daily_plans.id", ondelete="CASCADE"))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    reschedule_count: Mapped[int] = mapped_column(Integer, default=0)
    last_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]


class OutboundMessage(Base):
    """Every bot-initiated message. Source of truth for the 4/60 limit."""

    __tablename__ = "outbound_messages"
    __table_args__ = (Index("ix_outbound_limit", "counts_toward_limit", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    category: Mapped[str] = mapped_column(String(32))
    dedup_key: Mapped[str] = mapped_column(String(200), unique=True)
    reminder_id: Mapped[int | None] = mapped_column(
        ForeignKey("reminder_schedule.id", ondelete="SET NULL")
    )
    counts_toward_limit: Mapped[bool] = mapped_column(Boolean)
    send_state: Mapped[str] = mapped_column(String(16), default="sending")
    text: Mapped[str] = mapped_column(Text)
    reply_markup: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    telegram_message_id: Mapped[int | None]
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime]  # reservation time, used by the sliding window
    sent_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


# --------------------------------------------------------------------------- conversation


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    direction: Mapped[str] = mapped_column(String(8))  # in / out
    kind: Mapped[str] = mapped_column(String(16))  # text / voice / callback / command
    telegram_message_id: Mapped[int | None]
    update_id: Mapped[int | None]
    text: Mapped[str | None] = mapped_column(Text)
    transcript: Mapped[str | None] = mapped_column(Text)
    deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(index=True)


class ConversationState(Base):
    """aiogram FSM storage + any pending question the bot awaits an answer to."""

    __tablename__ = "conversation_states"

    key: Mapped[str] = mapped_column(String(200), primary_key=True)
    state: Mapped[str | None] = mapped_column(String(200))
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime]


# --------------------------------------------------------------------------- behaviour


class BehaviorObservation(Base):
    __tablename__ = "behavior_observations"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(64))  # e.g. start_delay_minutes, estimate_ratio
    task_id: Mapped[int | None] = mapped_column(ForeignKey("tasks.id", ondelete="SET NULL"))
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("work_sessions.id", ondelete="SET NULL")
    )
    value: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    time_source: Mapped[str] = mapped_column(String(16), default="exact")
    observed_at: Mapped[datetime] = mapped_column(index=True)


class Intervention(Base):
    __tablename__ = "interventions"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int | None] = mapped_column(ForeignKey("tasks.id", ondelete="SET NULL"))
    reminder_id: Mapped[int | None] = mapped_column(
        ForeignKey("reminder_schedule.id", ondelete="SET NULL")
    )
    kind: Mapped[str] = mapped_column(String(32))
    reason_category: Mapped[str | None] = mapped_column(String(32))  # ObstacleReason
    reason_text: Mapped[str | None] = mapped_column(Text)
    reason_origin: Mapped[str | None] = mapped_column(String(16))  # user_said / ai_inferred
    depth: Mapped[str] = mapped_column(String(16), default="quick")  # quick / deep
    suggestion: Mapped[str | None] = mapped_column(Text)
    used_ai: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime]


class InterventionOutcome(Base):
    __tablename__ = "intervention_outcomes"

    id: Mapped[int] = mapped_column(primary_key=True)
    intervention_id: Mapped[int] = mapped_column(ForeignKey("interventions.id", ondelete="CASCADE"))
    result: Mapped[str] = mapped_column(String(16))
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("work_sessions.id", ondelete="SET NULL")
    )
    time_source: Mapped[str] = mapped_column(String(16), default="exact")
    note: Mapped[str | None] = mapped_column(Text)
    recorded_at: Mapped[datetime]


# --------------------------------------------------------------------------- memory


class MemoryItem(Base):
    __tablename__ = "memory_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(16))
    origin: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="active")
    content: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    # how to check a hypothesis (FR-11 direction 6)
    verification_hint: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]


class MemorySource(Base):
    """Links a memory item to the primary evidence it was derived from."""

    __tablename__ = "memory_sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    memory_item_id: Mapped[int] = mapped_column(
        ForeignKey("memory_items.id", ondelete="CASCADE"), index=True
    )
    source_type: Mapped[str] = mapped_column(String(32))  # message/task_event/session/...
    source_id: Mapped[int]
    created_at: Mapped[datetime]


class MemorySummary(Base):
    __tablename__ = "memory_summaries"

    id: Mapped[int] = mapped_column(primary_key=True)
    scope: Mapped[str] = mapped_column(String(32))  # profile / patterns / recent
    content: Mapped[str] = mapped_column(Text)
    valid: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime]


class MemorySummaryItem(Base):
    """Which memory items a summary was built from (for cascade invalidation)."""

    __tablename__ = "memory_summary_items"

    summary_id: Mapped[int] = mapped_column(
        ForeignKey("memory_summaries.id", ondelete="CASCADE"), primary_key=True
    )
    memory_item_id: Mapped[int] = mapped_column(
        ForeignKey("memory_items.id", ondelete="CASCADE"), primary_key=True
    )


# --------------------------------------------------------------------------- calendar


class CalendarConnection(Base):
    __tablename__ = "calendar_connections"

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(16), default="google")
    status: Mapped[str] = mapped_column(String(16), default="active")  # active/disabled/error
    encrypted_token: Mapped[str | None] = mapped_column(Text)
    scopes: Mapped[list[Any]] = mapped_column(JSON, default=list)
    read_calendar_ids: Mapped[list[Any]] = mapped_column(JSON, default=list)
    write_calendar_id: Mapped[str | None] = mapped_column(String(300))
    last_sync_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]


class CalendarEventCache(Base):
    __tablename__ = "calendar_event_cache"
    __table_args__ = (UniqueConstraint("calendar_id", "external_id", name="uq_cal_event"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    connection_id: Mapped[int] = mapped_column(
        ForeignKey("calendar_connections.id", ondelete="CASCADE")
    )
    calendar_id: Mapped[str] = mapped_column(String(300))
    external_id: Mapped[str] = mapped_column(String(300))
    start_at: Mapped[datetime] = mapped_column(index=True)
    end_at: Mapped[datetime]
    all_day: Mapped[bool] = mapped_column(Boolean, default=False)
    busy: Mapped[bool] = mapped_column(Boolean, default=True)
    summary: Mapped[str | None] = mapped_column(Text)
    synced_at: Mapped[datetime]


class CalendarEventLink(Base):
    """Events this bot created (only after explicit approval)."""

    __tablename__ = "calendar_event_links"

    id: Mapped[int] = mapped_column(primary_key=True)
    operation_key: Mapped[str] = mapped_column(String(200), unique=True)
    approval_id: Mapped[int | None] = mapped_column(
        ForeignKey("pending_approvals.id", ondelete="SET NULL")
    )
    plan_item_id: Mapped[int | None] = mapped_column(
        ForeignKey("plan_items.id", ondelete="SET NULL")
    )
    task_id: Mapped[int | None] = mapped_column(ForeignKey("tasks.id", ondelete="SET NULL"))
    calendar_id: Mapped[str] = mapped_column(String(300))
    external_event_id: Mapped[str | None] = mapped_column(String(300))
    state: Mapped[str] = mapped_column(String(16), default="creating")  # creating/created/failed
    start_at: Mapped[datetime]
    end_at: Mapped[datetime]
    created_at: Mapped[datetime]


# --------------------------------------------------------------------------- approvals


class PendingApproval(Base):
    __tablename__ = "pending_approvals"

    id: Mapped[int] = mapped_column(primary_key=True)
    op_key: Mapped[str] = mapped_column(String(200), unique=True)
    action_type: Mapped[str] = mapped_column(String(64))
    action_payload: Mapped[dict[str, Any]] = mapped_column(JSON)  # typed, validated on apply
    diff: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # human-readable preview
    status: Mapped[str] = mapped_column(String(16), default="pending")
    origin: Mapped[str] = mapped_column(String(16))  # Actor that proposed it
    group_key: Mapped[str | None] = mapped_column(String(200), index=True)  # batch of proposals
    ai_call_id: Mapped[int | None] = mapped_column(ForeignKey("ai_calls.id", ondelete="SET NULL"))
    telegram_message_id: Mapped[int | None]
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime]
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    resolved_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class OperationKey(Base):
    """Idempotency keys for button presses and other one-shot operations."""

    __tablename__ = "operation_keys"

    key: Mapped[str] = mapped_column(String(200), primary_key=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime]


# --------------------------------------------------------------------------- AI


class AICredential(Base):
    __tablename__ = "ai_credentials"

    id: Mapped[int] = mapped_column(primary_key=True)
    label: Mapped[str] = mapped_column(String(64), unique=True)
    provider: Mapped[str] = mapped_column(String(32))  # openai/openai_compatible/anthropic/gemini
    base_url: Mapped[str | None] = mapped_column(String(500))
    encrypted_key: Mapped[str] = mapped_column(Text)
    key_last4: Mapped[str] = mapped_column(String(8))
    status: Mapped[str] = mapped_column(String(16), default="active")
    cooldown_until: Mapped[datetime | None] = mapped_column(UTCDateTime)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)
    spend_limit_usd: Mapped[float | None] = mapped_column(Float)
    # e.g. {"memory": true, "history": true}; empty = no restriction (default)
    allowed_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    source: Mapped[str] = mapped_column(String(8), default="env")  # env / chat
    last_error: Mapped[str | None] = mapped_column(Text)
    disabled_notified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]


class AIRoutingConfig(Base):
    """Versions of the gateway config that were in effect (audit)."""

    __tablename__ = "ai_routing_config"

    id: Mapped[int] = mapped_column(primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime]


class AICall(Base):
    __tablename__ = "ai_calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    task: Mapped[str] = mapped_column(String(64), index=True)
    initiator: Mapped[str] = mapped_column(String(64))  # service / handler name
    correlation_id: Mapped[str | None] = mapped_column(String(64))
    tier: Mapped[str] = mapped_column(String(16))
    escalation_reason: Mapped[str | None] = mapped_column(String(200))
    config_sha256: Mapped[str | None] = mapped_column(String(64))
    input_categories: Mapped[list[Any]] = mapped_column(JSON, default=list)
    provider: Mapped[str | None] = mapped_column(String(32))
    model: Mapped[str | None] = mapped_column(String(128))
    credential_label: Mapped[str | None] = mapped_column(String(64))
    outcome: Mapped[str] = mapped_column(String(16))
    response_type: Mapped[str | None] = mapped_column(String(64))
    validation_error: Mapped[str | None] = mapped_column(Text)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    cost_estimated: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    attempts_count: Mapped[int] = mapped_column(Integer, default=0)
    proposed_actions: Mapped[list[Any]] = mapped_column(JSON, default=list)
    actions_result: Mapped[list[Any]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(index=True)


class AICallAttempt(Base):
    __tablename__ = "ai_call_attempts"

    id: Mapped[int] = mapped_column(primary_key=True)
    call_id: Mapped[int] = mapped_column(ForeignKey("ai_calls.id", ondelete="CASCADE"), index=True)
    attempt_no: Mapped[int]
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(128))
    credential_label: Mapped[str] = mapped_column(String(64))
    model_alias: Mapped[str] = mapped_column(String(64))
    error_kind: Mapped[str | None] = mapped_column(String(32))
    http_status: Mapped[int | None]
    repair: Mapped[bool] = mapped_column(Boolean, default=False)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    cost_estimated: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(index=True)


# --------------------------------------------------------------------------- infrastructure


class ProcessedUpdate(Base):
    __tablename__ = "processed_updates"

    update_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    processed_at: Mapped[datetime]


class JobLease(Base):
    __tablename__ = "job_leases"

    name: Mapped[str] = mapped_column(String(200), primary_key=True)
    owner: Mapped[str] = mapped_column(String(100))
    acquired_at: Mapped[datetime]
    expires_at: Mapped[datetime]


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    actor: Mapped[str] = mapped_column(String(16))
    action: Mapped[str] = mapped_column(String(64))
    entity: Mapped[str | None] = mapped_column(String(64))
    entity_id: Mapped[str | None] = mapped_column(String(64))
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(index=True)


class BackupManifest(Base):
    __tablename__ = "backup_manifest"

    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(String(500))
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int]
    encrypted: Mapped[bool] = mapped_column(Boolean, default=True)
    status: Mapped[str] = mapped_column(String(16), default="created")  # created/verified/failed
    schema_revision: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime]
    verified_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
