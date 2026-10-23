"""Domain enumerations. Stored in the database as their string values."""

from __future__ import annotations

from enum import StrEnum


class Actor(StrEnum):
    USER = "user"
    SYSTEM = "system"
    AI = "ai"


class TaskStatus(StrEnum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"  # work confirmed at least once, not finished
    DONE = "done"
    SOMEDAY = "someday"  # consciously deferred for a long time
    CANCELLED = "cancelled"  # dropped by user decision


class EstimateSource(StrEnum):
    USER = "user"
    AI = "ai"
    HISTORY = "history"


class EntityStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    DONE = "done"
    ARCHIVED = "archived"


class TaskEventType(StrEnum):
    CREATED = "created"
    UPDATED = "updated"
    RESCHEDULED = "rescheduled"  # planned start / due moved
    POSTPONED = "postponed"  # user postponed without new date
    STARTED = "started"
    PROGRESSED = "progressed"
    COMPLETED = "completed"
    REOPENED = "reopened"
    CANCELLED = "cancelled"
    DEFERRED = "deferred"  # moved to someday
    REORDERED = "reordered"
    SUBTASKS_CHANGED = "subtasks_changed"
    OBSTACLE_REPORTED = "obstacle_reported"
    DELETED = "deleted"


class SessionMode(StrEnum):
    FREE = "free"
    POMODORO = "pomodoro"
    SHORT_START = "short_start"


class SessionStatus(StrEnum):
    UNCONFIRMED = "unconfirmed"  # planned/proposed, user did not confirm the start
    ACTIVE = "active"
    PAUSED = "paused"
    FINISHED = "finished"
    CANCELLED = "cancelled"


class TimeSource(StrEnum):
    EXACT = "exact"  # recorded live with buttons/timer
    RETRO = "retro"  # reported afterwards with explicit times
    APPROXIMATE = "approximate"  # reported afterwards, rough
    UNKNOWN = "unknown"


class SessionOutcome(StrEnum):
    STARTED = "started"
    PROGRESSED = "progressed"
    COMPLETED = "completed"


class MessageCategory(StrEnum):
    """Categories of bot-initiated messages. Only these six may be initiated (FR-07),
    plus timer signals the user set up and one-off system notices about failures."""

    MORNING_PLAN = "morning_plan"
    PRE_TASK = "pre_task"
    START_DELAY = "start_delay"
    REPLAN = "replan"
    EVENING_REVIEW = "evening_review"
    WEEKLY_REVIEW = "weekly_review"
    TIMER_SIGNAL = "timer_signal"  # pre-configured end of work/break signal
    SYSTEM_NOTICE = "system_notice"  # e.g. "AI key disabled", sent at most once per incident
    DIGEST = "digest"  # folded summary of stale reminders after quiet hours/downtime


INITIATIVE_CATEGORIES: frozenset[MessageCategory] = frozenset(
    {
        MessageCategory.MORNING_PLAN,
        MessageCategory.PRE_TASK,
        MessageCategory.START_DELAY,
        MessageCategory.REPLAN,
        MessageCategory.EVENING_REVIEW,
        MessageCategory.WEEKLY_REVIEW,
    }
)

# Reminder-type categories always count toward the strict 4-per-60-minutes limit.
REMINDER_CATEGORIES: frozenset[MessageCategory] = frozenset(
    {MessageCategory.PRE_TASK, MessageCategory.START_DELAY, MessageCategory.REPLAN}
)


class ReminderState(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"  # handed to outbound and sent
    RESCHEDULED = "rescheduled"
    FOLDED = "folded"  # merged into a digest
    CANCELLED = "cancelled"
    DROPPED = "dropped"  # obsolete, nothing to send


class SendState(StrEnum):
    SENDING = "sending"
    SENT = "sent"
    MAYBE_SENT = "maybe_sent"  # crashed between send and confirm
    FAILED = "failed"
    CANCELLED = "cancelled"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPLIED = "applied"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class ObstacleReason(StrEnum):
    """Hypotheses, not diagnoses (FR-04)."""

    UNCLEAR = "unclear"
    TOO_BIG = "too_big"
    UNPLEASANT = "unpleasant"
    DISTRACTIONS = "distractions"
    LOW_ENERGY = "low_energy"
    PRIORITY_CONFLICT = "priority_conflict"
    EXTERNAL = "external"
    OTHER = "other"


class InterventionKind(StrEnum):
    TINY_STEP = "tiny_step"
    SHORT_START = "short_start"
    SPLIT = "split"
    SIMPLIFY = "simplify"
    RESCHEDULE = "reschedule"
    DEFER_LONG = "defer_long"
    CHANGE_APPROACH = "change_approach"
    DROP = "drop"
    REST = "rest"
    DEEP_DIALOG = "deep_dialog"
    REMOVE_DISTRACTIONS = "remove_distractions"


class InterventionResult(StrEnum):
    STARTED = "started"
    PROGRESSED = "progressed"
    COMPLETED = "completed"
    NO_EFFECT = "no_effect"
    DECLINED = "declined"
    UNKNOWN = "unknown"


class MemoryKind(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    PATTERN = "pattern"
    HYPOTHESIS = "hypothesis"


class MemoryOrigin(StrEnum):
    USER_SAID = "user_said"
    SYSTEM_MEASURED = "system_measured"
    AI_INFERRED = "ai_inferred"
    USER_CONFIRMED = "user_confirmed"
    USER_REFUTED = "user_refuted"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    REFUTED = "refuted"
    DELETED = "deleted"


class PlanStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REVISED = "revised"
    CLOSED = "closed"
    REJECTED = "rejected"


class PlanItemKind(StrEnum):
    TASK = "task"
    HABIT = "habit"
    BREAK = "break"
    BUFFER = "buffer"
    EVENT = "event"  # calendar event shown for context (read-only)


class HabitOccurrenceStatus(StrEnum):
    DONE = "done"
    PARTIAL = "partial"
    SKIPPED = "skipped"  # consciously skipped
    MISSED = "missed"


class AIOutcome(StrEnum):
    OK = "ok"
    INVALID = "invalid"
    UNAVAILABLE = "unavailable"
    REFUSED = "refused"


class CredentialStatus(StrEnum):
    ACTIVE = "active"
    COOLING_DOWN = "cooling_down"
    DISABLED = "disabled"


class AIErrorKind(StrEnum):
    TIMEOUT = "timeout"
    NETWORK = "network"
    SERVER = "server"  # 5xx
    RATE_LIMIT = "rate_limit"  # 429 / quota
    AUTH = "auth"  # 401/403
    BAD_REQUEST = "bad_request"  # 400, incl. context too long
    CONTEXT_TOO_LONG = "context_too_long"
    REFUSAL = "refusal"
    INVALID_OUTPUT = "invalid_output"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"
