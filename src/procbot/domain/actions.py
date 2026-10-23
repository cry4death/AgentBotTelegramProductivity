"""Typed actions and the autonomy permission matrix (FR-15).

The AI never changes data directly. It returns `ProposedAction` objects; the
ActionExecutor (services.actions) looks up the permission for the action type and
either applies it, turns it into a pending approval, or drops it as forbidden.

An action that the user triggers explicitly (a button or command for that exact
change) is itself the consent, so `origin=USER` actions are applied directly.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class Permission(StrEnum):
    AUTO = "auto"  # allowed without separate confirmation (within constraints)
    NEEDS_APPROVAL = "needs_approval"
    FORBIDDEN = "forbidden"


class ActionType(StrEnum):
    CREATE_TASK = "create_task"
    CREATE_SUBTASKS = "create_subtasks"
    REPLACE_SUBTASKS = "replace_subtasks"
    REORDER_TASKS = "reorder_tasks"
    ADJUST_REMINDERS = "adjust_reminders"
    CHOOSE_INTERVENTION = "choose_intervention"
    RESCHEDULE_TASK = "reschedule_task"
    CHANGE_DEADLINE = "change_deadline"
    CHANGE_TASK_STATUS = "change_task_status"
    UPDATE_TASK = "update_task"
    CREATE_CALENDAR_EVENT = "create_calendar_event"
    UPDATE_CALENDAR_EVENT = "update_calendar_event"
    DELETE_CALENDAR_EVENT = "delete_calendar_event"
    CREATE_HABIT = "create_habit"
    CHANGE_HABIT = "change_habit"
    CREATE_GOAL = "create_goal"
    CHANGE_GOAL = "change_goal"
    CHANGE_MILESTONE = "change_milestone"
    APPLY_DAY_PLAN = "apply_day_plan"
    FILL_FREE_SLOTS = "fill_free_slots"
    SAVE_HYPOTHESIS = "save_hypothesis"
    CHANGE_PROFILE_SETTING = "change_profile_setting"


PERMISSIONS: dict[ActionType, Permission] = {
    # Allowed without confirmation (FR-15 table, "Да" rows)
    ActionType.REORDER_TASKS: Permission.AUTO,
    ActionType.ADJUST_REMINDERS: Permission.AUTO,
    ActionType.CREATE_SUBTASKS: Permission.AUTO,
    ActionType.REPLACE_SUBTASKS: Permission.AUTO,
    ActionType.CHOOSE_INTERVENTION: Permission.AUTO,
    # Hypotheses are stored as ai_inferred, never as facts; user can refute.
    ActionType.SAVE_HYPOTHESIS: Permission.AUTO,
    # Needs consent ("Нет" rows)
    ActionType.CREATE_TASK: Permission.NEEDS_APPROVAL,
    ActionType.RESCHEDULE_TASK: Permission.NEEDS_APPROVAL,
    ActionType.CHANGE_DEADLINE: Permission.NEEDS_APPROVAL,
    ActionType.CHANGE_TASK_STATUS: Permission.NEEDS_APPROVAL,
    ActionType.UPDATE_TASK: Permission.NEEDS_APPROVAL,
    ActionType.CREATE_CALENDAR_EVENT: Permission.NEEDS_APPROVAL,
    ActionType.CREATE_HABIT: Permission.NEEDS_APPROVAL,
    ActionType.CHANGE_HABIT: Permission.NEEDS_APPROVAL,
    ActionType.CREATE_GOAL: Permission.NEEDS_APPROVAL,
    ActionType.CHANGE_GOAL: Permission.NEEDS_APPROVAL,
    ActionType.CHANGE_MILESTONE: Permission.NEEDS_APPROVAL,
    ActionType.APPLY_DAY_PLAN: Permission.NEEDS_APPROVAL,
    ActionType.FILL_FREE_SLOTS: Permission.NEEDS_APPROVAL,
    ActionType.CHANGE_PROFILE_SETTING: Permission.NEEDS_APPROVAL,
    # Never automatically
    ActionType.UPDATE_CALENDAR_EVENT: Permission.FORBIDDEN,
    ActionType.DELETE_CALENDAR_EVENT: Permission.FORBIDDEN,
}

# Actions the bot must never perform even when the user asks via the AI flow;
# external calendar events are not ours to change.
ALWAYS_FORBIDDEN: frozenset[ActionType] = frozenset(
    {ActionType.UPDATE_CALENDAR_EVENT, ActionType.DELETE_CALENDAR_EVENT}
)


def permission_for(action_type: ActionType) -> Permission:
    # Unknown/unlisted actions are treated as needing approval (fail safe).
    return PERMISSIONS.get(action_type, Permission.NEEDS_APPROVAL)


class _Action(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskDraft(_Action):
    title: str = Field(min_length=1, max_length=500)
    description: str | None = None
    due_at: datetime | None = None
    due_date: date | None = None
    due_is_hard: bool = False
    planned_start_at: datetime | None = None
    planned_date: date | None = None
    estimate_minutes: int | None = Field(default=None, ge=1, le=24 * 60)
    priority: int | None = None
    importance: int | None = None
    done_criteria: str | None = None
    verification: Literal["button", "criteria"] = "button"
    is_backlog: bool = False
    goal_id: int | None = None
    project_id: int | None = None
    milestone_id: int | None = None
    category: str | None = None
    context: str | None = None
    recurrence: dict[str, object] | None = None
    parent_id: int | None = None


class CreateTask(_Action):
    type: Literal[ActionType.CREATE_TASK] = ActionType.CREATE_TASK
    task: TaskDraft


class SubtaskDraft(_Action):
    title: str = Field(min_length=1, max_length=500)
    estimate_minutes: int | None = Field(default=None, ge=1, le=24 * 60)
    done_criteria: str | None = None


class CreateSubtasks(_Action):
    type: Literal[ActionType.CREATE_SUBTASKS] = ActionType.CREATE_SUBTASKS
    parent_id: int
    subtasks: list[SubtaskDraft] = Field(min_length=1, max_length=30)


class ReplaceSubtasks(_Action):
    """Rebuild open subtasks of a parent (finer or coarser steps). Done subtasks stay."""

    type: Literal[ActionType.REPLACE_SUBTASKS] = ActionType.REPLACE_SUBTASKS
    parent_id: int
    subtasks: list[SubtaskDraft] = Field(min_length=1, max_length=30)


class ReorderTasks(_Action):
    type: Literal[ActionType.REORDER_TASKS] = ActionType.REORDER_TASKS
    task_ids: list[int] = Field(min_length=1)


class AdjustReminders(_Action):
    type: Literal[ActionType.ADJUST_REMINDERS] = ActionType.ADJUST_REMINDERS
    task_id: int
    pre_task_lead_min: int | None = Field(default=None, ge=0, le=240)
    start_delay_min: int | None = Field(default=None, ge=1, le=240)


class ChooseIntervention(_Action):
    type: Literal[ActionType.CHOOSE_INTERVENTION] = ActionType.CHOOSE_INTERVENTION
    task_id: int | None = None
    kind: str
    suggestion: str


class RescheduleTask(_Action):
    type: Literal[ActionType.RESCHEDULE_TASK] = ActionType.RESCHEDULE_TASK
    task_id: int
    planned_start_at: datetime | None = None
    planned_date: date | None = None
    to_backlog: bool = False
    reason: str | None = None


class ChangeDeadline(_Action):
    type: Literal[ActionType.CHANGE_DEADLINE] = ActionType.CHANGE_DEADLINE
    task_id: int
    due_at: datetime | None = None
    due_date: date | None = None
    reason: str | None = None


class ChangeTaskStatus(_Action):
    type: Literal[ActionType.CHANGE_TASK_STATUS] = ActionType.CHANGE_TASK_STATUS
    task_id: int
    status: Literal["someday", "cancelled", "todo"]
    reason: str | None = None


class UpdateTask(_Action):
    """Change scope-defining fields (title, criteria, estimate...). Needs consent."""

    type: Literal[ActionType.UPDATE_TASK] = ActionType.UPDATE_TASK
    task_id: int
    title: str | None = None
    description: str | None = None
    done_criteria: str | None = None
    estimate_minutes: int | None = None
    importance: int | None = None
    priority: int | None = None
    reason: str | None = None


class CreateCalendarEvent(_Action):
    type: Literal[ActionType.CREATE_CALENDAR_EVENT] = ActionType.CREATE_CALENDAR_EVENT
    title: str
    start_at: datetime
    end_at: datetime
    task_id: int | None = None
    plan_item_id: int | None = None


class UpdateCalendarEvent(_Action):
    type: Literal[ActionType.UPDATE_CALENDAR_EVENT] = ActionType.UPDATE_CALENDAR_EVENT
    external_event_id: str


class DeleteCalendarEvent(_Action):
    type: Literal[ActionType.DELETE_CALENDAR_EVENT] = ActionType.DELETE_CALENDAR_EVENT
    external_event_id: str


class CreateHabit(_Action):
    type: Literal[ActionType.CREATE_HABIT] = ActionType.CREATE_HABIT
    title: str
    schedule: dict[str, object]
    target: dict[str, object] = Field(default_factory=dict)
    preferred_time: str | None = None  # HH:MM local
    goal_id: int | None = None


class ChangeHabit(_Action):
    type: Literal[ActionType.CHANGE_HABIT] = ActionType.CHANGE_HABIT
    habit_id: int
    schedule: dict[str, object] | None = None
    target: dict[str, object] | None = None
    preferred_time: str | None = None
    reason: str | None = None


class CreateGoal(_Action):
    type: Literal[ActionType.CREATE_GOAL] = ActionType.CREATE_GOAL
    title: str
    measurable_outcome: str | None = None
    success_criteria: str | None = None
    horizon_date: date | None = None
    milestones: list[str] = Field(default_factory=list)


class ChangeGoal(_Action):
    type: Literal[ActionType.CHANGE_GOAL] = ActionType.CHANGE_GOAL
    goal_id: int
    title: str | None = None
    measurable_outcome: str | None = None
    success_criteria: str | None = None
    horizon_date: date | None = None
    status: str | None = None
    reason: str | None = None


class ChangeMilestone(_Action):
    type: Literal[ActionType.CHANGE_MILESTONE] = ActionType.CHANGE_MILESTONE
    milestone_id: int
    title: str | None = None
    due_date: date | None = None
    status: str | None = None
    reason: str | None = None


class ApplyDayPlan(_Action):
    type: Literal[ActionType.APPLY_DAY_PLAN] = ActionType.APPLY_DAY_PLAN
    plan_id: int
    revision_id: int


class FillFreeSlots(_Action):
    type: Literal[ActionType.FILL_FREE_SLOTS] = ActionType.FILL_FREE_SLOTS
    slots: list[dict[str, object]]


class SaveHypothesis(_Action):
    type: Literal[ActionType.SAVE_HYPOTHESIS] = ActionType.SAVE_HYPOTHESIS
    content: str
    verification_hint: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class ChangeProfileSetting(_Action):
    type: Literal[ActionType.CHANGE_PROFILE_SETTING] = ActionType.CHANGE_PROFILE_SETTING
    key: str
    value: object
    reason: str | None = None


ProposedAction = Annotated[
    CreateTask
    | CreateSubtasks
    | ReplaceSubtasks
    | ReorderTasks
    | AdjustReminders
    | ChooseIntervention
    | RescheduleTask
    | ChangeDeadline
    | ChangeTaskStatus
    | UpdateTask
    | CreateCalendarEvent
    | UpdateCalendarEvent
    | DeleteCalendarEvent
    | CreateHabit
    | ChangeHabit
    | CreateGoal
    | ChangeGoal
    | ChangeMilestone
    | ApplyDayPlan
    | FillFreeSlots
    | SaveHypothesis
    | ChangeProfileSetting,
    Field(discriminator="type"),
]
