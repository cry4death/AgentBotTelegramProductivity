"""All inline-button payloads in one place (one prefix per module).

Telegram limits callback_data to 64 bytes, so payloads carry ids, not text.
Owner module in brackets: only that module's router handles the prefix.
"""

from __future__ import annotations

from aiogram.filters.callback_data import CallbackData


class ApprovalCb(CallbackData, prefix="ap"):  # [handlers.approvals]
    action: str  # accept / reject / edit / accept_all / reject_all / view
    approval_id: int = 0
    group: str = ""  # short group key for batch cards


class TaskCb(CallbackData, prefix="tk"):  # [handlers.tasks]
    action: str  # view / done / partial / postpone / backlog / plan / delete / subtasks / list
    task_id: int = 0
    arg: str = ""


class DayCb(CallbackData, prefix="dy"):  # [handlers.day]
    action: str  # show / approve / revise / reject / replan
    plan_id: int = 0
    revision_id: int = 0


class SessionCb(CallbackData, prefix="ws"):  # [handlers.sessions]
    action: str  # pick / start / pause / resume / stop / outcome / retro / mode
    session_id: int = 0
    task_id: int = 0
    arg: str = ""


class ReminderCb(CallbackData, prefix="rm"):  # [handlers.reminders]
    action: str  # starting / later / reason / snooze / skip
    reminder_id: int = 0
    task_id: int = 0
    arg: str = ""


class SettingsCb(CallbackData, prefix="st"):  # [handlers.settings]
    action: str
    arg: str = ""


class OnboardingCb(CallbackData, prefix="ob"):  # [handlers.start]
    step: str
    arg: str = ""


class InterventionCb(CallbackData, prefix="iv"):  # [handlers.procrastination]
    action: str  # accept / other / deep / outcome
    intervention_id: int = 0
    task_id: int = 0
    arg: str = ""


class HabitCb(CallbackData, prefix="hb"):  # [handlers.habits]
    action: str
    habit_id: int = 0
    arg: str = ""


class GoalCb(CallbackData, prefix="gl"):  # [handlers.goals]
    action: str
    goal_id: int = 0
    arg: str = ""


class MemoryCb(CallbackData, prefix="mm"):  # [handlers.memory]
    action: str  # list / view / confirm / refute / edit / delete / wipe / export
    item_id: int = 0
    arg: str = ""


class ReportCb(CallbackData, prefix="rp"):  # [handlers.reports]
    action: str
    arg: str = ""


class AIKeyCb(CallbackData, prefix="ak"):  # [handlers.ai_keys]
    action: str
    credential_id: int = 0
    arg: str = ""


class CalendarCb(CallbackData, prefix="cl"):  # [handlers.calendar]
    action: str
    arg: str = ""
