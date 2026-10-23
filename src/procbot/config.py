"""Configuration.

Two layers:
- EnvSettings: secrets and deployment paths from the environment / .env;
- AppConfig: behaviour knobs from config/app.yaml (optional; defaults below).

Values the spec leaves open (section 10) are NOT invented here. They are either
None ("not set": the related initiative is disabled or the bot asks the user), or
a conservative default explicitly marked `UNCONFIRMED` in docs/open-questions.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from procbot.domain.enums import MessageCategory


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    telegram_bot_token: str = ""
    owner_telegram_id: int = 0
    master_key: str = ""  # Fernet key; `procbot gen-key` prints one
    database_path: Path = Path("data/procbot.db")
    app_config_path: Path = Path("config/app.yaml")
    ai_config_path: Path = Path("config/ai.yaml")
    backup_dir: Path = Path("backups")
    lock_path: Path = Path("data/procbot.lock")
    log_level: str = "INFO"
    # Google OAuth client (stage C); empty = calendar integration unavailable.
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = ""
    # Mini App / healthcheck HTTP server (stage D).
    http_host: str = "0.0.0.0"
    http_port: int = 8080
    miniapp_public_url: str = ""


class CategoryPolicy(BaseModel):
    # Open question 3: whether service reports count toward the strict 4/60 limit.
    # None = not decided; the gate treats None as True (the conservative choice)
    # and the README lists it as unconfirmed. Reminder categories always count.
    counts_toward_limit: bool | None = None
    enabled: bool = True


class GateConfig(BaseModel):
    # Hard product rule (FR-07): at most 4 reminders in any sliding 60 minutes.
    max_reminders: int = Field(default=4, ge=0, le=4)
    window_minutes: int = Field(default=60, ge=60)
    categories: dict[MessageCategory, CategoryPolicy] = Field(default_factory=dict)
    # Timer signals were set up by the user for an active session.
    timer_signals_respect_quiet_hours: bool = False
    # When a deferred reminder is older than this, it is folded into a digest.
    default_stale_after_minutes: int = 120
    # Busy calendar blocks initiative messages (FR-07) except timer signals.
    respect_calendar_busy: bool = True

    def policy(self, category: MessageCategory) -> CategoryPolicy:
        return self.categories.get(category, CategoryPolicy())


class SchedulerConfig(BaseModel):
    tick_seconds: float = 20.0
    lease_seconds: int = 120
    batch_size: int = 20


class ReminderConfig(BaseModel):
    # Open question 3: initial adaptive intervals are not decided. These are
    # technical defaults, editable in app.yaml, and listed as unconfirmed.
    pre_task_lead_minutes: int | None = None  # None = no pre-task warning until set
    start_delay_after_minutes: int | None = None  # None = no delay check until set
    max_start_delay_followups: int | None = None


class ApprovalConfig(BaseModel):
    ttl_hours: float = 24.0


class AISwitches(BaseModel):
    """Resource-heavy features can be switched off (spec section 7)."""

    weekly_analysis: bool = True
    memory_rebuild: bool = True
    voice: bool = True


class BackupConfig(BaseModel):
    # Open question 8 / 10: periodicity and retention not decided.
    interval_hours: float | None = 24.0  # UNCONFIRMED technical default
    retention_days: int | None = None  # None = never auto-delete
    encrypt: bool = True


class CalendarConfig(BaseModel):
    """Google Calendar (stage C, services.calendar). Open question 6: which calendars
    to read/write is chosen by the owner in /calendar; the OAuth scopes are here."""

    # Minimal default (UNCONFIRMED, see docs/deploy.md "Google Calendar"): read the
    # calendar list + events of every calendar, write (insert) only to owned ones.
    scopes: list[str] = Field(
        default_factory=lambda: [
            "https://www.googleapis.com/auth/calendar.calendarlist.readonly",
            "https://www.googleapis.com/auth/calendar.events.readonly",
            "https://www.googleapis.com/auth/calendar.events.owned",
        ]
    )
    sync_interval_minutes: float = Field(default=15.0, gt=0)  # UNCONFIRMED
    stale_after_minutes: float = Field(default=60.0, gt=0)  # UNCONFIRMED
    sync_past_days: int = Field(default=1, ge=0, le=31)
    sync_future_days: int = Field(default=14, ge=1, le=90)
    oauth_state_ttl_minutes: int = Field(default=10, ge=1, le=60)
    max_block_hours: float = Field(default=12.0, gt=0, le=24)


class AppConfig(BaseModel):
    gate: GateConfig = Field(default_factory=GateConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    reminders: ReminderConfig = Field(default_factory=ReminderConfig)
    approvals: ApprovalConfig = Field(default_factory=ApprovalConfig)
    ai_features: AISwitches = Field(default_factory=AISwitches)
    backup: BackupConfig = Field(default_factory=BackupConfig)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    # Open question 4: deep-dialog threshold (repeated postponements of one task).
    deep_dialog_after_postpones: int | None = None
    # Planner: share of free time left unplanned for breaks / surprises.
    planner_free_buffer_ratio: float = Field(default=0.25, ge=0.0, le=0.9)

    @field_validator("gate", mode="before")
    @classmethod
    def _gate_none(cls, v: Any) -> Any:
        return {} if v is None else v


def load_app_config(path: Path | None) -> AppConfig:
    if path is None or not path.exists():
        return AppConfig()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(data)
