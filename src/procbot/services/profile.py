"""Owner profile and settings (FR-01). Single-row table, id=1."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.db.models import OwnerProfile, Setting
from procbot.domain.enums import Actor
from procbot.services.audit import audit
from procbot.timeutil import LocalWindow, windows_from_raw

PROFILE_ID = 1

# Fields the user may edit through settings; anything else is internal.
EDITABLE_FIELDS: frozenset[str] = frozenset(
    {
        "timezone",
        "morning_time",
        "evening_time",
        "weekly_review_dow",
        "weekly_review_time",
        "quiet_hours",
        "routine",
        "pause_until",
        "notifications_enabled",
        "style_preset",
        "strictness",
        "humor",
        "profanity_allowed",
        "verbosity",
        "pomodoro_work_min",
        "pomodoro_break_min",
        "short_start_min",
        "pre_task_lead_min",
        "start_delay_min",
        "memory_settings",
        "autonomy_settings",
        "goals_note",
        "onboarding_step",
        "chat_id",
    }
)


async def get_profile(session: AsyncSession) -> OwnerProfile | None:
    return await session.get(OwnerProfile, PROFILE_ID)


async def get_or_create_profile(
    session: AsyncSession, *, telegram_user_id: int, now: datetime, chat_id: int | None = None
) -> tuple[OwnerProfile, bool]:
    profile = await get_profile(session)
    if profile is not None:
        if chat_id is not None and profile.chat_id != chat_id:
            profile.chat_id = chat_id
            profile.updated_at = now
        return profile, False
    profile = OwnerProfile(
        id=PROFILE_ID,
        telegram_user_id=telegram_user_id,
        chat_id=chat_id,
        onboarding_step="timezone",
        quiet_hours=[],
        routine={},
        memory_settings={},
        autonomy_settings={},
        created_at=now,
        updated_at=now,
    )
    session.add(profile)
    await session.flush()
    await audit(session, now=now, actor=Actor.USER, action="profile.created", entity="profile")
    return profile, True


async def update_profile(
    session: AsyncSession, *, now: datetime, actor: Actor = Actor.USER, **changes: Any
) -> OwnerProfile:
    """Change profile fields. Personality changes never touch tasks or memory (FR-08)."""
    unknown = set(changes) - EDITABLE_FIELDS
    if unknown:
        raise ValueError(f"not editable: {sorted(unknown)}")
    profile = await get_profile(session)
    if profile is None:
        raise LookupError("profile does not exist yet")
    before: dict[str, Any] = {}
    for key, value in changes.items():
        before[key] = _jsonable(getattr(profile, key))
        setattr(profile, key, value)
    profile.updated_at = now
    await audit(
        session,
        now=now,
        actor=actor,
        action="profile.updated",
        entity="profile",
        details={"before": before, "after": {k: _jsonable(v) for k, v in changes.items()}},
    )
    return profile


def quiet_windows(profile: OwnerProfile | None) -> list[LocalWindow]:
    return windows_from_raw(profile.quiet_hours if profile else None)


async def get_setting(session: AsyncSession, key: str, default: Any = None) -> Any:
    row = await session.get(Setting, key)
    return default if row is None else row.value


async def set_setting(session: AsyncSession, key: str, value: Any, *, now: datetime) -> None:
    row = await session.get(Setting, key)
    if row is None:
        session.add(Setting(key=key, value=value, updated_at=now))
    else:
        row.value = value
        row.updated_at = now
    await session.flush()


async def all_settings(session: AsyncSession) -> dict[str, Any]:
    rows = (await session.execute(select(Setting))).scalars().all()
    return {r.key: r.value for r in rows}


def _jsonable(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.isoformat()
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v
