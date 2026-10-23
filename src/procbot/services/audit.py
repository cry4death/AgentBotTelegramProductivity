"""Append-only audit log helper."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from procbot.db.models import AuditLog
from procbot.domain.enums import Actor


async def audit(
    session: AsyncSession,
    *,
    now: datetime,
    actor: Actor | str,
    action: str,
    entity: str | None = None,
    entity_id: object | None = None,
    details: dict[str, Any] | None = None,
) -> AuditLog:
    row = AuditLog(
        actor=str(actor),
        action=action,
        entity=entity,
        entity_id=None if entity_id is None else str(entity_id),
        details=details or {},
        created_at=now,
    )
    session.add(row)
    await session.flush()
    return row
