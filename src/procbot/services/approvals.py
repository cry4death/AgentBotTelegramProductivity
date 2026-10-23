"""ApprovalService: queue of proposed changes that need explicit consent (FR-15).

Owner of `pending_approvals`. A card stores the typed action payload, a
human-readable diff, the origin, an optional group (batch preview, applied per
item or all at once) and an expiry. Silence is never consent: a periodic job
turns overdue cards into EXPIRED and an expired card can never be applied.

Every resolution is idempotent: approving/rejecting a card that is no longer
pending (double button press) changes nothing and reports the earlier state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.models import PendingApproval
from procbot.domain.enums import Actor, ApprovalStatus
from procbot.services import actions
from procbot.services.audit import audit
from procbot.services.jobs import PeriodicJob, register_job

EXPIRE_JOB = "approvals.expire"
EXPIRE_INTERVAL_SECONDS = 300.0

STATUS_TEXT: dict[str, str] = {
    ApprovalStatus.PENDING: "ждёт подтверждения",
    ApprovalStatus.APPLIED: "применено",
    ApprovalStatus.REJECTED: "отклонено",
    ApprovalStatus.EXPIRED: "срок истёк — не применено",
    ApprovalStatus.FAILED: "не удалось применить",
}


@dataclass
class Resolution:
    approval: PendingApproval
    status: ApprovalStatus
    changed: bool  # this call changed the card's status
    message: str  # short Russian text for the user (callback answer)
    after_commit: list[actions.AfterCommit] = field(default_factory=list)


# --------------------------------------------------------------------------- queries


async def get_approval(session: AsyncSession, approval_id: int) -> PendingApproval | None:
    return await session.get(PendingApproval, approval_id)


async def get_by_op_key(session: AsyncSession, op_key: str) -> PendingApproval | None:
    return (
        await session.execute(select(PendingApproval).where(PendingApproval.op_key == op_key))
    ).scalar_one_or_none()


async def list_group(session: AsyncSession, group_key: str) -> list[PendingApproval]:
    stmt = (
        select(PendingApproval)
        .where(PendingApproval.group_key == group_key)
        .order_by(PendingApproval.id)
    )
    return list((await session.execute(stmt)).scalars())


async def group_of(session: AsyncSession, approval: PendingApproval) -> list[PendingApproval]:
    """All cards shown together with `approval` (itself when it has no group)."""
    if approval.group_key is None:
        return [approval]
    return await list_group(session, approval.group_key)


async def list_pending(session: AsyncSession) -> list[PendingApproval]:
    stmt = (
        select(PendingApproval)
        .where(PendingApproval.status == ApprovalStatus.PENDING)
        .order_by(PendingApproval.id)
    )
    return list((await session.execute(stmt)).scalars())


async def find_by_message(session: AsyncSession, message_id: int) -> list[PendingApproval]:
    stmt = (
        select(PendingApproval)
        .where(PendingApproval.telegram_message_id == message_id)
        .order_by(PendingApproval.id)
    )
    return list((await session.execute(stmt)).scalars())


# --------------------------------------------------------------------------- propose


async def propose(
    session: AsyncSession,
    ctx: AppContext,
    action: Any,
    *,
    origin: Actor,
    op_key: str,
    diff: dict[str, Any] | None = None,
    group_key: str | None = None,
    ai_call_id: int | None = None,
) -> PendingApproval:
    """Store a proposal. Idempotent by op_key (returns the existing card)."""
    existing = await get_by_op_key(session, op_key)
    if existing is not None:
        return existing
    now = ctx.clock.now()
    ttl = ctx.config.approvals.ttl_hours
    row = PendingApproval(
        op_key=op_key,
        action_type=str(action.type),
        action_payload=actions.dump_action(action),
        diff=diff or {"summary": str(action.type), "lines": []},
        status=ApprovalStatus.PENDING,
        origin=str(origin),
        group_key=group_key,
        ai_call_id=ai_call_id,
        created_at=now,
        expires_at=now + timedelta(hours=ttl) if ttl and ttl > 0 else None,
    )
    session.add(row)
    await session.flush()
    return row


async def set_card_message(
    session: AsyncSession, approval_ids: Sequence[int], message_id: int
) -> None:
    """Remember which Telegram message shows these cards (for "да" replies)."""
    for aid in approval_ids:
        row = await session.get(PendingApproval, aid)
        if row is not None:
            row.telegram_message_id = message_id


async def update_pending_payload(
    session: AsyncSession, ctx: AppContext, approval_id: int, action: Any, diff: dict[str, Any]
) -> PendingApproval:
    """The user edited a proposal before accepting it (still pending)."""
    row = await session.get(PendingApproval, approval_id)
    if row is None:
        raise LookupError(approval_id)
    if row.status != ApprovalStatus.PENDING or _is_expired(ctx, row):
        raise actions.ActionError("Карточка уже не ожидает подтверждения.")
    if str(action.type) != row.action_type:
        raise actions.ActionError("Нельзя поменять тип действия.")
    before = row.action_payload
    row.action_payload = actions.dump_action(action)
    row.diff = diff
    await audit(
        session,
        now=ctx.clock.now(),
        actor=Actor.USER,
        action="approval.edited",
        entity="approval",
        entity_id=row.id,
        details={"before": before, "after": row.action_payload},
    )
    return row


async def add_diff_notes(session: AsyncSession, approval_id: int, notes: Sequence[str]) -> None:
    """Append informational lines (e.g. "❓ Неясно: срок") to a pending card's diff.

    Display only: the action payload is untouched. Idempotent (a line already
    present is not added twice); resolved cards are left as they are."""
    row = await session.get(PendingApproval, approval_id)
    if row is None or row.status != ApprovalStatus.PENDING or not notes:
        return
    diff = dict(row.diff or {})
    lines = list(diff.get("lines") or [])
    added = [n for n in notes if n and n not in lines]
    if added:
        diff["lines"] = lines + added
        row.diff = diff


# --------------------------------------------------------------------------- resolve


def _is_expired(ctx: AppContext, row: PendingApproval) -> bool:
    return row.expires_at is not None and row.expires_at <= ctx.clock.now()


def _final(row: PendingApproval) -> Resolution:
    status = ApprovalStatus(row.status)
    msg = {
        ApprovalStatus.APPLIED: "Уже применено.",
        ApprovalStatus.REJECTED: "Уже отклонено.",
        ApprovalStatus.EXPIRED: "Срок подтверждения истёк — ничего не изменено.",
        ApprovalStatus.FAILED: "Это предложение уже не применить.",
    }.get(status, "")
    return Resolution(row, status, False, msg)


async def _resolve(
    session: AsyncSession,
    ctx: AppContext,
    row: PendingApproval,
    status: ApprovalStatus,
    *,
    result: dict[str, Any] | None = None,
) -> None:
    now = ctx.clock.now()
    row.status = status
    row.resolved_at = now
    if result is not None:
        row.result = result
    await audit(
        session,
        now=now,
        actor=Actor.USER
        if status in (ApprovalStatus.APPLIED, ApprovalStatus.REJECTED)
        else Actor.SYSTEM,
        action=f"approval.{status.value}",
        entity="approval",
        entity_id=row.id,
        details={"action_type": row.action_type, "origin": row.origin, "result": result or {}},
    )


async def _expire_if_due(session: AsyncSession, ctx: AppContext, row: PendingApproval) -> bool:
    if row.status == ApprovalStatus.PENDING and _is_expired(ctx, row):
        await _resolve(session, ctx, row, ApprovalStatus.EXPIRED)
        return True
    return False


async def approve(session: AsyncSession, ctx: AppContext, approval_id: int) -> Resolution:
    """Apply a pending card as the user's consent. Repeated calls are no-ops."""
    row = await session.get(PendingApproval, approval_id)
    if row is None:
        raise LookupError(f"approval {approval_id} not found")
    if row.status != ApprovalStatus.PENDING:
        return _final(row)
    if await _expire_if_due(session, ctx, row):
        return Resolution(
            row, ApprovalStatus.EXPIRED, True, "Срок подтверждения истёк — ничего не изменено."
        )
    try:
        action = actions.parse_action(row.action_payload)
    except ValidationError as exc:
        await _resolve(
            session,
            ctx,
            row,
            ApprovalStatus.FAILED,
            result={"error": "invalid payload", "details": str(exc)[:500]},
        )
        return Resolution(
            row, ApprovalStatus.FAILED, True, "Предложение повреждено — не применено."
        )
    if actions.get_handler(ctx, action.type) is None:
        await _resolve(session, ctx, row, ApprovalStatus.FAILED, result={"error": "unsupported"})
        return Resolution(row, ApprovalStatus.FAILED, True, actions.UNSUPPORTED_TEXT)
    try:
        outcome = await actions.apply_action(session, ctx, action, actor=Actor.USER)
    except actions.ActionError as exc:
        await _resolve(session, ctx, row, ApprovalStatus.FAILED, result={"error": str(exc)})
        return Resolution(row, ApprovalStatus.FAILED, True, f"Не удалось применить: {exc}")
    await _resolve(session, ctx, row, ApprovalStatus.APPLIED, result=outcome.result)
    hooks = [outcome.after_commit] if outcome.after_commit else []
    return Resolution(row, ApprovalStatus.APPLIED, True, "Готово.", hooks)


async def reject(session: AsyncSession, ctx: AppContext, approval_id: int) -> Resolution:
    row = await session.get(PendingApproval, approval_id)
    if row is None:
        raise LookupError(f"approval {approval_id} not found")
    if row.status != ApprovalStatus.PENDING:
        return _final(row)
    if await _expire_if_due(session, ctx, row):
        return Resolution(row, ApprovalStatus.EXPIRED, True, "Срок истёк — ничего не изменено.")
    await _resolve(session, ctx, row, ApprovalStatus.REJECTED)
    return Resolution(row, ApprovalStatus.REJECTED, True, "Отклонено — ничего не изменено.")


async def approve_group(
    session: AsyncSession, ctx: AppContext, approval_id: int
) -> list[Resolution]:
    """Accept every still-pending card of the group `approval_id` belongs to."""
    row = await session.get(PendingApproval, approval_id)
    if row is None:
        raise LookupError(f"approval {approval_id} not found")
    return [await approve(session, ctx, r.id) for r in await group_of(session, row)]


async def reject_group(
    session: AsyncSession, ctx: AppContext, approval_id: int
) -> list[Resolution]:
    row = await session.get(PendingApproval, approval_id)
    if row is None:
        raise LookupError(f"approval {approval_id} not found")
    return [await reject(session, ctx, r.id) for r in await group_of(session, row)]


def summarize(resolutions: Sequence[Resolution]) -> str:
    """One callback-answer line for a batch resolution."""
    changed = [r for r in resolutions if r.changed]
    if not changed:
        return resolutions[0].message if resolutions else "Нечего применять."
    applied = sum(1 for r in changed if r.status == ApprovalStatus.APPLIED)
    rejected = sum(1 for r in changed if r.status == ApprovalStatus.REJECTED)
    expired = sum(1 for r in changed if r.status == ApprovalStatus.EXPIRED)
    failed = sum(1 for r in changed if r.status == ApprovalStatus.FAILED)
    parts: list[str] = []
    if applied:
        parts.append(f"применено: {applied}")
    if rejected:
        parts.append(f"отклонено: {rejected}")
    if expired:
        parts.append(f"истёк срок: {expired}")
    if failed:
        parts.append(f"не удалось: {failed}")
    return ", ".join(parts).capitalize() + "."


# --------------------------------------------------------------------------- expiry


async def expire_pending(ctx: AppContext) -> int:
    """Periodic job: overdue pending cards become EXPIRED (silence != consent)."""
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        stmt = select(PendingApproval).where(
            PendingApproval.status == ApprovalStatus.PENDING,
            PendingApproval.expires_at.is_not(None),
            PendingApproval.expires_at <= now,
        )
        rows = list((await s.execute(stmt)).scalars())
        for row in rows:
            await _resolve(s, ctx, row, ApprovalStatus.EXPIRED)
    return len(rows)


async def _expire_job(ctx: AppContext) -> None:
    await expire_pending(ctx)


def register(ctx: AppContext) -> None:
    """Register the expiry job."""
    register_job(
        ctx,
        PeriodicJob(
            name=EXPIRE_JOB,
            interval_seconds=EXPIRE_INTERVAL_SECONDS,
            run=_expire_job,
            run_on_start=True,
        ),
    )
