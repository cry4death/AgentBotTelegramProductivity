"""ActionExecutor: the only path by which proposed typed actions change data.

For every `ProposedAction` the executor looks up the FR-15 permission:

- ALWAYS_FORBIDDEN (e.g. changing external calendar events) -> dropped + audit,
  whoever asks;
- origin USER (an explicit user command for exactly this change) -> applied;
- AUTO -> applied; the applier enforces the constraints (subtasks only under an
  open parent, reorder touches only sort_order, ...);
- NEEDS_APPROVAL -> a pending approval card (services.approvals);
- FORBIDDEN -> dropped + audit.

Appliers are registered per ActionType by the owning module's `register(ctx)`.
An applier runs inside the caller's transaction (in a SAVEPOINT, so a failing
action leaves nothing behind) and may return an `after_commit` hook for network
side effects (e.g. a calendar insert), which the CALLER runs after commit.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import TypeAdapter, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.models import OperationKey
from procbot.domain.actions import (
    ALWAYS_FORBIDDEN,
    ActionType,
    Permission,
    ProposedAction,
    permission_for,
)
from procbot.domain.enums import Actor
from procbot.services.audit import audit

log = logging.getLogger(__name__)

AfterCommit = Callable[[], Awaitable[None]]


class ActionError(Exception):
    """The action is not valid against the current data (constraint violated,
    entity missing...). `str(exc)` is a short Russian message for the user."""


@dataclass
class ApplyOutcome:
    result: dict[str, Any] = field(default_factory=dict)
    after_commit: AfterCommit | None = None


# fn(session, ctx, action, actor) -> ApplyOutcome. `actor` is who the change is
# attributed to: USER for explicit commands and approved cards, else the origin.
Applier = Callable[[AsyncSession, AppContext, Any, Actor], Awaitable[ApplyOutcome]]
# fn(session, ctx, action) -> {"summary": str, "lines": [str, ...]}; raises
# ActionError when the action cannot be applied (used to validate proposals).
Describer = Callable[[AsyncSession, AppContext, Any], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ActionHandler:
    apply: Applier
    describe: Describer | None = None


class ExecStatus(StrEnum):
    APPLIED = "applied"
    PENDING_APPROVAL = "pending_approval"
    FORBIDDEN = "forbidden"
    INVALID = "invalid"
    UNSUPPORTED = "unsupported"


@dataclass
class ExecResult:
    status: ExecStatus
    action_type: ActionType | None
    op_key: str
    result: dict[str, Any] = field(default_factory=dict)
    approval_id: int | None = None
    error: str | None = None
    after_commit: list[AfterCommit] = field(default_factory=list)
    repeated: bool = False  # op_key was already executed; nothing changed now

    @property
    def ok(self) -> bool:
        return self.status in (ExecStatus.APPLIED, ExecStatus.PENDING_APPROVAL)


ACTION_ADAPTER: TypeAdapter[ProposedAction] = TypeAdapter(ProposedAction)

UNSUPPORTED_TEXT = "Это действие пока не поддерживается — ничего не изменено."


# --------------------------------------------------------------------------- registry


def register_applier(
    ctx: AppContext, action_type: ActionType, fn: Applier, *, describe: Describer | None = None
) -> None:
    handlers: dict[ActionType, ActionHandler] = ctx.registry.setdefault("action_handlers", {})
    handlers[action_type] = ActionHandler(apply=fn, describe=describe)


def get_handler(ctx: AppContext, action_type: ActionType | str) -> ActionHandler | None:
    try:
        key = ActionType(action_type)
    except ValueError:
        return None
    return ctx.registry.get("action_handlers", {}).get(key)


# --------------------------------------------------------------------------- payloads


def parse_action(payload: Any) -> Any:
    """dict -> typed ProposedAction (raises pydantic.ValidationError)."""
    return ACTION_ADAPTER.validate_python(payload)


def dump_action(action: Any) -> dict[str, Any]:
    """JSON payload that round-trips through `parse_action`.

    Only explicitly set fields are stored (plus the discriminator), so defaults
    that depend on what the proposer set (e.g. TaskDraft.verification) survive.
    """
    data = action.model_dump(mode="json", exclude_unset=True)
    data["type"] = str(action.type)
    return data


# --------------------------------------------------------------------------- operation keys


async def get_operation(session: AsyncSession, key: str) -> OperationKey | None:
    return await session.get(OperationKey, key)


async def claim_operation(
    session: AsyncSession, ctx: AppContext, key: str, result: dict[str, Any] | None = None
) -> bool:
    """Record a one-shot operation. False if it was already recorded (no-op)."""
    if await session.get(OperationKey, key) is not None:
        return False
    session.add(OperationKey(key=key[:200], result=result, created_at=ctx.clock.now()))
    await session.flush()
    return True


# --------------------------------------------------------------------------- execution


async def describe_action(session: AsyncSession, ctx: AppContext, action: Any) -> dict[str, Any]:
    handler = get_handler(ctx, action.type)
    if handler is not None and handler.describe is not None:
        return await handler.describe(session, ctx, action)
    return {"summary": str(action.type), "lines": []}


async def apply_action(
    session: AsyncSession, ctx: AppContext, action: Any, *, actor: Actor
) -> ApplyOutcome:
    """Run the registered applier in a SAVEPOINT. Raises ActionError (nothing
    changed) or LookupError when no applier is registered."""
    handler = get_handler(ctx, action.type)
    if handler is None:
        raise LookupError(f"no applier for {action.type}")
    async with session.begin_nested():
        return await handler.apply(session, ctx, action, actor)


async def execute(
    session: AsyncSession,
    ctx: AppContext,
    action: Any,
    *,
    origin: Actor,
    op_key: str,
    group_key: str | None = None,
    ai_call_id: int | None = None,
) -> ExecResult:
    """Decide and perform one proposed action inside the caller's transaction.

    Idempotent by `op_key`: a repeated call returns the earlier outcome without
    applying again. Run `result.after_commit` hooks only after the commit.
    """
    origin = Actor(origin)
    # 1. Typed validation (the AI may hand us raw dicts).
    if not hasattr(action, "type") or not hasattr(action, "model_dump"):
        try:
            action = parse_action(action)
        except ValidationError as exc:
            await _audit(
                session,
                ctx,
                "action.invalid",
                origin,
                op_key,
                None,
                {"error": exc.errors(include_url=False)[:5]},
            )
            return ExecResult(ExecStatus.INVALID, None, op_key, error="invalid payload")
    action_type = ActionType(action.type)
    perm = permission_for(action_type)

    # 2. Forbidden actions are dropped whoever asks.
    if action_type in ALWAYS_FORBIDDEN or (perm is Permission.FORBIDDEN and origin != Actor.USER):
        await _audit(
            session,
            ctx,
            "action.forbidden",
            origin,
            op_key,
            action_type,
            {"payload": dump_action(action)},
        )
        return ExecResult(ExecStatus.FORBIDDEN, action_type, op_key, error="forbidden")

    # 3. Repeated op_key -> earlier outcome.
    done = await get_operation(session, _op(op_key))
    if done is not None:
        prev = done.result or {}
        return ExecResult(
            ExecStatus(prev.get("status", ExecStatus.APPLIED)),
            action_type,
            op_key,
            result=prev.get("result", {}),
            approval_id=prev.get("approval_id"),
            repeated=True,
        )
    from procbot.services import approvals  # local import: approvals imports this module

    existing = await approvals.get_by_op_key(session, op_key)
    if existing is not None:
        return ExecResult(
            ExecStatus.PENDING_APPROVAL
            if existing.status == "pending"
            else ExecStatus(_status_from_approval(existing.status)),
            action_type,
            op_key,
            result=existing.result or {},
            approval_id=existing.id,
            repeated=True,
        )

    handler = get_handler(ctx, action_type)
    if handler is None:
        await _audit(session, ctx, "action.unsupported", origin, op_key, action_type, {})
        return ExecResult(ExecStatus.UNSUPPORTED, action_type, op_key, error=UNSUPPORTED_TEXT)

    # 4. Apply directly or ask for consent.
    if origin == Actor.USER or perm is Permission.AUTO:
        try:
            outcome = await apply_action(session, ctx, action, actor=origin)
        except ActionError as exc:
            await _audit(
                session, ctx, "action.invalid", origin, op_key, action_type, {"error": str(exc)}
            )
            return ExecResult(ExecStatus.INVALID, action_type, op_key, error=str(exc))
        await claim_operation(
            session,
            ctx,
            _op(op_key),
            {"status": ExecStatus.APPLIED.value, "result": outcome.result},
        )
        await _audit(
            session,
            ctx,
            "action.applied",
            origin,
            op_key,
            action_type,
            {"permission": perm.value, "result": outcome.result, "ai_call_id": ai_call_id},
        )
        hooks = [outcome.after_commit] if outcome.after_commit else []
        return ExecResult(
            ExecStatus.APPLIED, action_type, op_key, result=outcome.result, after_commit=hooks
        )

    try:
        diff = await describe_action(session, ctx, action)
    except ActionError as exc:
        await _audit(
            session, ctx, "action.invalid", origin, op_key, action_type, {"error": str(exc)}
        )
        return ExecResult(ExecStatus.INVALID, action_type, op_key, error=str(exc))
    row = await approvals.propose(
        session,
        ctx,
        action,
        origin=origin,
        op_key=op_key,
        diff=diff,
        group_key=group_key,
        ai_call_id=ai_call_id,
    )
    await _audit(
        session,
        ctx,
        "action.pending",
        origin,
        op_key,
        action_type,
        {"approval_id": row.id, "group_key": group_key},
    )
    return ExecResult(ExecStatus.PENDING_APPROVAL, action_type, op_key, approval_id=row.id)


async def execute_many(
    session: AsyncSession,
    ctx: AppContext,
    actions: Sequence[Any],
    *,
    origin: Actor,
    op_prefix: str,
    group_key: str | None = None,
    ai_call_id: int | None = None,
) -> list[ExecResult]:
    """Execute a batch; NEEDS_APPROVAL items share `group_key` (one card)."""
    out: list[ExecResult] = []
    for i, action in enumerate(actions):
        out.append(
            await execute(
                session,
                ctx,
                action,
                origin=origin,
                op_key=f"{op_prefix}:{i}",
                group_key=group_key,
                ai_call_id=ai_call_id,
            )
        )
    return out


async def run_after_commit(hooks: Sequence[AfterCommit]) -> list[BaseException]:
    """Run post-commit side effects; failures are logged and returned, not raised."""
    errors: list[BaseException] = []
    for hook in hooks:
        try:
            await hook()
        except Exception as exc:  # side effects must not break the reply
            log.exception("after_commit hook failed")
            errors.append(exc)
    return errors


class ActionExecutor:
    """Object-style facade over `execute` for callers that prefer one."""

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx

    async def execute(
        self,
        session: AsyncSession,
        action: Any,
        *,
        origin: Actor,
        op_key: str,
        group_key: str | None = None,
        ai_call_id: int | None = None,
    ) -> ExecResult:
        return await execute(
            session,
            self.ctx,
            action,
            origin=origin,
            op_key=op_key,
            group_key=group_key,
            ai_call_id=ai_call_id,
        )


# --------------------------------------------------------------------------- helpers


def _op(op_key: str) -> str:
    return f"exec:{op_key}"


def _status_from_approval(status: str) -> str:
    return {
        "applied": ExecStatus.APPLIED.value,
        "rejected": ExecStatus.INVALID.value,
        "expired": ExecStatus.INVALID.value,
        "failed": ExecStatus.INVALID.value,
    }.get(status, ExecStatus.PENDING_APPROVAL.value)


async def _audit(
    session: AsyncSession,
    ctx: AppContext,
    action: str,
    origin: Actor,
    op_key: str,
    action_type: ActionType | None,
    details: dict[str, Any],
) -> None:
    await audit(
        session,
        now=ctx.clock.now(),
        actor=origin,
        action=action,
        entity="action",
        entity_id=None if action_type is None else action_type.value,
        details={"op_key": op_key, **details},
    )
