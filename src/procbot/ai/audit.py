"""AI audit (spec section 7): ai_calls, ai_call_attempts, ai_routing_config.

Written in their own short transactions, after the network part of a call.
Never stores keys; prompts and answers are not stored here (retention is open
question 8).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select

from procbot.ai.config import AIConfig
from procbot.context import AppContext
from procbot.db.models import AICall, AICallAttempt, AIRoutingConfig


@dataclass
class AttemptRecord:
    attempt_no: int
    provider: str
    model: str
    credential_label: str
    model_alias: str
    created_at: datetime
    error_kind: str | None = None
    http_status: int | None = None
    repair: bool = False
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    cost_estimated: bool = False
    latency_ms: int = 0


async def store_routing_config(ctx: AppContext, config: AIConfig) -> str:
    sha = config.sha256()
    async with ctx.db.transaction() as s:
        exists = (
            await s.execute(select(AIRoutingConfig.id).where(AIRoutingConfig.sha256 == sha))
        ).scalar_one_or_none()
        if exists is None:
            s.add(
                AIRoutingConfig(
                    sha256=sha, content=config.canonical_json(), created_at=ctx.clock.now()
                )
            )
    return sha


async def write_call(
    ctx: AppContext,
    *,
    task: str,
    initiator: str,
    correlation_id: str | None,
    tier: str,
    escalation_reason: str | None,
    config_sha256: str | None,
    input_categories: list[str],
    outcome: str,
    attempts: list[AttemptRecord],
    latency_ms: int,
    final: AttemptRecord | None = None,
    response_type: str | None = None,
    validation_error: str | None = None,
) -> int:
    async with ctx.db.transaction() as s:
        call = AICall(
            task=task,
            initiator=initiator,
            correlation_id=correlation_id,
            tier=tier,
            escalation_reason=escalation_reason,
            config_sha256=config_sha256,
            input_categories=sorted(input_categories),
            provider=final.provider if final else None,
            model=final.model if final else None,
            credential_label=final.credential_label if final else None,
            outcome=outcome,
            response_type=response_type,
            validation_error=(validation_error or None) and validation_error[:2000],
            tokens_in=sum(a.tokens_in for a in attempts),
            tokens_out=sum(a.tokens_out for a in attempts),
            cost_usd=round(sum(a.cost_usd for a in attempts), 8),
            cost_estimated=any(a.cost_estimated for a in attempts),
            latency_ms=latency_ms,
            attempts_count=len(attempts),
            proposed_actions=[],
            actions_result=[],
            created_at=ctx.clock.now(),
        )
        s.add(call)
        await s.flush()
        for a in attempts:
            s.add(
                AICallAttempt(
                    call_id=call.id,
                    attempt_no=a.attempt_no,
                    provider=a.provider,
                    model=a.model,
                    credential_label=a.credential_label,
                    model_alias=a.model_alias,
                    error_kind=a.error_kind,
                    http_status=a.http_status,
                    repair=a.repair,
                    tokens_in=a.tokens_in,
                    tokens_out=a.tokens_out,
                    cost_usd=a.cost_usd,
                    cost_estimated=a.cost_estimated,
                    latency_ms=a.latency_ms,
                    created_at=a.created_at,
                )
            )
        return call.id


def _jsonable(items: list[Any]) -> list[Any]:
    out: list[Any] = []
    for it in items:
        if isinstance(it, BaseModel):
            out.append(it.model_dump(mode="json"))
        else:
            out.append(it)
    return out


async def record_actions(
    ctx: AppContext, call_id: int, proposed: list[Any], results: list[Any]
) -> None:
    """Domain code records what the AI proposed and what happened to it
    (applied / pending approval / rejected). Items: dicts or pydantic models.
    Must be called outside of a transaction or joins the caller's one."""
    async with ctx.db.transaction() as s:
        call = await s.get(AICall, call_id)
        if call is None:
            raise KeyError(f"ai_call {call_id} not found")
        call.proposed_actions = _jsonable(proposed)
        call.actions_result = _jsonable(results)
