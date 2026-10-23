"""Ready-made fake gateway for tests of domain code.

    gw = await make_fake_gateway(ctx, {"extract_tasks": [{"tasks": [...]}]})
    ...  # code under test calls ctx.ai.run("extract_tasks", ...)
    assert "<<<DATA" in gw.fake.calls_for("extract_tasks")[0].user
    gw.fake.push("chat_reply", "Привет!")
    gw.fake.push("plan_day", FakeError(AIErrorKind.SERVER))   # -> retries, then AIUnavailable

Every tier (fast/standard/strong/stt) has one fake model; structured tasks use
`native_schema` unless `strategy=` says otherwise. Backoff sleeps are no-ops,
notices are recorded in `gw.notices` instead of going through the gate.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from procbot.ai.adapters.fake import FakeAI, FakeCall, FakeError, FakeItem, FakeRefusal
from procbot.ai.config import AIConfig
from procbot.ai.gateway import AIGateway
from procbot.ai.types import StructuredStrategy
from procbot.context import AppContext
from procbot.db.models import AICall, AICallAttempt

__all__ = [
    "FakeAI",
    "FakeCall",
    "FakeError",
    "FakeRefusal",
    "add_spend",
    "fake_config",
    "make_fake_gateway",
]


def fake_config(
    *,
    strategy: StructuredStrategy = StructuredStrategy.NATIVE_SCHEMA,
    overrides: Mapping[str, Any] | None = None,
) -> AIConfig:
    data: dict[str, Any] = {
        "credentials": {"fake-main": {"provider": "fake", "api_key_env": None}},
        "models": {
            "fake-fast": {
                "credential": "fake-main",
                "model": "fake-fast",
                "structured_output": strategy,
            },
            "fake-standard": {
                "credential": "fake-main",
                "model": "fake-standard",
                "structured_output": strategy,
            },
            "fake-strong": {
                "credential": "fake-main",
                "model": "fake-strong",
                "structured_output": strategy,
            },
            "fake-stt": {"credential": "fake-main", "model": "fake-stt", "audio_input": True},
        },
        "tiers": {
            "fast": ["fake-fast"],
            "standard": ["fake-standard"],
            "strong": ["fake-strong"],
            "stt": ["fake-stt"],
        },
    }
    for key, value in (overrides or {}).items():
        data[key] = value
    return AIConfig.model_validate(data)


async def make_fake_gateway(
    ctx: AppContext,
    script: Mapping[str, Iterable[FakeItem] | FakeItem] | None = None,
    *,
    config: AIConfig | None = None,
    set_ctx: bool = True,
    environ: dict[str, str] | None = None,
) -> AIGateway:
    notices: list[tuple[str, str]] = []

    async def record_notice(text: str, dedup_key: str) -> None:
        notices.append((text, dedup_key))

    async def no_sleep(_seconds: float) -> None:
        return None

    gw = AIGateway(
        ctx,
        config or fake_config(),
        fake=FakeAI(script),
        notifier=record_notice,
        sleep=no_sleep,
        environ=environ or {},
    )
    gw.notices = notices  # type: ignore[attr-defined]
    await gw.start()
    if set_ctx:
        ctx.ai = gw
    return gw


async def add_spend(
    ctx: AppContext,
    *,
    at: datetime,
    cost: float,
    label: str = "fake-main",
    provider: str = "fake",
    alias: str = "fake-fast",
    task: str = "chat_reply",
) -> int:
    """Insert one finished AI call with one attempt costing `cost` (budget tests)."""
    async with ctx.db.transaction() as s:
        call = AICall(
            task=task,
            initiator="test",
            tier="fast",
            outcome="ok",
            input_categories=[],
            proposed_actions=[],
            actions_result=[],
            cost_usd=cost,
            created_at=at,
        )
        s.add(call)
        await s.flush()
        s.add(
            AICallAttempt(
                call_id=call.id,
                attempt_no=1,
                provider=provider,
                model="m",
                credential_label=label,
                model_alias=alias,
                cost_usd=cost,
                created_at=at,
            )
        )
        return call.id
