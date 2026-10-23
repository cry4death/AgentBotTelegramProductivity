"""Circuit breaker, retries/backoff, budget, no-AI mode + recovery probe,
cross-provider fallback over real adapters."""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from typing import Any

import httpx
import pytest
from sqlalchemy import select

from procbot.ai import AIUnavailable
from procbot.ai.gateway import AIGateway, gate_notifier
from procbot.ai.jobs import probe_job
from procbot.ai.jobs import register as register_ai_jobs
from procbot.ai.testing import FakeError, add_spend
from procbot.clock import FrozenClock
from procbot.context import AppContext
from procbot.db.models import AICredential
from procbot.domain.enums import AIErrorKind, CredentialStatus
from procbot.services.jobs import registered_jobs
from test_ai_gateway import BASE, ENV, INTENT, calls, cfg, gw_for, msg


async def cred_row(ctx: AppContext, label: str) -> AICredential:
    async with ctx.db.transaction() as s:
        row = (
            await s.execute(select(AICredential).where(AICredential.label == label))
        ).scalar_one()
    return row


def notices(gw: AIGateway) -> list[tuple[str, str]]:
    return gw.notices  # type: ignore[attr-defined]


class SleepRecorder:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


# --------------------------------------------------------------------------- retries


async def test_transient_errors_retried_with_exponential_backoff(ctx: AppContext) -> None:
    gw = await gw_for(ctx)
    gw.sleep = sleeper = SleepRecorder()
    gw.fake.push(
        "classify_intent",
        FakeError(AIErrorKind.SERVER, 503),
        FakeError(AIErrorKind.TIMEOUT),
        INTENT,
        model="m-a1",
    )
    res = await gw.run("classify_intent", msg(), initiator="t")
    assert res.model_alias == "a1"
    assert sleeper.delays == [1.0, 2.0]
    [row] = await calls(ctx)
    assert row.attempts_count == 3 and row.outcome == "ok"


async def test_retries_exhausted_then_next_candidate(ctx: AppContext) -> None:
    gw = await gw_for(ctx)
    gw.sleep = sleeper = SleepRecorder()
    gw.fake.push("classify_intent", *[FakeError(AIErrorKind.NETWORK)] * 3, model="m-a1")
    gw.fake.push("classify_intent", INTENT, model="m-a2")
    res = await gw.run("classify_intent", msg(), initiator="t")
    assert res.model_alias == "a2"
    assert sleeper.delays == [1.0, 2.0]
    row = await cred_row(ctx, "fa")
    assert row.status == "active" and row.consecutive_failures == 1  # 5xx never opens the breaker


async def test_attempt_limit_per_call(ctx: AppContext) -> None:
    gw = await gw_for(ctx, cfg(limits={"max_attempts_per_call": 2}))
    gw.fake.set_default("classify_intent", FakeError(AIErrorKind.SERVER, 500))
    with pytest.raises(AIUnavailable):
        await gw.run("classify_intent", msg(), initiator="t")
    [row] = await calls(ctx)
    assert row.attempts_count == 2


async def test_bad_request_not_retried_on_same_model(ctx: AppContext) -> None:
    gw = await gw_for(ctx)
    gw.sleep = sleeper = SleepRecorder()
    gw.fake.push("classify_intent", FakeError(AIErrorKind.CONTEXT_TOO_LONG, 400), model="m-a1")
    gw.fake.push("classify_intent", INTENT, model="m-a2")
    assert (await gw.run("classify_intent", msg(), initiator="t")).model_alias == "a2"
    assert sleeper.delays == []
    assert (await cred_row(ctx, "fa")).status == "active"


# --------------------------------------------------------------------------- circuit breaker


async def test_429_cools_key_down_for_retry_after_then_recovers(
    ctx: AppContext, clock: FrozenClock
) -> None:
    gw = await gw_for(ctx)
    gw.fake.set_default("classify_intent", INTENT)
    gw.fake.push(
        "classify_intent", FakeError(AIErrorKind.RATE_LIMIT, 429, retry_after=120), model="m-a1"
    )
    res = await gw.run("classify_intent", msg(), initiator="t")
    assert res.model_alias == "a2"  # immediately the next candidate, no retry
    row = await cred_row(ctx, "fa")
    assert row.status == CredentialStatus.COOLING_DOWN
    assert row.cooldown_until == clock.now() + timedelta(seconds=120)
    # While cooling down the key is skipped.
    res2 = await gw.run("classify_intent", msg(), initiator="t")
    assert res2.credential_label == "fb"
    assert len([c for c in gw.fake.calls if c.model == "m-a1"]) == 1
    # After the cooldown the breaker is half-open; success closes it.
    clock.advance(seconds=121)
    res3 = await gw.run("classify_intent", msg(), initiator="t")
    assert res3.model_alias == "a1"
    row = await cred_row(ctx, "fa")
    assert row.status == "active" and row.cooldown_until is None and row.consecutive_failures == 0


async def test_429_without_retry_after_uses_exponential_cooldown(
    ctx: AppContext, clock: FrozenClock
) -> None:
    gw = await gw_for(ctx)
    gw.fake.set_default("classify_intent", INTENT)
    gw.fake.push("classify_intent", FakeError(AIErrorKind.RATE_LIMIT, 429), model="m-a1")
    await gw.run("classify_intent", msg(), initiator="t")
    assert (await cred_row(ctx, "fa")).cooldown_until == clock.now() + timedelta(seconds=60)
    clock.advance(seconds=61)
    gw.fake.push("classify_intent", FakeError(AIErrorKind.RATE_LIMIT, 429), model="m-a1")
    await gw.run("classify_intent", msg(), initiator="t")
    assert (await cred_row(ctx, "fa")).cooldown_until == clock.now() + timedelta(seconds=120)


async def test_401_disables_key_and_notifies_owner_exactly_once(ctx: AppContext) -> None:
    gw = await gw_for(ctx)
    gw.fake.set_default("classify_intent", INTENT)
    gw.fake.push("classify_intent", FakeError(AIErrorKind.AUTH, 401), model="m-a1")
    res = await gw.run("classify_intent", msg(), initiator="t")
    assert res.model_alias == "a2"
    row = await cred_row(ctx, "fa")
    assert row.status == "disabled" and row.disabled_notified is True
    for _ in range(3):
        await gw.run("classify_intent", msg(), initiator="t")
    # A restart re-syncing the same env key keeps it disabled and silent.
    await gw.credentials.sync()
    gw.fake.push("classify_intent", FakeError(AIErrorKind.AUTH, 403), model="m-a1")
    await gw.run("classify_intent", msg(), initiator="t")
    assert len([n for n in notices(gw) if "fa" in n[0]]) == 1
    text = notices(gw)[0][0]
    assert "fa" in text and "****1111" in text and ENV["FA_KEY"] not in text
    assert len([c for c in gw.fake.calls if c.model == "m-a1"]) == 1


async def test_rotated_env_key_reactivates_disabled_credential(ctx: AppContext) -> None:
    gw = await gw_for(ctx)
    gw.fake.set_default("classify_intent", INTENT)
    gw.fake.push("classify_intent", FakeError(AIErrorKind.AUTH, 401), model="m-a1")
    await gw.run("classify_intent", msg(), initiator="t")
    assert (await cred_row(ctx, "fa")).status == "disabled"
    gw.credentials.environ = {**ENV, "FA_KEY": "fake-key-NEW-9999"}
    await gw.credentials.sync()
    row = await cred_row(ctx, "fa")
    assert row.status == "active" and row.key_last4 == "9999" and not row.disabled_notified


async def test_several_keys_for_one_model_rotate(ctx: AppContext) -> None:
    config = cfg(
        credentials={
            **BASE["credentials"],
            "fa2": {"provider": "fake", "api_key_env": "FA2_KEY"},
        },
        models={**BASE["models"], "a1": {**BASE["models"]["a1"], "credential": ["fa", "fa2"]}},
    )
    from procbot.ai.testing import make_fake_gateway

    gw = await make_fake_gateway(ctx, config=config, environ={**ENV, "FA2_KEY": "fake-key-2222"})
    gw.fake.set_default("classify_intent", INTENT)
    gw.fake.push("classify_intent", FakeError(AIErrorKind.RATE_LIMIT, 429), model="m-a1")
    res = await gw.run("classify_intent", msg(), initiator="t")
    assert (res.model_alias, res.credential_label) == ("a1", "fa2")


async def test_missing_env_key_is_skipped(ctx: AppContext) -> None:
    from procbot.ai.testing import make_fake_gateway

    gw = await make_fake_gateway(ctx, config=cfg(), environ={"FB_KEY": ENV["FB_KEY"]})
    gw.fake.set_default("classify_intent", INTENT)
    res = await gw.run("classify_intent", msg(), initiator="t")
    assert res.credential_label == "fb"
    row = await cred_row(ctx, "fa")
    assert row.last_error and "FA_KEY" in row.last_error


# --------------------------------------------------------------------------- budget


async def test_soft_budget_downgrades_one_tier(ctx: AppContext, clock: FrozenClock) -> None:
    gw = await gw_for(ctx, cfg(budget={"monthly_soft_usd": 1.0}))
    gw.fake.set_default("chat_reply", "ответ")
    assert (await gw.run("chat_reply", msg(), initiator="t")).tier == "standard"
    await add_spend(ctx, at=clock.now() - timedelta(hours=1), cost=1.5)
    res = await gw.run("chat_reply", msg(), initiator="t")
    assert res.tier == "fast"
    assert (await calls(ctx))[-1].escalation_reason == "budget_soft_limit"


async def test_hard_budget_means_no_ai_and_one_notice(ctx: AppContext, clock: FrozenClock) -> None:
    gw = await gw_for(ctx, cfg(budget={"monthly_hard_usd": 2.0}))
    gw.fake.set_default("chat_reply", "ответ")
    await add_spend(ctx, at=clock.now() - timedelta(hours=1), cost=2.5)
    # Spend from last month does not count.
    await add_spend(ctx, at=clock.now() - timedelta(days=40), cost=100.0)
    for _ in range(2):
        with pytest.raises(AIUnavailable) as ei:
            await gw.run("chat_reply", msg(), initiator="t")
        assert ei.value.reason == "budget_hard_limit"
    assert gw.fake.calls == []
    assert len(notices(gw)) == 1 and "лимит" in notices(gw)[0][0]


async def test_per_key_spend_limit(ctx: AppContext, clock: FrozenClock) -> None:
    config = cfg(
        credentials={
            "fa": {"provider": "fake", "api_key_env": "FA_KEY", "spend_limit_usd": 1.0},
            "fb": {"provider": "fake", "api_key_env": "FB_KEY"},
        }
    )
    gw = await gw_for(ctx, config)
    gw.fake.set_default("classify_intent", INTENT)
    await add_spend(ctx, at=clock.now(), cost=1.0, label="fa")
    assert (await gw.run("classify_intent", msg(), initiator="t")).credential_label == "fb"


# --------------------------------------------------------------------------- no-AI mode


async def test_no_ai_mode_entered_once_and_probe_recovers(
    ctx: AppContext, clock: FrozenClock
) -> None:
    config = cfg(
        no_ai={
            "failed_calls_threshold": 2,
            "min_failure_seconds": 60,
            "probe_interval_seconds": 300,
        }
    )
    gw = await gw_for(ctx, config)
    gw.fake.set_default("classify_intent", FakeError(AIErrorKind.SERVER, 503))
    gw.fake.set_default("probe", FakeError(AIErrorKind.SERVER, 503))
    with pytest.raises(AIUnavailable):
        await gw.run("classify_intent", msg(), initiator="t")
    assert not gw.no_ai_mode
    clock.advance(seconds=61)
    with pytest.raises(AIUnavailable):
        await gw.run("classify_intent", msg(), initiator="t")
    assert gw.no_ai_mode
    n_calls = len(gw.fake.calls)
    with pytest.raises(AIUnavailable) as ei:
        await gw.run("classify_intent", msg(), initiator="t")
    assert ei.value.reason == "no_ai_mode"
    assert len(gw.fake.calls) == n_calls  # no provider traffic in no-AI mode
    assert len(notices(gw)) == 1 and "без ИИ" in notices(gw)[0][0]

    # Probe is rate limited by probe_interval_seconds.
    assert await gw.maybe_probe() is False
    clock.advance(seconds=301)
    assert await gw.maybe_probe() is False  # providers still down
    assert gw.no_ai_mode
    gw.fake.set_default("probe", "ok")
    clock.advance(seconds=301)
    ctx.ai = gw
    await probe_job(ctx)
    assert not gw.no_ai_mode
    assert len(notices(gw)) == 2 and "снова доступен" in notices(gw)[1][0]
    gw.fake.set_default("classify_intent", INTENT)
    assert (await gw.run("classify_intent", msg(), initiator="t")).model_alias == "a1"


async def test_all_keys_cooling_down_counts_toward_no_ai_mode(
    ctx: AppContext, clock: FrozenClock
) -> None:
    config = cfg(no_ai={"failed_calls_threshold": 2, "min_failure_seconds": 0})
    gw = await gw_for(ctx, config)
    gw.fake.set_default("classify_intent", FakeError(AIErrorKind.RATE_LIMIT, 429, retry_after=3600))
    with pytest.raises(AIUnavailable):
        await gw.run("classify_intent", msg(), initiator="t")
    with pytest.raises(AIUnavailable) as ei:
        await gw.run("classify_intent", msg(), initiator="t")
    assert ei.value.reason == "no_candidates"
    assert gw.no_ai_mode


def test_ai_jobs_registered(ctx: AppContext) -> None:
    register_ai_jobs(ctx)
    names = {j.name for j in registered_jobs(ctx)}
    assert {"ai_probe", "ai_spend_anomaly"} <= names


async def test_gate_notifier_tolerates_missing_gate(
    ctx: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "procbot.services.gate", None)
    await gate_notifier(ctx)("text", "key")  # ImportError is logged, not raised


async def test_gate_notifier_uses_system_notice(
    ctx: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types

    sent: list[dict[str, Any]] = []

    async def deliver(ctx: AppContext, **kw: Any) -> None:
        sent.append(kw)

    fake_gate = types.ModuleType("procbot.services.gate")
    fake_gate.deliver = deliver  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "procbot.services.gate", fake_gate)
    await gate_notifier(ctx)("hello", "k1")
    assert sent == [{"category": "system_notice", "text": "hello", "dedup_key": "k1"}]


# --------------------------------------------------------------------------- real adapters


async def test_cross_provider_fallback_openai_down_anthropic_answers(ctx: AppContext) -> None:
    from procbot.ai.config import AIConfig

    config = AIConfig.model_validate(
        {
            "credentials": {
                "oa": {"provider": "openai", "api_key_env": "OA"},
                "an": {"provider": "anthropic", "api_key_env": "AN"},
            },
            "models": {
                "o": {"credential": "oa", "model": "<o>", "structured_output": "native_schema"},
                "c": {"credential": "an", "model": "<c>", "structured_output": "tool_call"},
            },
            "tiers": {"fast": ["o", "c"]},
        }
    )
    hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        if request.url.host == "api.openai.com":
            return httpx.Response(503, json={"error": {"message": "down"}})
        body = json.loads(request.content)
        assert body["tool_choice"]["name"] == "IntentClassification"
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "tool_use", "id": "t", "name": "IntentClassification", "input": INTENT}
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    sleeper = SleepRecorder()
    gw = AIGateway(
        ctx,
        config,
        transport=httpx.MockTransport(handler),
        environ={"OA": "sk-oa-xxxxxxxxxxxx", "AN": "sk-an-yyyyyyyyyyyy"},
        notifier=lambda t, k: _noop(),
        sleep=sleeper,
    )
    await gw.start()
    res = await gw.run("classify_intent", msg(), initiator="t")
    assert (res.provider, res.model_alias) == ("anthropic", "c")
    assert hosts == ["api.openai.com"] * 3 + ["api.anthropic.com"]
    assert sleeper.delays == [1.0, 2.0]
    [row] = await calls(ctx)
    assert (row.provider, row.credential_label, row.tokens_out) == ("anthropic", "an", 5)
    await gw.aclose()


async def _noop() -> None:
    return None
