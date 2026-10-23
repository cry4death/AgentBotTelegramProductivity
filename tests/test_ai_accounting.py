"""Audit rows, cost accounting, spend report, anomaly notice, secrets hygiene."""

from __future__ import annotations

import json
import logging
from datetime import timedelta

import httpx
import pytest
from sqlalchemy import select

from procbot.ai import AIUnavailable, record_actions
from procbot.ai.config import AIConfig
from procbot.ai.cost import compute_cost, spend_report
from procbot.ai.gateway import AIGateway
from procbot.ai.testing import FakeError, add_spend
from procbot.ai.types import Usage
from procbot.clock import FrozenClock
from procbot.context import AppContext
from procbot.db.models import AICall, AICallAttempt, AICredential
from procbot.domain.enums import AIErrorKind
from test_ai_gateway import ENV, INTENT, calls, cfg, gw_for, msg

SECRET = "sk-live-TOPSECRET-0123456789abcdefXYZ9"


async def test_cost_from_prices_and_tokens(ctx: AppContext) -> None:
    gw = await gw_for(ctx)
    gw.fake.push("classify_intent", INTENT, model="m-a1")  # a1: $1 in / $2 out per Mtok
    res = await gw.run("classify_intent", msg(), initiator="t")
    [row] = await calls(ctx)
    expected = row.tokens_in / 1e6 * 1.0 + row.tokens_out / 1e6 * 2.0
    assert row.cost_usd == pytest.approx(expected) and row.cost_usd > 0
    assert row.cost_estimated is False and res.cost_usd == row.cost_usd


def test_compute_cost_rules() -> None:
    m = cfg().models["a1"]
    assert compute_cost(m, Usage(tokens_in=1_000_000, tokens_out=500_000)) == (2.0, False)
    assert compute_cost(m, Usage(tokens_in=10, tokens_out=10, estimated=True))[1] is True
    unpriced = cfg().models["a2"]
    assert compute_cost(unpriced, Usage(tokens_in=100, tokens_out=100)) == (0.0, True)
    stt = cfg(
        models={
            "stt1": {
                "credential": "fa",
                "model": "x",
                "audio_input": True,
                "price_per_audio_minute": 0.5,
            }
        }
    ).models["stt1"]
    assert compute_cost(stt, Usage(audio_seconds=90)) == (0.75, False)


async def test_failed_attempts_and_repairs_are_recorded(ctx: AppContext) -> None:
    gw = await gw_for(ctx)
    gw.fake.push("classify_intent", FakeError(AIErrorKind.BAD_REQUEST, 400), model="m-a1")
    gw.fake.push("classify_intent", "bad", INTENT, model="m-a2")
    await gw.run("classify_intent", msg(), initiator="t")
    [row] = await calls(ctx)
    async with ctx.db.transaction() as s:
        tries = list(
            (
                await s.execute(select(AICallAttempt).where(AICallAttempt.call_id == row.id))
            ).scalars()
        )
    assert [(t.model_alias, t.error_kind, t.repair) for t in tries] == [
        ("a1", "bad_request", False),
        ("a2", "invalid_output", False),
        ("a2", None, True),
    ]
    assert row.tokens_in == sum(t.tokens_in for t in tries)


async def test_record_actions(ctx: AppContext) -> None:
    gw = await gw_for(ctx)
    gw.fake.push("classify_intent", INTENT)
    res = await gw.run("classify_intent", msg(), initiator="t")
    await record_actions(
        ctx,
        res.call_id,
        [{"type": "create_task", "title": "молоко"}],
        [{"status": "pending_approval"}],
    )
    async with ctx.db.transaction() as s:
        row = await s.get(AICall, res.call_id)
    assert row is not None
    assert row.proposed_actions == [{"type": "create_task", "title": "молоко"}]
    assert row.actions_result == [{"status": "pending_approval"}]
    with pytest.raises(KeyError):
        await record_actions(ctx, 99999, [], [])


async def test_spend_report_groups(ctx: AppContext, clock: FrozenClock) -> None:
    now = clock.now()
    await add_spend(
        ctx, at=now, cost=0.5, label="k1", provider="openai", alias="m1", task="plan_day"
    )
    await add_spend(
        ctx, at=now, cost=0.25, label="k2", provider="anthropic", alias="m2", task="chat_reply"
    )
    await add_spend(
        ctx, at=now - timedelta(days=2), cost=1.0, label="k1", provider="openai", alias="m1"
    )
    day = await spend_report(ctx, "day")
    assert day.total_usd == pytest.approx(0.75) and day.calls == 2
    assert day.by_provider == {"openai": 0.5, "anthropic": 0.25}
    assert day.by_task == {"plan_day": 0.5, "chat_reply": 0.25}
    month = await spend_report(ctx, "month")
    assert month.total_usd == pytest.approx(1.75)
    assert month.by_credential == {"k1": 1.5, "k2": 0.25}
    assert month.by_model == {"m1": 1.5, "m2": 0.25}


async def test_anomaly_notice_once_per_day(ctx: AppContext, clock: FrozenClock) -> None:
    gw = await gw_for(ctx)
    now = clock.now()
    for d in range(1, 8):
        await add_spend(ctx, at=now - timedelta(days=d), cost=0.1)
    assert await gw.check_spend_anomaly() is False
    await add_spend(ctx, at=now, cost=0.5)  # > 3 x 0.1
    assert await gw.check_spend_anomaly() is True
    assert await gw.check_spend_anomaly() is False  # once per day
    notices = gw.notices  # type: ignore[attr-defined]
    assert len(notices) == 1 and "необычно" in notices[0][0]
    assert notices[0][1] == f"ai-anomaly:{now.date().isoformat()}"


async def test_no_anomaly_without_history(ctx: AppContext, clock: FrozenClock) -> None:
    gw = await gw_for(ctx)
    await add_spend(ctx, at=clock.now(), cost=5.0)
    assert await gw.check_spend_anomaly() is False


async def test_env_keys_stored_encrypted_with_last4(ctx: AppContext) -> None:
    await gw_for(ctx)
    async with ctx.db.transaction() as s:
        rows = {r.label: r for r in (await s.execute(select(AICredential))).scalars()}
    assert set(rows) == {"fa", "fb"}
    fa = rows["fa"]
    assert fa.encrypted_key != ENV["FA_KEY"] and ENV["FA_KEY"] not in fa.encrypted_key
    assert ctx.secrets.decrypt(fa.encrypted_key) == ENV["FA_KEY"]
    assert (fa.key_last4, fa.source, fa.status) == ("1111", "env", "active")


async def test_key_never_appears_in_logs_or_audit(
    ctx: AppContext, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    seen_auth: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("authorization", ""))
        # Provider echoes the key back in its error message.
        return httpx.Response(
            401, json={"error": {"message": f"Incorrect API key provided: {SECRET}"}}
        )

    config = AIConfig.model_validate(
        {
            "credentials": {"oa": {"provider": "openai", "api_key_env": "OA"}},
            "models": {
                "o": {"credential": "oa", "model": "<o>", "structured_output": "native_schema"}
            },
            "tiers": {"fast": ["o"]},
        }
    )
    sent: list[tuple[str, str]] = []

    async def notifier(text: str, key: str) -> None:
        sent.append((text, key))

    gw = AIGateway(
        ctx,
        config,
        transport=httpx.MockTransport(handler),
        environ={"OA": SECRET},
        notifier=notifier,
    )
    await gw.start()
    with pytest.raises(AIUnavailable) as ei:
        await gw.run("classify_intent", msg(), initiator="t")
    assert seen_auth == [f"Bearer {SECRET}"]  # the key was used, only in the header
    assert SECRET not in str(ei.value) and SECRET not in repr(ei.value)
    assert SECRET not in caplog.text
    for rec in caplog.records:
        assert SECRET not in json.dumps(rec.__dict__, default=str)
    assert all(SECRET not in t for t, _ in sent) and len(sent) == 1
    async with ctx.db.transaction() as s:
        dump = []
        for model in (AICall, AICallAttempt):
            for row in (await s.execute(select(model))).scalars():
                dump.append(
                    json.dumps(
                        {c.name: getattr(row, c.name) for c in model.__table__.columns}, default=str
                    )
                )
        cred = (await s.execute(select(AICredential))).scalar_one()
    assert dump and all(SECRET not in d for d in dump)
    assert SECRET not in (cred.last_error or "") and SECRET not in cred.encrypted_key
    assert cred.status == "disabled"
    cred_obj = await gw.credentials.auth("oa")
    assert SECRET not in repr(cred_obj) and SECRET not in str(cred_obj)
    await gw.aclose()
