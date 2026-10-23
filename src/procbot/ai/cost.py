"""Cost accounting and the "Расходы ИИ" report.

Spend is the sum of `ai_call_attempts.cost_usd` (every attempt, failed ones and
repairs included). Days and months are the owner's local calendar days (IANA
zone from the profile; UTC until onboarding sets it).
Unknown prices (None in ai.yaml) -> cost 0 with `estimated=True`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.ai.config import ModelConfig
from procbot.ai.types import Usage
from procbot.context import AppContext
from procbot.db.models import AICall, AICallAttempt
from procbot.services.profile import get_profile
from procbot.timeutil import local_date, local_day_bounds, start_of_local_day


def compute_cost(model: ModelConfig, usage: Usage) -> tuple[float, bool]:
    """(cost_usd, estimated). Estimated = usage was guessed or a price is unknown."""
    if usage.audio_seconds is not None and model.price_per_audio_minute is not None:
        return round(usage.audio_seconds / 60.0 * model.price_per_audio_minute, 8), usage.estimated
    pin, pout = model.price_in_per_mtok, model.price_out_per_mtok
    if not (usage.tokens_in or usage.tokens_out) or (pin is None and pout is None):
        return 0.0, True
    cost = usage.tokens_in / 1_000_000 * (pin or 0.0) + usage.tokens_out / 1_000_000 * (pout or 0.0)
    return round(cost, 8), usage.estimated or pin is None or pout is None


async def owner_tz(s: AsyncSession) -> str | None:
    p = await get_profile(s)
    return p.timezone if p is not None else None


def month_start(d: date) -> date:
    return d.replace(day=1)


def next_month(d: date) -> date:
    return (d.replace(day=28) + timedelta(days=4)).replace(day=1)


def period_bounds(
    now: datetime, tz: str | None, period: Literal["day", "month"]
) -> tuple[datetime, datetime]:
    today = local_date(now, tz)
    if period == "day":
        return local_day_bounds(today, tz)
    return start_of_local_day(month_start(today), tz), start_of_local_day(next_month(today), tz)


async def spend_between(
    s: AsyncSession, start: datetime, end: datetime, *, credential_label: str | None = None
) -> float:
    q = select(func.coalesce(func.sum(AICallAttempt.cost_usd), 0.0)).where(
        AICallAttempt.created_at >= start, AICallAttempt.created_at < end
    )
    if credential_label is not None:
        q = q.where(AICallAttempt.credential_label == credential_label)
    return float((await s.execute(q)).scalar_one())


async def month_spend(ctx: AppContext, *, credential_label: str | None = None) -> float:
    async with ctx.db.transaction() as s:
        tz = await owner_tz(s)
        start, end = period_bounds(ctx.clock.now(), tz, "month")
        return await spend_between(s, start, end, credential_label=credential_label)


async def month_spend_by_credential(ctx: AppContext) -> dict[str, float]:
    async with ctx.db.transaction() as s:
        tz = await owner_tz(s)
        start, end = period_bounds(ctx.clock.now(), tz, "month")
        rows = await s.execute(
            select(AICallAttempt.credential_label, func.sum(AICallAttempt.cost_usd))
            .where(AICallAttempt.created_at >= start, AICallAttempt.created_at < end)
            .group_by(AICallAttempt.credential_label)
        )
        return {str(label): float(total or 0.0) for label, total in rows.all()}


@dataclass
class SpendReport:
    period: str
    start: datetime
    end: datetime
    total_usd: float = 0.0
    attempts: int = 0
    calls: int = 0
    estimated: bool = False  # at least one attempt had unknown price / usage
    by_provider: dict[str, float] = field(default_factory=dict)
    by_credential: dict[str, float] = field(default_factory=dict)
    by_model: dict[str, float] = field(default_factory=dict)
    by_task: dict[str, float] = field(default_factory=dict)


async def spend_report(ctx: AppContext, period: Literal["day", "month"] = "month") -> SpendReport:
    async with ctx.db.transaction() as s:
        tz = await owner_tz(s)
        start, end = period_bounds(ctx.clock.now(), tz, period)
        rows = (
            await s.execute(
                select(
                    AICallAttempt.provider,
                    AICallAttempt.credential_label,
                    AICallAttempt.model_alias,
                    AICall.task,
                    AICallAttempt.cost_usd,
                    AICallAttempt.cost_estimated,
                    AICallAttempt.call_id,
                )
                .join(AICall, AICall.id == AICallAttempt.call_id)
                .where(AICallAttempt.created_at >= start, AICallAttempt.created_at < end)
            )
        ).all()
    rep = SpendReport(period=period, start=start, end=end)
    groups: dict[str, defaultdict[str, float]] = {
        k: defaultdict(float) for k in ("provider", "credential", "model", "task")
    }
    call_ids: set[int] = set()
    for provider, label, alias, task, cost, est, call_id in rows:
        c = float(cost or 0.0)
        rep.total_usd += c
        rep.attempts += 1
        rep.estimated = rep.estimated or bool(est)
        call_ids.add(int(call_id))
        groups["provider"][provider] += c
        groups["credential"][label] += c
        groups["model"][alias] += c
        groups["task"][task] += c
    rep.calls = len(call_ids)
    rep.by_provider = dict(groups["provider"])
    rep.by_credential = dict(groups["credential"])
    rep.by_model = dict(groups["model"])
    rep.by_task = dict(groups["task"])
    return rep


async def anomaly_check(
    ctx: AppContext, *, factor: float, trailing_days: int
) -> tuple[bool, float, float]:
    """(is_anomaly, today_usd, trailing_daily_avg_usd).

    Needs some history: with a zero trailing average nothing is flagged.
    """
    async with ctx.db.transaction() as s:
        tz = await owner_tz(s)
        today = local_date(ctx.clock.now(), tz)
        t_start, t_end = local_day_bounds(today, tz)
        today_spend = await spend_between(s, t_start, t_end)
        h_start = start_of_local_day(today - timedelta(days=trailing_days), tz)
        hist = await spend_between(s, h_start, t_start)
    avg = hist / trailing_days
    return (avg > 0 and today_spend > factor * avg), today_spend, avg


def fmt_usd(v: float) -> str:
    if v == 0:
        return "$0"
    if v < 0.01:
        return f"${v:.4f}"
    return f"${v:.2f}"
