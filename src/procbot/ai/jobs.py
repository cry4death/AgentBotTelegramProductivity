"""Periodic AI jobs; `register(ctx)` is listed in procbot.wiring.REGISTRANTS.

- ai_probe: while in no-AI mode, a tiny request every
  `no_ai.probe_interval_seconds` brings the bot back automatically;
- ai_spend_anomaly: one owner notice per day when spend is anomalous.
Jobs run outside transactions (the gateway refuses to run inside one).
"""

from __future__ import annotations

import logging

from procbot.context import AppContext
from procbot.services.jobs import PeriodicJob, register_job

log = logging.getLogger(__name__)


async def probe_job(ctx: AppContext) -> None:
    from procbot.ai.gateway import get_gateway

    gw = get_gateway(ctx)
    if gw is not None and gw.configured:
        await gw.maybe_probe()


async def anomaly_job(ctx: AppContext) -> None:
    from procbot.ai.gateway import get_gateway

    gw = get_gateway(ctx)
    if gw is not None and gw.configured:
        await gw.check_spend_anomaly()


def register(ctx: AppContext) -> None:
    # The job ticks every minute; the gateway decides whether a probe is due.
    register_job(ctx, PeriodicJob(name="ai_probe", interval_seconds=60.0, run=probe_job))
    register_job(
        ctx, PeriodicJob(name="ai_spend_anomaly", interval_seconds=3600.0, run=anomaly_job)
    )
