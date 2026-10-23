"""Periodic background jobs, run by services.scheduler with a lease per run.

Jobs must be idempotent: a crash mid-run means the job simply runs again.
Daily "at local time" behaviour is implemented by interval jobs that check
whether something is due (e.g. the initiative seeder), not by in-memory timers.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from procbot.context import AppContext

JobFn = Callable[[AppContext], Awaitable[None]]


@dataclass(frozen=True)
class PeriodicJob:
    name: str
    interval_seconds: float
    run: JobFn
    # Run once right after startup (after recovery) in addition to the interval.
    run_on_start: bool = False


def register_job(ctx: AppContext, job: PeriodicJob) -> None:
    jobs: dict[str, PeriodicJob] = ctx.registry.setdefault("jobs", {})
    jobs[job.name] = job


def registered_jobs(ctx: AppContext) -> list[PeriodicJob]:
    return list(ctx.registry.get("jobs", {}).values())
