"""job_leases: cross-process mutual exclusion for background work.

A lease is a row (name, owner, expires_at). Acquisition happens inside a
``BEGIN IMMEDIATE`` transaction, so two schedulers (even in two processes on the
same SQLite file) can never both hold the same unexpired lease. Leases expire on
their own, so a crashed holder blocks others for at most its TTL.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.models import JobLease


async def acquire_in(
    session: AsyncSession, *, name: str, owner: str, now: datetime, ttl_seconds: float
) -> bool:
    """Take `name` if it is free or expired. A held, unexpired lease is never shared,
    not even with the same owner (two coroutines of one process are distinct users)."""
    row = await session.get(JobLease, name)
    expires = now + timedelta(seconds=ttl_seconds)
    if row is None:
        session.add(JobLease(name=name, owner=owner, acquired_at=now, expires_at=expires))
        await session.flush()
        return True
    if row.expires_at <= now:
        row.owner = owner
        row.acquired_at = now
        row.expires_at = expires
        await session.flush()
        return True
    return False


async def acquire(
    ctx: AppContext, name: str, *, ttl_seconds: float | None = None, owner: str | None = None
) -> bool:
    ttl = ttl_seconds if ttl_seconds is not None else ctx.config.scheduler.lease_seconds
    async with ctx.db.transaction() as s:
        return await acquire_in(
            s, name=name, owner=owner or ctx.instance_id, now=ctx.clock.now(), ttl_seconds=ttl
        )


async def renew(
    ctx: AppContext, name: str, *, ttl_seconds: float | None = None, owner: str | None = None
) -> bool:
    """Extend a lease we hold; False if it is not ours any more."""
    ttl = ttl_seconds if ttl_seconds is not None else ctx.config.scheduler.lease_seconds
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        row = await s.get(JobLease, name)
        if row is None or row.owner != (owner or ctx.instance_id):
            return False
        row.expires_at = now + timedelta(seconds=ttl)
        return True


async def release(ctx: AppContext, name: str, *, owner: str | None = None) -> bool:
    async with ctx.db.transaction() as s:
        res = await s.execute(
            delete(JobLease).where(
                JobLease.name == name, JobLease.owner == (owner or ctx.instance_id)
            )
        )
        return (res.rowcount or 0) > 0  # type: ignore[attr-defined]


async def expire_stale(ctx: AppContext) -> int:
    """Delete expired leases (startup recovery). Returns how many were removed."""
    async with ctx.db.transaction() as s:
        res = await s.execute(delete(JobLease).where(JobLease.expires_at <= ctx.clock.now()))
        return res.rowcount or 0  # type: ignore[attr-defined]


async def is_held_in(session: AsyncSession, name: str, now: datetime) -> bool:
    row = await session.get(JobLease, name)
    return row is not None and row.expires_at > now
