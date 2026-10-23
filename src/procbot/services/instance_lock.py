"""Single-instance guard: long polling must never run twice (spec section 6).

Two layers:
1. an exclusive, non-blocking `fcntl.flock` on `env.lock_path` (same host /
   same volume), released automatically by the OS if the process dies;
2. a `job_leases` row named "poller" with a short TTL, renewed by a heartbeat.
   It protects against a second instance that shares the database but not the
   lock file. A lease left by a crashed process simply expires.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
from pathlib import Path
from typing import IO

from procbot.context import AppContext
from procbot.services import leases

log = logging.getLogger(__name__)

POLLER_LEASE = "poller"


class InstanceLockError(RuntimeError):
    """Another instance already runs (or holds the lease)."""


class InstanceLock:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fh: IO[str] | None = None

    @property
    def held(self) -> bool:
        return self._fh is not None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+", encoding="utf-8")  # noqa: SIM115 - kept open while held
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            fh.close()
            raise InstanceLockError(
                f"another procbot instance holds {self.path}; "
                "уже запущен другой экземпляр бота — второй запуск запрещён"
            ) from exc
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()}\n")
        fh.flush()
        self._fh = fh

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> InstanceLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def lock_is_held(path: Path) -> bool:
    """True when some process currently holds the instance lock at `path`."""
    if not Path(path).exists():
        return False
    probe = InstanceLock(path)
    try:
        probe.acquire()
    except InstanceLockError:
        return True
    probe.release()
    return False


async def acquire_poller_lease(
    ctx: AppContext,
    *,
    ttl_seconds: float = 60.0,
    wait_seconds: float = 0.0,
    poll_interval: float = 1.0,
) -> None:
    """Acquire the poller lease or raise InstanceLockError.

    `wait_seconds` lets a restart after a crash wait for the stale lease to
    expire; a live instance keeps renewing it, so the wait ends in a refusal.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + wait_seconds
    while True:
        if await leases.acquire(ctx, POLLER_LEASE, ttl_seconds=ttl_seconds):
            return
        if loop.time() >= deadline:
            raise InstanceLockError(
                "the 'poller' lease is held by another live instance; "
                "другой экземпляр бота работает с этой базой — запуск отменён"
            )
        await asyncio.sleep(poll_interval)


async def release_poller_lease(ctx: AppContext) -> None:
    await leases.release(ctx, POLLER_LEASE)


async def lease_heartbeat(
    ctx: AppContext, stop: asyncio.Event, *, ttl_seconds: float = 60.0
) -> bool:
    """Renew the poller lease every ttl/3 until `stop`.

    Returns True if the lease was lost (taken over by another instance); then it
    also sets `stop` so the process shuts down instead of polling twice.
    """
    interval = max(ttl_seconds / 3, 1.0)
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return False
        except TimeoutError:
            pass
        try:
            ok = await leases.renew(ctx, POLLER_LEASE, ttl_seconds=ttl_seconds)
        except Exception:
            log.exception("poller lease renewal failed")
            continue
        if not ok:
            log.critical("poller lease lost (taken over by another instance); stopping")
            stop.set()
            return True
    return False
