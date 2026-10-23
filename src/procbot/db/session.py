"""Engine and transaction management.

SQLite in WAL mode. Every transaction starts with BEGIN IMMEDIATE so that a
read-check-write sequence (e.g. the 4-per-60-minutes limit) is serialized across
concurrent tasks and processes.

Rule for callers: never await network I/O (Telegram, AI, Calendar) inside a
transaction; commit first, call out, then open a new transaction.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from procbot.db.base import Base


def sqlite_url(path: str | Path) -> str:
    return f"sqlite+aiosqlite:///{Path(path)}"


def _install_sqlite_hooks(engine: AsyncEngine, busy_timeout_ms: int) -> None:
    @event.listens_for(engine.sync_engine, "connect")
    def _on_connect(dbapi_connection: Any, _record: Any) -> None:
        # Let SQLAlchemy (not the driver) control transactions.
        dbapi_connection.isolation_level = None
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        cur.close()

    @event.listens_for(engine.sync_engine, "begin")
    def _on_begin(conn: Any) -> None:
        conn.exec_driver_sql("BEGIN IMMEDIATE")


class Database:
    """Owns the engine and hands out transactional sessions."""

    def __init__(self, url: str, *, busy_timeout_ms: int = 10_000, echo: bool = False) -> None:
        self.url = url
        self.engine = create_async_engine(url, echo=echo)
        _install_sqlite_hooks(self.engine, busy_timeout_ms)
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)
        # In-process write serialization: avoids SQLITE_BUSY between coroutines of
        # this process; BEGIN IMMEDIATE still protects against other processes.
        self._write_lock = asyncio.Lock()
        self._current: contextvars.ContextVar[AsyncSession | None] = contextvars.ContextVar(
            f"procbot_tx_{id(self)}", default=None
        )

    @classmethod
    def for_path(cls, path: str | Path, **kwargs: Any) -> Database:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return cls(sqlite_url(path), **kwargs)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[AsyncSession]:
        """One unit of work: commits on success, rolls back on error.

        Reply "done" to the user only after this context exits without error.
        A nested call in the same task joins the outer transaction (no commit of
        its own) instead of deadlocking on the write lock.
        """
        outer = self._current.get()
        if outer is not None:
            yield outer
            return
        async with self._write_lock, self.sessionmaker() as session:
            token = self._current.set(session)
            try:
                async with session.begin():
                    yield session
            finally:
                self._current.reset(token)

    def in_transaction(self) -> bool:
        return self._current.get() is not None

    async def create_all(self) -> None:
        """Create tables directly from metadata (tests). Production uses Alembic."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self.engine.dispose()
