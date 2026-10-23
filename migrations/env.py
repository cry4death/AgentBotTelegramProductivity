"""Alembic environment (async, SQLite, batch mode).

URL resolution order:
1. `config.attributes["connection"]`: an existing sync connection (tests);
2. `sqlalchemy.url` set programmatically (procbot.main.alembic_config);
3. DATABASE_PATH from the environment / .env (procbot.config.EnvSettings).

`render_as_batch=True` because SQLite cannot ALTER most things in place.
Custom column types (UTCDateTime, LocalTime, LocalDate) are rendered as their
plain storage types so migrations never import application code that may change.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig
from typing import Any, Literal

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy.types import TypeDecorator

import procbot.db.models  # noqa: F401  (registers all tables on Base.metadata)
from procbot.db.base import Base

config = context.config

if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _url() -> str:
    url = config.get_main_option("sqlalchemy.url")
    if url:
        return url
    from procbot.config import EnvSettings
    from procbot.db.session import sqlite_url

    path = EnvSettings().database_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite_url(path)


def render_item(type_: str, obj: Any, autogen_context: Any) -> str | Literal[False]:
    """Render application TypeDecorators as their underlying storage type."""
    if type_ == "type" and isinstance(obj, TypeDecorator):
        return f"sa.{obj.impl_instance!r}"
    return False


def _configure(connection: Connection | None = None, **kw: Any) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,
        compare_type=True,
        render_item=render_item,
        **kw,
    )


def run_migrations_offline() -> None:
    _configure(url=_url(), literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    _configure(connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _url()
    engine = async_engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        do_run_migrations(connection)
        return
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
