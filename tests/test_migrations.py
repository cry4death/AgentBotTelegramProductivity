"""Alembic: head on an empty file matches the models (guards model/migration drift)."""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, inspect

import procbot.db.models  # noqa: F401
from procbot.db.base import Base
from procbot.db.session import Database
from procbot.main import alembic_config, current_revision, head_revision, upgrade_to_head

# The models have an FK cycle (work_sessions -> interventions -> reminder_schedule ->
# work_sessions). SQLite does not care, but metadata sorting warns about it.
pytestmark = pytest.mark.filterwarnings("ignore:Cannot correctly sort tables")


def _diff(db_path: Path) -> list[object]:
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with engine.connect() as conn:
            mc = MigrationContext.configure(conn, opts={"compare_type": True})
            return list(compare_metadata(mc, Base.metadata))
    finally:
        engine.dispose()


def test_migrations_match_models(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    rev = upgrade_to_head(db)
    assert rev is not None and rev == head_revision(db)
    diff = _diff(db)
    assert diff == [], (
        "models and migrations differ; regenerate with scripts/regen_initial_migration.sh "
        f"(before the first release) or add a revision: {diff}"
    )


def test_upgrade_is_idempotent_and_downgrade_works(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    upgrade_to_head(db)
    upgrade_to_head(db)  # second run is a no-op
    command.downgrade(alembic_config(db), "base")
    engine = create_engine(f"sqlite:///{db}")
    try:
        assert set(inspect(engine).get_table_names()) <= {"alembic_version"}
    finally:
        engine.dispose()
    assert current_revision(db) is None
    upgrade_to_head(db)
    assert _diff(db) == []


async def test_app_works_on_migrated_database(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from procbot.services.profile import get_or_create_profile

    path = tmp_path / "app.db"
    upgrade_to_head(path)
    db = Database.for_path(path)
    try:
        async with db.transaction() as s:
            await get_or_create_profile(s, telegram_user_id=1, now=datetime.now(UTC))
        async with db.transaction() as s:
            from procbot.services.profile import get_profile

            p = await get_profile(s)
            assert p is not None and p.quiet_hours == []
    finally:
        await db.dispose()
