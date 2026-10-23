"""Backups: create -> verify -> restore into a clean path (acceptance 24)."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from procbot.config import AppConfig, BackupConfig
from procbot.crypto import SecretBox
from procbot.db.models import BackupManifest, OwnerProfile, Task
from procbot.db.session import Database
from procbot.services import backup
from procbot.services.backup import BackupError, restore_backup, run_backup, verify_backup
from procbot.services.jobs import registered_jobs
from procbot.testing import TEST_MASTER_KEY, make_context


async def _seed(ctx) -> None:
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        s.add(Task(title="Написать отчёт", created_at=now, updated_at=now))
        s.add(Task(title="Позвонить", created_at=now, updated_at=now, status="done"))


async def _tasks(db: Database) -> list[tuple[str, str]]:
    async with db.transaction() as s:
        rows = (await s.execute(select(Task).order_by(Task.id))).scalars().all()
        return [(t.title, t.status) for t in rows]


async def test_backup_restore_on_clean_path(ctx, profile, tmp_path: Path):
    await _seed(ctx)
    bf, manifest_id = await run_backup(ctx)
    assert bf.path.exists() and bf.encrypted and bf.path.name.endswith(".db.gz.enc")
    assert (bf.path.parent / (bf.path.name + ".sha256")).exists()
    assert TEST_MASTER_KEY.encode() not in bf.path.read_bytes()
    async with ctx.db.transaction() as s:
        row = await s.get(BackupManifest, manifest_id)
        assert row is not None and row.status == "verified" and row.sha256 == bf.sha256

    res = verify_backup(bf.path, TEST_MASTER_KEY)
    assert res.integrity == "ok" and res.tables > 10

    # Data written after the backup is not in it.
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        s.add(Task(title="после бэкапа", created_at=now, updated_at=now))

    clean = tmp_path / "clean-env" / "restored.db"
    restored = restore_backup(bf.path, clean, TEST_MASTER_KEY)
    assert restored.target == clean and restored.safety_copy is None
    db2 = Database.for_path(clean)
    try:
        assert await _tasks(db2) == [("Написать отчёт", "todo"), ("Позвонить", "done")]
        async with db2.transaction() as s:
            p = await s.get(OwnerProfile, 1)
            assert p is not None and p.timezone == "Europe/Berlin"
    finally:
        await db2.dispose()


async def test_tampered_and_wrong_key_backups_are_rejected(ctx, profile, tmp_path: Path):
    bf, _ = await run_backup(ctx)
    other_key = SecretBox.generate_key()
    with pytest.raises(BackupError, match="wrong MASTER_KEY"):
        verify_backup(bf.path, other_key)
    with pytest.raises(BackupError, match="MASTER_KEY is required"):
        verify_backup(bf.path, None)

    data = bytearray(bf.path.read_bytes())
    data[len(data) // 2] ^= 0x01
    tampered = tmp_path / "tampered.db.gz.enc"
    tampered.write_bytes(bytes(data))
    with pytest.raises(BackupError, match="wrong MASTER_KEY or the file was modified"):
        verify_backup(tampered, TEST_MASTER_KEY)
    # Same tampering next to the original sidecar: caught by sha256 first.
    bf.path.write_bytes(bytes(data))
    with pytest.raises(BackupError, match="sha256 mismatch"):
        verify_backup(bf.path, TEST_MASTER_KEY)
    with pytest.raises(BackupError):
        restore_backup(tampered, tmp_path / "x.db", TEST_MASTER_KEY)
    assert not (tmp_path / "x.db").exists()


async def test_restore_refuses_to_overwrite_without_force(ctx, profile, tmp_path: Path):
    await _seed(ctx)
    bf, _ = await run_backup(ctx)
    target = tmp_path / "live.db"
    restore_backup(bf.path, target, TEST_MASTER_KEY)
    with pytest.raises(BackupError, match="--force"):
        restore_backup(bf.path, target, TEST_MASTER_KEY)
    res = restore_backup(bf.path, target, TEST_MASTER_KEY, force=True, now=ctx.clock.now())
    assert res.safety_copy is not None and res.safety_copy.exists()
    db2 = Database.for_path(res.safety_copy)
    try:
        assert len(await _tasks(db2)) == 2
    finally:
        await db2.dispose()


async def test_unencrypted_backup_when_disabled(tmp_path: Path):
    ctx = await make_context(tmp_path, config=AppConfig(backup=BackupConfig(encrypt=False)))
    try:
        bf, _ = await run_backup(ctx)
        assert not bf.encrypted and bf.path.name.endswith(".db.gz")
        assert verify_backup(bf.path, None).encrypted is False
    finally:
        await ctx.db.dispose()


async def test_backup_job_due_logic_and_registration(ctx, clock, tmp_path: Path):
    backup.register(ctx)
    jobs = {j.name: j for j in registered_jobs(ctx)}
    assert "backup" in jobs and jobs["backup"].run_on_start
    assert await backup.backup_due(ctx)
    await backup.backup_job(ctx)
    assert not await backup.backup_due(ctx)
    await backup.backup_job(ctx)  # not due: no second file
    files = list(ctx.env.backup_dir.glob("procbot-*.db.gz.enc"))
    assert len(files) == 1
    clock.advance(hours=24)
    assert await backup.backup_due(ctx)

    off = await make_context(
        tmp_path / "off", config=AppConfig(backup=BackupConfig(interval_hours=None))
    )
    try:
        backup.register(off)
        assert "backup" not in {j.name for j in registered_jobs(off)}
        assert not await backup.backup_due(off)
    finally:
        await off.db.dispose()


async def test_retention_none_never_deletes_and_days_deletes_old(ctx, clock):
    bf, _ = await run_backup(ctx)
    clock.advance(days=400)
    assert await backup.apply_retention(ctx) == 0  # open question 8: keep forever
    assert bf.path.exists()
    ctx.config = AppConfig(backup=BackupConfig(retention_days=30))
    assert await backup.apply_retention(ctx) == 1
    assert not bf.path.exists()
    async with ctx.db.transaction() as s:
        assert (await s.execute(select(BackupManifest.status))).scalar_one() == "deleted"
    assert await backup.last_backup(ctx.db) is None
