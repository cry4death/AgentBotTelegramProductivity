"""Backups (spec section 7, acceptance 24).

Pipeline: SQLite Online Backup API (consistent snapshot while the bot runs, WAL
included) -> gzip -> Fernet encryption with MASTER_KEY (unless
`backup.encrypt: false`) -> file in `env.backup_dir` + `<file>.sha256` sidecar
-> `backup_manifest` row.

The backup holds only the database. Secrets inside it (AI keys, OAuth tokens)
are already encrypted with the master key; the master key itself is never in a
backup, so restoring needs the same MASTER_KEY in the environment.

Open questions 8 and 10: periodicity and retention are not decided.
`backup.interval_hours` (UNCONFIRMED default 24h, None = off) and
`backup.retention_days` (None = never auto-delete) live in app.yaml.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import logging
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import select

from procbot.context import AppContext
from procbot.db.models import BackupManifest
from procbot.db.session import Database
from procbot.services.jobs import PeriodicJob, register_job

log = logging.getLogger(__name__)

GZIP_MAGIC = b"\x1f\x8b"
SQLITE_MAGIC = b"SQLite format 3\x00"
# How often the backup job wakes up to check whether a backup is due. The due
# check reads backup_manifest, so restarts never skip or duplicate backups.
CHECK_INTERVAL_SECONDS = 3600.0


class BackupError(Exception):
    pass


@dataclass(frozen=True)
class BackupFile:
    path: Path
    sha256: str
    size_bytes: int
    encrypted: bool
    schema_revision: str | None
    created_at: datetime


@dataclass(frozen=True)
class VerifyResult:
    path: Path
    sha256: str
    encrypted: bool
    integrity: str
    tables: int
    schema_revision: str | None


@dataclass(frozen=True)
class RestoreResult:
    target: Path
    safety_copy: Path | None
    schema_revision: str | None


# --------------------------------------------------------------------------- helpers


def _fernet(master_key: str) -> Fernet:
    try:
        return Fernet(master_key.encode())
    except (ValueError, TypeError) as exc:
        raise BackupError("MASTER_KEY is not a valid Fernet key") from exc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _snapshot(src_path: Path, dest_path: Path) -> None:
    """Consistent copy via the SQLite Online Backup API (works while others write)."""
    src = sqlite3.connect(src_path)
    try:
        dst = sqlite3.connect(dest_path)
        try:
            src.backup(dst)
            dst.execute("PRAGMA journal_mode=DELETE")  # single self-contained file
        finally:
            dst.close()
    finally:
        src.close()


def _schema_revision(db_file: Path) -> str | None:
    con = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    try:
        try:
            row = con.execute("SELECT version_num FROM alembic_version").fetchone()
        except sqlite3.DatabaseError:
            return None
        return None if row is None else str(row[0])
    finally:
        con.close()


def _integrity(db_file: Path) -> tuple[str, int]:
    con = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    try:
        result = con.execute("PRAGMA integrity_check").fetchone()
        tables = con.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()
    except sqlite3.DatabaseError as exc:
        raise BackupError(f"backup does not contain a valid SQLite database: {exc}") from exc
    finally:
        con.close()
    return (str(result[0]) if result else "unknown"), int(tables[0] if tables else 0)


def _decode(blob: bytes, master_key: str | None) -> tuple[bytes, bool]:
    """Backup file bytes -> raw SQLite bytes. Rejects tampering and wrong keys."""
    encrypted = not blob.startswith(GZIP_MAGIC)
    if encrypted:
        if not master_key:
            raise BackupError("backup is encrypted; MASTER_KEY is required")
        try:
            blob = _fernet(master_key).decrypt(blob)
        except InvalidToken as exc:
            raise BackupError(
                "cannot decrypt backup: wrong MASTER_KEY or the file was modified"
            ) from exc
    try:
        raw = gzip.decompress(blob)
    except (OSError, EOFError) as exc:
        raise BackupError(f"backup is corrupted (gzip): {exc}") from exc
    if not raw.startswith(SQLITE_MAGIC):
        raise BackupError("backup payload is not a SQLite database")
    return raw, encrypted


def _check_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    side = _sidecar(path)
    if side.exists():
        expected = side.read_text(encoding="utf-8").split()[0].strip()
        if expected != digest:
            raise BackupError("sha256 mismatch: the backup file was modified or truncated")
    return digest


# --------------------------------------------------------------------------- create


def create_backup_file(
    *,
    db_path: Path,
    backup_dir: Path,
    master_key: str | None,
    now: datetime,
) -> BackupFile:
    """Blocking: snapshot + gzip + (encrypt) + sha256. Run in a thread."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise BackupError(f"database not found: {db_path}")
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    suffix = ".db.gz.enc" if master_key else ".db.gz"
    final = backup_dir / f"procbot-{stamp}{suffix}"
    n = 1
    while final.exists():
        final = backup_dir / f"procbot-{stamp}-{n}{suffix}"
        n += 1
    with tempfile.TemporaryDirectory(dir=backup_dir, prefix=".tmp-") as tmp:
        snap = Path(tmp) / "snapshot.db"
        _snapshot(db_path, snap)
        revision = _schema_revision(snap)
        blob = gzip.compress(snap.read_bytes(), compresslevel=6)
        if master_key:
            blob = _fernet(master_key).encrypt(blob)
        part = Path(tmp) / "out.part"
        part.write_bytes(blob)
        os.chmod(part, 0o600)
        os.replace(part, final)
    digest = sha256_file(final)
    side = _sidecar(final)
    side.write_text(f"{digest}  {final.name}\n", encoding="utf-8")
    os.chmod(side, 0o600)
    return BackupFile(
        path=final,
        sha256=digest,
        size_bytes=final.stat().st_size,
        encrypted=bool(master_key),
        schema_revision=revision,
        created_at=now,
    )


# --------------------------------------------------------------------------- verify


def verify_backup(path: Path, master_key: str | None) -> VerifyResult:
    """Decrypt + decompress into a temp dir and run PRAGMA integrity_check."""
    path = Path(path)
    if not path.exists():
        raise BackupError(f"backup file not found: {path}")
    digest = _check_sidecar(path)
    raw, encrypted = _decode(path.read_bytes(), master_key)
    with tempfile.TemporaryDirectory(prefix="procbot-verify-") as tmp:
        db_file = Path(tmp) / "verify.db"
        db_file.write_bytes(raw)
        integrity, tables = _integrity(db_file)
        revision = _schema_revision(db_file)
    if integrity != "ok":
        raise BackupError(f"integrity_check failed: {integrity}")
    if tables == 0:
        raise BackupError("backup database has no tables")
    return VerifyResult(
        path=path,
        sha256=digest,
        encrypted=encrypted,
        integrity=integrity,
        tables=tables,
        schema_revision=revision,
    )


# --------------------------------------------------------------------------- restore


def restore_backup(
    path: Path,
    target: Path,
    master_key: str | None,
    *,
    force: bool = False,
    now: datetime | None = None,
) -> RestoreResult:
    """Restore a backup into `target`.

    Refuses to overwrite an existing database unless `force`; with `force` the
    current database is first copied to `<target>.pre-restore-<ts>`. The caller
    must make sure the bot is stopped (the CLI checks the instance lock).
    """
    path, target = Path(path), Path(target)
    verify_backup(path, master_key)
    raw, _ = _decode(path.read_bytes(), master_key)
    target.parent.mkdir(parents=True, exist_ok=True)
    safety: Path | None = None
    if target.exists():
        if not force:
            raise BackupError(
                f"target database exists: {target}; refusing to overwrite without --force"
            )
        stamp = (now.strftime("%Y%m%dT%H%M%SZ") if now else "now").replace(":", "")
        safety = target.with_name(f"{target.name}.pre-restore-{stamp}")
        _snapshot(target, safety)
    tmp = target.with_name(f".{target.name}.restore-tmp")
    tmp.write_bytes(raw)
    os.chmod(tmp, 0o600)
    integrity, _ = _integrity(tmp)
    if integrity != "ok":
        tmp.unlink(missing_ok=True)
        raise BackupError(f"integrity_check failed after write: {integrity}")
    for ext in ("-wal", "-shm"):
        Path(f"{target}{ext}").unlink(missing_ok=True)
    os.replace(tmp, target)
    return RestoreResult(
        target=target, safety_copy=safety, schema_revision=_schema_revision(target)
    )


# --------------------------------------------------------------------------- manifest


async def record_backup(db: Database, bf: BackupFile, *, status: str, now: datetime) -> int:
    async with db.transaction() as s:
        row = BackupManifest(
            path=str(bf.path),
            sha256=bf.sha256,
            size_bytes=bf.size_bytes,
            encrypted=bf.encrypted,
            status=status,
            schema_revision=bf.schema_revision,
            created_at=bf.created_at,
            verified_at=now if status == "verified" else None,
        )
        s.add(row)
        await s.flush()
        return row.id


async def last_backup(db: Database) -> BackupManifest | None:
    async with db.transaction() as s:
        return (
            await s.execute(
                select(BackupManifest)
                .where(BackupManifest.status.in_(("created", "verified")))
                .order_by(BackupManifest.created_at.desc(), BackupManifest.id.desc())
                .limit(1)
            )
        ).scalar_one_or_none()


async def make_backup(
    db: Database,
    *,
    db_path: Path,
    backup_dir: Path,
    master_key: str | None,
    now: datetime,
) -> tuple[BackupFile, int]:
    """Create, verify and record one backup. Never runs inside a transaction."""
    if db.in_transaction():
        raise RuntimeError("make_backup must not run inside a transaction")
    bf = await asyncio.to_thread(
        create_backup_file,
        db_path=db_path,
        backup_dir=backup_dir,
        master_key=master_key,
        now=now,
    )
    try:
        await asyncio.to_thread(verify_backup, bf.path, master_key)
        status = "verified"
    except BackupError:
        log.exception("fresh backup failed verification", extra={"path": str(bf.path)})
        status = "failed"
    manifest_id = await record_backup(db, bf, status=status, now=now)
    log.info("backup created", extra={"path": str(bf.path), "status": status})
    return bf, manifest_id


def _master_key_for(ctx: AppContext) -> str | None:
    return ctx.env.master_key if ctx.config.backup.encrypt else None


async def run_backup(ctx: AppContext) -> tuple[BackupFile, int]:
    return await make_backup(
        ctx.db,
        db_path=ctx.env.database_path,
        backup_dir=ctx.env.backup_dir,
        master_key=_master_key_for(ctx),
        now=ctx.clock.now(),
    )


async def apply_retention(ctx: AppContext) -> int:
    """Delete backup files older than retention_days. None = never (open question 8)."""
    days = ctx.config.backup.retention_days
    if days is None:
        return 0
    cutoff = ctx.clock.now() - timedelta(days=days)
    async with ctx.db.transaction() as s:
        rows = (
            await s.execute(
                select(BackupManifest.id, BackupManifest.path).where(
                    BackupManifest.created_at < cutoff, BackupManifest.status != "deleted"
                )
            )
        ).all()
    if not rows:
        return 0

    def _unlink(paths: list[str]) -> None:
        for raw in paths:
            p = Path(raw)
            p.unlink(missing_ok=True)
            _sidecar(p).unlink(missing_ok=True)

    await asyncio.to_thread(_unlink, [r.path for r in rows])
    ids = [r.id for r in rows]
    async with ctx.db.transaction() as s:
        for row in (
            (await s.execute(select(BackupManifest).where(BackupManifest.id.in_(ids))))
            .scalars()
            .all()
        ):
            row.status = "deleted"
    log.info("old backups deleted", extra={"count": len(ids)})
    return len(ids)


async def backup_due(ctx: AppContext) -> bool:
    hours = ctx.config.backup.interval_hours
    if hours is None:
        return False
    last = await last_backup(ctx.db)
    return last is None or ctx.clock.now() - last.created_at >= timedelta(hours=hours)


async def backup_job(ctx: AppContext) -> None:
    """Periodic job: idempotent; the due check reads backup_manifest."""
    if await backup_due(ctx):
        await run_backup(ctx)
    await apply_retention(ctx)


def register(ctx: AppContext) -> None:
    hours = ctx.config.backup.interval_hours
    if hours is None:
        return  # backups off (open question 8/10); the CLI still works
    register_job(
        ctx,
        PeriodicJob(
            name="backup",
            interval_seconds=min(CHECK_INTERVAL_SECONDS, hours * 3600),
            run=backup_job,
            run_on_start=True,
        ),
    )


__all__ = [
    "BackupError",
    "BackupFile",
    "RestoreResult",
    "VerifyResult",
    "apply_retention",
    "backup_due",
    "backup_job",
    "create_backup_file",
    "last_backup",
    "make_backup",
    "record_backup",
    "register",
    "restore_backup",
    "run_backup",
    "verify_backup",
]
