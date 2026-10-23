"""Entry point: `procbot` CLI and the bot process startup (architecture scenario G).

Startup order for `procbot run`:
1. load EnvSettings + AppConfig, validate required secrets;
2. take the single-instance file lock (long polling must never run twice);
3. apply Alembic migrations up to head;
4. build AppContext, take the "poller" lease in job_leases;
5. wire modules, run recovery, restore active work sessions;
6. only then start the poller, the scheduler loop and the HTTP server;
7. on SIGTERM / SIGINT stop polling, let the scheduler finish its tick, clean up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine

from procbot.config import AppConfig, EnvSettings, load_app_config
from procbot.context import AppContext

if TYPE_CHECKING:
    from aiogram import Bot
from procbot.crypto import SecretBox, SecretBoxError
from procbot.logging_setup import CorrelationMiddleware, correlation_scope, setup_logging

log = logging.getLogger("procbot.main")

EXIT_OK = 0
EXIT_CRASH = 1
EXIT_CONFIG = 2
EXIT_ALREADY_RUNNING = 3

LEASE_TTL_SECONDS = 60.0
SHUTDOWN_TIMEOUT_SECONDS = 30.0


class ConfigError(Exception):
    pass


# --------------------------------------------------------------------------- migrations


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[2], Path.cwd()):
        if (candidate / "migrations" / "env.py").exists():
            return candidate
    raise ConfigError("migrations/ directory not found (expected next to src/)")


def alembic_config(db_path: Path) -> Config:
    root = _project_root()
    ini = root / "alembic.ini"
    cfg = Config(str(ini)) if ini.exists() else Config()
    cfg.set_main_option("script_location", str(root / "migrations"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{Path(db_path)}")
    cfg.attributes["configure_logger"] = False  # keep our JSON logging
    return cfg


def _upgrade(db_path: Path) -> str | None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    command.upgrade(alembic_config(db_path), "head")
    return current_revision(db_path)


def upgrade_to_head(db_path: Path) -> str | None:
    """Blocking. env.py runs its own event loop, so from async code prefer
    `await asyncio.to_thread(upgrade_to_head, path)`; if called inside a running
    loop anyway, the migration runs in a worker thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _upgrade(db_path)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_upgrade, db_path).result()


def current_revision(db_path: Path) -> str | None:
    engine = create_engine(f"sqlite:///{Path(db_path)}")
    try:
        with engine.connect() as conn:
            return MigrationContext.configure(conn).get_current_revision()
    finally:
        engine.dispose()


def head_revision(db_path: Path) -> str | None:
    return ScriptDirectory.from_config(alembic_config(db_path)).get_current_head()


# --------------------------------------------------------------------------- validation


def validate_env(env: EnvSettings, *, need_token: bool = True) -> list[str]:
    """Human-readable problems with the environment (Russian / English)."""
    errors: list[str] = []
    if need_token:
        token = env.telegram_bot_token.strip()
        if not token or ":" not in token:
            errors.append(
                "TELEGRAM_BOT_TOKEN не задан или неверного формата (получите у @BotFather) / "
                "TELEGRAM_BOT_TOKEN is missing or malformed"
            )
    if env.owner_telegram_id <= 0:
        errors.append(
            "OWNER_TELEGRAM_ID не задан: укажите числовой Telegram ID владельца / "
            "OWNER_TELEGRAM_ID is missing (numeric Telegram user id of the owner)"
        )
    errors.extend(_master_key_errors(env))
    return errors


def _master_key_errors(env: EnvSettings) -> list[str]:
    if not env.master_key:
        return [
            "MASTER_KEY не задан: сгенерируйте командой `procbot gen-key` / "
            "MASTER_KEY is missing: generate one with `procbot gen-key`"
        ]
    try:
        SecretBox(env.master_key)
    except SecretBoxError:
        return [
            "MASTER_KEY неверного формата (нужен Fernet-ключ из `procbot gen-key`) / "
            "MASTER_KEY is not a valid Fernet key"
        ]
    return []


def load_settings() -> tuple[EnvSettings, AppConfig]:
    env = EnvSettings()
    try:
        config = load_app_config(env.app_config_path)
    except Exception as exc:
        raise ConfigError(f"cannot read {env.app_config_path}: {exc}") from exc
    return env, config


# --------------------------------------------------------------------------- run


async def prepare(ctx: AppContext) -> None:
    """Step 5 of scenario G: wiring, recovery, session restore (before polling)."""
    from procbot.services.recovery import run_recovery
    from procbot.services.sessions import restore_sessions
    from procbot.wiring import register_all

    register_all(ctx)
    with correlation_scope(prefix="recovery"):
        await run_recovery(ctx)
        restored = await restore_sessions(ctx)
    log.info("startup recovery finished", extra={"count": restored})


async def run_bot(
    env: EnvSettings,
    config: AppConfig,
    *,
    bot: Bot | None = None,
    stop: asyncio.Event | None = None,
    install_signals: bool = True,
) -> int:
    """Run until SIGTERM/SIGINT (or `stop` is set). `bot` is injectable for tests."""
    from aiogram import Bot

    from procbot.clock import SystemClock
    from procbot.db.session import Database
    from procbot.http_server import mark_started, start_http_server
    from procbot.services.instance_lock import (
        InstanceLock,
        InstanceLockError,
        acquire_poller_lease,
        lease_heartbeat,
        release_poller_lease,
    )
    from procbot.telegram.app import build_dispatcher
    from procbot.telegram.messenger import AiogramMessenger

    errors = validate_env(env)
    if errors:
        for e in errors:
            print(f"config error: {e}", file=sys.stderr)
        return EXIT_CONFIG

    lock = InstanceLock(env.lock_path)
    try:
        lock.acquire()
    except InstanceLockError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_ALREADY_RUNNING

    db: Database | None = None
    ctx: AppContext | None = None
    runner = None
    lease_taken = False
    try:
        revision = await asyncio.to_thread(upgrade_to_head, env.database_path)
        log.info("database migrated", extra={"status": revision or "none"})

        if bot is None:
            bot = Bot(token=env.telegram_bot_token)
        db = Database.for_path(env.database_path)
        ctx = AppContext(
            env=env,
            config=config,
            db=db,
            clock=SystemClock(),
            secrets=SecretBox(env.master_key),
            messenger=AiogramMessenger(bot),
        )
        try:
            await acquire_poller_lease(
                ctx, ttl_seconds=LEASE_TTL_SECONDS, wait_seconds=LEASE_TTL_SECONDS + 5
            )
        except InstanceLockError as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_ALREADY_RUNNING
        lease_taken = True
        mark_started(ctx)

        await prepare(ctx)

        from procbot.services.scheduler import Scheduler

        dp = build_dispatcher(ctx)
        dp.update.outer_middleware(CorrelationMiddleware())
        runner = await start_http_server(ctx, env.http_host, env.http_port)

        stop = stop or asyncio.Event()
        loop = asyncio.get_running_loop()
        if install_signals:
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    loop.add_signal_handler(sig, stop.set)
                except (NotImplementedError, RuntimeError):  # pragma: no cover - non-unix
                    pass

        poller = asyncio.create_task(
            dp.start_polling(
                bot,
                handle_as_tasks=False,
                handle_signals=False,
                close_bot_session=False,
                allowed_updates=dp.resolve_used_update_types(),
            ),
            name="poller",
        )
        scheduler = asyncio.create_task(Scheduler(ctx).run_forever(stop), name="scheduler")
        heartbeat = asyncio.create_task(
            lease_heartbeat(ctx, stop, ttl_seconds=LEASE_TTL_SECONDS), name="lease-heartbeat"
        )
        stopper = asyncio.create_task(stop.wait(), name="stop-wait")
        log.info("procbot started", extra={"instance_id": ctx.instance_id})

        done, _ = await asyncio.wait(
            {poller, scheduler, heartbeat, stopper}, return_when=asyncio.FIRST_COMPLETED
        )
        exit_code = EXIT_OK
        for task in done:
            if task is stopper or task.cancelled():
                continue
            exc = task.exception()
            if exc is not None:
                log.error("task crashed: %s", task.get_name(), exc_info=exc)
                exit_code = EXIT_CRASH
            elif task is heartbeat:
                if task.result():
                    exit_code = EXIT_ALREADY_RUNNING  # lease lost to another instance
            else:
                log.error("task exited unexpectedly: %s", task.get_name())
                exit_code = EXIT_CRASH

        log.info("shutting down")
        stop.set()
        try:
            await dp.stop_polling()
        except RuntimeError:
            pass  # polling already finished
        tasks: list[asyncio.Task[Any]] = [poller, scheduler, heartbeat, stopper]
        pending = [t for t in tasks if not t.done()]
        if pending:
            _, still = await asyncio.wait(pending, timeout=SHUTDOWN_TIMEOUT_SECONDS)
            for t in still:
                t.cancel()
            await asyncio.gather(*still, return_exceptions=True)
        return exit_code
    finally:
        if runner is not None:
            await runner.cleanup()
        if ctx is not None and lease_taken:
            try:
                await release_poller_lease(ctx)
            except Exception:
                log.exception("cannot release poller lease")
        if bot is not None:
            await bot.session.close()
        if db is not None:
            await db.dispose()
        lock.release()


# --------------------------------------------------------------------------- CLI commands


def _cmd_run(_args: argparse.Namespace) -> int:
    env, config = load_settings()
    setup_logging(env.log_level)
    try:
        return asyncio.run(run_bot(env, config))
    except Exception:
        log.exception("procbot crashed during startup or shutdown")
        return EXIT_CRASH


def _cmd_migrate(_args: argparse.Namespace) -> int:
    env = EnvSettings()
    setup_logging(env.log_level)
    revision = upgrade_to_head(env.database_path)
    print(f"database {env.database_path} is at revision {revision}")
    return EXIT_OK


def _cmd_gen_key(_args: argparse.Namespace) -> int:
    print(SecretBox.generate_key())
    return EXIT_OK


def _backup_key(env: EnvSettings, config: AppConfig) -> str | None:
    if not config.backup.encrypt:
        return None
    problems = _master_key_errors(env)
    if problems:
        raise ConfigError("; ".join(problems))
    return env.master_key


def _cmd_backup(_args: argparse.Namespace) -> int:
    from procbot.clock import SystemClock
    from procbot.db.session import Database
    from procbot.services.backup import make_backup

    env, config = load_settings()
    setup_logging(env.log_level)
    key = _backup_key(env, config)
    if not env.database_path.exists():
        raise ConfigError(f"database not found: {env.database_path}")

    async def _go() -> int:
        db = Database.for_path(env.database_path)
        try:
            bf, manifest_id = await make_backup(
                db,
                db_path=env.database_path,
                backup_dir=env.backup_dir,
                master_key=key,
                now=SystemClock().now(),
            )
        finally:
            await db.dispose()
        print(
            json.dumps(
                {
                    "path": str(bf.path),
                    "sha256": bf.sha256,
                    "size_bytes": bf.size_bytes,
                    "encrypted": bf.encrypted,
                    "manifest_id": manifest_id,
                },
                ensure_ascii=False,
            )
        )
        return EXIT_OK

    return asyncio.run(_go())


def _cmd_verify(args: argparse.Namespace) -> int:
    from procbot.services.backup import BackupError, verify_backup

    env = EnvSettings()
    try:
        res = verify_backup(Path(args.file), env.master_key or None)
    except BackupError as exc:
        print(f"backup INVALID: {exc}", file=sys.stderr)
        return EXIT_CRASH
    print(
        json.dumps(
            {
                "path": str(res.path),
                "ok": True,
                "sha256": res.sha256,
                "encrypted": res.encrypted,
                "tables": res.tables,
                "schema_revision": res.schema_revision,
            },
            ensure_ascii=False,
        )
    )
    return EXIT_OK


def _cmd_restore(args: argparse.Namespace) -> int:
    from procbot.clock import SystemClock
    from procbot.services.backup import BackupError, restore_backup
    from procbot.services.instance_lock import lock_is_held

    env = EnvSettings()
    target = Path(args.target) if args.target else env.database_path
    if target.resolve() == env.database_path.resolve() and lock_is_held(env.lock_path):
        print(
            "бот запущен — остановите его перед восстановлением / "
            "the bot is running; stop it before restoring",
            file=sys.stderr,
        )
        return EXIT_ALREADY_RUNNING
    try:
        res = restore_backup(
            Path(args.file),
            target,
            env.master_key or None,
            force=args.force,
            now=SystemClock().now(),
        )
    except BackupError as exc:
        print(f"restore failed: {exc}", file=sys.stderr)
        return EXIT_CRASH
    print(
        json.dumps(
            {
                "target": str(res.target),
                "safety_copy": None if res.safety_copy is None else str(res.safety_copy),
                "schema_revision": res.schema_revision,
            },
            ensure_ascii=False,
        )
    )
    print(
        "restored; the next `procbot run` applies pending migrations",
        file=sys.stderr,
    )
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="procbot", description="Personal anti-procrastination bot")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("run", help="run the bot (default)").set_defaults(func=_cmd_run)
    sub.add_parser("migrate", help="apply database migrations").set_defaults(func=_cmd_migrate)
    sub.add_parser("gen-key", help="print a new MASTER_KEY").set_defaults(func=_cmd_gen_key)
    sub.add_parser("backup", help="create an encrypted backup now").set_defaults(func=_cmd_backup)
    r = sub.add_parser("restore", help="restore a backup into a database file")
    r.add_argument("file")
    r.add_argument("--target", help="database path (default: DATABASE_PATH)")
    r.add_argument("--force", action="store_true", help="overwrite an existing database")
    r.set_defaults(func=_cmd_restore)
    v = sub.add_parser("verify-backup", help="decrypt and integrity-check a backup")
    v.add_argument("file")
    v.set_defaults(func=_cmd_verify)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    func = getattr(args, "func", _cmd_run)
    try:
        return int(func(args))
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except KeyboardInterrupt:  # pragma: no cover
        return EXIT_OK


def cli() -> None:
    sys.exit(main())


if __name__ == "__main__":  # pragma: no cover
    cli()
