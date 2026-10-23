"""CLI, startup sequence, single-instance guard, /healthz, logging."""

from __future__ import annotations

import asyncio
import io
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from aiogram import Bot
from aiogram.methods import GetUpdates, TelegramMethod
from aiohttp.test_utils import TestClient, TestServer
from cryptography.fernet import Fernet

from procbot.config import AppConfig, EnvSettings
from procbot.http_server import build_app, mark_started
from procbot.logging_setup import JsonFormatter, correlation_scope
from procbot.main import EXIT_ALREADY_RUNNING, EXIT_CONFIG, EXIT_OK, main, run_bot, validate_env
from procbot.services.instance_lock import (
    InstanceLock,
    InstanceLockError,
    acquire_poller_lease,
    lock_is_held,
    release_poller_lease,
)
from procbot.testing import OWNER_ID, TEST_MASTER_KEY, MockedBotSession, make_bot, make_context

pytestmark = pytest.mark.filterwarnings("ignore:Cannot correctly sort tables")

ENV_KEYS = (
    "TELEGRAM_BOT_TOKEN",
    "OWNER_TELEGRAM_ID",
    "MASTER_KEY",
    "DATABASE_PATH",
    "BACKUP_DIR",
    "LOCK_PATH",
    "APP_CONFIG_PATH",
)


@pytest.fixture
def clean_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)  # no stray .env
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "data" / "procbot.db"))
    monkeypatch.setenv("BACKUP_DIR", str(tmp_path / "backups"))
    monkeypatch.setenv("LOCK_PATH", str(tmp_path / "data" / "procbot.lock"))
    monkeypatch.setenv("APP_CONFIG_PATH", str(tmp_path / "app.yaml"))
    return tmp_path


class PollingSession(MockedBotSession):
    """getUpdates returns an empty batch after a short wait (like long polling)."""

    async def make_request(
        self, bot: Bot, method: TelegramMethod[Any], timeout: int | None = None
    ) -> Any:
        if isinstance(method, GetUpdates):
            self.requests.append(method)
            await asyncio.sleep(0.02)
            return []
        return await super().make_request(bot, method, timeout)


def _polling_bot() -> tuple[Bot, PollingSession]:
    session = PollingSession()
    return Bot(token="42:TEST", session=session), session


def _env(tmp_path: Path) -> EnvSettings:
    return EnvSettings(
        telegram_bot_token="42:TEST",
        owner_telegram_id=OWNER_ID,
        master_key=TEST_MASTER_KEY,
        database_path=tmp_path / "data" / "procbot.db",
        backup_dir=tmp_path / "backups",
        lock_path=tmp_path / "data" / "procbot.lock",
        app_config_path=tmp_path / "missing.yaml",
        http_host="127.0.0.1",
        http_port=0,
    )


def test_gen_key_prints_valid_fernet_key(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["gen-key"]) == EXIT_OK
    key = capsys.readouterr().out.strip()
    Fernet(key.encode())  # does not raise
    assert len(key) == 44


def test_validate_env_reports_missing_values(clean_env: Path) -> None:
    errors = validate_env(EnvSettings())
    joined = " ".join(errors)
    assert "TELEGRAM_BOT_TOKEN" in joined and "OWNER_TELEGRAM_ID" in joined
    assert "MASTER_KEY" in joined and "gen-key" in joined
    bad = EnvSettings(telegram_bot_token="1:x", owner_telegram_id=5, master_key="nope")
    assert [e for e in validate_env(bad) if "MASTER_KEY" in e]
    assert validate_env(_env(clean_env)) == []


def test_run_refuses_without_config(clean_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run"]) == EXIT_CONFIG
    assert "TELEGRAM_BOT_TOKEN" in capsys.readouterr().err
    assert not (clean_env / "data" / "procbot.db").exists()  # nothing touched


def test_file_lock_allows_only_one_holder(tmp_path: Path) -> None:
    path = tmp_path / "x.lock"
    first = InstanceLock(path)
    first.acquire()
    try:
        assert lock_is_held(path)
        with pytest.raises(InstanceLockError):
            InstanceLock(path).acquire()
    finally:
        first.release()
    assert not lock_is_held(path)
    with InstanceLock(path):
        pass


async def test_poller_lease_refuses_second_instance(tmp_path: Path) -> None:
    a = await make_context(tmp_path)
    b = await make_context(tmp_path / "b")
    b.db = a.db  # same database, different instance
    try:
        await acquire_poller_lease(a, ttl_seconds=60)
        with pytest.raises(InstanceLockError):
            await acquire_poller_lease(b, ttl_seconds=60, wait_seconds=0)
        await release_poller_lease(a)
        await acquire_poller_lease(b, ttl_seconds=60)
    finally:
        await a.db.dispose()


async def test_healthz_reports_version_uptime_and_db(ctx, clock) -> None:
    mark_started(ctx)
    clock.advance(seconds=90)
    async with TestClient(TestServer(build_app(ctx))) as client:
        resp = await client.get("/healthz")
        assert resp.status == 200
        body = await resp.json()
    assert body["status"] == "ok" and body["db"] is True
    assert body["uptime_seconds"] == 90.0 and body["version"]


async def test_full_startup_and_graceful_shutdown(tmp_path: Path) -> None:
    env = _env(tmp_path)
    bot, session = _polling_bot()
    stop = asyncio.Event()
    task = asyncio.create_task(run_bot(env, AppConfig(), bot=bot, stop=stop, install_signals=False))
    for _ in range(100):
        await asyncio.sleep(0.05)
        if any(isinstance(r, GetUpdates) for r in session.requests):  # polling started
            break
    assert lock_is_held(env.lock_path)
    # A second instance must refuse to start: long polling never runs twice.
    second = await run_bot(env, AppConfig(), bot=make_bot()[0], install_signals=False)
    assert second == EXIT_ALREADY_RUNNING
    stop.set()
    assert await asyncio.wait_for(task, timeout=15) == EXIT_OK
    assert not lock_is_held(env.lock_path)
    assert env.database_path.exists()

    from procbot.main import current_revision, head_revision

    assert current_revision(env.database_path) == head_revision(env.database_path)
    import sqlite3

    con = sqlite3.connect(env.database_path)
    try:
        assert con.execute("SELECT count(*) FROM job_leases WHERE name='poller'").fetchone() == (0,)
    finally:
        con.close()


def test_cli_migrate_backup_verify_restore(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MASTER_KEY", TEST_MASTER_KEY)
    assert main(["migrate"]) == EXIT_OK
    capsys.readouterr()
    assert main(["backup"]) == EXIT_OK
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    path = out["path"]
    assert out["encrypted"] is True
    assert main(["verify-backup", path]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["ok"] is True
    target = clean_env / "restored" / "procbot.db"
    assert main(["restore", path, "--target", str(target)]) == EXIT_OK
    assert target.exists()
    # Overwriting the live DB while the bot runs is refused.
    lock = InstanceLock(clean_env / "data" / "procbot.lock")
    lock.acquire()
    try:
        assert main(["restore", path, "--force"]) == EXIT_ALREADY_RUNNING
    finally:
        lock.release()
    monkeypatch.setenv("MASTER_KEY", Fernet.generate_key().decode())
    assert main(["verify-backup", path]) != EXIT_OK


def test_json_logs_redact_secrets_and_drop_texts() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("procbot.test.logging")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    token = "123456789:AAH" + "x" * 32
    try:
        with correlation_scope("u42"):
            logger.info(
                "calling https://api.telegram.org/bot%s/getMe",
                token,
                extra={"update_id": 42, "text": "секретное сообщение"},
            )
    finally:
        logger.removeHandler(handler)
    record = json.loads(stream.getvalue())
    assert record["correlation_id"] == "u42" and record["update_id"] == 42
    assert token not in stream.getvalue() and "[REDACTED]" in record["msg"]
    assert "секретное" not in stream.getvalue()


async def test_status_command(ctx, dp, bot_and_session, profile) -> None:
    from procbot.services.backup import run_backup
    from procbot.testing import message_update

    bot, session = bot_and_session
    await dp.feed_update(bot, message_update("/status"))
    text = session.sent_texts()[0]
    assert "База данных: ок" in text and "Последний бэкап: ещё не было" in text
    assert "ИИ не настроен" in text and "Запланированных напоминаний: 0" in text
    await run_backup(ctx)
    session.clear()
    await dp.feed_update(bot, message_update("/status"))
    assert "(проверен)" in session.sent_texts()[0]


def test_example_app_config_matches_defaults() -> None:
    from procbot.config import load_app_config

    root = Path(__file__).resolve().parents[1]
    loaded = load_app_config(root / "config" / "app.example.yaml")
    defaults = AppConfig()
    assert loaded.model_dump(exclude={"gate"}) == defaults.model_dump(exclude={"gate"})
    assert loaded.gate.model_dump(exclude={"categories"}) == defaults.gate.model_dump(
        exclude={"categories"}
    )
    for policy in loaded.gate.categories.values():
        assert policy.counts_toward_limit is None and policy.enabled


async def test_heartbeat_stops_process_when_lease_is_lost(ctx, clock) -> None:
    from datetime import timedelta

    from procbot.db.models import JobLease
    from procbot.services.instance_lock import lease_heartbeat

    await acquire_poller_lease(ctx, ttl_seconds=60)
    async with ctx.db.transaction() as s:
        row = await s.get(JobLease, "poller")
        assert row is not None
        row.owner = "someone-else"
        row.expires_at = clock.now() + timedelta(minutes=5)
    stop = asyncio.Event()
    lost = await asyncio.wait_for(lease_heartbeat(ctx, stop, ttl_seconds=0.5), timeout=5)
    assert lost is True and stop.is_set()
    stop2 = asyncio.Event()
    stop2.set()
    assert await lease_heartbeat(ctx, stop2, ttl_seconds=0.5) is False
