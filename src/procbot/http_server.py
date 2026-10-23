"""Small aiohttp server inside the bot process.

Stage A: `GET /healthz` for the Docker HEALTHCHECK and external monitoring
(open question 10: whether to alert on downtime is not decided; the endpoint is
ready for it). Stage D adds the Mini App API to the same app (`build_app`).
"""

from __future__ import annotations

import logging
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from aiohttp import web

from procbot.context import AppContext
from procbot.db.session import Database

log = logging.getLogger(__name__)

CTX_KEY: web.AppKey[AppContext] = web.AppKey("procbot_ctx", AppContext)
STARTED_AT_KEY = "started_at"  # ctx.registry key, set by procbot.main


def app_version() -> str:
    try:
        return version("procbot")
    except PackageNotFoundError:  # pragma: no cover - running from a bare checkout
        return "0.0.0+unknown"


def mark_started(ctx: AppContext) -> None:
    ctx.registry.setdefault(STARTED_AT_KEY, ctx.clock.now())


def uptime_seconds(ctx: AppContext) -> float | None:
    started: datetime | None = ctx.registry.get(STARTED_AT_KEY)
    if started is None:
        return None
    return max((ctx.clock.now() - started).total_seconds(), 0.0)


async def db_healthy(db: Database) -> bool:
    """Read-only probe that bypasses BEGIN IMMEDIATE (does not take a write lock)."""
    try:
        async with db.engine.connect() as conn:
            raw = await conn.get_raw_connection()
            driver: Any = raw.driver_connection
            cursor = await driver.execute("SELECT 1")
            row = await cursor.fetchone()
            await cursor.close()
            return row is not None and row[0] == 1
    except Exception:
        log.exception("database health probe failed")
        return False


async def healthz(request: web.Request) -> web.Response:
    ctx = request.app[CTX_KEY]
    ok = await db_healthy(ctx.db)
    uptime = uptime_seconds(ctx)
    body = {
        "status": "ok" if ok else "degraded",
        "version": app_version(),
        "uptime_seconds": None if uptime is None else round(uptime, 1),
        "db": ok,
        "instance_id": ctx.instance_id,
    }
    return web.json_response(body, status=200 if ok else 503)


async def google_oauth_callback(request: web.Request) -> web.Response:
    """Google OAuth redirect target (services.calendar_oauth, WEB mode)."""
    from procbot.services.calendar_oauth import handle_web_callback

    status, page = await handle_web_callback(request.app[CTX_KEY], dict(request.query))
    return web.Response(
        status=status,
        text=page,
        content_type="text/html",
        headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
    )


def build_app(ctx: AppContext) -> web.Application:
    app = web.Application()
    app[CTX_KEY] = ctx
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/oauth/google/callback", google_oauth_callback)  # stage C calendar
    from procbot.miniapp.api import add_routes  # stage D

    add_routes(app, CTX_KEY)
    return app


async def start_http_server(ctx: AppContext, host: str, port: int) -> web.AppRunner:
    runner = web.AppRunner(build_app(ctx), access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()
    log.info("http server listening", extra={"path": f"{host}:{port}"})
    return runner
