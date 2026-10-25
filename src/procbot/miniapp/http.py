"""Shared plumbing for the Mini App API: owner check, JSON body, input parsing
and mapping of service errors to JSON answers with Russian text."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import date, time
from typing import Any

from aiohttp import web

from procbot.context import AppContext
from procbot.miniapp.auth import InitDataError, verify_init_data
from procbot.services import actions, sessions
from procbot.services import tasks as task_service

CtxKey = web.AppKey[AppContext]
Handler = Callable[[web.Request, AppContext], Awaitable[web.StreamResponse]]


def error(status: int, message: str, **extra: Any) -> web.Response:
    return web.json_response({"error": message, **extra}, status=status)


def authorize(request: web.Request, ctx: AppContext) -> None:
    header = request.headers.get("Authorization", "")
    scheme, _, raw = header.partition(" ")
    if scheme.lower() != "tma" or not raw:
        raise web.HTTPUnauthorized(text="initData required")
    try:
        verify_init_data(
            raw,
            bot_token=ctx.env.telegram_bot_token,
            owner_id=ctx.owner_id,
            now=ctx.clock.now(),
        )
    except InitDataError as exc:
        raise web.HTTPForbidden(text=str(exc)) from exc


async def body(request: web.Request) -> dict[str, Any]:
    if not request.can_read_body:
        return {}
    try:
        data = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise web.HTTPBadRequest(text="bad json") from exc
    if not isinstance(data, dict):
        raise web.HTTPBadRequest(text="json object expected")
    return data


def int_id(request: web.Request, name: str) -> int:
    try:
        return int(request.match_info[name])
    except ValueError as exc:
        raise web.HTTPBadRequest(text=f"bad {name}") from exc


def parse_date(value: Any, field: str) -> date | None:
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"Неверная дата: {field}") from exc


def parse_hhmm(value: Any, field: str) -> time | None:
    if value in (None, ""):
        return None
    try:
        t = time.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"Неверное время: {field}") from exc
    return t.replace(second=0, microsecond=0, tzinfo=None)


def parse_minutes(value: Any, field: str, *, high: int = 240) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise ValueError(f"Нужно число минут: {field}")
    try:
        n = int(value)
    except ValueError as exc:
        raise ValueError(f"Нужно число минут: {field}") from exc
    if not 0 <= n <= high:
        raise ValueError(f"Значение вне диапазона 0–{high}: {field}")
    return n


def route(key: CtxKey, fn: Handler) -> Callable[[web.Request], Awaitable[web.StreamResponse]]:
    """Auth + mapping of service errors to JSON answers with Russian text."""

    async def wrapped(request: web.Request) -> web.StreamResponse:
        ctx = request.app[key]
        authorize(request, ctx)
        try:
            return await fn(request, ctx)
        except sessions.ActiveSessionExists as exc:
            return error(409, "Уже идёт другая рабочая сессия.", session_id=exc.session_id)
        except sessions.SettingsMissing as exc:
            return error(409, "Для этого режима не заданы длительности.", missing=list(exc.fields))
        except sessions.CriteriaCheckRequired:
            return error(409, "Сначала сверь результат с критерием готовности.", criteria=True)
        except sessions.SessionError as exc:
            return error(409, str(exc))
        except task_service.TaskNotFound:
            return error(404, "Задача не найдена.")
        except actions.ActionError as exc:
            return error(400, str(exc))
        except LookupError:
            return error(404, "Не найдено.")
        except ValueError as exc:
            return error(400, str(exc))

    return wrapped
