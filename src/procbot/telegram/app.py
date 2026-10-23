"""Builds the aiogram Dispatcher: middlewares + routers in a fixed order.

Router order matters: specific handlers first, free-text fallback last.
"""

from __future__ import annotations

from aiogram import Dispatcher

from procbot.context import AppContext
from procbot.logging_setup import CorrelationMiddleware
from procbot.telegram.handlers import (
    ai_keys,
    approvals,
    calendar,
    chat,
    day,
    goals,
    habits,
    memory,
    procrastination,
    reminders,
    reports,
    sessions,
    settings,
    start,
    status,
    tasks,
    voice,
)
from procbot.telegram.middlewares import AuthMiddleware, DedupMiddleware
from procbot.telegram.storage import SQLiteStorage

# Order: commands/onboarding, buttons of each module, voice, then free text.
ROUTER_MODULES = (
    start,
    status,
    settings,
    approvals,
    reminders,
    sessions,
    procrastination,
    tasks,
    day,
    habits,
    goals,
    memory,
    reports,
    calendar,
    ai_keys,
    voice,
    chat,  # must stay last: catches any remaining text
)


def build_dispatcher(ctx: AppContext) -> Dispatcher:
    dp = Dispatcher(storage=SQLiteStorage(ctx))
    dp["ctx"] = ctx
    dp.update.outer_middleware(CorrelationMiddleware())  # first: ids on every log line
    dp.update.outer_middleware(AuthMiddleware(ctx))
    dp.update.outer_middleware(DedupMiddleware(ctx))
    for module in ROUTER_MODULES:
        # Module-level routers are singletons; allow re-attaching them to a fresh
        # Dispatcher (tests build one per test, production builds one).
        module.router._parent_router = None  # type: ignore[attr-defined]
        dp.include_router(module.router)
    return dp
