"""Registers cross-module hooks (renderers, action appliers, periodic jobs).

Called once by main() and by tests that need the full app behaviour.
Each listed module exposes `register(ctx)`.
"""

from __future__ import annotations

import importlib

from procbot.context import AppContext

REGISTRANTS = (
    "procbot.services.reminders",
    "procbot.services.initiative",
    "procbot.services.tasks",
    "procbot.services.approvals",
    "procbot.services.sessions",
    "procbot.services.backup",
    "procbot.ai.jobs",
)


def register_all(ctx: AppContext) -> AppContext:
    if ctx.registry.get("_wired"):
        return ctx
    for name in REGISTRANTS:
        importlib.import_module(name).register(ctx)
    ctx.registry["_wired"] = True
    return ctx
