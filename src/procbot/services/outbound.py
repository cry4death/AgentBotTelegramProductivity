"""Rendering contract for bot-initiated messages.

A due reminder is turned into text by the renderer registered for its category.
Renderers run OUTSIDE any transaction (they may call the AI gateway) and return
None when the reminder became obsolete (e.g. the task already started).

The Outbound Gate (services.gate) is the only code that actually sends.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from aiogram.types import InlineKeyboardMarkup

from procbot.context import AppContext
from procbot.db.models import ReminderSchedule
from procbot.domain.enums import MessageCategory


@dataclass
class Outgoing:
    text: str
    reply_markup: InlineKeyboardMarkup | None = None


Renderer = Callable[[AppContext, ReminderSchedule], Awaitable[Outgoing | None]]


def register_renderer(ctx: AppContext, category: MessageCategory, renderer: Renderer) -> None:
    ctx.registry.setdefault("renderers", {})[category] = renderer


def get_renderer(ctx: AppContext, category: MessageCategory) -> Renderer | None:
    return ctx.registry.get("renderers", {}).get(category)
