"""Application context: the explicit dependency container passed to services."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from procbot.clock import Clock
from procbot.config import AppConfig, EnvSettings
from procbot.crypto import SecretBox
from procbot.db.session import Database

if TYPE_CHECKING:
    from procbot.telegram.messenger import Messenger


@dataclass
class AppContext:
    env: EnvSettings
    config: AppConfig
    db: Database
    clock: Clock
    secrets: SecretBox
    messenger: Messenger
    # Filled in by later stages; None means the integration is not configured.
    ai: Any = None  # procbot.ai.gateway.AIGateway
    calendar: Any = None  # procbot.services.calendar client factory
    instance_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    # Free-form registry for cross-module hooks (renderers, action appliers...)
    registry: dict[str, Any] = field(default_factory=dict)

    @property
    def owner_id(self) -> int:
        return self.env.owner_telegram_id
