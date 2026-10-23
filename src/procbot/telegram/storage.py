"""aiogram FSM storage backed by the conversation_states table, so dialogue state
survives restarts (scenario G)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aiogram.fsm.state import State
from aiogram.fsm.storage.base import BaseStorage, StateType, StorageKey

from procbot.context import AppContext
from procbot.db.models import ConversationState


def _key(key: StorageKey) -> str:
    return f"{key.bot_id}:{key.chat_id}:{key.user_id}:{key.thread_id or 0}:{key.destiny}"


class SQLiteStorage(BaseStorage):
    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx

    async def _row(self, s: Any, key: StorageKey) -> ConversationState:
        row = await s.get(ConversationState, _key(key))
        if row is None:
            row = ConversationState(
                key=_key(key), state=None, data={}, updated_at=self.ctx.clock.now()
            )
            s.add(row)
        return row

    async def set_state(self, key: StorageKey, state: StateType = None) -> None:
        value = state.state if isinstance(state, State) else state
        async with self.ctx.db.transaction() as s:
            row = await self._row(s, key)
            row.state = value
            row.updated_at = self.ctx.clock.now()

    async def get_state(self, key: StorageKey) -> str | None:
        async with self.ctx.db.transaction() as s:
            row = await s.get(ConversationState, _key(key))
            return None if row is None else row.state

    async def set_data(self, key: StorageKey, data: Mapping[str, Any]) -> None:
        async with self.ctx.db.transaction() as s:
            row = await self._row(s, key)
            row.data = dict(data)
            row.updated_at = self.ctx.clock.now()

    async def get_data(self, key: StorageKey) -> dict[str, Any]:
        async with self.ctx.db.transaction() as s:
            row = await s.get(ConversationState, _key(key))
            return {} if row is None else dict(row.data or {})

    async def close(self) -> None:
        return None
