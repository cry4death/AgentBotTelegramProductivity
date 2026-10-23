"""Credential registry: encrypted keys, health and the per-key circuit breaker.

- Keys come from the environment (`api_key_env` in ai.yaml) and are imported
  into `ai_credentials` encrypted with the master key (SecretBox); keys added via
  chat (if allowed) are stored the same way with source=chat.
- Several keys per provider: a model may list several credentials, and keys
  added via chat join the pool of every model whose credential has the same
  provider + base_url.
- Circuit breaker: 429/quota -> cooling_down until Retry-After (or exponential
  backoff); 401/403 -> disabled + ONE owner notice (disabled_notified flag);
  success closes the breaker. Disabled keys come back only when the owner
  enables them or the env key changes.
- Only labels ever reach logs and ai_calls; decrypted keys live in memory only.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select

from procbot.ai.config import AIConfig, ModelConfig
from procbot.ai.errors import ProviderError
from procbot.ai.types import Credential
from procbot.context import AppContext
from procbot.crypto import SecretBoxError, last4
from procbot.db.models import AICredential
from procbot.domain.enums import AIErrorKind, CredentialStatus

log = logging.getLogger(__name__)

Notifier = Callable[[str, str], Awaitable[None]]  # (text, dedup_key)


def _norm_url(url: str | None) -> str:
    return (url or "").rstrip("/").lower()


@dataclass(frozen=True)
class CredentialState:
    id: int
    label: str
    provider: str
    base_url: str | None
    status: CredentialStatus
    cooldown_until: datetime | None
    spend_limit_usd: float | None
    allowed_data: frozenset[str] | None  # None = no restriction
    has_key: bool
    source: str
    key_last4: str
    last_error: str | None

    def available(self, now: datetime) -> bool:
        if not self.has_key or self.status == CredentialStatus.DISABLED:
            return False
        if self.status == CredentialStatus.COOLING_DOWN:
            return self.cooldown_until is None or self.cooldown_until <= now
        return True

    def allows(self, categories: frozenset[str]) -> bool:
        return self.allowed_data is None or categories <= self.allowed_data


def default_environ(env_file: str | os.PathLike[str] | None = ".env") -> dict[str, str]:
    """os.environ plus the .env file (pydantic-settings does not export extras)."""
    merged: dict[str, str] = {}
    if env_file and os.path.exists(env_file):
        try:
            from dotenv import dotenv_values

            merged.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
        except ImportError:  # pragma: no cover - python-dotenv ships with pydantic-settings
            pass
    merged.update(os.environ)
    return merged


class CredentialRegistry:
    def __init__(
        self,
        ctx: AppContext,
        config: AIConfig,
        *,
        notify: Notifier,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.ctx = ctx
        self.config = config
        self.notify = notify
        self.environ = environ if environ is not None else default_environ()
        self.states: dict[str, CredentialState] = {}
        self._keys: dict[int, tuple[str, str]] = {}  # id -> (encrypted, plaintext)

    # ------------------------------------------------------------------ loading

    async def sync(self) -> None:
        """Import env keys into the DB (encrypted) and load every credential."""
        now = self.ctx.clock.now()
        box = self.ctx.secrets
        async with self.ctx.db.transaction() as s:
            rows = {r.label: r for r in (await s.execute(select(AICredential))).scalars()}
            for label, cc in self.config.credentials.items():
                row = rows.get(label)
                key: str | None
                if cc.api_key_env is None:
                    key = ""  # local server without a key
                else:
                    key = (self.environ.get(cc.api_key_env) or "").strip() or None
                allowed = {c: True for c in cc.allowed_data} if cc.allowed_data is not None else {}
                if row is None:
                    row = AICredential(
                        label=label,
                        provider=cc.provider,
                        base_url=cc.base_url,
                        encrypted_key=box.encrypt(key or ""),
                        key_last4=last4(key) if key else "",
                        status=CredentialStatus.ACTIVE.value,
                        consecutive_failures=0,
                        spend_limit_usd=cc.spend_limit_usd,
                        allowed_data=allowed,
                        source="env",
                        last_error=None
                        if key is not None
                        else f"переменная {cc.api_key_env} не задана",
                        disabled_notified=False,
                        created_at=now,
                        updated_at=now,
                    )
                    s.add(row)
                    continue
                row.provider = cc.provider
                row.base_url = cc.base_url
                row.spend_limit_usd = cc.spend_limit_usd
                row.allowed_data = allowed
                if key is not None and row.source == "env" and self._decrypt(row) != key:
                    # New / rotated env key: reset health.
                    row.encrypted_key = box.encrypt(key)
                    row.key_last4 = last4(key) if key else ""
                    row.status = CredentialStatus.ACTIVE.value
                    row.cooldown_until = None
                    row.consecutive_failures = 0
                    row.disabled_notified = False
                    row.last_error = None
                elif key is None and not self._decrypt(row):
                    row.last_error = f"переменная {cc.api_key_env} не задана"
                row.updated_at = now
        await self.refresh()

    async def refresh(self) -> dict[str, CredentialState]:
        async with self.ctx.db.transaction() as s:
            rows = list((await s.execute(select(AICredential))).scalars())
        states: dict[str, CredentialState] = {}
        for r in rows:
            cc = self.config.credentials.get(r.label)
            needs_key = cc is None or cc.api_key_env is not None
            plaintext = self._decrypt(r)
            allowed = frozenset(k for k, v in (r.allowed_data or {}).items() if v)
            states[r.label] = CredentialState(
                id=r.id,
                label=r.label,
                provider=r.provider,
                base_url=r.base_url,
                status=CredentialStatus(r.status),
                cooldown_until=r.cooldown_until,
                spend_limit_usd=r.spend_limit_usd,
                allowed_data=allowed if r.allowed_data else None,
                has_key=bool(plaintext) or not needs_key,
                source=r.source,
                key_last4=r.key_last4,
                last_error=r.last_error,
            )
        self.states = states
        return states

    def _decrypt(self, row: AICredential) -> str:
        cached = self._keys.get(row.id) if row.id is not None else None
        if cached is not None and cached[0] == row.encrypted_key:
            return cached[1]
        try:
            plaintext = self.ctx.secrets.decrypt(row.encrypted_key)
        except SecretBoxError:
            log.error("cannot decrypt AI credential", extra={"credential": row.label})
            return ""
        if row.id is not None:
            self._keys[row.id] = (row.encrypted_key, plaintext)
        return plaintext

    # ------------------------------------------------------------------ lookup

    def pool_for(self, model: ModelConfig) -> list[str]:
        """Credential labels a model may use, in order: configured, then chat-added
        keys of the same provider + base_url."""
        labels = list(model.credentials)
        first = self.config.credentials[labels[0]]
        for st in self.states.values():
            if (
                st.source == "chat"
                and st.label not in labels
                and st.provider == first.provider
                and _norm_url(st.base_url) == _norm_url(first.base_url)
            ):
                labels.append(st.label)
        return labels

    async def auth(self, label: str) -> Credential:
        st = self.states[label]
        async with self.ctx.db.transaction() as s:
            row = await s.get(AICredential, st.id)
        key = self._decrypt(row) if row is not None else ""
        cc = self.config.credentials.get(label)
        return Credential(
            id=st.id,
            label=label,
            provider=st.provider,
            api_key=key,
            base_url=st.base_url,
            extra_headers=cc.headers if cc else None,
        )

    # ------------------------------------------------------------------ breaker

    async def record_success(self, label: str) -> None:
        st = self.states.get(label)
        if st is None or (
            st.status == CredentialStatus.ACTIVE
            and st.last_error is None
            and st.cooldown_until is None
        ):
            return
        async with self.ctx.db.transaction() as s:
            row = await s.get(AICredential, st.id)
            if row is None or row.status == CredentialStatus.DISABLED.value:
                return
            row.status = CredentialStatus.ACTIVE.value
            row.cooldown_until = None
            row.consecutive_failures = 0
            row.last_error = None
            row.updated_at = self.ctx.clock.now()
        await self.refresh()

    async def record_failure(self, label: str, err: ProviderError, *, notify: bool = True) -> None:
        st = self.states.get(label)
        if st is None:
            return
        now = self.ctx.clock.now()
        lim = self.config.limits
        notice: str | None = None
        async with self.ctx.db.transaction() as s:
            row = await s.get(AICredential, st.id)
            if row is None:
                return
            row.consecutive_failures = (row.consecutive_failures or 0) + 1
            row.last_error = f"{err.kind.value} {err.http_status or ''}".strip()
            row.updated_at = now
            if err.kind == AIErrorKind.RATE_LIMIT:
                if err.retry_after is not None:
                    seconds = min(err.retry_after, lim.cooldown_max_seconds)
                else:
                    seconds = min(
                        lim.cooldown_base_seconds * 2 ** max(0, row.consecutive_failures - 1),
                        lim.cooldown_max_seconds,
                    )
                row.status = CredentialStatus.COOLING_DOWN.value
                row.cooldown_until = now + timedelta(seconds=seconds)
                log.warning(
                    "AI credential cooling down",
                    extra={"credential": label, "seconds": seconds},
                )
            elif err.kind == AIErrorKind.AUTH:
                row.status = CredentialStatus.DISABLED.value
                row.cooldown_until = None
                log.error("AI credential disabled (auth)", extra={"credential": label})
                if not row.disabled_notified:
                    row.disabled_notified = True
                    notice = (
                        None
                        if not notify
                        else (
                            f"🔑 Ключ ИИ «{label}» ({row.provider}, ****{row.key_last4}) отклонён "
                            f"провайдером (HTTP {err.http_status or '401/403'}) и отключён. "
                            "Проверьте ключ и включите его снова в /ai."
                        )
                    )
        await self.refresh()
        if notice is not None:
            await self.notify(notice, f"ai-key-disabled:{st.id}:{now:%Y%m%d%H%M%S}")

    async def set_enabled(self, credential_id: int, enabled: bool) -> CredentialState | None:
        now = self.ctx.clock.now()
        async with self.ctx.db.transaction() as s:
            row = await s.get(AICredential, credential_id)
            if row is None:
                return None
            if enabled:
                row.status = CredentialStatus.ACTIVE.value
                row.cooldown_until = None
                row.consecutive_failures = 0
                row.disabled_notified = False
                row.last_error = None
            else:
                row.status = CredentialStatus.DISABLED.value
                row.last_error = "отключён владельцем"
                row.disabled_notified = True  # no auto notice for a manual disable
            row.updated_at = now
            label = row.label
        await self.refresh()
        return self.states.get(label)

    async def add_key(
        self, *, provider: str, api_key: str, base_url: str | None = None, label: str | None = None
    ) -> CredentialState:
        """Store a key added via chat (encrypted). Returns the new state."""
        now = self.ctx.clock.now()
        async with self.ctx.db.transaction() as s:
            existing = {r.label for r in (await s.execute(select(AICredential))).scalars()}
            if label is None:
                n = 1
                while f"{provider}-chat-{n}" in existing:
                    n += 1
                label = f"{provider}-chat-{n}"
            elif label in existing:
                raise ValueError(f"credential {label!r} already exists")
            s.add(
                AICredential(
                    label=label,
                    provider=provider,
                    base_url=base_url,
                    encrypted_key=self.ctx.secrets.encrypt(api_key),
                    key_last4=last4(api_key),
                    status=CredentialStatus.ACTIVE.value,
                    consecutive_failures=0,
                    allowed_data={},
                    source="chat",
                    disabled_notified=False,
                    created_at=now,
                    updated_at=now,
                )
            )
        await self.refresh()
        return self.states[label]
