"""Gateway configuration (config/ai.yaml), validated with pydantic.

Nothing here names a real model or a real price: providers, models and budget
are open question 7. Budget limits default to None = accounting only.
See config/ai.example.yaml for an annotated example.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from procbot.ai.prompting import DATA_CATEGORIES
from procbot.ai.types import StructuredStrategy, Tier

Provider = Literal["openai", "openai_compatible", "anthropic", "gemini", "fake"]
PROVIDERS: tuple[str, ...] = ("openai", "openai_compatible", "anthropic", "gemini", "fake")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CredentialConfig(_Strict):
    provider: Provider
    base_url: str | None = None
    # Name of the env variable holding the key; None = no key (local servers).
    api_key_env: str | None = None
    spend_limit_usd: float | None = Field(default=None, ge=0)
    # Privacy (architecture proposal, not an approved requirement): None = every
    # data category may be sent. A list = only these categories may be sent.
    allowed_data: list[str] | None = None
    # Extra HTTP headers (e.g. OpenRouter attribution). Never put keys here.
    headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("allowed_data", mode="before")
    @classmethod
    def _allowed(cls, v: Any) -> Any:
        if isinstance(v, dict):  # {"memory": false, "message": true} form
            return [k for k, ok in v.items() if ok]
        return v

    @field_validator("allowed_data")
    @classmethod
    def _known(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            unknown = set(v) - DATA_CATEGORIES
            if unknown:
                raise ValueError(f"unknown data categories: {sorted(unknown)}")
        return v

    @model_validator(mode="after")
    def _base_url_needed(self) -> CredentialConfig:
        if self.provider == "openai_compatible" and not self.base_url:
            raise ValueError("openai_compatible credentials need base_url")
        return self


class ModelConfig(_Strict):
    # One credential label, or several (tried in order: key rotation).
    credential: str | list[str]
    model: str
    structured_output: StructuredStrategy = StructuredStrategy.NONE
    max_context_tokens: int | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    audio_input: bool = False
    vision: bool = False
    # USD; None = unknown -> cost is recorded as estimated (0 until priced).
    price_in_per_mtok: float | None = Field(default=None, ge=0)
    price_out_per_mtok: float | None = Field(default=None, ge=0)
    price_per_audio_minute: float | None = Field(default=None, ge=0)
    timeout_seconds: float | None = Field(default=None, gt=0)

    @property
    def credentials(self) -> list[str]:
        return [self.credential] if isinstance(self.credential, str) else list(self.credential)


class EscalationConfig(_Strict):
    # After a model fails validation twice (answer + repair) -> try the next tier.
    validation_failed_twice: Literal["next_tier", "none"] = "next_tier"
    # Tier used when the caller passes escalate=True.
    caller_escalate_to: Tier = Tier.STRONG


class BudgetConfig(_Strict):
    # Open question 7: not decided -> None = accounting only, no limits.
    monthly_soft_usd: float | None = Field(default=None, ge=0)
    monthly_hard_usd: float | None = Field(default=None, ge=0)
    on_soft_limit: Literal["downgrade_one_tier", "none"] = "downgrade_one_tier"
    on_hard_limit: Literal["no_ai_mode"] = "no_ai_mode"


class LimitsConfig(_Strict):
    # Technical defaults (not product decisions); tune in ai.yaml.
    max_attempts_per_call: int = Field(default=6, ge=1)
    call_timeout_seconds: float = Field(default=90.0, gt=0)
    request_timeout_seconds: float = Field(default=40.0, gt=0)
    retries_per_candidate: int = Field(default=2, ge=0)  # on timeouts/5xx/network
    backoff_base_seconds: float = Field(default=1.0, ge=0)
    backoff_max_seconds: float = Field(default=20.0, ge=0)
    cooldown_base_seconds: float = Field(default=60.0, ge=0)  # 429 without Retry-After
    cooldown_max_seconds: float = Field(default=3600.0, ge=0)


class NoAIModeConfig(_Strict):
    # Flip to no-AI mode when calls keep failing for this long / this many times.
    failed_calls_threshold: int = Field(default=3, ge=1)
    min_failure_seconds: float = Field(default=300.0, ge=0)
    probe_interval_seconds: float = Field(default=300.0, gt=0)


class FeaturesConfig(_Strict):
    # Open question: adding keys via chat. Default off -> keys only via .env.
    allow_keys_via_chat: bool = False
    # Anomaly warning: today's spend > factor x trailing daily average.
    # UNCONFIRMED technical default.
    anomaly_factor: float | None = Field(default=3.0, gt=1)
    anomaly_trailing_days: int = Field(default=7, ge=1)


class AIConfig(_Strict):
    credentials: dict[str, CredentialConfig] = Field(default_factory=dict)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    tiers: dict[Tier, list[str]] = Field(default_factory=dict)
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    no_ai: NoAIModeConfig = Field(default_factory=NoAIModeConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)

    @field_validator("credentials", "models", "tiers", mode="before")
    @classmethod
    def _none_is_empty(cls, v: Any) -> Any:
        return {} if v is None else v

    @model_validator(mode="after")
    def _refs(self) -> AIConfig:
        for alias, m in self.models.items():
            for label in m.credentials:
                if label not in self.credentials:
                    raise ValueError(f"model {alias}: unknown credential {label!r}")
        for tier, aliases in self.tiers.items():
            for alias in aliases:
                if alias not in self.models:
                    raise ValueError(f"tier {tier}: unknown model alias {alias!r}")
                if tier == Tier.STT and not self.models[alias].audio_input:
                    raise ValueError(f"tier stt: model {alias!r} lacks audio_input: true")
        return self

    @property
    def configured(self) -> bool:
        return any(self.tiers.get(t) for t in Tier)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


def load_ai_config(path: Path | None) -> AIConfig | None:
    """None when the file does not exist (the bot then works without AI)."""
    if path is None or not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AIConfig.model_validate(data)
