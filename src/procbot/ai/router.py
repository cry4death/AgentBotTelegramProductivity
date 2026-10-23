"""Router: task + situation -> ordered candidates "model alias + credential".

Steps (architecture doc, "Маршрутизация и эскалация"):
1. tier of the task, then escalation (caller escalate=True -> strong;
   validation failed twice -> next tier, applied by the gateway mid-call);
2. budget: soft limit -> one tier down, hard limit -> AIUnavailable (gateway);
3. filter by capabilities, context size, credential health, allowed_data
   privacy and the per-key spend limit;
4. the gateway tries the candidates in order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from procbot.ai import registry as R
from procbot.ai.config import AIConfig, ModelConfig
from procbot.ai.credentials import CredentialRegistry
from procbot.ai.registry import AITaskSpec
from procbot.ai.types import TEXT_TIERS, StructuredStrategy, Tier

AUDIO_PROVIDERS = frozenset({"openai", "openai_compatible", "gemini", "fake"})


@dataclass(frozen=True)
class Candidate:
    alias: str
    model: ModelConfig
    label: str
    provider: str
    tier: Tier

    @property
    def key(self) -> tuple[str, str]:
        return (self.alias, self.label)


@dataclass
class Selection:
    candidates: list[Candidate] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # ("alias/label", reason)


def next_tier(tier: Tier) -> Tier | None:
    if tier not in TEXT_TIERS:
        return None
    i = TEXT_TIERS.index(tier)
    return TEXT_TIERS[i + 1] if i + 1 < len(TEXT_TIERS) else None


def lower_tier(tier: Tier) -> Tier:
    if tier not in TEXT_TIERS:
        return tier
    i = TEXT_TIERS.index(tier)
    return TEXT_TIERS[max(0, i - 1)]


class Router:
    def __init__(self, config: AIConfig, credentials: CredentialRegistry) -> None:
        self.config = config
        self.credentials = credentials

    def initial_tier(self, spec: AITaskSpec, *, escalate: bool) -> tuple[Tier, str | None]:
        if escalate and spec.tier in TEXT_TIERS:
            target = self.config.escalation.caller_escalate_to
            if TEXT_TIERS.index(target) > TEXT_TIERS.index(spec.tier):
                return target, "caller"
        return spec.tier, None

    def model_skip_reason(
        self, spec: AITaskSpec, model: ModelConfig, est_tokens: int
    ) -> str | None:
        caps = spec.required_capabilities
        if R.STRUCTURED in caps and model.structured_output == StructuredStrategy.NONE:
            return "no_structured_output"
        if R.AUDIO_INPUT in caps and not model.audio_input:
            return "no_audio_input"
        if R.VISION in caps and not model.vision:
            return "no_vision"
        if model.max_context_tokens is not None:
            out_tokens = min(
                spec.max_output_tokens, model.max_output_tokens or spec.max_output_tokens
            )
            if est_tokens + out_tokens > model.max_context_tokens:
                return "context_too_small"
        return None

    def select(
        self,
        tier: Tier,
        spec: AITaskSpec,
        *,
        categories: frozenset[str],
        est_tokens: int,
        now: datetime,
        spend_by_credential: dict[str, float],
        exclude: set[tuple[str, str]] | None = None,
    ) -> Selection:
        sel = Selection()
        exclude = exclude or set()
        for alias in self.config.tiers.get(tier, []):
            model = self.config.models[alias]
            reason = self.model_skip_reason(spec, model, est_tokens)
            if reason is not None:
                sel.skipped.append((alias, reason))
                continue
            pool = self.credentials.pool_for(model)
            first_cfg = self.config.credentials[model.credentials[0]]
            for label in pool:
                if (alias, label) in exclude:
                    continue
                st = self.credentials.states.get(label)
                where = f"{alias}/{label}"
                if st is None:
                    sel.skipped.append((where, "unknown_credential"))
                    continue
                if (
                    R.AUDIO_INPUT in spec.required_capabilities
                    and st.provider not in AUDIO_PROVIDERS
                ):
                    sel.skipped.append((where, "provider_has_no_audio"))
                    continue
                if not st.has_key:
                    sel.skipped.append((where, "no_key"))
                    continue
                if not st.available(now):
                    sel.skipped.append((where, f"credential_{st.status.value}"))
                    continue
                # Chat-added keys inherit the privacy rule of the configured credential.
                allowed = st.allowed_data
                if st.source == "chat" and first_cfg.allowed_data is not None:
                    allowed = frozenset(first_cfg.allowed_data)
                if allowed is not None and not categories <= allowed:
                    sel.skipped.append((where, "allowed_data"))
                    continue
                if (
                    st.spend_limit_usd is not None
                    and spend_by_credential.get(label, 0.0) >= st.spend_limit_usd
                ):
                    sel.skipped.append((where, "credential_spend_limit"))
                    continue
                sel.candidates.append(
                    Candidate(
                        alias=alias, model=model, label=label, provider=st.provider, tier=tier
                    )
                )
        return sel
