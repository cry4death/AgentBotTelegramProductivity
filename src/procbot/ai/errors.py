"""Gateway errors.

Domain code only ever needs `AIUnavailable`: catch it and take the template
branch (hard rule 8). `ProviderError` never leaves the gateway.
"""

from __future__ import annotations

from procbot.domain.enums import AIErrorKind


class AIUnavailable(Exception):
    """No valid AI answer could be obtained. Domain code falls back to templates.

    `reason` is a short stable code:
    - not_configured: no ai.yaml / no models for the tier;
    - no_ai_mode: every provider failed for a while; the probe job will recover;
    - budget_hard_limit: the monthly hard budget is reached;
    - no_candidates: every configured candidate was filtered out (health,
      capabilities, privacy, context size, per-key spend limit);
    - all_failed: candidates were tried and all failed (API errors);
    - invalid_output: answers did not pass validation even after repair;
    - refused: the model(s) refused to answer;
    - timeout: the per-call time limit was exhausted.
    """

    def __init__(self, reason: str, detail: str = "", *, call_id: int | None = None) -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail
        self.call_id = call_id


class AIUsageError(RuntimeError):
    """Programming error in the caller (e.g. running the gateway inside a transaction)."""


class ProviderError(Exception):
    """A normalized provider failure. Messages are scrubbed of secrets."""

    def __init__(
        self,
        kind: AIErrorKind,
        message: str = "",
        *,
        http_status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(f"{kind.value}: {message}" if message else kind.value)
        self.kind = kind
        self.message = message
        self.http_status = http_status
        self.retry_after = retry_after

    @property
    def transient(self) -> bool:
        """Worth retrying on the same candidate after a backoff."""
        return self.kind in (AIErrorKind.TIMEOUT, AIErrorKind.NETWORK, AIErrorKind.SERVER)
