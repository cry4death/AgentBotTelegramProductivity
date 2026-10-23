"""Adapter interface and shared HTTP plumbing.

Adapters are thin translators on httpx (no provider SDKs): one neutral request
in, one neutral response out, every failure normalized into ProviderError
(AIErrorKind + HTTP status + Retry-After). Error messages are scrubbed of the
key and truncated; the key is only ever placed in a request header.
"""

from __future__ import annotations

import email.utils
import json
import re
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

import httpx

from procbot.ai.config import ModelConfig
from procbot.ai.errors import ProviderError
from procbot.ai.prompting import estimate_tokens
from procbot.ai.types import (
    CompletionRequest,
    CompletionResponse,
    Credential,
    TranscriptionRequest,
    TranscriptionResponse,
    Usage,
)
from procbot.domain.enums import AIErrorKind

_CONTEXT_HINTS = re.compile(
    r"context[_ ]length|context window|too many tokens|maximum context|too long|"
    r"prompt is too long|exceeds the maximum|token limit",
    re.IGNORECASE,
)


class ProviderAdapter(ABC):
    provider: str = ""
    supports_audio: bool = False

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    @abstractmethod
    async def complete(
        self, cred: Credential, model: ModelConfig, req: CompletionRequest
    ) -> CompletionResponse: ...

    async def transcribe(
        self, cred: Credential, model: ModelConfig, req: TranscriptionRequest
    ) -> TranscriptionResponse:
        raise ProviderError(AIErrorKind.UNSUPPORTED, f"{self.provider} has no audio input")

    async def list_models(self, cred: Credential, *, timeout: float = 20.0) -> list[str]:
        raise ProviderError(AIErrorKind.UNSUPPORTED, "listing models is not supported")

    # ------------------------------------------------------------------ helpers

    async def _send(
        self,
        cred: Credential,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        timeout: float,
        json_body: Any = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
    ) -> Any:
        try:
            resp = await self.client.request(
                method,
                url,
                headers={**cred.extra_headers, **headers},
                json=json_body,
                data=data,
                files=files,
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(AIErrorKind.TIMEOUT, type(exc).__name__) from None
        except httpx.TransportError as exc:
            raise ProviderError(AIErrorKind.NETWORK, type(exc).__name__) from None
        if resp.status_code >= 400:
            raise self.normalize_error(resp, cred)
        try:
            return resp.json()
        except ValueError:
            raise ProviderError(
                AIErrorKind.SERVER, "non-JSON response", http_status=resp.status_code
            ) from None

    def error_message(self, body: Any) -> str:
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                return str(err.get("message") or err.get("type") or err)
            if isinstance(err, str):
                return err
            if "message" in body:
                return str(body["message"])
        return ""

    def classify_status(self, status: int, message: str, body: Any) -> AIErrorKind:
        if status in (401, 403):
            return AIErrorKind.AUTH
        if status == 429:
            return AIErrorKind.RATE_LIMIT
        if status == 408:
            return AIErrorKind.TIMEOUT
        if status == 413 or (status in (400, 422) and _CONTEXT_HINTS.search(message)):
            return AIErrorKind.CONTEXT_TOO_LONG
        if status in (409, 529) or status >= 500:
            return AIErrorKind.SERVER
        if status in (400, 404, 422):
            return AIErrorKind.BAD_REQUEST
        return AIErrorKind.UNKNOWN

    def retry_after_from(self, resp: httpx.Response, body: Any) -> float | None:
        return parse_retry_after(resp.headers.get("retry-after"))

    def normalize_error(self, resp: httpx.Response, cred: Credential) -> ProviderError:
        try:
            body: Any = resp.json()
        except ValueError:
            body = resp.text
        message = scrub(self.error_message(body) or str(body)[:200], cred.api_key)
        kind = self.classify_status(resp.status_code, message, body)
        return ProviderError(
            kind,
            message,
            http_status=resp.status_code,
            retry_after=self.retry_after_from(resp, body)
            if kind == AIErrorKind.RATE_LIMIT
            else None,
        )


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def scrub(message: str, secret: str) -> str:
    if secret and len(secret) >= 4:
        message = message.replace(secret, "****")
    # Generic key-looking tokens (sk-..., AIza...) are masked too.
    message = re.sub(r"(sk-[A-Za-z0-9_\-]{4})[A-Za-z0-9_\-]{8,}", r"\1****", message)
    message = re.sub(r"(AIza)[A-Za-z0-9_\-]{10,}", r"\1****", message)
    return message[:300]


def parse_json_text(text: str) -> Any:
    """Parse model JSON text, tolerating ```json fences``` some models add."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return json.loads(t)


def estimated_usage(req: CompletionRequest, out_text: str) -> Usage:
    return Usage(
        tokens_in=estimate_tokens(req.system, *(m.content for m in req.messages)),
        tokens_out=estimate_tokens(out_text),
        estimated=True,
    )


def join_url(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")
