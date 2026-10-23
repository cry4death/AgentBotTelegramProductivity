"""Anthropic Messages API (`POST /v1/messages`, REST, httpx).

Structured output uses ONE forced tool: `tools=[{name, description,
input_schema}]` + `tool_choice={"type": "tool", "name": ...}`; the answer is the
`tool_use` block's `input`. `native_schema` in ai.yaml is served the same way
(the forced tool is the long-stable mechanism). `json_mode` = schema in prompt,
plain text answer. No audio input.
"""

from __future__ import annotations

from typing import Any

from procbot.ai.adapters.base import ProviderAdapter, estimated_usage, join_url
from procbot.ai.config import ModelConfig
from procbot.ai.errors import ProviderError
from procbot.ai.schema import anthropic_schema
from procbot.ai.types import (
    CompletionRequest,
    CompletionResponse,
    Credential,
    StructuredStrategy,
    Usage,
)
from procbot.domain.enums import AIErrorKind

ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicAdapter(ProviderAdapter):
    provider = "anthropic"
    supports_audio = False

    def _headers(self, cred: Credential) -> dict[str, str]:
        return {
            "x-api-key": cred.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def _url(self, cred: Credential, path: str) -> str:
        return join_url(cred.base_url or ANTHROPIC_BASE_URL, path)

    def build_body(self, model: ModelConfig, req: CompletionRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model.model,
            "max_tokens": req.max_output_tokens,
            "system": req.system,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
        }
        if req.json_schema is not None and req.strategy in (
            StructuredStrategy.TOOL_CALL,
            StructuredStrategy.NATIVE_SCHEMA,
        ):
            body["tools"] = [
                {
                    "name": req.schema_name,
                    "description": req.schema_description or "Return the result.",
                    "input_schema": anthropic_schema(req.json_schema),
                }
            ]
            body["tool_choice"] = {"type": "tool", "name": req.schema_name}
        return body

    async def complete(
        self, cred: Credential, model: ModelConfig, req: CompletionRequest
    ) -> CompletionResponse:
        data = await self._send(
            cred,
            "POST",
            self._url(cred, "/v1/messages"),
            headers=self._headers(cred),
            timeout=req.timeout_seconds,
            json_body=self.build_body(model, req),
        )
        if not isinstance(data, dict) or "content" not in data:
            raise ProviderError(AIErrorKind.SERVER, "response has no content")
        stop = data.get("stop_reason")
        out = CompletionResponse(finish_reason=stop, model=data.get("model"))
        texts: list[str] = []
        for block in data.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and out.data is None:
                inp = block.get("input")
                if isinstance(inp, dict):
                    out.data = inp
            elif block.get("type") == "text":
                texts.append(str(block.get("text") or ""))
        if out.data is None:
            out.text = "".join(texts)
        if stop == "refusal":
            out.refusal = "".join(texts) or "refusal"
        out.truncated = stop == "max_tokens"
        usage = data.get("usage") or {}
        if "input_tokens" in usage:
            out.usage = Usage(
                tokens_in=int(usage.get("input_tokens") or 0)
                + int(usage.get("cache_creation_input_tokens") or 0)
                + int(usage.get("cache_read_input_tokens") or 0),
                tokens_out=int(usage.get("output_tokens") or 0),
            )
        else:
            out.usage = estimated_usage(req, out.raw_for_repair())
        return out

    def error_message(self, body: Any) -> str:
        # {"type": "error", "error": {"type": "rate_limit_error", "message": "..."}}
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            err = body["error"]
            return f"{err.get('type', '')}: {err.get('message', '')}".strip(": ")
        return super().error_message(body)

    def classify_status(self, status: int, message: str, body: Any) -> AIErrorKind:
        if message.startswith("overloaded_error"):
            return AIErrorKind.SERVER
        return super().classify_status(status, message, body)

    async def list_models(self, cred: Credential, *, timeout: float = 20.0) -> list[str]:
        data = await self._send(
            cred, "GET", self._url(cred, "/v1/models"), headers=self._headers(cred), timeout=timeout
        )
        items = data.get("data", []) if isinstance(data, dict) else []
        return [str(m.get("id")) for m in items if isinstance(m, dict) and m.get("id")]
