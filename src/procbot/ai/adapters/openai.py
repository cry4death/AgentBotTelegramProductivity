"""OpenAI Chat Completions + Audio Transcriptions (REST, httpx).

Structured output:
- native_schema: `response_format: {"type": "json_schema", "json_schema": {name, schema, strict}}`;
- tool_call: one function tool + `tool_choice` forcing it;
- json_mode: `response_format: {"type": "json_object"}` (schema is in the prompt).
"""

from __future__ import annotations

import json
from typing import Any

from procbot.ai.adapters.base import ProviderAdapter, estimated_usage, join_url
from procbot.ai.config import ModelConfig
from procbot.ai.errors import ProviderError
from procbot.ai.schema import openai_strict
from procbot.ai.types import (
    CompletionRequest,
    CompletionResponse,
    Credential,
    StructuredStrategy,
    TranscriptionRequest,
    TranscriptionResponse,
    Usage,
)
from procbot.domain.enums import AIErrorKind

OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAIAdapter(ProviderAdapter):
    provider = "openai"
    supports_audio = True
    default_base_url = OPENAI_BASE_URL
    # OpenAI deprecated `max_tokens` for chat completions; compatible servers
    # still expect `max_tokens`.
    max_tokens_param = "max_completion_tokens"
    strict_schemas = True
    strict_tools = True

    def base_url(self, cred: Credential) -> str:
        return cred.base_url or self.default_base_url

    def auth_headers(self, cred: Credential) -> dict[str, str]:
        return {"Authorization": f"Bearer {cred.api_key}"} if cred.api_key else {}

    def adapt_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        return openai_strict(schema) if self.strict_schemas else schema

    def build_body(self, model: ModelConfig, req: CompletionRequest) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": req.system}]
        messages += [{"role": m.role, "content": m.content} for m in req.messages]
        body: dict[str, Any] = {
            "model": model.model,
            "messages": messages,
            self.max_tokens_param: req.max_output_tokens,
        }
        if req.json_schema is not None:
            schema = self.adapt_schema(req.json_schema)
            if req.strategy == StructuredStrategy.NATIVE_SCHEMA:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": req.schema_name,
                        "schema": schema,
                        "strict": self.strict_schemas,
                    },
                }
            elif req.strategy == StructuredStrategy.TOOL_CALL:
                fn: dict[str, Any] = {
                    "name": req.schema_name,
                    "description": req.schema_description or "Return the result.",
                    "parameters": schema,
                }
                if self.strict_tools:
                    fn["strict"] = True
                body["tools"] = [{"type": "function", "function": fn}]
                body["tool_choice"] = {"type": "function", "function": {"name": req.schema_name}}
            elif req.strategy == StructuredStrategy.JSON_MODE:
                body["response_format"] = {"type": "json_object"}
        return body

    async def complete(
        self, cred: Credential, model: ModelConfig, req: CompletionRequest
    ) -> CompletionResponse:
        data = await self._send(
            cred,
            "POST",
            join_url(self.base_url(cred), "/chat/completions"),
            headers={**self.auth_headers(cred), "Content-Type": "application/json"},
            timeout=req.timeout_seconds,
            json_body=self.build_body(model, req),
        )
        return self.parse_completion(data, req)

    def parse_completion(self, data: Any, req: CompletionRequest) -> CompletionResponse:
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            raise ProviderError(AIErrorKind.SERVER, "response has no choices")
        choice = choices[0] or {}
        msg = choice.get("message") or {}
        finish = choice.get("finish_reason")
        out = CompletionResponse(finish_reason=finish, model=data.get("model"))
        if msg.get("refusal"):
            out.refusal = str(msg["refusal"])
        elif finish == "content_filter":
            out.refusal = "content_filter"
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            args = (tool_calls[0].get("function") or {}).get("arguments")
            if isinstance(args, dict):
                out.data = args
            elif isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except ValueError:
                    out.text = args  # invalid JSON -> validation/repair path
                else:
                    if isinstance(parsed, dict):
                        out.data = parsed
                    else:
                        out.text = args
        content = msg.get("content")
        if out.data is None and out.text is None and isinstance(content, str):
            out.text = content
        out.truncated = finish == "length"
        usage = data.get("usage") or {}
        if "prompt_tokens" in usage or "completion_tokens" in usage:
            out.usage = Usage(
                tokens_in=int(usage.get("prompt_tokens") or 0),
                tokens_out=int(usage.get("completion_tokens") or 0),
            )
        else:
            out.usage = estimated_usage(req, out.raw_for_repair())
        return out

    async def transcribe(
        self, cred: Credential, model: ModelConfig, req: TranscriptionRequest
    ) -> TranscriptionResponse:
        form: dict[str, Any] = {"model": model.model, "response_format": "json"}
        if req.language:
            form["language"] = req.language
        data = await self._send(
            cred,
            "POST",
            join_url(self.base_url(cred), "/audio/transcriptions"),
            headers=self.auth_headers(cred),
            timeout=req.timeout_seconds,
            data=form,
            files={"file": (req.filename, req.audio, req.mime_type)},
        )
        if not isinstance(data, dict) or not isinstance(data.get("text"), str):
            raise ProviderError(AIErrorKind.SERVER, "transcription response has no text")
        usage = Usage(estimated=True)
        raw_usage = data.get("usage") or {}
        if raw_usage.get("type") == "duration" or "seconds" in raw_usage:
            usage = Usage(audio_seconds=float(raw_usage.get("seconds") or 0))
        elif "input_tokens" in raw_usage:
            usage = Usage(
                tokens_in=int(raw_usage.get("input_tokens") or 0),
                tokens_out=int(raw_usage.get("output_tokens") or 0),
            )
        elif "duration" in data:
            usage = Usage(audio_seconds=float(data["duration"]))
        return TranscriptionResponse(text=data["text"].strip(), usage=usage)

    async def list_models(self, cred: Credential, *, timeout: float = 20.0) -> list[str]:
        data = await self._send(
            cred,
            "GET",
            join_url(self.base_url(cred), "/models"),
            headers=self.auth_headers(cred),
            timeout=timeout,
        )
        items = data.get("data", []) if isinstance(data, dict) else []
        return [str(m.get("id")) for m in items if isinstance(m, dict) and m.get("id")]
