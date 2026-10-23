"""Google Gemini API: `models/{model}:generateContent` (REST, httpx).

Auth: `x-goog-api-key` header (never a `?key=` query parameter, so the key
cannot end up in URL logs).
Structured output:
- native_schema: `generationConfig.responseMimeType = application/json` +
  `responseSchema` (OpenAPI subset, see procbot.ai.schema.gemini_schema);
- tool_call: one function declaration + `toolConfig.functionCallingConfig`
  mode ANY restricted to it;
- json_mode: `responseMimeType = application/json` only (schema in the prompt).
Audio (stt): the audio goes inline as `inlineData` (base64) with a short
transcription instruction.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx

from procbot.ai.adapters.base import ProviderAdapter, estimated_usage, join_url
from procbot.ai.config import ModelConfig
from procbot.ai.errors import ProviderError
from procbot.ai.schema import gemini_schema
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

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_REFUSAL_FINISH = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}

TRANSCRIBE_INSTRUCTION = (
    "Распознай речь в этом аудио и верни только дословный текст, без пояснений."
)


class GeminiAdapter(ProviderAdapter):
    provider = "gemini"
    supports_audio = True

    def _headers(self, cred: Credential) -> dict[str, str]:
        return {"x-goog-api-key": cred.api_key, "Content-Type": "application/json"}

    def _url(self, cred: Credential, model: str, method: str) -> str:
        name = model if model.startswith("models/") else f"models/{model}"
        return join_url(cred.base_url or GEMINI_BASE_URL, f"/{name}:{method}")

    def build_body(self, model: ModelConfig, req: CompletionRequest) -> dict[str, Any]:
        gen: dict[str, Any] = {"maxOutputTokens": req.max_output_tokens}
        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": req.system}]},
            "contents": [
                {
                    "role": "model" if m.role == "assistant" else "user",
                    "parts": [{"text": m.content}],
                }
                for m in req.messages
            ],
            "generationConfig": gen,
        }
        if req.json_schema is not None:
            if req.strategy == StructuredStrategy.NATIVE_SCHEMA:
                gen["responseMimeType"] = "application/json"
                gen["responseSchema"] = gemini_schema(req.json_schema)
            elif req.strategy == StructuredStrategy.TOOL_CALL:
                body["tools"] = [
                    {
                        "functionDeclarations": [
                            {
                                "name": req.schema_name,
                                "description": req.schema_description or "Return the result.",
                                "parameters": gemini_schema(req.json_schema),
                            }
                        ]
                    }
                ]
                body["toolConfig"] = {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [req.schema_name],
                    }
                }
            elif req.strategy == StructuredStrategy.JSON_MODE:
                gen["responseMimeType"] = "application/json"
        return body

    def _usage(self, data: dict[str, Any]) -> Usage | None:
        meta = data.get("usageMetadata") or {}
        if "promptTokenCount" not in meta and "candidatesTokenCount" not in meta:
            return None
        return Usage(
            tokens_in=int(meta.get("promptTokenCount") or 0)
            + int(meta.get("toolUsePromptTokenCount") or 0),
            # thinking tokens are billed as output
            tokens_out=int(meta.get("candidatesTokenCount") or 0)
            + int(meta.get("thoughtsTokenCount") or 0),
        )

    async def complete(
        self, cred: Credential, model: ModelConfig, req: CompletionRequest
    ) -> CompletionResponse:
        data = await self._send(
            cred,
            "POST",
            self._url(cred, model.model, "generateContent"),
            headers=self._headers(cred),
            timeout=req.timeout_seconds,
            json_body=self.build_body(model, req),
        )
        out = self.parse(data)
        out.usage = self._usage(data) or estimated_usage(req, out.raw_for_repair())
        return out

    def parse(self, data: Any) -> CompletionResponse:
        if not isinstance(data, dict):
            raise ProviderError(AIErrorKind.SERVER, "unexpected response")
        block = (data.get("promptFeedback") or {}).get("blockReason")
        candidates = data.get("candidates") or []
        if block and not candidates:
            return CompletionResponse(refusal=f"blocked: {block}", finish_reason=str(block))
        if not candidates:
            raise ProviderError(AIErrorKind.SERVER, "response has no candidates")
        cand = candidates[0] or {}
        finish = cand.get("finishReason")
        out = CompletionResponse(finish_reason=finish, model=data.get("modelVersion"))
        texts: list[str] = []
        for part in (cand.get("content") or {}).get("parts") or []:
            if not isinstance(part, dict) or part.get("thought"):
                continue
            fc = part.get("functionCall") or part.get("function_call")
            if isinstance(fc, dict) and out.data is None:
                args = fc.get("args")
                out.data = args if isinstance(args, dict) else {}
            elif "text" in part:
                texts.append(str(part["text"]))
        if out.data is None:
            out.text = "".join(texts)
        if finish in _REFUSAL_FINISH:
            out.refusal = f"finish: {finish}"
        out.truncated = finish == "MAX_TOKENS"
        return out

    async def transcribe(
        self, cred: Credential, model: ModelConfig, req: TranscriptionRequest
    ) -> TranscriptionResponse:
        instruction = TRANSCRIBE_INSTRUCTION
        if req.language:
            instruction += f" Язык: {req.language}."
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": req.mime_type,
                                "data": base64.b64encode(req.audio).decode(),
                            }
                        },
                        {"text": instruction},
                    ],
                }
            ],
            "generationConfig": {"temperature": 0},
        }
        data = await self._send(
            cred,
            "POST",
            self._url(cred, model.model, "generateContent"),
            headers=self._headers(cred),
            timeout=req.timeout_seconds,
            json_body=body,
        )
        out = self.parse(data)
        if out.refusal:
            raise ProviderError(AIErrorKind.REFUSAL, out.refusal)
        usage = self._usage(data) or Usage(estimated=True)
        return TranscriptionResponse(text=(out.text or "").strip(), usage=usage)

    async def list_models(self, cred: Credential, *, timeout: float = 20.0) -> list[str]:
        data = await self._send(
            cred,
            "GET",
            join_url(cred.base_url or GEMINI_BASE_URL, "/models"),
            headers=self._headers(cred),
            timeout=timeout,
        )
        items = data.get("models", []) if isinstance(data, dict) else []
        return [str(m.get("name")) for m in items if isinstance(m, dict) and m.get("name")]

    # ------------------------------------------------------------------ errors

    def classify_status(self, status: int, message: str, body: Any) -> AIErrorKind:
        err = body.get("error") if isinstance(body, dict) else None
        if isinstance(err, dict):
            reasons = {
                str(d.get("reason"))
                for d in err.get("details") or []
                if isinstance(d, dict) and d.get("reason")
            }
            if "API_KEY_INVALID" in reasons or "API key not valid" in message:
                return AIErrorKind.AUTH
            if err.get("status") == "RESOURCE_EXHAUSTED":
                return AIErrorKind.RATE_LIMIT
        return super().classify_status(status, message, body)

    def retry_after_from(self, resp: httpx.Response, body: Any) -> float | None:
        header = super().retry_after_from(resp, body)
        if header is not None:
            return header
        err = body.get("error") if isinstance(body, dict) else None
        if isinstance(err, dict):
            for d in err.get("details") or []:
                delay = d.get("retryDelay") if isinstance(d, dict) else None
                if isinstance(delay, str) and delay.endswith("s"):
                    try:
                        return float(delay[:-1])
                    except ValueError:
                        return None
        return None
