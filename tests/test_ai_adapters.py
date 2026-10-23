"""Contract tests for the real adapters on httpx.MockTransport.

Fixtures in tests/fixtures/ai/ follow the providers' public API references
(hand-assembled from documented examples; no real keys or model names).
Each test checks the request shape (URL, auth header, model, structured-output
strategy), response parsing, usage tokens and error normalization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from procbot.ai.adapters.anthropic import AnthropicAdapter
from procbot.ai.adapters.base import parse_retry_after
from procbot.ai.adapters.gemini import GeminiAdapter
from procbot.ai.adapters.openai import OpenAIAdapter
from procbot.ai.adapters.openai_compatible import OpenAICompatibleAdapter
from procbot.ai.config import ModelConfig
from procbot.ai.errors import ProviderError
from procbot.ai.registry import IntentClassification
from procbot.ai.schema import model_schema
from procbot.ai.types import (
    ChatMessage,
    CompletionRequest,
    Credential,
    StructuredStrategy,
    TranscriptionRequest,
)
from procbot.domain.enums import AIErrorKind

FIX = Path(__file__).parent / "fixtures" / "ai"
KEY = "sk-test-SECRET-KEY-1234567890abcd"


def fixture(name: str) -> Any:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


class Recorder:
    def __init__(self, status: int = 200, body: Any = None, headers: dict[str, str] | None = None):
        self.status = status
        self.body = body
        self.headers = headers or {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json=self.body, headers=self.headers)

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]

    def last_json(self) -> dict[str, Any]:
        return json.loads(self.last.content)


def client(rec: Recorder) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(rec))


def cred(provider: str, base_url: str | None = None, key: str = KEY) -> Credential:
    return Credential(
        id=1, label=f"{provider}-main", provider=provider, api_key=key, base_url=base_url
    )


def model(
    strategy: StructuredStrategy = StructuredStrategy.NATIVE_SCHEMA, **kw: Any
) -> ModelConfig:
    return ModelConfig(credential="x", model="test-model", structured_output=strategy, **kw)


def structured_req(strategy: StructuredStrategy) -> CompletionRequest:
    return CompletionRequest(
        task="classify_intent",
        system="SYS",
        messages=[ChatMessage("user", "USER")],
        max_output_tokens=300,
        strategy=strategy,
        json_schema=model_schema(IntentClassification),
        schema_name="IntentClassification",
    )


def text_req() -> CompletionRequest:
    return CompletionRequest(
        task="chat_reply",
        system="SYS",
        messages=[ChatMessage("user", "USER")],
        max_output_tokens=50,
    )


# --------------------------------------------------------------------------- OpenAI


async def test_openai_native_schema_request_and_parse() -> None:
    rec = Recorder(body=fixture("openai_chat_json_schema.json"))
    a = OpenAIAdapter(client(rec))
    resp = await a.complete(
        cred("openai"), model(), structured_req(StructuredStrategy.NATIVE_SCHEMA)
    )
    req = rec.last
    assert str(req.url) == "https://api.openai.com/v1/chat/completions"
    assert req.headers["authorization"] == f"Bearer {KEY}"
    body = rec.last_json()
    assert body["model"] == "test-model"
    assert body["max_completion_tokens"] == 300
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    rf = body["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    schema = rf["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert json.loads(resp.text or "")["intent"] == "add_tasks"
    assert (resp.usage.tokens_in, resp.usage.tokens_out, resp.usage.estimated) == (412, 27, False)


async def test_openai_tool_call_strategy() -> None:
    rec = Recorder(body=fixture("openai_chat_tool_call.json"))
    a = OpenAIAdapter(client(rec))
    resp = await a.complete(cred("openai"), model(), structured_req(StructuredStrategy.TOOL_CALL))
    body = rec.last_json()
    assert body["tools"][0]["function"]["name"] == "IntentClassification"
    assert body["tools"][0]["function"]["strict"] is True
    assert body["tool_choice"] == {"type": "function", "function": {"name": "IntentClassification"}}
    assert "response_format" not in body
    assert resp.data == {
        "intent": "plan_day",
        "confidence": 0.7,
        "needs_clarification": False,
        "clarifying_question": None,
    }


async def test_openai_json_mode_and_refusal() -> None:
    rec = Recorder(body=fixture("openai_chat_refusal.json"))
    a = OpenAIAdapter(client(rec))
    resp = await a.complete(cred("openai"), model(), structured_req(StructuredStrategy.JSON_MODE))
    assert rec.last_json()["response_format"] == {"type": "json_object"}
    assert resp.refusal and "can't" in resp.refusal


async def test_openai_compatible_local_without_key_and_max_tokens() -> None:
    rec = Recorder(body=fixture("openai_chat_text.json"))
    a = OpenAICompatibleAdapter(client(rec))
    resp = await a.complete(
        cred("openai_compatible", base_url="http://localhost:11434/v1/", key=""),
        model(),
        text_req(),
    )
    assert str(rec.last.url) == "http://localhost:11434/v1/chat/completions"
    assert "authorization" not in rec.last.headers
    body = rec.last_json()
    assert body["max_tokens"] == 50 and "max_completion_tokens" not in body
    assert "response_format" not in body and "tools" not in body
    assert resp.text and resp.text.startswith("Привет")


async def test_openai_compatible_tool_has_no_strict_flag() -> None:
    rec = Recorder(body=fixture("openai_chat_tool_call.json"))
    a = OpenAICompatibleAdapter(client(rec))
    await a.complete(
        cred("openai_compatible", base_url="https://openrouter.example/api/v1"),
        model(),
        structured_req(StructuredStrategy.TOOL_CALL),
    )
    assert "strict" not in rec.last_json()["tools"][0]["function"]


@pytest.mark.parametrize(
    ("status", "fixture_name", "headers", "kind", "retry_after"),
    [
        (429, "openai_error_429.json", {"retry-after": "17"}, AIErrorKind.RATE_LIMIT, 17.0),
        (401, "openai_error_401.json", {}, AIErrorKind.AUTH, None),
        (403, "openai_error_401.json", {}, AIErrorKind.AUTH, None),
        (400, "openai_error_context.json", {}, AIErrorKind.CONTEXT_TOO_LONG, None),
        (500, "openai_error_429.json", {}, AIErrorKind.SERVER, None),
        (503, "openai_error_429.json", {}, AIErrorKind.SERVER, None),
        (404, "openai_error_429.json", {}, AIErrorKind.BAD_REQUEST, None),
    ],
)
async def test_openai_error_normalization(
    status: int,
    fixture_name: str,
    headers: dict[str, str],
    kind: AIErrorKind,
    retry_after: float | None,
) -> None:
    rec = Recorder(status=status, body=fixture(fixture_name), headers=headers)
    a = OpenAIAdapter(client(rec))
    with pytest.raises(ProviderError) as ei:
        await a.complete(cred("openai"), model(), text_req())
    err = ei.value
    assert err.kind == kind and err.http_status == status and err.retry_after == retry_after
    assert KEY not in str(err) and KEY not in err.message


async def test_openai_timeout_and_network_errors() -> None:
    def timeout(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    def network(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    for handler, kind in ((timeout, AIErrorKind.TIMEOUT), (network, AIErrorKind.NETWORK)):
        a = OpenAIAdapter(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        with pytest.raises(ProviderError) as ei:
            await a.complete(cred("openai"), model(), text_req())
        assert ei.value.kind == kind and ei.value.transient


async def test_openai_transcription_multipart() -> None:
    rec = Recorder(body=fixture("openai_transcription.json"))
    a = OpenAIAdapter(client(rec))
    out = await a.transcribe(
        cred("openai"),
        model(audio_input=True),
        TranscriptionRequest(audio=b"OGGDATA", mime_type="audio/ogg", language="ru"),
    )
    req = rec.last
    assert str(req.url) == "https://api.openai.com/v1/audio/transcriptions"
    assert req.headers["authorization"] == f"Bearer {KEY}"
    assert req.headers["content-type"].startswith("multipart/form-data")
    raw = req.content
    assert b"OGGDATA" in raw and b'name="model"' in raw and b"test-model" in raw
    assert b'name="language"' in raw
    assert out.text.startswith("Завтра в десять")
    assert out.usage.audio_seconds == 7


async def test_openai_list_models() -> None:
    rec = Recorder(body=fixture("openai_models.json"))
    models = await OpenAIAdapter(client(rec)).list_models(cred("openai"))
    assert models == ["model-a", "model-b"]
    assert str(rec.last.url) == "https://api.openai.com/v1/models"


# --------------------------------------------------------------------------- Anthropic


async def test_anthropic_forced_tool_request_and_parse() -> None:
    rec = Recorder(body=fixture("anthropic_tool_use.json"))
    a = AnthropicAdapter(client(rec))
    resp = await a.complete(
        cred("anthropic"),
        model(StructuredStrategy.TOOL_CALL),
        structured_req(StructuredStrategy.TOOL_CALL),
    )
    req = rec.last
    assert str(req.url) == "https://api.anthropic.com/v1/messages"
    assert req.headers["x-api-key"] == KEY
    assert req.headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in req.headers
    body = rec.last_json()
    assert body["system"] == "SYS" and body["max_tokens"] == 300
    assert body["messages"] == [{"role": "user", "content": "USER"}]
    assert body["tool_choice"] == {"type": "tool", "name": "IntentClassification"}
    assert body["tools"][0]["input_schema"]["type"] == "object"
    assert resp.data == {"intent": "start_work", "confidence": 0.8, "needs_clarification": False}
    assert (resp.usage.tokens_in, resp.usage.tokens_out) == (620, 45)


async def test_anthropic_text_and_refusal() -> None:
    rec = Recorder(body=fixture("anthropic_text.json"))
    resp = await AnthropicAdapter(client(rec)).complete(cred("anthropic"), model(), text_req())
    assert "tools" not in rec.last_json()
    assert resp.text and resp.text.startswith("Хорошо")
    rec2 = Recorder(body=fixture("anthropic_refusal.json"))
    resp2 = await AnthropicAdapter(client(rec2)).complete(cred("anthropic"), model(), text_req())
    assert resp2.refusal


@pytest.mark.parametrize(
    ("status", "name", "kind"),
    [
        (529, "anthropic_error_overloaded.json", AIErrorKind.SERVER),
        (429, "anthropic_error_rate_limit.json", AIErrorKind.RATE_LIMIT),
        (401, "anthropic_error_auth.json", AIErrorKind.AUTH),
    ],
)
async def test_anthropic_errors(status: int, name: str, kind: AIErrorKind) -> None:
    rec = Recorder(status=status, body=fixture(name), headers={"retry-after": "5"})
    with pytest.raises(ProviderError) as ei:
        await AnthropicAdapter(client(rec)).complete(cred("anthropic"), model(), text_req())
    assert ei.value.kind == kind
    if kind == AIErrorKind.RATE_LIMIT:
        assert ei.value.retry_after == 5.0


async def test_anthropic_has_no_audio() -> None:
    rec = Recorder(body={})
    with pytest.raises(ProviderError) as ei:
        await AnthropicAdapter(client(rec)).transcribe(
            cred("anthropic"), model(), TranscriptionRequest(audio=b"x", mime_type="audio/ogg")
        )
    assert ei.value.kind == AIErrorKind.UNSUPPORTED
    assert rec.requests == []


# --------------------------------------------------------------------------- Gemini


async def test_gemini_native_schema_request_and_parse() -> None:
    rec = Recorder(body=fixture("gemini_json.json"))
    a = GeminiAdapter(client(rec))
    resp = await a.complete(
        cred("gemini"), model(), structured_req(StructuredStrategy.NATIVE_SCHEMA)
    )
    req = rec.last
    assert (
        str(req.url)
        == "https://generativelanguage.googleapis.com/v1beta/models/test-model:generateContent"
    )
    assert req.headers["x-goog-api-key"] == KEY
    assert "key=" not in str(req.url)
    body = rec.last_json()
    assert body["systemInstruction"] == {"parts": [{"text": "SYS"}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "USER"}]}]
    gen = body["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    assert gen["maxOutputTokens"] == 300
    assert gen["responseSchema"]["type"] == "OBJECT"
    assert "additionalProperties" not in json.dumps(gen["responseSchema"])
    assert json.loads(resp.text or "")["intent"] == "question"
    assert (resp.usage.tokens_in, resp.usage.tokens_out) == (380, 71)


async def test_gemini_function_calling_strategy() -> None:
    rec = Recorder(body=fixture("gemini_function_call.json"))
    resp = await GeminiAdapter(client(rec)).complete(
        cred("gemini"),
        model(StructuredStrategy.TOOL_CALL),
        structured_req(StructuredStrategy.TOOL_CALL),
    )
    body = rec.last_json()
    decl = body["tools"][0]["functionDeclarations"][0]
    assert decl["name"] == "IntentClassification" and decl["parameters"]["type"] == "OBJECT"
    assert body["toolConfig"]["functionCallingConfig"] == {
        "mode": "ANY",
        "allowedFunctionNames": ["IntentClassification"],
    }
    assert resp.data == {"intent": "postpone", "confidence": 0.9}


async def test_gemini_blocked_prompt_is_refusal() -> None:
    rec = Recorder(body=fixture("gemini_blocked.json"))
    resp = await GeminiAdapter(client(rec)).complete(cred("gemini"), model(), text_req())
    assert resp.refusal and "SAFETY" in resp.refusal


async def test_gemini_errors_quota_retry_delay_and_invalid_key() -> None:
    rec = Recorder(status=429, body=fixture("gemini_error_429.json"))
    with pytest.raises(ProviderError) as ei:
        await GeminiAdapter(client(rec)).complete(cred("gemini"), model(), text_req())
    assert ei.value.kind == AIErrorKind.RATE_LIMIT and ei.value.retry_after == 42.0
    rec2 = Recorder(status=400, body=fixture("gemini_error_key_invalid.json"))
    with pytest.raises(ProviderError) as ei2:
        await GeminiAdapter(client(rec2)).complete(cred("gemini"), model(), text_req())
    assert ei2.value.kind == AIErrorKind.AUTH


async def test_gemini_transcription_inline_audio() -> None:
    rec = Recorder(body=fixture("gemini_transcription.json"))
    out = await GeminiAdapter(client(rec)).transcribe(
        cred("gemini"),
        model(audio_input=True),
        TranscriptionRequest(audio=b"\x00\x01OGG", mime_type="audio/ogg", language="ru"),
    )
    parts = rec.last_json()["contents"][0]["parts"]
    assert parts[0]["inlineData"] == {"mimeType": "audio/ogg", "data": "AAFPR0c="}
    assert "Распознай" in parts[1]["text"]
    assert out.text == "Напомни мне позвонить маме в семь."
    assert out.usage.tokens_in == 180


def test_parse_retry_after_forms() -> None:
    assert parse_retry_after("12") == 12.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("garbage") is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0  # in the past
