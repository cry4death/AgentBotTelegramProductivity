"""Provider-neutral request/response types and the public result objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Tier(StrEnum):
    FAST = "fast"
    STANDARD = "standard"
    STRONG = "strong"
    STT = "stt"


# Order used for "next tier" escalation and for "downgrade one tier".
TEXT_TIERS: tuple[Tier, ...] = (Tier.FAST, Tier.STANDARD, Tier.STRONG)


class StructuredStrategy(StrEnum):
    NATIVE_SCHEMA = "native_schema"
    TOOL_CALL = "tool_call"
    JSON_MODE = "json_mode"
    NONE = "none"


class TextOutput(BaseModel):
    """Output of plain-text tasks (e.g. chat_reply)."""

    text: str


# --------------------------------------------------------------------------- public results


@dataclass(frozen=True)
class AIResult:
    output: BaseModel  # validated schema instance, or TextOutput for text tasks
    call_id: int
    task: str
    tier: str
    model_alias: str
    provider: str
    credential_label: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    cost_estimated: bool = False

    def as_(self, cls: type[T]) -> T:
        """Typed access: `plan = result.as_(DayPlanProposal)`."""
        if not isinstance(self.output, cls):
            raise TypeError(f"AI output is {type(self.output).__name__}, not {cls.__name__}")
        return self.output

    @property
    def text(self) -> str:
        """Text of a text task (raises for structured tasks)."""
        return self.as_(TextOutput).text


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    call_id: int
    model_alias: str
    provider: str
    credential_label: str
    language: str | None = None
    cost_usd: float = 0.0
    cost_estimated: bool = False


# --------------------------------------------------------------------------- adapter I/O


@dataclass
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0
    audio_seconds: float | None = None
    estimated: bool = False  # provider did not report usage; counts are our estimate


@dataclass
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class CompletionRequest:
    task: str  # for logs / the fake adapter only; real adapters ignore it
    system: str
    messages: list[ChatMessage]
    max_output_tokens: int
    strategy: StructuredStrategy = StructuredStrategy.NONE
    # Plain JSON Schema from pydantic (adapters adapt it per provider) or None for text.
    json_schema: dict[str, Any] | None = None
    schema_name: str = "output"
    schema_description: str = ""
    timeout_seconds: float = 30.0


@dataclass
class CompletionResponse:
    text: str | None = None  # raw text content (JSON text for native/json modes)
    data: dict[str, Any] | None = None  # parsed tool/function arguments
    refusal: str | None = None
    finish_reason: str | None = None
    truncated: bool = False
    usage: Usage = field(default_factory=Usage)
    model: str | None = None  # model id reported by the provider

    def raw_for_repair(self) -> str:
        if self.data is not None:
            import json

            return json.dumps(self.data, ensure_ascii=False)
        return self.text or ""


@dataclass
class TranscriptionRequest:
    audio: bytes
    mime_type: str
    language: str | None = None
    timeout_seconds: float = 60.0
    filename: str = "audio.ogg"


@dataclass
class TranscriptionResponse:
    text: str
    usage: Usage = field(default_factory=Usage)


class Credential:
    """What an adapter needs to authenticate. The key never appears in repr()."""

    __slots__ = ("api_key", "base_url", "extra_headers", "id", "label", "provider")

    def __init__(
        self,
        *,
        id: int,
        label: str,
        provider: str,
        api_key: str,
        base_url: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.id = id
        self.label = label
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.extra_headers = dict(extra_headers or {})

    def __repr__(self) -> str:
        return f"Credential(label={self.label!r}, provider={self.provider!r})"

    __str__ = __repr__
