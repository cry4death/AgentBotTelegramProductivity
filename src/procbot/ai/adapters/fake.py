"""Scripted adapter for tests (never used in production configs by accident:
it only answers what a test pushed into its FakeAI script).

Script items, consumed in order per task (or per "task@model-id"):
- dict           -> structured answer (or {"text": ...} for text tasks)
- str            -> raw text (for structured tasks it is parsed as JSON, so
                    "not json" exercises the invalid-output/repair path)
- FakeError(...) -> the adapter raises a normalized ProviderError
- FakeRefusal()  -> the model refuses
- callable(req)  -> returns one of the above (lets a test inspect the request)
When a task has no scripted item left, the fake raises UNSUPPORTED, which the
gateway turns into AIUnavailable once candidates are exhausted.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

from procbot.ai.adapters.base import ProviderAdapter
from procbot.ai.config import ModelConfig
from procbot.ai.errors import ProviderError
from procbot.ai.prompting import estimate_tokens
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


@dataclass(frozen=True)
class FakeError:
    kind: AIErrorKind = AIErrorKind.SERVER
    status: int | None = None
    retry_after: float | None = None
    message: str = "scripted error"


@dataclass(frozen=True)
class FakeRefusal:
    text: str = "Я не могу помочь с этим."


FakeItem = dict[str, Any] | str | FakeError | FakeRefusal | Callable[[Any], Any]


@dataclass
class FakeCall:
    kind: str  # complete / transcribe / list_models
    task: str
    model: str
    credential: str
    system: str = ""
    messages: list[tuple[str, str]] = field(default_factory=list)
    strategy: str = ""
    json_schema: dict[str, Any] | None = None
    audio: bytes | None = None
    repair: bool = False

    @property
    def user(self) -> str:
        """Text of the first user message (rendered data blocks + instruction)."""
        return next((c for r, c in self.messages if r == "user"), "")

    @property
    def all_text(self) -> str:
        return "\n".join([self.system, *(c for _, c in self.messages)])


class FakeAI:
    """Script + request recorder shared by every fake credential of a gateway."""

    def __init__(self, script: Mapping[str, Iterable[FakeItem] | FakeItem] | None = None) -> None:
        self.queues: dict[str, deque[FakeItem]] = {}
        self.defaults: dict[str, FakeItem] = {}
        self.calls: list[FakeCall] = []
        self.models: list[str] = ["fake-model"]
        for key, items in (script or {}).items():
            if isinstance(items, list | tuple):
                self.push(key, *items)
            else:
                self.push(key, items)  # type: ignore[arg-type]

    def push(self, task: str, *items: FakeItem, model: str | None = None) -> FakeAI:
        key = f"{task}@{model}" if model else task
        self.queues.setdefault(key, deque()).extend(items)
        return self

    def set_default(self, task: str, item: FakeItem) -> FakeAI:
        """Answer used whenever the task's queue is empty (sticky)."""
        self.defaults[task] = item
        return self

    def calls_for(self, task: str, kind: str = "complete") -> list[FakeCall]:
        return [c for c in self.calls if c.task == task and c.kind == kind]

    def remaining(self, task: str) -> int:
        return len(self.queues.get(task, ()))

    def next_item(self, task: str, model: str) -> FakeItem:
        for key in (f"{task}@{model}", task):
            q = self.queues.get(key)
            if q:
                return q.popleft()
        if task in self.defaults:
            return self.defaults[task]
        raise ProviderError(AIErrorKind.UNSUPPORTED, f"fake: no scripted answer for {task!r}")


class FakeAdapter(ProviderAdapter):
    provider = "fake"
    supports_audio = True

    def __init__(self, client: httpx.AsyncClient, fake: FakeAI) -> None:
        super().__init__(client)
        self.fake = fake

    async def complete(
        self, cred: Credential, model: ModelConfig, req: CompletionRequest
    ) -> CompletionResponse:
        self.fake.calls.append(
            FakeCall(
                kind="complete",
                task=req.task,
                model=model.model,
                credential=cred.label,
                system=req.system,
                messages=[(m.role, m.content) for m in req.messages],
                strategy=str(req.strategy),
                json_schema=req.json_schema,
                repair=len(req.messages) > 1,
            )
        )
        item: Any = self.fake.next_item(req.task, model.model)
        if callable(item) and not isinstance(item, FakeError | FakeRefusal):
            item = item(req)
        if isinstance(item, FakeError):
            raise ProviderError(
                item.kind, item.message, http_status=item.status, retry_after=item.retry_after
            )
        tokens_in = estimate_tokens(req.system, *(m.content for m in req.messages))
        if isinstance(item, FakeRefusal):
            return CompletionResponse(
                refusal=item.text, usage=Usage(tokens_in=tokens_in, tokens_out=5)
            )
        out = CompletionResponse(finish_reason="stop", model=model.model)
        if isinstance(item, dict):
            if req.json_schema is None:
                out.text = str(item.get("text", json.dumps(item, ensure_ascii=False)))
            elif req.strategy == StructuredStrategy.TOOL_CALL:
                out.data = item
            else:
                out.text = json.dumps(item, ensure_ascii=False)
        else:
            out.text = str(item)
        out.usage = Usage(tokens_in=tokens_in, tokens_out=estimate_tokens(out.raw_for_repair()))
        return out

    async def transcribe(
        self, cred: Credential, model: ModelConfig, req: TranscriptionRequest
    ) -> TranscriptionResponse:
        self.fake.calls.append(
            FakeCall(
                kind="transcribe",
                task="transcribe",
                model=model.model,
                credential=cred.label,
                audio=req.audio,
            )
        )
        item: Any = self.fake.next_item("transcribe", model.model)
        if callable(item) and not isinstance(item, FakeError | FakeRefusal):
            item = item(req)
        if isinstance(item, FakeError):
            raise ProviderError(
                item.kind, item.message, http_status=item.status, retry_after=item.retry_after
            )
        if isinstance(item, FakeRefusal):
            raise ProviderError(AIErrorKind.REFUSAL, item.text)
        text = item.get("text", "") if isinstance(item, dict) else str(item)
        return TranscriptionResponse(text=text, usage=Usage(estimated=True))

    async def list_models(self, cred: Credential, *, timeout: float = 20.0) -> list[str]:
        self.fake.calls.append(
            FakeCall(kind="list_models", task="list_models", model="", credential=cred.label)
        )
        q = self.fake.queues.get("list_models")
        if q:
            item = q.popleft()
            if isinstance(item, FakeError):
                raise ProviderError(
                    item.kind, item.message, http_status=item.status, retry_after=item.retry_after
                )
        return list(self.fake.models)
