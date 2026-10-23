"""Registry of AI tasks.

A task names WHAT is asked (schema, tier, budget, allowed data), never which
model answers. Domain modules register their tasks at import time:

    register_task(AITaskSpec(name="extract_tasks", tier=Tier.STANDARD,
                             output=ExtractedTasks, data_categories={"message", "tasks"}))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from procbot.ai import prompting as P
from procbot.ai.types import Tier

# Capabilities a task may require from a model.
STRUCTURED = "structured"
AUDIO_INPUT = "audio_input"
VISION = "vision"


@dataclass(frozen=True)
class AITaskSpec:
    name: str
    tier: Tier
    # Pydantic model of the answer; None = plain text (TextOutput).
    output: type[BaseModel] | None = None
    # Extra required capabilities ("structured" is implied by `output`).
    capabilities: frozenset[str] = field(default_factory=frozenset)
    max_output_tokens: int = 1024
    # Max estimated prompt tokens this task may send; None = model limit only.
    context_budget_tokens: int | None = None
    # Data categories the task may send (see procbot.ai.prompting).
    data_categories: frozenset[str] = field(default_factory=lambda: frozenset({P.MESSAGE}))
    # Trusted task-level system instructions.
    system: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        object.__setattr__(self, "data_categories", frozenset(self.data_categories))
        unknown = self.data_categories - P.DATA_CATEGORIES
        if unknown:
            raise ValueError(f"task {self.name}: unknown data categories {sorted(unknown)}")

    @property
    def required_capabilities(self) -> frozenset[str]:
        caps = set(self.capabilities)
        if self.output is not None:
            caps.add(STRUCTURED)
        return frozenset(caps)


_TASKS: dict[str, AITaskSpec] = {}


def register_task(spec: AITaskSpec, *, replace: bool = False) -> AITaskSpec:
    """Register a task. Re-registering the same spec is a no-op (module reloads)."""
    existing = _TASKS.get(spec.name)
    if existing is not None and existing != spec and not replace:
        raise ValueError(f"AI task {spec.name!r} is already registered with another spec")
    _TASKS[spec.name] = spec
    return spec


def get_task(name: str) -> AITaskSpec:
    try:
        return _TASKS[name]
    except KeyError:
        raise KeyError(f"unknown AI task {name!r}; register it with register_task()") from None


def registered_tasks() -> list[AITaskSpec]:
    return list(_TASKS.values())


# --------------------------------------------------------------------------- built-in tasks


class IntentClassification(BaseModel):
    """Result of `classify_intent`: what the owner wants when no command/button matched."""

    intent: Literal[
        "add_tasks",  # wants to add one or more tasks
        "plan_day",  # asks for / about the day plan
        "start_work",  # wants to start working on something
        "report_progress",  # reports what was done / retro entry
        "postpone",  # wants to postpone / cannot start
        "question",  # asks the bot something
        "chat",  # small talk / emotions
        "other",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    needs_clarification: bool = False
    clarifying_question: str | None = None


CHAT_REPLY = register_task(
    AITaskSpec(
        name="chat_reply",
        tier=Tier.STANDARD,
        output=None,
        max_output_tokens=800,
        data_categories=frozenset({P.MESSAGE, P.HISTORY, P.MEMORY, P.TASKS, P.PROFILE}),
        description="Обычный ответ в диалоге с учётом стиля (FR-08)",
    )
)

CLASSIFY_INTENT = register_task(
    AITaskSpec(
        name="classify_intent",
        tier=Tier.FAST,
        output=IntentClassification,
        max_output_tokens=300,
        data_categories=frozenset({P.MESSAGE, P.HISTORY}),
        system="Определи намерение пользователя. Не придумывай задачи и не выполняй действий.",
        description="Понять, что хочет пользователь, если не сработали команды и кнопки",
    )
)

TRANSCRIBE = register_task(
    AITaskSpec(
        name="transcribe",
        tier=Tier.STT,
        output=None,
        capabilities=frozenset({AUDIO_INPUT}),
        data_categories=frozenset({P.AUDIO}),
        description="Распознавание голосовых сообщений",
    )
)

# Internal: tiny request used by the recovery probe and the key test.
PROBE = register_task(
    AITaskSpec(
        name="probe",
        tier=Tier.FAST,
        output=None,
        max_output_tokens=16,
        data_categories=frozenset(),
        description="Пробный запрос (проверка доступности)",
    )
)
