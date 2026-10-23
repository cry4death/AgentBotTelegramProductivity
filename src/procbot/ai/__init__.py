"""AI gateway (provider-agnostic). Public API for domain code:

    from procbot.ai import AIUnavailable, AITaskSpec, DataBlock, Prompt, Tier, register_task
    result = await ctx.ai.run("task_name", Prompt(...), initiator="services.x")

See procbot.ai.gateway for details. Importing this package registers the
built-in tasks (chat_reply, classify_intent, transcribe, probe).
"""

from procbot.ai.audit import record_actions
from procbot.ai.errors import AIUnavailable, AIUsageError
from procbot.ai.prompting import DataBlock, Prompt
from procbot.ai.registry import AITaskSpec, IntentClassification, get_task, register_task
from procbot.ai.types import AIResult, TextOutput, Tier, TranscriptResult

__all__ = [
    "AIResult",
    "AITaskSpec",
    "AIUnavailable",
    "AIUsageError",
    "DataBlock",
    "IntentClassification",
    "Prompt",
    "TextOutput",
    "Tier",
    "TranscriptResult",
    "get_task",
    "record_actions",
    "register_task",
]
