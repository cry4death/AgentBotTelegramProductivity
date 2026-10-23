"""Any OpenAI-format API by base_url: OpenRouter, DeepSeek, Groq, Mistral,
local Ollama / LM Studio / vLLM.

Same wire format as OpenAI chat completions, with the conservative choices
most compatible servers accept: `max_tokens` (not `max_completion_tokens`),
no `strict` flag on tools. Local servers may have no key: then no
Authorization header is sent. Which strategy a server supports is declared per
model in ai.yaml (`structured_output`), not guessed.
"""

from __future__ import annotations

from procbot.ai.adapters.openai import OpenAIAdapter
from procbot.ai.errors import ProviderError
from procbot.ai.types import Credential
from procbot.domain.enums import AIErrorKind


class OpenAICompatibleAdapter(OpenAIAdapter):
    provider = "openai_compatible"
    supports_audio = True  # only if the server implements /audio/transcriptions
    max_tokens_param = "max_tokens"
    strict_tools = False

    def base_url(self, cred: Credential) -> str:
        if not cred.base_url:
            raise ProviderError(AIErrorKind.BAD_REQUEST, "openai_compatible needs base_url")
        return cred.base_url
