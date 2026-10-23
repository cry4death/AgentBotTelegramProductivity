"""Persona / style text for AI prompts from the owner's personality knobs (FR-08).

    from procbot.ai.persona import persona_for, load_persona
    prompt = Prompt(instruction=..., data=blocks, persona=await load_persona(ctx))

The style (preset + strictness / humor / profanity / verbosity, each independent)
changes only HOW the bot talks. The guardrails are ALWAYS appended, whatever the
style (FR-08, spec section 1, acceptance 10): no humiliation, no false
accusations, no manipulative pressure, tiredness is never called a lie or
"simulation", no diagnoses, hypotheses are stated as hypotheses, and the model
never claims it changed data (only the application does, after confirmation).

`tone()` is a tiny helper for template (non-AI) replies that want a slightly
different wording for strict vs. soft styles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from procbot.context import AppContext
from procbot.services.profile import get_profile

__all__ = [
    "GUARDRAILS",
    "SAFE_REPLY",
    "PersonaKnobs",
    "guard_reply",
    "guardrails_text",
    "load_persona",
    "persona_for",
    "tone",
]

PRESET_STYLE: dict[str, str] = {
    "neutral": "Спокойный деловой помощник: дружелюбно, по существу, без лишних эмоций.",
    "coach": (
        "Строгий тренер: прямолинейно, требовательно, с упором на следующий конкретный шаг. "
        "Жёсткость — к задаче, не к человеку."
    ),
    "ironic": (
        "Ироничный напарник: лёгкая самоирония и подколки по поводу ситуации, "
        "но никогда не над человеком и его состоянием."
    ),
    "soft": "Мягкий и поддерживающий: бережно, с признанием усилий, без давления.",
}

_STRICTNESS = (
    "Строгость минимальная: только предлагай, не настаивай.",
    "Строгость низкая: мягко напоминай о договорённостях.",
    "Строгость средняя: напоминай о договорённостях прямо, предлагай конкретный шаг.",
    "Строгость высокая: говори прямо, не принимай расплывчатых ответов, проси конкретики.",
    "Строгость максимальная: очень прямолинейно и требовательно к делу — но без унижений.",
)
_HUMOR = (
    "Без шуток.",
    "Юмор изредка и очень аккуратно.",
    "Умеренный юмор, если к месту.",
    "Юмор приветствуется, можно иронизировать над ситуацией.",
    "Много юмора и иронии над ситуацией (не над человеком).",
)
_VERBOSITY = (
    "Отвечай предельно кратко: одно-два предложения.",
    "Отвечай кратко: до трёх предложений.",
    "Средняя длина ответа: коротко, но с нужными пояснениями.",
    "Можно подробнее: объясняй рассуждения и варианты.",
    "Подробные ответы с пояснениями и примерами.",
)

GUARDRAILS: tuple[str, ...] = (
    "Никаких унижений, оскорблений и насмешек над человеком — при любом стиле.",
    "Никаких ложных обвинений: не утверждай того, чего нет в данных.",
    "Никакого манипулятивного давления: без шантажа, стыжения, запугивания и чувства вины.",
    "Никогда не называй усталость или плохое самочувствие ложью, симуляцией или отговоркой; "
    "если человек устал — помоги уменьшить объём или пересмотреть план.",
    "Не ставь диагнозов (медицинских и психологических), ты не врач и не психотерапевт.",
    "Гипотезы о причинах прокрастинации формулируй как предположения («возможно…», "
    "«похоже, что…») и предлагай способ проверить; никогда не выдавай их за факт.",
    "Не требуй раскрывать личные сведения.",
    "Ты не меняешь данные сам и не говоришь, что уже создал, перенёс, удалил или изменил "
    "задачи, события календаря или настройки: изменения делает только приложение после "
    "подтверждения пользователя.",
)


@dataclass(frozen=True)
class PersonaKnobs:
    style_preset: str = "neutral"
    strictness: int = 2
    humor: int = 1
    profanity_allowed: bool = False
    verbosity: int = 2

    @classmethod
    def from_profile(cls, profile: Any | None) -> PersonaKnobs:
        if profile is None:
            return cls()
        return cls(
            style_preset=getattr(profile, "style_preset", None) or "neutral",
            strictness=_knob(getattr(profile, "strictness", None), 2),
            humor=_knob(getattr(profile, "humor", None), 1),
            profanity_allowed=bool(getattr(profile, "profanity_allowed", False)),
            verbosity=_knob(getattr(profile, "verbosity", None), 2),
        )


def _knob(value: Any, default: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, min(4, v))


def guardrails_text() -> str:
    return "Обязательные правила (важнее стиля):\n" + "\n".join(f"- {g}" for g in GUARDRAILS)


def persona_for(profile: Any | None = None, *, knobs: PersonaKnobs | None = None) -> str:
    """Trusted persona text for `Prompt.persona` (style + guardrails)."""
    k = knobs or PersonaKnobs.from_profile(profile)
    lines = [
        "Ты общаешься на русском, на «ты».",
        PRESET_STYLE.get(k.style_preset, PRESET_STYLE["neutral"]),
        _STRICTNESS[k.strictness],
        _HUMOR[k.humor],
        _VERBOSITY[k.verbosity],
        (
            "Ненормативная лексика допустима изредка и к месту — никогда не в адрес человека."
            if k.profanity_allowed
            else "Без ненормативной лексики."
        ),
    ]
    return "\n".join(lines) + "\n\n" + guardrails_text()


async def load_persona(ctx: AppContext) -> str:
    """Read the profile in a short transaction and build the persona text."""
    async with ctx.db.transaction() as s:
        profile = await get_profile(s)
        knobs = PersonaKnobs.from_profile(profile)
    return persona_for(knobs=knobs)


def tone(
    profile: Any | None, *, neutral: str, strict: str | None = None, soft: str | None = None
) -> str:
    """Pick a template wording by style: strict for coach / strictness >= 3,
    soft for the soft preset / strictness <= 0, else neutral."""
    k = PersonaKnobs.from_profile(profile)
    if strict is not None and (k.style_preset == "coach" or k.strictness >= 3):
        return strict
    if soft is not None and (k.style_preset == "soft" or k.strictness <= 0):
        return soft
    return neutral


# --------------------------------------------------------------------------- post-filter

# A model that ignores the guardrails still must not reach the owner with an
# accusation of faking tiredness or a diagnosis. Deliberately narrow patterns:
# they target statements ABOUT the user, not mentions of the words.
_FORBIDDEN_REPLY = re.compile(
    r"(?:"
    r"\bты\s+(?:просто\s+)?(?:симулиру\w*|притворя\w*|прикидыва\w*|врёшь|врешь|лжёшь|лжешь"
    r"|ленив\w*|лентяй\w*|ничтожеств\w*)"
    r"|(?:твоя|твоё|твое|эта)\s+усталость\s+[—–-]?\s*(?:это\s+)?(?:ложь|отговорк\w*|симуляц\w*|выдумк\w*)"
    r"|усталость\s+[—–-]\s*(?:это\s+)?(?:ложь|отговорк\w*|симуляц\w*)"
    r"|\bу\s+тебя\s+(?:явно\s+|точно\s+|определённо\s+|определенно\s+)?"
    r"(?:депресси\w*|сдвг|adhd|биполяр\w*|тревожное\s+расстройств\w*|расстройств\w*)"
    r"|(?:ставлю|мой)\s+диагноз"
    r")",
    re.IGNORECASE,
)

SAFE_REPLY = (
    "Похоже, сейчас тяжело — это нормально, и это не повод себя ругать. "
    "Давай уменьшим объём: выбери один самый маленький шаг на 5–10 минут "
    "или перенесём часть дел. Решать тебе."
)


def guard_reply(text: str) -> tuple[str, bool]:
    """(text to show, filtered?). Replaces replies that accuse the owner of
    faking tiredness / lying or state a diagnosis (FR-08, acceptance 10)."""
    if _FORBIDDEN_REPLY.search(text or ""):
        return SAFE_REPLY, True
    return text, False
