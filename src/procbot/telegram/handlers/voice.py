"""Voice pipeline (FR-02, acceptance 5): voice/audio -> `transcribe` (stt tier)
-> "🎙 Распознал: …" -> the same text pipeline as typed text (handlers.chat).

The bot answers ONLY with text: no voice, no audio, no TTS. Network work
(download, transcription) never runs inside a DB transaction. Feature switch:
AppConfig.ai_features.voice. AIUnavailable -> a polite request to type it.
"""

from __future__ import annotations

import io
import logging

from aiogram import Bot, F, Router
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from procbot.ai.errors import AIUnavailable
from procbot.ai.gateway import feature_enabled, get_gateway
from procbot.context import AppContext
from procbot.telegram.handlers.chat import process_text

log = logging.getLogger(__name__)

router = Router(name="voice")

# UNCONFIRMED technical limits (docs/open-questions.md). Telegram bots cannot
# download files over 20 MB anyway.
MAX_VOICE_SECONDS = 300
MAX_VOICE_BYTES = 20 * 1024 * 1024

VOICE_OFF_TEXT = "Голосовые сейчас выключены — напиши, пожалуйста, текстом."
TOO_LONG_TEXT = "Голосовое слишком длинное (до {minutes} мин). Раздели его или напиши текстом."
UNAVAILABLE_TEXT = (
    "Не получилось распознать голосовое — сейчас это недоступно. Напиши, пожалуйста, текстом."
)
EMPTY_TEXT = "Не расслышал слов в голосовом. Повтори или напиши текстом."
BUSY_TEXT = "Сейчас жду ответ на свой вопрос — напиши его текстом (или /cancel)."


@router.message(StateFilter(None), F.voice | F.audio)
async def on_voice(
    message: Message, ctx: AppContext, state: FSMContext, update_id: int, bot: Bot
) -> None:
    media = message.voice or message.audio
    if media is None:  # pragma: no cover - filtered above
        return
    if not feature_enabled(ctx, "voice"):
        await message.answer(VOICE_OFF_TEXT)
        return
    duration = int(media.duration or 0)
    if duration > MAX_VOICE_SECONDS or (media.file_size or 0) > MAX_VOICE_BYTES:
        await message.answer(TOO_LONG_TEXT.format(minutes=MAX_VOICE_SECONDS // 60))
        return
    gw = get_gateway(ctx)
    if gw is None:
        await message.answer(UNAVAILABLE_TEXT)
        return
    mime = media.mime_type or "audio/ogg"
    filename = getattr(media, "file_name", None) or "voice.ogg"
    try:
        buf = io.BytesIO()
        await bot.download(media, destination=buf)
        audio = buf.getvalue()
    except Exception:
        log.warning("voice download failed", exc_info=True)
        await message.answer(UNAVAILABLE_TEXT)
        return
    if not audio or len(audio) > MAX_VOICE_BYTES:
        await message.answer(
            UNAVAILABLE_TEXT if not audio else TOO_LONG_TEXT.format(minutes=MAX_VOICE_SECONDS // 60)
        )
        return
    try:
        result = await gw.transcribe(
            audio,
            mime_type=mime,
            initiator="handlers.voice",
            correlation_id=f"u{update_id}",
            duration_seconds=float(duration) if duration else None,
            filename=filename,
        )
    except AIUnavailable as exc:
        log.info("transcription unavailable: %s", exc.reason)
        await message.answer(UNAVAILABLE_TEXT)
        return
    text = " ".join((result.text or "").split())
    if not text:
        await message.answer(EMPTY_TEXT)
        return
    ctx.registry.setdefault("update_transcripts", {})[update_id] = text
    await message.answer(f"🎙 Распознал: «{text}»")
    await process_text(message, ctx, state, update_id, text)


@router.message(F.voice | F.audio)
async def on_voice_busy(message: Message) -> None:
    """A voice message while another dialog waits for a typed answer."""
    await message.answer(BUSY_TEXT)
