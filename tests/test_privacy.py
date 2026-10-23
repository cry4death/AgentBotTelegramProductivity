"""Privacy (spec section 7, FR-12): export without secrets, history scrub, full
wipe with strong confirmation, honest backup/provider notice."""

from __future__ import annotations

import json
from datetime import timedelta

from aiogram.methods import EditMessageText, SendDocument, SendMessage
from aiogram.types import BufferedInputFile, InlineKeyboardMarkup
from sqlalchemy import func, select

from procbot.db.models import (
    AICredential,
    AuditLog,
    CalendarConnection,
    ConversationMessage,
    MemoryItem,
    OutboundMessage,
    Setting,
    Task,
    WorkSession,
)
from procbot.domain.actions import TaskDraft
from procbot.domain.enums import Actor
from procbot.services import memory, privacy
from procbot.services import tasks as ts
from procbot.services.memory import SourceRef, SourceType
from procbot.telegram.keyboards import Menu
from procbot.testing import callback_update, message_update

SECRET_KEY = "sk-SECRET-1234567890"
SECRET_TOKEN = "ya29.TOKEN-abcdef"


async def seed(ctx) -> dict[str, int]:
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        task = await ts.create_task(s, ctx, TaskDraft(title="Написать отчёт"), actor=Actor.USER)
        s.add(
            WorkSession(
                task_id=task.id,
                mode="free",
                status="finished",
                started_at=now - timedelta(hours=2),
                ended_at=now - timedelta(hours=1),
                time_source="retro",
                created_at=now,
                updated_at=now,
            )
        )
        msg = ConversationMessage(
            direction="in", kind="text", text="Я люблю работать утром", created_at=now
        )
        gone = ConversationMessage(
            direction="in", kind="text", text=None, deleted=True, created_at=now
        )
        s.add_all([msg, gone])
        s.add(
            OutboundMessage(
                category="evening_review",
                dedup_key="o1",
                counts_toward_limit=True,
                send_state="sent",
                text="Итоги дня: Написать отчёт",
                created_at=now,
            )
        )
        s.add(
            AICredential(
                label="main",
                provider="openai",
                encrypted_key=SECRET_KEY,  # raw value: proves the column is never exported
                key_last4="7890",
                created_at=now,
                updated_at=now,
            )
        )
        s.add(
            CalendarConnection(
                encrypted_token=SECRET_TOKEN,
                read_calendar_ids=["primary"],
                created_at=now,
                updated_at=now,
            )
        )
        s.add(Setting(key="ai.openai_api_key", value=SECRET_KEY, updated_at=now))
        s.add(Setting(key="ui.lang", value="ru", updated_at=now))
        await s.flush()
        item = await memory.add_fact(
            s, ctx, "Люблю работать утром", sources=[SourceRef(SourceType.MESSAGE, msg.id)]
        )
        deleted = await memory.add_fact(s, ctx, "Удалённый факт")
        assert item and deleted
        await memory.delete_item(s, ctx, deleted.id)
        return {"task": task.id, "item": item.id, "msg": msg.id}


# --------------------------------------------------------------------------- export


async def test_export_has_sections_and_no_secrets(ctx, profile):
    ids = await seed(ctx)
    name, raw = await privacy.export_json(ctx)
    assert name.startswith("procbot-export-") and name.endswith(".json")
    text = raw.decode("utf-8")
    data = json.loads(text)
    for section in (
        "meta",
        "profile",
        "settings",
        "tasks",
        "task_events",
        "goals",
        "habits",
        "plans",
        "sessions",
        "interventions",
        "memory",
        "history",
        "integrations",
    ):
        assert section in data, section
    assert data["profile"]["timezone"] == "Europe/Berlin"
    assert [t["title"] for t in data["tasks"]] == ["Написать отчёт"]
    assert data["sessions"][0]["time_source"] == "retro"
    [item] = data["memory"]["items"]
    assert item["id"] == ids["item"] and item["origin_label"] == "вы сказали"
    assert item["sources"] == [{"type": "message", "id": ids["msg"]}]
    assert [m["text"] for m in data["history"]] == ["Я люблю работать утром"]
    assert data["settings"] == {"ui.lang": "ru"}
    assert data["integrations"]["ai_keys"] == [
        {"label": "main", "provider": "openai", "status": "active"}
    ]
    # no secrets anywhere
    for secret in (SECRET_KEY, SECRET_TOKEN, "7890", "encrypted"):
        assert secret not in text, secret
    assert "Удалённый факт" not in text


async def test_export_document_through_dispatcher(ctx, profile, dp, bot_and_session):
    bot, tg = bot_and_session
    await seed(ctx)
    await dp.feed_update(bot, message_update("/export"))
    [doc] = [r for r in tg.requests if isinstance(r, SendDocument)]
    assert isinstance(doc.document, BufferedInputFile)
    data = json.loads(doc.document.data.decode("utf-8"))
    assert data["meta"]["format"] == "procbot-export"
    assert "Ключей ИИ и токенов в нём нет" in (doc.caption or "")


# --------------------------------------------------------------------------- wipe


def _markup(tg) -> InlineKeyboardMarkup:
    for r in reversed(tg.requests):
        if isinstance(r, SendMessage | EditMessageText) and isinstance(
            r.reply_markup, InlineKeyboardMarkup
        ):
            return r.reply_markup
    raise AssertionError("no keyboard")


def _btn(markup: InlineKeyboardMarkup, prefix: str) -> str:
    for row in markup.inline_keyboard:
        for b in row:
            if b.text.startswith(prefix) and b.callback_data:
                return b.callback_data
    raise AssertionError(prefix)


async def test_wipe_memory_and_history_scrubs_text(ctx, profile, dp, bot_and_session):
    bot, tg = bot_and_session
    await seed(ctx)
    await dp.feed_update(bot, message_update(Menu.MEMORY))
    await dp.feed_update(bot, callback_update(_btn(_markup(tg), "🧹 Очистить")))
    wipe_text = tg.sent_texts()[-1]
    assert "автоматически не удаляются" in wipe_text and "провайдер" in wipe_text
    await dp.feed_update(bot, callback_update(_btn(_markup(tg), "🧠 Память и история")))
    await dp.feed_update(bot, callback_update(_btn(_markup(tg), "🧹 Да, удалить")))
    assert "Готово" in tg.sent_texts()[-1]
    async with ctx.db.transaction() as s:
        rows = list((await s.execute(select(ConversationMessage))).scalars())
        out = list((await s.execute(select(OutboundMessage))).scalars())
        n_mem = (await s.execute(select(func.count(MemoryItem.id)))).scalar_one()
        n_tasks = (await s.execute(select(func.count(Task.id)))).scalar_one()
    seeded = [r for r in rows if r.update_id is None]  # rows logged before the wipe
    assert seeded and all(r.deleted and r.text is None and r.transcript is None for r in seeded)
    assert all("Написать отчёт" not in o.text for o in out)
    assert n_mem == 0 and n_tasks == 1  # tasks stay


async def test_wipe_everything_needs_typed_phrase(ctx, profile, dp, bot_and_session):
    bot, tg = bot_and_session
    ids = await seed(ctx)
    async with ctx.db.transaction() as s:
        other = await ts.create_task(s, ctx, TaskDraft(title="Секретный проект"), actor=Actor.USER)
        await ts.delete_task(s, ctx, other.id, actor=Actor.USER)  # snapshot lands in the audit log
    await dp.feed_update(bot, message_update(Menu.MEMORY))
    await dp.feed_update(bot, callback_update(_btn(_markup(tg), "🧹 Очистить")))
    wipe_kb = _markup(tg)
    await dp.feed_update(bot, callback_update(_btn(wipe_kb, "⚠️ Всё")))
    assert privacy.WIPE_ALL_PHRASE in tg.sent_texts()[-1]
    await dp.feed_update(bot, message_update("да"))
    assert "ничего не удалено" in tg.sent_texts()[-1]
    async with ctx.db.transaction() as s:
        assert await s.get(Task, ids["task"]) is not None

    await dp.feed_update(bot, callback_update(_btn(wipe_kb, "⚠️ Всё")))
    await dp.feed_update(bot, message_update(privacy.WIPE_ALL_PHRASE))
    assert "Всё удалено" in tg.sent_texts()[-1]
    async with ctx.db.transaction() as s:
        assert (await s.execute(select(func.count(Task.id)))).scalar_one() == 0
        assert (await s.execute(select(func.count(WorkSession.id)))).scalar_one() == 0
        assert (await s.execute(select(func.count(MemoryItem.id)))).scalar_one() == 0
        details = (await s.execute(select(AuditLog.details))).scalars().all()
        creds = (await s.execute(select(func.count(AICredential.id)))).scalar_one()
    assert "Секретный проект" not in json.dumps(details, ensure_ascii=False)
    assert creds == 1  # keys and settings stay


def test_retention_notice_follows_config(ctx):
    assert "автоматически не удаляются" in privacy.retention_notice(ctx)
    ctx.config.backup.retention_days = 30
    text = privacy.retention_notice(ctx)
    assert "30 дн." in text and "мгновенно стереть" in text
