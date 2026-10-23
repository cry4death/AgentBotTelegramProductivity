"""What memory reaches the model context (FR-12): settings toggles, the stale
summary guard and the full wipe, checked through the ContextBuilder."""

from __future__ import annotations

import pytest
from sqlalchemy import update

from procbot.ai.context import ContextBuilder
from procbot.db.models import MemoryItem, MemorySummary, MemorySummaryItem
from procbot.domain.enums import MemoryStatus
from procbot.services import memory
from procbot.services.profile import update_profile

HYP = "Возможно, ты откладываешь задачи без первого шага"
FACT = "Лучше всего работается утром"


@pytest.fixture
def wired(ctx, profile):
    memory.register(ctx)
    return ctx


async def _context_text(ctx) -> str:
    blocks = await ContextBuilder(ctx, task="chat_reply").build(
        message="как дела", history=0, tasks=0, memory=True, profile=False
    )
    return "\n".join(b.text() for b in blocks)


async def _seed(ctx) -> tuple[int, int]:
    async with ctx.db.transaction() as s:
        fact = await memory.add_fact(s, ctx, FACT)
        hyp = await memory.add_hypothesis(s, ctx, HYP)
        assert fact is not None and hyp is not None
        return fact.id, hyp.id


async def test_settings_toggles_control_what_reaches_the_model(wired):
    ctx = wired
    await _seed(ctx)
    text = await _context_text(ctx)
    assert FACT in text and HYP in text

    async with ctx.db.transaction() as s:
        await update_profile(s, now=ctx.clock.now(), memory_settings={"hypotheses": False})
    text = await _context_text(ctx)
    assert FACT in text
    assert HYP not in text

    async with ctx.db.transaction() as s:
        await update_profile(s, now=ctx.clock.now(), memory_settings={"enabled": False})
    text = await _context_text(ctx)
    assert FACT not in text and HYP not in text


async def test_summary_from_inactive_item_is_never_used_even_if_still_flagged_valid(wired):
    ctx = wired
    fact_id, _ = await _seed(ctx)
    async with ctx.db.transaction() as s:
        sm = MemorySummary(
            scope="profile", content="СВОДКА-ПРОФИЛЬ", valid=True, created_at=ctx.clock.now()
        )
        s.add(sm)
        await s.flush()
        s.add(MemorySummaryItem(summary_id=sm.id, memory_item_id=fact_id))
    assert "СВОДКА-ПРОФИЛЬ" in await _context_text(ctx)

    # The item leaves "active" without the service invalidating the summary
    # (e.g. a bug or a manual edit): the context builder must still skip it.
    async with ctx.db.transaction() as s:
        await s.execute(
            update(MemoryItem).where(MemoryItem.id == fact_id).values(status=MemoryStatus.REFUTED)
        )
    text = await _context_text(ctx)
    assert "СВОДКА-ПРОФИЛЬ" not in text
    assert FACT not in text


async def test_wipe_all_leaves_nothing_for_the_model(wired):
    ctx = wired
    await _seed(ctx)
    async with ctx.db.transaction() as s:
        res = await memory.wipe_all(s, ctx)
    assert res.items == 2
    text = await _context_text(ctx)
    assert FACT not in text and HYP not in text
    async with ctx.db.transaction() as s:
        assert await memory.list_items(s) == []
