"""MemoryService (FR-12): provenance, corrections, cascade invalidation of summaries
(acceptance 20, end to end through the ContextBuilder), rebuild, AIUnavailable."""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from procbot.ai.context import ContextBuilder
from procbot.ai.testing import FakeError, make_fake_gateway
from procbot.db.models import AuditLog, MemoryItem, MemorySource, MemorySummary, MemorySummaryItem
from procbot.domain.actions import SaveHypothesis
from procbot.domain.enums import Actor, AIErrorKind, MemoryKind, MemoryOrigin, MemoryStatus
from procbot.services import actions, memory
from procbot.services.memory import SourceRef, SourceType
from procbot.services.profile import update_profile

HYP = "Возможно, ты откладываешь задачи без первого шага"
FACT = "Лучше всего работается утром"


@pytest.fixture
def wired(ctx, profile):
    memory.register(ctx)
    return ctx


async def _add(ctx, content=HYP, *, kind="hypothesis") -> int:
    async with ctx.db.transaction() as s:
        if kind == "hypothesis":
            item = await memory.add_hypothesis(
                s, ctx, content, verification_hint="Неделю начинать с шага на 5 минут"
            )
        else:
            item = await memory.add_fact(s, ctx, content, kind=kind)
        assert item is not None
        return item.id


async def _item(ctx, item_id) -> MemoryItem:
    async with ctx.db.transaction() as s:
        item = await s.get(MemoryItem, item_id)
        assert item is not None
        return item


async def _summaries(ctx) -> list[MemorySummary]:
    async with ctx.db.transaction() as s:
        return list((await s.execute(select(MemorySummary).order_by(MemorySummary.id))).scalars())


async def _context_text(ctx) -> str:
    blocks = await ContextBuilder(ctx, task="chat_reply").build(
        message="как дела", history=0, tasks=0, memory=True, profile=False
    )
    return "\n".join(b.text() for b in blocks)


def _summary_answer(text: str, ids=()):
    return {"summary": text, "used_item_ids": list(ids)}


# --------------------------------------------------------------------------- provenance


async def test_items_keep_kind_origin_and_sources(wired):
    ctx = wired
    async with ctx.db.transaction() as s:
        fact = await memory.add_fact(
            s, ctx, FACT, kind=MemoryKind.PREFERENCE, sources=[SourceRef(SourceType.MESSAGE, 7)]
        )
        measured, created = await memory.add_measured_pattern(
            s, ctx, "Часто отмечаемая причина: «нет сил»", sources=[SourceRef("intervention", 3)]
        )
        hyp = await memory.add_hypothesis(
            s, ctx, HYP, verification_hint="проверить", confidence=0.4
        )
        assert fact and measured and hyp and created
        again = await memory.add_hypothesis(
            s, ctx, HYP + ".", sources=[SourceRef(SourceType.SESSION, 9)]
        )
        assert again is not None and again.id == hyp.id  # dedup, new source linked
        srcs = await memory.item_sources(s, hyp.id)
    assert (fact.kind, fact.origin) == ("preference", "user_said")
    assert (measured.kind, measured.origin) == ("pattern", "system_measured")
    assert (hyp.kind, hyp.origin, hyp.confidence) == ("hypothesis", "ai_inferred", 0.4)
    assert [(x.source_type, x.source_id) for x in srcs] == [("session", 9)]
    assert memory.ORIGIN_LABELS[hyp.origin] == "предположение ИИ"
    assert memory.ORIGIN_LABELS[measured.origin] == "измерено"


async def test_save_hypothesis_action_is_auto_and_ai_inferred(wired):
    ctx = wired
    async with ctx.db.transaction() as s:
        res = await actions.execute(
            s,
            ctx,
            SaveHypothesis(content=HYP, verification_hint="три дня подряд", confidence=0.3),
            origin=Actor.AI,
            op_key="t:h1",
        )
    assert res.status == actions.ExecStatus.APPLIED
    item = await _item(ctx, res.result["memory_item_id"])
    assert (item.kind, item.origin, item.status) == ("hypothesis", "ai_inferred", "active")


async def test_hypotheses_toggle_off_means_nothing_saved(wired):
    ctx = wired
    async with ctx.db.transaction() as s:
        await update_profile(s, now=ctx.clock.now(), memory_settings={"hypotheses": False})
        assert await memory.add_hypothesis(s, ctx, HYP) is None
        res = await actions.execute(
            s, ctx, SaveHypothesis(content=HYP), origin=Actor.AI, op_key="t:h2"
        )
        assert res.status == actions.ExecStatus.INVALID
        await update_profile(s, now=ctx.clock.now(), memory_settings={"enabled": False})
        assert await memory.add_fact(s, ctx, FACT) is None


# --------------------------------------------------------------------------- corrections


async def test_confirm_refute_correct_delete(wired):
    ctx = wired
    hid = await _add(ctx)
    async with ctx.db.transaction() as s:
        r1 = await memory.confirm(s, ctx, hid)
        r2 = await memory.confirm(s, ctx, hid)
    assert r1.changed and not r2.changed
    assert (await _item(ctx, hid)).origin == MemoryOrigin.USER_CONFIRMED

    async with ctx.db.transaction() as s:
        r = await memory.refute(s, ctx, hid)
        again = await memory.refute(s, ctx, hid)
    assert r.changed and not again.changed
    item = await _item(ctx, hid)
    assert (item.status, item.origin) == ("refuted", "user_refuted")

    async with ctx.db.transaction() as s:
        r = await memory.correct(s, ctx, hid, "Откладываю только неясные задачи")
    item = await _item(ctx, hid)
    assert r.changed
    assert (item.status, item.origin, item.content) == (
        "active",
        "user_confirmed",
        "Откладываю только неясные задачи",
    )

    async with ctx.db.transaction() as s:
        await memory.add_sources(s, ctx, item, [SourceRef(SourceType.TASK, 1)])
        assert await memory.delete_item(s, ctx, hid) is True
        assert await memory.delete_item(s, ctx, hid) is False
        n_src = (
            await s.execute(
                select(func.count(MemorySource.id)).where(MemorySource.memory_item_id == hid)
            )
        ).scalar_one()
    item = await _item(ctx, hid)
    assert item.status == MemoryStatus.DELETED and item.content == "" and n_src == 0
    with pytest.raises(memory.MemoryServiceError):
        async with ctx.db.transaction() as s:
            await memory.refute(s, ctx, hid)


async def test_refuted_or_deleted_statement_is_not_re_added(wired):
    ctx = wired
    hid = await _add(ctx)
    async with ctx.db.transaction() as s:
        await memory.refute(s, ctx, hid)
        assert await memory.add_hypothesis(s, ctx, HYP) is None
        assert (await memory.add_measured_pattern(s, ctx, HYP))[0] is None
    other = await _add(ctx, "Возможно, вечером мало сил")
    async with ctx.db.transaction() as s:
        await memory.delete_item(s, ctx, other)
        assert await memory.add_hypothesis(s, ctx, "возможно, вечером мало сил!") is None
        # the audit log keeps only a hash, never the deleted text
        details = (
            (await s.execute(select(AuditLog.details).where(AuditLog.action == "memory.deleted")))
            .scalars()
            .all()
        )
    assert all("вечером" not in str(d) for d in details)


# --------------------------------------------------------------------------- acceptance 20


async def test_acceptance_20_summary_invalidated_and_rebuilt_without_refuted(wired):
    """A hypothesis can be viewed, corrected, refuted and deleted; a summary built
    from it is invalidated in the same transaction, the ContextBuilder never sees
    it, and the rebuilt summary does not contain it."""
    ctx = wired
    fid = await _add(ctx, FACT, kind="fact")
    hid = await _add(ctx)
    seen: list[str] = []

    def answer(req):
        seen.append("\n".join(m.content for m in req.messages))
        return _summary_answer(f"ИИ предполагает: {HYP}", [hid])

    await make_fake_gateway(
        ctx,
        {
            "memory_summarize": [
                _summary_answer(f"Пользователь: {FACT}", [fid]),
                answer,
                lambda req: (
                    seen.append("\n".join(m.content for m in req.messages)),
                    _summary_answer("Гипотез нет."),
                )[1],
            ]
        },
    )
    res = await memory.rebuild_summaries(ctx)
    assert [r.status for r in res] == ["rebuilt", "rebuilt"]
    assert HYP in seen[0] and "<<<DATA" in seen[0]
    assert HYP in await _context_text(ctx)

    # view: the hypothesis is listed with its origin
    async with ctx.db.transaction() as s:
        listed = await memory.list_items(s, kinds=["hypothesis"])
    assert [i.id for i in listed] == [hid]

    async with ctx.db.transaction() as s:
        await memory.refute(s, ctx, hid)
        # same transaction: the summary built from it is already invalid
        rows = list((await s.execute(select(MemorySummary))).scalars())
        assert {r.scope: r.valid for r in rows} == {"profile": True, "patterns": False}
    assert HYP not in await _context_text(ctx)
    assert FACT in await _context_text(ctx)

    # rebuild: only active items go to the model, the new summary is valid
    res = await memory.rebuild_summaries(ctx)
    assert {r.scope: r.status for r in res} == {"profile": "up_to_date", "patterns": "empty"}
    assert len(seen) == 1  # nothing to summarise in "patterns" any more
    summaries = await _summaries(ctx)
    assert all(HYP not in sm.content for sm in summaries)
    assert [sm.scope for sm in summaries] == ["profile"]

    # correction brings it back as user_confirmed; delete scrubs it everywhere
    async with ctx.db.transaction() as s:
        await memory.correct(s, ctx, hid, "Откладываю, если нет первого шага")
    res = await memory.rebuild_summaries(ctx)
    assert {r.scope: r.status for r in res}["patterns"] == "rebuilt"
    assert "Откладываю, если нет первого шага" in seen[-1] and HYP not in seen[-1]
    async with ctx.db.transaction() as s:
        await memory.delete_item(s, ctx, hid)
        n_links = (
            await s.execute(
                select(func.count())
                .select_from(MemorySummaryItem)
                .where(MemorySummaryItem.memory_item_id == hid)
            )
        ).scalar_one()
    assert n_links == 0
    assert [sm.scope for sm in await _summaries(ctx)] == ["profile"]
    assert "первого шага" not in await _context_text(ctx)


async def test_ai_unavailable_keeps_summary_invalid(wired):
    ctx = wired
    hid = await _add(ctx)
    await make_fake_gateway(
        ctx,
        {"memory_summarize": [_summary_answer(HYP, [hid]), *[FakeError(AIErrorKind.SERVER)] * 10]},
    )
    await memory.rebuild_scope(ctx, "patterns")
    other = await _add(ctx, "Возможно, помогает короткий старт")
    async with ctx.db.transaction() as s:
        await memory.refute(s, ctx, hid)
    res = await memory.rebuild_scope(ctx, "patterns")
    assert res.status == "unavailable"
    [sm] = await _summaries(ctx)
    assert sm.valid is False  # stays invalid: not used instead of stale
    text = await _context_text(ctx)
    assert HYP not in text and "короткий старт" in text  # active item still visible
    assert other


async def test_rebuild_discarded_when_item_changes_meanwhile(wired):
    ctx = wired
    hid = await _add(ctx)

    calls: list[int] = []

    def answer(req):
        calls.append(1)
        return _summary_answer(HYP, [hid])

    await make_fake_gateway(ctx, {"memory_summarize": [answer]})
    # emulate a refute committed while the model answers: patch run_task
    import procbot.services.memory as mm

    real = mm.run_task

    async def run_and_refute(*a, **kw):
        result = await real(*a, **kw)
        async with ctx.db.transaction() as s:
            await memory.refute(s, ctx, hid)
        return result

    mm.run_task = run_and_refute  # type: ignore[assignment]
    try:
        res = await memory.rebuild_scope(ctx, "patterns")
    finally:
        mm.run_task = real  # type: ignore[assignment]
    assert res.status == "discarded" and calls == [1]
    assert await _summaries(ctx) == []


async def test_validator_rejects_unknown_item_ids(wired):
    ctx = wired
    hid = await _add(ctx)
    gw = await make_fake_gateway(
        ctx,
        {
            "memory_summarize": [
                _summary_answer("x", [hid, 999]),
                _summary_answer("Сводка", [hid]),
            ]
        },
    )
    res = await memory.rebuild_scope(ctx, "patterns")
    assert res.status == "rebuilt"
    assert len(gw.fake.calls_for("memory_summarize")) == 2  # repaired once
    [sm] = await _summaries(ctx)
    assert sm.content == "Сводка" and sm.valid


async def test_rebuild_job_respects_feature_switch(wired):
    ctx = wired
    await _add(ctx)
    gw = await make_fake_gateway(ctx, {"memory_summarize": [_summary_answer("S")]})
    ctx.config.ai_features.memory_rebuild = False
    await memory.rebuild_job(ctx)
    assert gw.fake.calls_for("memory_summarize") == []
    ctx.config.ai_features.memory_rebuild = True
    await memory.rebuild_job(ctx)
    assert len(gw.fake.calls_for("memory_summarize")) == 1


async def test_wipe_all_clears_items_and_summaries(wired):
    ctx = wired
    hid = await _add(ctx)
    await make_fake_gateway(ctx, {"memory_summarize": [_summary_answer(HYP, [hid])]})
    await memory.rebuild_scope(ctx, "patterns")
    async with ctx.db.transaction() as s:
        res = await memory.wipe_all(s, ctx)
    assert (res.items, res.summaries) == (1, 1)
    async with ctx.db.transaction() as s:
        assert (await s.execute(select(func.count(MemoryItem.id)))).scalar_one() == 0
        assert (await s.execute(select(func.count(MemorySummary.id)))).scalar_one() == 0
    assert HYP not in await _context_text(ctx)
