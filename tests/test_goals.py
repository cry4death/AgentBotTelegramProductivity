"""GoalService (FR-10): creation with roadmap, honest progress (measured vs
reported), alignment of the daily load with goals, critical review whose
proposals only become approvals (FR-15), AIUnavailable fallback, handlers."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from aiogram.methods import EditMessageText, SendMessage
from aiogram.types import InlineKeyboardMarkup
from sqlalchemy import func, select

from procbot.ai.adapters.fake import FakeError
from procbot.ai.testing import make_fake_gateway
from procbot.db.models import Goal, Milestone, PendingApproval, Task, WorkSession
from procbot.domain.actions import ChangeGoal, ChangeMilestone, CreateGoal, TaskDraft
from procbot.domain.enums import Actor, AIErrorKind, ApprovalStatus
from procbot.services import actions, approvals
from procbot.services import goals as gs
from procbot.services import structure as st
from procbot.services import tasks as ts
from procbot.telegram.callbacks import GoalCb
from procbot.telegram.keyboards import Menu
from procbot.testing import callback_update, message_update

NOW = datetime(2026, 10, 26, 9, 0, tzinfo=UTC)


@pytest.fixture
def wired(ctx, profile):
    ts.register(ctx)
    approvals.register(ctx)
    gs.register(ctx)
    return ctx


async def new_goal(ctx, **kw) -> Goal:
    async with ctx.db.transaction() as s:
        return await gs.create_goal(s, ctx, actor=Actor.USER, **kw)


async def new_task(ctx, **kw) -> Task:
    async with ctx.db.transaction() as s:
        return await ts.create_task(s, ctx, TaskDraft(**kw), actor=Actor.USER)


async def complete(ctx, clock, task_id: int, at: datetime) -> None:
    saved = clock.now()
    clock.set(at)
    async with ctx.db.transaction() as s:
        await ts.complete_task(s, ctx, task_id, actor=Actor.USER)
    clock.set(saved)


# --------------------------------------------------------------------------- create


async def test_create_goal_user_applies_ai_needs_approval(wired) -> None:
    ctx = wired
    action = CreateGoal(
        title="Испанский B1",
        measurable_outcome="Сдать DELE B1",
        horizon_date=date(2027, 6, 30),
        milestones=["!Пройти A2", "Разговорный клуб"],
    )
    async with ctx.db.transaction() as s:
        res = await actions.execute(s, ctx, action, origin=Actor.USER, op_key="t:1")
        assert res.status is actions.ExecStatus.APPLIED
        gid = int(res.result["goal_id"])
        ms = await gs.goal_milestones(s, gid)
        assert [(m.title, m.is_key) for m in ms] == [
            ("Пройти A2", True),
            ("Разговорный клуб", False),
        ]
        ai = await actions.execute(
            s, ctx, CreateGoal(title="Марафон"), origin=Actor.AI, op_key="t:2"
        )
        assert ai.status is actions.ExecStatus.PENDING_APPROVAL
        assert (await s.execute(select(func.count()).select_from(Goal))).scalar_one() == 1
        # Changing a goal or a key milestone proposed by AI waits for consent too.
        for i, a in enumerate(
            [
                ChangeGoal(goal_id=gid, horizon_date=date(2027, 12, 31), reason="нереально"),
                ChangeMilestone(milestone_id=ms[0].id, status="done"),
            ]
        ):
            r = await actions.execute(s, ctx, a, origin=Actor.AI, op_key=f"t:c{i}")
            assert r.status is actions.ExecStatus.PENDING_APPROVAL
        goal = await s.get(Goal, gid)
        assert goal is not None and goal.horizon_date == date(2027, 6, 30)
        assert ms[0].status == "active"
        card = (await approvals.list_pending(s))[-1]
        assert card.diff["summary"] == "Изменить ключевой этап «Пройти A2»"


# --------------------------------------------------------------------------- progress


async def test_goal_progress_keeps_measured_and_reported_apart(wired, clock) -> None:
    ctx = wired
    g = await new_goal(ctx, title="Запустить сайт", milestones=["!Дизайн", "Вёрстка", "Запуск"])
    async with ctx.db.transaction() as s:
        project = await st.create_project(s, ctx, title="Сайт", actor=Actor.USER, goal_id=g.id)
        pm = await st.create_milestone(
            s, ctx, title="Тексты", actor=Actor.USER, project_id=project.id
        )
        ms = await gs.goal_milestones(s, g.id)
        await gs.update_milestone(s, ctx, ms[0].id, actor=Actor.USER, status="done")
    direct = await new_task(ctx, title="Купить домен", goal_id=g.id)
    via_project = await new_task(ctx, title="Сверстать главную", project_id=project.id)
    via_milestone = await new_task(ctx, title="Написать о нас", milestone_id=pm.id)
    cancelled = await new_task(ctx, title="Лишнее", goal_id=g.id)
    sub_parent = await new_task(ctx, title="Отдельная", goal_id=None)
    await complete(ctx, clock, direct.id, NOW - timedelta(days=2))
    await complete(ctx, clock, via_milestone.id, NOW - timedelta(days=1))
    async with ctx.db.transaction() as s:
        await ts.cancel_task(s, ctx, cancelled.id, actor=Actor.USER)
        await gs.update_goal(s, ctx, g.id, actor=Actor.USER, progress_note="примерно 40%")
        p = await gs.goal_progress(s, ctx, g.id)
    assert (p.milestones_done, p.milestones_total, p.key_done, p.key_total) == (1, 4, 1, 1)
    assert (p.tasks_done, p.tasks_total, p.tasks_open) == (2, 3, 1)
    assert p.reported == "примерно 40%"
    assert p.last_activity_at == NOW - timedelta(days=1)
    lines = gs.progress_lines(p)
    assert lines[0] == "📏 Измерено: этапы 1/4 (ключевые 1/1) · связанные задачи 2/3 закрыто"
    assert lines[1] == "🗣 Со слов: примерно 40%"
    d = p.as_dict()
    assert d["reported"] == "примерно 40%" and d["measured"]["tasks_done"] == 2
    assert via_project.id in {t.id for t in await _open_linked(ctx, g.id)}
    assert sub_parent.goal_id is None

    empty = await new_goal(ctx, title="Пустая")
    async with ctx.db.transaction() as s:
        pe = await gs.goal_progress(s, ctx, empty.id)
    assert not pe.has_measure
    assert gs.progress_lines(pe) == [
        "📏 Измерено: пока нечего — нет этапов и связанных задач",
        "🗣 Со слов: не указано",
    ]


async def _open_linked(ctx, gid: int) -> list[Task]:
    async with ctx.db.transaction() as s:
        return await gs.open_linked_tasks(s, gid)


# --------------------------------------------------------------------------- alignment


async def test_alignment_detects_goals_without_activity(wired, clock) -> None:
    ctx = wired
    a = await new_goal(ctx, title="Испанский")
    b = await new_goal(ctx, title="Марафон")
    t1 = await new_task(ctx, title="Урок 1", goal_id=a.id)
    t2 = await new_task(ctx, title="Урок 2", goal_id=a.id)
    t3 = await new_task(ctx, title="Купить молоко")
    old = await new_task(ctx, title="Старая пробежка", goal_id=b.id)
    for t in (t1, t2, t3):
        await complete(ctx, clock, t.id, NOW - timedelta(days=3))
    await complete(ctx, clock, old.id, NOW - timedelta(days=30))
    async with ctx.db.transaction() as s:
        s.add(
            WorkSession(
                task_id=t1.id,
                mode="free",
                status="finished",
                started_at=NOW - timedelta(days=3, minutes=50),
                ended_at=NOW - timedelta(days=3, minutes=5),
                time_source="exact",
                created_at=NOW,
                updated_at=NOW,
            )
        )
    async with ctx.db.transaction() as s:
        rep = await gs.goal_alignment(s, ctx, NOW - timedelta(days=14))
    assert (rep.total_done, rep.unlinked_done) == (3, 1)
    by_id = {x.goal_id: x for x in rep.goals}
    assert by_id[a.id].tasks_done == 2 and by_id[a.id].share_of_done == pytest.approx(2 / 3, 1e-3)
    assert by_id[a.id].minutes_exact == 45
    assert [x.goal_id for x in rep.idle_goals] == [b.id]
    assert rep.contradiction
    text = "\n".join(gs.alignment_lines(rep))
    assert "по целям «Марафон» за период не было ничего" in text
    assert rep.as_dict()["idle_goals"] == [{"goal_id": b.id, "title": "Марафон"}]


# --------------------------------------------------------------------------- review


async def _review_fixture(ctx) -> tuple[Goal, Milestone]:
    g = await new_goal(
        ctx,
        title="Испанский B1",
        measurable_outcome="DELE B1",
        horizon_date=date(2026, 12, 1),
        milestones=["!A2", "Клуб"],
    )
    async with ctx.db.transaction() as s:
        ms = await gs.goal_milestones(s, g.id)
    return g, ms[1]


async def test_goal_review_proposals_become_approvals(wired) -> None:
    ctx = wired
    g, ms = await _review_fixture(ctx)
    script = {
        "summary": "Срок выглядит слишком оптимистичным.",
        "observations": ["За 14 дней ни одной задачи по цели."],
        "changes": [
            {"target": "goal", "goal_id": g.id, "horizon_date": "2027-06-30", "reason": "темп"},
            {
                "target": "milestone",
                "milestone_id": ms.id,
                "status": "archived",
                "reason": "не нужен",
            },
            {"target": "goal", "goal_id": 9999, "title": "Чужая", "reason": "x"},
            {"target": "goal", "goal_id": g.id, "measurable_outcome": "DELE B1", "reason": "same"},
        ],
        "questions": ["Сколько часов в неделю реально есть?"],
    }
    gw = await make_fake_gateway(ctx, {"goal_review": [script]})
    out = await gs.run_goal_review(ctx, op_prefix="u1:goalrev", goal_id=g.id)
    assert out.used_ai and len(out.approval_ids) == 2 and len(out.dropped) == 2
    assert "Наблюдения" in out.text and "Сколько часов" in out.text
    call = gw.fake.calls_for("goal_review")[0]
    assert "<<<DATA" in call.user and "Испанский B1" in call.user
    async with ctx.db.transaction() as s:
        cards = [await approvals.get_approval(s, i) for i in out.approval_ids]
        assert {c.action_type for c in cards if c} == {"change_goal", "change_milestone"}
        assert all(
            c is not None and c.origin == "ai" and c.group_key == "u1:goalrev" for c in cards
        )
        goal = await s.get(Goal, g.id)
        assert goal is not None and goal.horizon_date == date(2026, 12, 1)  # nothing applied
        row = await s.get(Milestone, ms.id)
        assert row is not None and row.status == "active"
    async with ctx.db.transaction() as s:
        res = await approvals.approve_group(s, ctx, out.approval_ids[0])
        assert all(r.status == ApprovalStatus.APPLIED for r in res)
        goal = await s.get(Goal, g.id)
        assert goal is not None and goal.horizon_date == date(2027, 6, 30)
        row = await s.get(Milestone, ms.id)
        assert row is not None and row.status == "archived"
    # Running the same review op again does not duplicate cards.
    gw.fake.push("goal_review", script)
    await gs.run_goal_review(ctx, op_prefix="u1:goalrev", goal_id=g.id)
    async with ctx.db.transaction() as s:
        n = (await s.execute(select(func.count()).select_from(PendingApproval))).scalar_one()
    assert n == 2


async def test_goal_review_without_ai_is_code_only(wired) -> None:
    ctx = wired
    g, _ = await _review_fixture(ctx)
    bare = await new_goal(ctx, title="Без плана")
    out = await gs.run_goal_review(ctx, op_prefix="u2:goalrev")  # ctx.ai is None
    assert not out.used_ai and out.approval_ids == []
    assert "ИИ сейчас недоступен" in out.text
    assert "«Без плана»: нет дорожной карты" in out.text
    assert "«Испанский B1»: нет критериев успеха." in out.text
    await make_fake_gateway(ctx, {"goal_review": [FakeError(AIErrorKind.SERVER)] * 10})
    out = await gs.run_goal_review(ctx, op_prefix="u3:goalrev", goal_id=bare.id)
    assert not out.used_ai and "ИИ сейчас недоступен" in out.text
    async with ctx.db.transaction() as s:
        n = (await s.execute(select(func.count()).select_from(PendingApproval))).scalar_one()
    assert n == 0
    assert g.id


# --------------------------------------------------------------------------- handlers


@pytest.fixture
def bot(bot_and_session):
    return bot_and_session[0]


@pytest.fixture
def tg(bot_and_session):
    return bot_and_session[1]


def last_markup(tg) -> InlineKeyboardMarkup:
    for r in reversed(tg.requests):
        if isinstance(r, SendMessage | EditMessageText) and isinstance(
            r.reply_markup, InlineKeyboardMarkup
        ):
            return r.reply_markup
    raise AssertionError("no inline keyboard sent")


def button(markup: InlineKeyboardMarkup, prefix: str) -> str:
    for row in markup.inline_keyboard:
        for b in row:
            if b.text.startswith(prefix):
                assert b.callback_data is not None
                return b.callback_data
    raise AssertionError(
        f"no {prefix!r} in {[[b.text for b in r] for r in markup.inline_keyboard]}"
    )


def last_text(tg) -> str:
    return tg.sent_texts()[-1]


async def test_goal_menu_create_milestone_link_and_review(wired, dp, bot, tg) -> None:
    ctx = wired
    await dp.feed_update(bot, message_update(Menu.GOALS))
    assert "Целей пока нет" in last_text(tg)
    await dp.feed_update(bot, callback_update(button(last_markup(tg), "➕ Новая цель")))
    await dp.feed_update(bot, message_update("Пробежать полумарафон"))
    await dp.feed_update(bot, callback_update(button(last_markup(tg), "Пропустить")))
    assert "К какому сроку" in last_text(tg)
    await dp.feed_update(bot, message_update("31.05.2027"))
    await dp.feed_update(bot, message_update("!10 км без остановки\nЗабег 15 км"))
    card = last_text(tg)
    assert "Цель добавлена" in card and "🔑 10 км без остановки" in card
    assert "Горизонт: 31.05.2027" in card and "Со слов: не указано" in card
    async with ctx.db.transaction() as s:
        (goal,) = list((await s.execute(select(Goal))).scalars())
    assert goal.measurable_outcome is None and goal.horizon_date == date(2027, 5, 31)

    # Mark a milestone done; a repeated press is a no-op, not a toggle.
    await dp.feed_update(bot, callback_update(button(last_markup(tg), "✅ Отметить этап")))
    done_cb = button(last_markup(tg), "○ 🔑")
    await dp.feed_update(bot, callback_update(done_cb))
    await dp.feed_update(bot, callback_update(done_cb))
    async with ctx.db.transaction() as s:
        ms = await gs.goal_milestones(s, goal.id)
    assert [m.status for m in ms] == ["done", "active"]

    # Link an open task to the goal.
    task = await new_task(ctx, title="Купить кроссовки")
    await dp.feed_update(bot, callback_update(GoalCb(action="link", goal_id=goal.id).pack()))
    await dp.feed_update(bot, callback_update(button(last_markup(tg), "Купить кроссовки")))
    assert "Привязал «Купить кроссовки»" in last_text(tg)
    async with ctx.db.transaction() as s:
        t = await s.get(Task, task.id)
    assert t is not None and t.goal_id == goal.id
    assert "• Купить кроссовки" in last_text(tg)

    # Reported progress is stored as the owner's words.
    await dp.feed_update(
        bot, callback_update(GoalCb(action="ef", goal_id=goal.id, arg="reported").pack())
    )
    await dp.feed_update(bot, message_update("бегаю 3 раза в неделю"))
    assert "🗣 Со слов: бегаю 3 раза в неделю" in last_text(tg)

    # Review without AI: code-only summary, no cards.
    await dp.feed_update(bot, callback_update(GoalCb(action="review", goal_id=goal.id).pack()))
    assert "ИИ сейчас недоступен" in last_text(tg)
    assert "не записан измеримый результат" in last_text(tg)


async def test_review_button_sends_approval_cards(wired, dp, bot, tg) -> None:
    ctx = wired
    g, _ = await _review_fixture(ctx)
    await make_fake_gateway(
        ctx,
        {
            "goal_review": [
                {
                    "summary": "Пересмотреть срок.",
                    "changes": [
                        {
                            "target": "goal",
                            "goal_id": g.id,
                            "horizon_date": "2027-03-01",
                            "reason": "мало времени",
                        }
                    ],
                }
            ]
        },
    )
    await dp.feed_update(bot, callback_update(GoalCb(action="review", goal_id=g.id).pack()))
    texts = tg.sent_texts()
    assert any("Пересмотреть срок" in t for t in texts)
    assert "Предлагаемые изменения целей" in last_text(tg)
    assert "Горизонт: 01.12.2026 → 01.03.2027" in last_text(tg)
    await dp.feed_update(bot, callback_update(button(last_markup(tg), "✅ Принять")))
    async with ctx.db.transaction() as s:
        goal = await s.get(Goal, g.id)
    assert goal is not None and goal.horizon_date == date(2027, 3, 1)


async def test_projects_create_and_attach_to_goal(wired, dp, bot, tg) -> None:
    ctx = wired
    g = await new_goal(ctx, title="Сайт")
    await dp.feed_update(bot, callback_update(GoalCb(action="projects").pack()))
    assert "Проектов пока нет" in last_text(tg)
    await dp.feed_update(bot, callback_update(button(last_markup(tg), "➕ Новый проект")))
    await dp.feed_update(bot, message_update("Лендинг"))
    assert "Проект создан" in last_text(tg) and "Цель: не привязан" in last_text(tg)
    await dp.feed_update(bot, callback_update(button(last_markup(tg), "🎯 Привязать к цели")))
    await dp.feed_update(bot, callback_update(button(last_markup(tg), "Сайт")))
    assert "Цель: Сайт" in last_text(tg)
    async with ctx.db.transaction() as s:
        projects = await gs.goal_projects(s, g.id)
    assert [p.title for p in projects] == ["Лендинг"]
