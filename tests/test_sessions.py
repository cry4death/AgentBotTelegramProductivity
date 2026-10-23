"""WorkSessionService (FR-06): modes, timer signals, pauses, outcomes, retro entries,
stats and restart recovery. Acceptance criteria 2, 14, 15, 16, 17."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, select

from procbot.context import AppContext
from procbot.db.models import (
    Intervention,
    ReminderSchedule,
    SessionCorrection,
    Task,
    TaskEvent,
    WorkSession,
)
from procbot.db.session import Database
from procbot.domain.enums import (
    MessageCategory,
    ObstacleReason,
    ReminderState,
    SessionMode,
    SessionOutcome,
    SessionStatus,
    TaskStatus,
    TimeSource,
)
from procbot.services import sessions as svc
from procbot.services.outbound import Outgoing, get_renderer, register_renderer
from procbot.services.profile import update_profile
from procbot.services.reminders import schedule_reminder

T0 = datetime(2026, 10, 26, 9, 0, tzinfo=UTC)  # 10:00 in Berlin
BERLIN = "Europe/Berlin"


async def _task(ctx: AppContext, title: str = "Отчёт", **kw: object) -> int:
    now = ctx.clock.now()
    async with ctx.db.transaction() as s:
        t = Task(title=title, created_at=now, updated_at=now, **kw)
        s.add(t)
        await s.flush()
        return t.id


async def _settings(ctx: AppContext, **fields: object) -> None:
    async with ctx.db.transaction() as s:
        await update_profile(s, now=ctx.clock.now(), **fields)


async def _timers(ctx: AppContext, *, state: str | None = None) -> list[ReminderSchedule]:
    async with ctx.db.transaction() as s:
        stmt = select(ReminderSchedule).where(
            ReminderSchedule.category == str(MessageCategory.TIMER_SIGNAL)
        )
        if state is not None:
            stmt = stmt.where(ReminderSchedule.state == state)
        return list((await s.execute(stmt.order_by(ReminderSchedule.id))).scalars().all())


async def _pending(ctx: AppContext) -> list[ReminderSchedule]:
    return await _timers(ctx, state=str(ReminderState.PENDING))


async def _start(ctx: AppContext, mode: SessionMode, task_id: int | None = None) -> int:
    async with ctx.db.transaction() as s:
        ws = await svc.start_session(s, ctx, task_id=task_id, mode=mode)
        return ws.id


async def _op(ctx: AppContext, fn, sid: int, **kw):  # type: ignore[no-untyped-def]
    async with ctx.db.transaction() as s:
        return await fn(s, ctx, session_id=sid, **kw)


async def _stats(ctx: AppContext) -> svc.SessionStats:
    async with ctx.db.transaction() as s:
        return await svc.session_stats(
            s, start=T0 - timedelta(days=1), end=T0 + timedelta(days=1), now=ctx.clock.now()
        )


# --------------------------------------------------------------------------- start / modes


async def test_free_session_starts_exact_and_marks_task_started(ctx, profile) -> None:
    tid = await _task(ctx)
    sid = await _start(ctx, SessionMode.FREE, tid)
    async with ctx.db.transaction() as s:
        ws = await s.get(WorkSession, sid)
        task = await s.get(Task, tid)
        events = (await s.execute(select(TaskEvent).where(TaskEvent.task_id == tid))).scalars()
        assert ws is not None and task is not None
        assert ws.status == SessionStatus.ACTIVE
        assert ws.started_at == T0 and ws.time_source == TimeSource.EXACT
        assert task.status == TaskStatus.IN_PROGRESS
        assert [e.event_type for e in events] == ["started"]
    assert await _pending(ctx) == []  # free timer without a target: no signals at all


async def test_pomodoro_needs_settings_and_never_invents_them(ctx, profile) -> None:
    with pytest.raises(svc.SettingsMissing) as exc:
        await _start(ctx, SessionMode.POMODORO)
    assert exc.value.fields == ("pomodoro_work_min", "pomodoro_break_min")
    with pytest.raises(svc.SettingsMissing):
        await _start(ctx, SessionMode.SHORT_START)
    async with ctx.db.transaction() as s:
        assert await svc.get_active_session(s) is None
    await _settings(ctx, pomodoro_work_min=25, pomodoro_break_min=5)
    sid = await _start(ctx, SessionMode.POMODORO)
    [row] = await _pending(ctx)
    assert row.session_id == sid
    assert row.due_at == T0 + timedelta(minutes=25)
    assert row.dedup_key == f"timer:{sid}:work_end:25:0"


async def test_only_one_active_session(ctx, profile) -> None:
    sid = await _start(ctx, SessionMode.FREE)
    with pytest.raises(svc.ActiveSessionExists) as exc:
        await _start(ctx, SessionMode.FREE)
    assert exc.value.session_id == sid
    await _op(ctx, svc.pause_session, sid)
    with pytest.raises(svc.ActiveSessionExists):
        await _start(ctx, SessionMode.FREE)
    await _op(ctx, svc.stop_session, sid)
    assert await _start(ctx, SessionMode.FREE) != sid


async def test_pause_resume_net_duration_and_timer_shift(ctx, profile, clock) -> None:
    await _settings(ctx, pomodoro_work_min=25, pomodoro_break_min=5)
    sid = await _start(ctx, SessionMode.POMODORO)
    clock.advance(minutes=10)
    await _op(ctx, svc.pause_session, sid)
    await _op(ctx, svc.pause_session, sid)  # repeated press is a no-op
    assert await _pending(ctx) == []  # mid-period pause: signal cancelled, nothing pending
    clock.advance(minutes=5)
    await _op(ctx, svc.resume_session, sid)
    await _op(ctx, svc.resume_session, sid)
    [row] = await _pending(ctx)
    assert row.due_at == T0 + timedelta(minutes=30)  # 25 net minutes + 5 paused
    clock.advance(minutes=20)
    await _op(ctx, svc.stop_session, sid)
    assert await _pending(ctx) == []
    stats = await _stats(ctx)
    [item] = stats.items
    assert item.gross_minutes == 35 and item.pause_minutes == 5 and item.net_minutes == 30
    assert item.is_exact and stats.exact_minutes == 30


async def test_pomodoro_break_cycle_and_renderer(ctx, profile, clock) -> None:
    await _settings(ctx, pomodoro_work_min=25, pomodoro_break_min=5)
    tid = await _task(ctx, "Сайт")
    sid = await _start(ctx, SessionMode.POMODORO, tid)
    [work_end] = await _pending(ctx)
    clock.advance(minutes=25)
    out = await svc.render_timer_signal(ctx, work_end)
    assert out is not None and "перерыв" in out.text.lower() and "Сайт" in out.text
    assert out.reply_markup is not None
    actions = [b.callback_data for row in out.reply_markup.inline_keyboard for b in row]
    assert any(a and a.startswith("ws:brk:") for a in actions)
    # the user takes the break -> a "break over" signal after 5 minutes
    await _op(ctx, svc.take_break, sid)
    [break_end] = await _pending(ctx)
    assert break_end.due_at == T0 + timedelta(minutes=30)
    assert await svc.render_timer_signal(ctx, work_end) is None  # phase moved on
    out = await svc.render_timer_signal(ctx, break_end)
    assert out is not None and "Перерыв окончен" in out.text
    clock.advance(minutes=6)
    await _op(ctx, svc.resume_session, sid)  # resuming after a break = new period
    [nxt] = await _pending(ctx)
    assert nxt.due_at == clock.now() + timedelta(minutes=25)
    async with ctx.db.transaction() as s:
        ws = await s.get(WorkSession, sid)
        assert ws is not None and ws.planned_minutes == 50
    # work time excludes the break
    clock.advance(minutes=10)
    await _op(ctx, svc.stop_session, sid)
    [item] = (await _stats(ctx)).items
    assert item.net_minutes == 35 and item.pause_minutes == 6


async def test_pomodoro_skip_break(ctx, profile, clock) -> None:
    await _settings(ctx, pomodoro_work_min=25, pomodoro_break_min=5)
    sid = await _start(ctx, SessionMode.POMODORO)
    await _op(ctx, svc.continue_without_break, sid)  # too early: no-op
    [row] = await _pending(ctx)
    assert row.due_at == T0 + timedelta(minutes=25)
    clock.advance(minutes=26)
    await _op(ctx, svc.continue_without_break, sid)
    pending = await _pending(ctx)
    assert [r.due_at for r in pending] == [clock.now() + timedelta(minutes=25)]


async def test_short_start_extend_and_switch_to_free(ctx, profile, clock) -> None:
    await _settings(ctx, short_start_min=5)
    sid = await _start(ctx, SessionMode.SHORT_START)
    [first] = await _pending(ctx)
    assert first.due_at == T0 + timedelta(minutes=5)
    clock.advance(minutes=5)
    out = await svc.render_timer_signal(ctx, first)
    assert out is not None and "Продолжаем" in out.text and out.reply_markup is not None
    labels = [b.text for row in out.reply_markup.inline_keyboard for b in row]
    assert "➕ Ещё 5 мин" in labels and "▶️ Свободный режим" in labels
    await _op(ctx, svc.extend_session, sid)
    [second] = await _pending(ctx)
    assert second.due_at == T0 + timedelta(minutes=10)
    assert await svc.render_timer_signal(ctx, first) is None
    await _op(ctx, svc.switch_to_free, sid)
    assert await _pending(ctx) == []
    assert await svc.render_timer_signal(ctx, second) is None
    async with ctx.db.transaction() as s:
        ws = await s.get(WorkSession, sid)
        assert ws is not None and ws.mode == SessionMode.FREE and ws.status == SessionStatus.ACTIVE


async def test_free_target_signal(ctx, profile) -> None:
    async with ctx.db.transaction() as s:
        ws = await svc.start_session(
            s, ctx, task_id=None, mode=SessionMode.FREE, planned_minutes=40
        )
    [row] = await _pending(ctx)
    assert row.due_at == T0 + timedelta(minutes=40)
    out = await svc.render_timer_signal(ctx, row)
    assert out is not None and "40 мин" in out.text
    await _op(ctx, svc.stop_session, ws.id)
    assert await svc.render_timer_signal(ctx, row) is None


async def test_recommendation_rules(ctx, profile) -> None:
    tid = await _task(ctx)
    async with ctx.db.transaction() as s:
        assert await svc.recommend_mode(s, ctx, task_id=tid) is None  # no history: none
        rec = await svc.recommend_mode(s, ctx, task_id=tid, obstacle=ObstacleReason.TOO_BIG)
        assert rec is not None and rec.mode == SessionMode.SHORT_START
        s.add(
            Intervention(
                task_id=tid,
                kind="tiny_step",
                reason_category=str(ObstacleReason.UNCLEAR),
                created_at=ctx.clock.now(),
            )
        )
    async with ctx.db.transaction() as s:
        rec = await svc.recommend_mode(s, ctx, task_id=tid)
        assert rec is not None and rec.mode == SessionMode.SHORT_START
    sid = await _start(ctx, SessionMode.FREE)
    await _op(ctx, svc.stop_session, sid)
    async with ctx.db.transaction() as s:
        rec = await svc.recommend_mode(s, ctx, task_id=None)  # the user's last choice
        assert rec is not None and rec.mode == SessionMode.FREE


async def test_session_links_recent_intervention_and_records_outcome(ctx, profile) -> None:
    from procbot.db.models import InterventionOutcome

    tid = await _task(ctx)
    async with ctx.db.transaction() as s:
        iv = Intervention(task_id=tid, kind="short_start", created_at=ctx.clock.now())
        s.add(iv)
        await s.flush()
        iv_id = iv.id
    sid = await _start(ctx, SessionMode.FREE, tid)
    await _op(ctx, svc.stop_session, sid)
    await _op(ctx, svc.record_outcome, sid, outcome=SessionOutcome.PROGRESSED)
    async with ctx.db.transaction() as s:
        ws = await s.get(WorkSession, sid)
        assert ws is not None and ws.intervention_id == iv_id
        results = (
            (
                await s.execute(
                    select(InterventionOutcome.result).where(
                        InterventionOutcome.intervention_id == iv_id
                    )
                )
            )
            .scalars()
            .all()
        )
        assert sorted(results) == ["progressed", "started"]


# --------------------------------------------------------------------------- acceptance 14


async def test_acceptance_14_only_timer_signals_reach_the_user_during_a_session(
    ctx, profile, clock, messenger
) -> None:
    """During an active session no progress checks are sent; timer signals are."""
    from procbot.services.scheduler import process_due_reminders

    svc.register(ctx)

    async def nagging(_ctx: AppContext, _r: ReminderSchedule) -> Outgoing:
        return Outgoing(text="Как успехи? Работаешь?")

    register_renderer(ctx, MessageCategory.PRE_TASK, nagging)
    await _settings(ctx, pomodoro_work_min=25, pomodoro_break_min=5, chat_id=1001)
    other = await _task(ctx, "Другое")
    sid = await _start(ctx, SessionMode.POMODORO)
    async with ctx.db.transaction() as s:
        await schedule_reminder(
            s,
            now=ctx.clock.now(),
            category=MessageCategory.PRE_TASK,
            due_at=T0 + timedelta(minutes=10),
            dedup_key="test:pre_task",
            task_id=other,
        )
    for _ in range(28):
        clock.advance(minutes=1)
        await process_due_reminders(ctx)
    texts = messenger.texts()
    assert len(texts) == 1 and "Помодоро готово" in texts[0]
    assert not any("Как успехи" in t for t in texts)
    # every reminder the session created is a timer signal
    async with ctx.db.transaction() as s:
        cats = (
            (
                await s.execute(
                    select(ReminderSchedule.category).where(ReminderSchedule.session_id == sid)
                )
            )
            .scalars()
            .all()
        )
    assert cats and set(cats) == {str(MessageCategory.TIMER_SIGNAL)}
    # ignoring the signal never leads to more messages
    for _ in range(30):
        clock.advance(minutes=2)
        await process_due_reminders(ctx)
    assert len(messenger.texts()) == 1
    assert get_renderer(ctx, MessageCategory.TIMER_SIGNAL) is svc.render_timer_signal


# --------------------------------------------------------------------------- acceptance 15


async def test_acceptance_15_unconfirmed_session_is_never_counted(ctx, profile, clock) -> None:
    tid = await _task(ctx)
    async with ctx.db.transaction() as s:
        planned = await svc.create_unconfirmed_session(
            s, ctx, task_id=tid, planned_start_at=T0, planned_minutes=60
        )
    clock.advance(hours=2)
    stats = await _stats(ctx)
    assert stats.items == [] and stats.total_minutes == 0 and stats.unconfirmed_count == 1
    async with ctx.db.transaction() as s:
        ws = await s.get(WorkSession, planned.id)
        assert ws is not None and ws.status == SessionStatus.UNCONFIRMED and ws.started_at is None
        task = await s.get(Task, tid)
        assert task is not None and task.status == TaskStatus.TODO
    assert await _timers(ctx) == []
    # confirming the start counts time from the confirmation only
    async with ctx.db.transaction() as s:
        await svc.start_session(s, ctx, task_id=None, mode=SessionMode.FREE, session_id=planned.id)
    clock.advance(minutes=30)
    await _op(ctx, svc.stop_session, planned.id)
    stats = await _stats(ctx)
    [item] = stats.items
    assert item.net_minutes == 30 and item.task_id == tid and stats.unconfirmed_count == 0


# --------------------------------------------------------------------------- acceptance 16


async def test_acceptance_16_retro_keeps_source_confidence_and_journal(ctx, profile, clock) -> None:
    tid = await _task(ctx)
    start, end = T0 - timedelta(hours=5), T0 - timedelta(hours=3)
    async with ctx.db.transaction() as s:
        ws = await svc.create_retro_session(
            s,
            ctx,
            task_id=tid,
            started_at=start,
            ended_at=end,
            time_source=TimeSource.RETRO,
            confidence="medium",
            breaks=[svc.RetroBreak(start + timedelta(minutes=60), start + timedelta(minutes=75))],
        )
        sid = ws.id
    stats = await _stats(ctx)
    [item] = stats.items
    assert item.time_source == TimeSource.RETRO and item.confidence == "medium"
    assert not item.is_exact and not stats.all_exact
    assert item.net_minutes == 105 and stats.exact_minutes == 0 and stats.not_exact_minutes == 105
    assert "задним числом" in item.source_label and "примерно" in item.source_label
    assert stats.minutes_by_source() == {"retro": 105}
    async with ctx.db.transaction() as s:
        [c] = (
            (await s.execute(select(SessionCorrection).where(SessionCorrection.session_id == sid)))
            .scalars()
            .all()
        )
        assert (
            c.field == "created_retro"
            and c.actor == "user"
            and '"confidence": "medium"' in (c.new_value or "")
        )
    # later edit: journaled with old/new, source kept, flagged corrected
    async with ctx.db.transaction() as s:
        changes = await svc.correct_session_times(
            s, ctx, session_id=sid, ended_at=end + timedelta(minutes=30), reason="забыл"
        )
    assert [(c.field, c.old_value) for c in changes] == [("ended_at", end.isoformat())]
    [item] = (await _stats(ctx)).items
    assert item.time_source == TimeSource.RETRO and item.corrected
    assert "(исправлено)" in item.source_label


async def test_retro_approximate_and_validation(ctx, profile) -> None:
    async with ctx.db.transaction() as s:
        with pytest.raises(ValueError):
            await svc.create_retro_session(
                s, ctx, task_id=None, started_at=T0 - timedelta(hours=1), ended_at=T0,
                time_source=TimeSource.EXACT, confidence="high",
            )  # fmt: skip
        with pytest.raises(ValueError):
            await svc.create_retro_session(
                s, ctx, task_id=None, started_at=T0, ended_at=T0 + timedelta(hours=1),
                time_source=TimeSource.RETRO, confidence="high",
            )  # fmt: skip
        await svc.create_retro_session(
            s, ctx, task_id=None, started_at=T0 - timedelta(hours=1), ended_at=T0,
            time_source=TimeSource.APPROXIMATE, confidence="low",
        )  # fmt: skip
    [item] = (await _stats(ctx)).items
    assert item.time_source == TimeSource.APPROXIMATE and not item.is_exact


async def test_corrected_exact_session_keeps_exact_source(ctx, profile, clock) -> None:
    sid = await _start(ctx, SessionMode.FREE)
    clock.advance(minutes=50)
    await _op(ctx, svc.stop_session, sid)
    async with ctx.db.transaction() as s:
        await svc.correct_session_times(
            s, ctx, session_id=sid, started_at=T0 + timedelta(minutes=10)
        )
        rows = (
            (await s.execute(select(SessionCorrection).where(SessionCorrection.session_id == sid)))
            .scalars()
            .all()
        )
        assert [r.field for r in rows] == ["started_at"]
    [item] = (await _stats(ctx)).items
    assert item.time_source == TimeSource.EXACT and item.corrected and item.net_minutes == 40


# --------------------------------------------------------------------------- acceptance 17


async def test_acceptance_17_partial_is_not_done_and_criteria_are_checked(ctx, profile) -> None:
    important = await _task(ctx, "Релиз", verification="criteria", done_criteria="Тесты зелёные")
    sid = await _start(ctx, SessionMode.FREE, important)
    with pytest.raises(svc.InvalidSessionState):  # outcome only after stop
        await _op(ctx, svc.record_outcome, sid, outcome=SessionOutcome.PROGRESSED)
    await _op(ctx, svc.stop_session, sid)
    with pytest.raises(svc.CriteriaCheckRequired):
        await _op(ctx, svc.record_outcome, sid, outcome=SessionOutcome.COMPLETED)
    ws, applied = await _op(
        ctx, svc.record_outcome, sid, outcome=SessionOutcome.PROGRESSED, note="criteria: partial"
    )
    assert applied and ws.outcome == SessionOutcome.PROGRESSED
    _, again = await _op(ctx, svc.record_outcome, sid, outcome=SessionOutcome.COMPLETED)
    assert again is False  # repeated press: no-op
    async with ctx.db.transaction() as s:
        task = await s.get(Task, important)
        assert task is not None and task.status == TaskStatus.IN_PROGRESS
    # a confirmed check closes it
    sid2 = await _start(ctx, SessionMode.FREE, important)
    await _op(ctx, svc.stop_session, sid2)
    await _op(
        ctx, svc.record_outcome, sid2, outcome=SessionOutcome.COMPLETED, criteria_confirmed=True
    )
    await _op(ctx, svc.save_evidence, sid2, evidence="https://example.com/pr/1")
    async with ctx.db.transaction() as s:
        task = await s.get(Task, important)
        ws2 = await s.get(WorkSession, sid2)
        assert task is not None and task.status == TaskStatus.DONE
        assert ws2 is not None and ws2.evidence == "https://example.com/pr/1"
    # a simple chore closes by a tap
    chore = await _task(ctx, "Вынести мусор")
    sid3 = await _start(ctx, SessionMode.FREE, chore)
    await _op(ctx, svc.stop_session, sid3)
    await _op(ctx, svc.record_outcome, sid3, outcome=SessionOutcome.COMPLETED)
    async with ctx.db.transaction() as s:
        task = await s.get(Task, chore)
        assert task is not None and task.status == TaskStatus.DONE


# --------------------------------------------------------------------------- restart (G / 2)


async def _reopen(ctx: AppContext) -> None:
    await ctx.db.dispose()
    ctx.db = Database.for_path(ctx.env.database_path)


async def test_restart_restores_session_and_signal_once(ctx, profile, clock) -> None:
    await _settings(ctx, pomodoro_work_min=25, pomodoro_break_min=5)
    sid = await _start(ctx, SessionMode.POMODORO)
    clock.advance(minutes=10)
    await _reopen(ctx)
    assert await svc.restore_sessions(ctx) == 1
    assert await svc.restore_sessions(ctx) == 1
    [row] = await _pending(ctx)
    assert row.due_at == T0 + timedelta(minutes=25)
    async with ctx.db.transaction() as s:
        ws = await svc.get_active_session(s)
        assert ws is not None and ws.id == sid and ws.started_at == T0


async def test_restart_phase_ended_during_downtime_is_scheduled_once(ctx, profile, clock) -> None:
    await _settings(ctx, short_start_min=5)
    sid = await _start(ctx, SessionMode.SHORT_START)
    # simulate a lost schedule row (e.g. written by an older version) + 40 min downtime
    async with ctx.db.transaction() as s:
        await s.execute(delete(ReminderSchedule))
    clock.advance(minutes=40)
    await _reopen(ctx)
    await svc.restore_sessions(ctx)
    await svc.restore_sessions(ctx)
    rows = await _timers(ctx)
    assert len(rows) == 1 and rows[0].state == ReminderState.PENDING
    assert rows[0].due_at == T0 + timedelta(minutes=5)  # original moment: the gate folds stale
    assert rows[0].dedup_key == f"timer:{sid}:short_end:5:0"
    # a paused session gets no signals after restart
    await _op(ctx, svc.pause_session, sid)
    await svc.restore_sessions(ctx)
    assert await _pending(ctx) == []


async def test_restore_does_not_resignal_a_boundary_already_sent(ctx, profile, clock) -> None:
    await _settings(ctx, short_start_min=5)
    sid = await _start(ctx, SessionMode.SHORT_START)
    [row] = await _pending(ctx)
    async with ctx.db.transaction() as s:
        r = await s.get(ReminderSchedule, row.id)
        assert r is not None
        r.state = str(ReminderState.DONE)  # the scheduler sent it
    clock.advance(minutes=7)
    await _op(ctx, svc.pause_session, sid)
    clock.advance(minutes=1)
    await _op(ctx, svc.resume_session, sid)
    await svc.restore_sessions(ctx)
    assert await _pending(ctx) == []  # boundary signalled before the pause: not again


# --------------------------------------------------------------------------- parsing


def test_parse_interval_formats() -> None:
    now = datetime(2026, 10, 26, 17, 0, tzinfo=UTC)  # 18:00 Berlin (CET)
    p = svc.parse_interval("14:00-16:30", now=now, tz=BERLIN)
    assert p is not None and p.explicit and p.time_source == TimeSource.RETRO
    assert (p.start, p.end) == (
        datetime(2026, 10, 26, 13, 0, tzinfo=UTC),
        datetime(2026, 10, 26, 15, 30, tzinfo=UTC),
    )
    p = svc.parse_interval("с 14 до 16", now=now, tz=BERLIN)
    assert p is not None and p.start == datetime(2026, 10, 26, 13, 0, tzinfo=UTC)
    p = svc.parse_interval("вчера 10–12", now=now, tz=BERLIN)
    assert p is not None and p.start == datetime(2026, 10, 25, 9, 0, tzinfo=UTC)
    p = svc.parse_interval("с 2 до 4 дня", now=now, tz=BERLIN)
    assert p is not None and p.end == datetime(2026, 10, 26, 15, 0, tzinfo=UTC)
    p = svc.parse_interval("примерно с 9 до 11", now=now, tz=BERLIN)
    assert p is not None and p.time_source == TimeSource.APPROXIMATE
    # in the future today -> yesterday
    p = svc.parse_interval("19:00-20:00", now=now, tz=BERLIN)
    assert p is not None and p.start == datetime(2026, 10, 25, 18, 0, tzinfo=UTC)
    for text, minutes in (
        ("2 часа", 120),
        ("примерно час утром", 60),
        ("полтора часа", 90),
        ("90 минут", 90),
        ("1.5 ч", 90),
        ("2 часа 30 минут", 150),
        ("полчаса", 30),
    ):
        p = svc.parse_interval(text, now=now, tz=BERLIN)
        assert p is not None and not p.explicit, text
        assert p.duration == timedelta(minutes=minutes), text
        assert p.time_source == TimeSource.APPROXIMATE
    assert svc.parse_interval("1-2 часа", now=now, tz=BERLIN).duration == timedelta(hours=2)  # type: ignore[union-attr]
    assert svc.parse_interval("не помню", now=now, tz=BERLIN) is None


def test_parse_clock_and_breaks() -> None:
    now = datetime(2026, 10, 26, 17, 0, tzinfo=UTC)
    assert svc.parse_clock("только что", now=now, tz=BERLIN) == now
    assert svc.parse_clock("в 12:30", now=now, tz=BERLIN) == datetime(
        2026, 10, 26, 11, 30, tzinfo=UTC
    )
    start, end = (
        datetime(2026, 10, 26, 12, 0, tzinfo=UTC),
        datetime(2026, 10, 26, 14, 0, tzinfo=UTC),
    )
    assert svc.parse_breaks("без перерывов", start=start, end=end, tz=BERLIN) == []
    assert svc.parse_breaks("0", start=start, end=end, tz=BERLIN) == []
    [b] = svc.parse_breaks("15", start=start, end=end, tz=BERLIN) or []
    assert b.ended_at - b.started_at == timedelta(minutes=15)
    assert b.time_source == TimeSource.APPROXIMATE and start < b.started_at < end
    [b] = svc.parse_breaks("14:00-14:20", start=start, end=end, tz=BERLIN) or []
    assert b.started_at == datetime(2026, 10, 26, 13, 0, tzinfo=UTC)
    assert b.time_source == TimeSource.RETRO
    assert svc.parse_breaks("18:00-18:20", start=start, end=end, tz=BERLIN) is None
    assert svc.parse_breaks("3 часа", start=start, end=end, tz=BERLIN) is None


def test_classify_phrase_is_conservative() -> None:
    assert svc.classify_phrase("Пауза") == "pause"
    assert svc.classify_phrase("продолжаю!") == "resume"
    assert svc.classify_phrase("стоп") == "stop"
    assert svc.classify_phrase("Готово.") == "stop"
    assert svc.classify_phrase("закончил") == "stop"
    assert svc.classify_phrase("начинаю") == "start"
    assert svc.classify_phrase("пауза в переговорах затянулась, надо подумать") is None
    assert svc.classify_phrase("готово ли письмо") is None
    assert svc.classify_phrase("/stop") is None
    assert svc.classify_phrase(None) is None


def test_net_seconds_clips_pauses() -> None:
    ws = WorkSession(
        id=1, mode="free", status="finished", started_at=T0, ended_at=T0 + timedelta(hours=1),
        time_source="exact", created_at=T0, updated_at=T0,
    )  # fmt: skip
    from procbot.db.models import SessionPause

    pauses = [
        SessionPause(session_id=1, started_at=T0 + timedelta(minutes=50),
                     ended_at=T0 + timedelta(minutes=70)),
    ]  # fmt: skip
    assert svc.net_seconds(ws, pauses, T0 + timedelta(hours=2)) == 50 * 60
    assert svc.wall_time_at_net(ws, pauses, 55 * 60) == T0 + timedelta(minutes=75)
