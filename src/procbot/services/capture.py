"""CaptureService: free text / voice transcript -> intent -> proposals (scenario A).

AI tasks registered here (at import):
- `route_message` (fast): what the owner wants — new tasks, chat/question,
  session control, retro report, postpone/obstacle, settings, unclear;
- `extract_tasks` (standard): task candidates with uncertainties, or "this is
  only a reflection" + a clarifying question.

Rules (spec FR-02, FR-14, FR-15, section 7, open question 36):
- the model only PROPOSES: candidates become pending approvals through
  `services.tasks.propose_task_candidates` (one card, per-item and all/none
  buttons). Nothing is created by itself; silence is never consent;
- dates/times are resolved by CODE in the owner's timezone (`services.when`
  plus a few vague phrases); the model copies the words. Invalid output (bad
  dates, unknown task ids) is rejected by a business validator -> repair path;
- reflection-only or ambiguous text -> a clarifying question, or the draft
  offered for confirmation, never silent creation;
- `AIUnavailable` -> the caller falls back to stage A (`fallback_capture`:
  line-split candidates with confirmation); no data changes by guess;
- chat replies are text only (persona + guardrails, `guard_reply` post-filter)
  and are never executed.

No aiogram here: handlers (telegram.handlers.chat / voice) call these functions.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.ai import prompting as P
from procbot.ai.audit import record_actions
from procbot.ai.context import ContextBuilder
from procbot.ai.gateway import get_gateway, run_task
from procbot.ai.persona import guard_reply, load_persona
from procbot.ai.prompting import Prompt
from procbot.ai.registry import AITaskSpec, register_task
from procbot.ai.types import Tier
from procbot.context import AppContext
from procbot.db.models import Goal, PendingApproval, Project, Task
from procbot.domain.actions import CreateTask, TaskDraft
from procbot.domain.enums import Actor, TaskStatus
from procbot.services import actions, approvals
from procbot.services import tasks as task_service
from procbot.services.sessions import ParsedInterval, parse_interval
from procbot.services.when import parse_when
from procbot.timeutil import local_date, resolve_local, to_local

# UNCONFIRMED technical defaults (docs/open-questions.md).
ROUTE_MIN_CONFIDENCE = 0.5  # below this a non-task intent is treated as "unclear"
MAX_FUTURE_DAYS = 3 * 366  # a resolved date further away is treated as a model error
MAX_TEXT_CHARS = 4000

AI_DOWN_NOTE = "ИИ сейчас недоступен — разобрал сообщение по строкам."
INITIATOR = "services.capture"

UNCERTAINTY_LABELS: dict[str, str] = {
    "date": "срок / дата",
    "time": "время",
    "priority": "приоритет",
    "scope": "объём",
    "estimate": "длительность",
    "intent": "точно ли это задача",
    "goal": "цель / проект",
}

# --------------------------------------------------------------------------- schemas


class TaskCandidate(BaseModel):
    """One task the owner may want. Dates are WORDS from the message; code resolves them."""

    model_config = ConfigDict(extra="ignore")

    title: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    # Words from the message ("завтра", "в пятницу", "на выходных", "25.10") or
    # YYYY-MM-DD; null when no date was mentioned.
    when: str | None = Field(default=None, max_length=60)
    # "HH:MM" (24h) or words ("вечером"); null when no time was mentioned.
    at: str | None = Field(default=None, max_length=30)
    # planned = when to do it; due = deadline ("к пятнице", "до 25.10").
    date_kind: Literal["planned", "due"] = "planned"
    estimate_minutes: int | None = Field(default=None, ge=1, le=24 * 60)
    # Only if the owner said it explicitly; 1 = most urgent.
    priority: int | None = Field(default=None, ge=1, le=5)
    goal_hint: str | None = Field(default=None, max_length=200)
    project_hint: str | None = Field(default=None, max_length=200)
    is_backlog: bool = False
    uncertainties: list[
        Literal["date", "time", "priority", "scope", "estimate", "intent", "goal"]
    ] = Field(default_factory=list, max_length=7)
    uncertainty_note: str | None = Field(default=None, max_length=200)


class ExtractedTasks(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidates: list[TaskCandidate] = Field(default_factory=list, max_length=15)
    # The text is thinking aloud / a wish, not a commitment.
    is_reflection_only: bool = False
    clarifying_question: str | None = Field(default=None, max_length=300)


Intent = Literal[
    "new_tasks",
    "chat",
    "question",
    "session_control",
    "retro_report",
    "postpone",
    "obstacle",
    "settings_request",
    "unclear",
]


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    session_action: Literal["start", "pause", "resume", "stop"] | None = None
    # An existing open task the message is about (id from the task list), else null.
    task_id: int | None = None
    retro_day: Literal["today", "yesterday", "day_before_yesterday"] | None = None
    retro_start: str | None = Field(default=None, max_length=5)  # "HH:MM" 24h
    retro_end: str | None = Field(default=None, max_length=5)
    retro_duration_min: int | None = Field(default=None, ge=1, le=24 * 60)
    retro_approximate: bool = False
    clarifying_question: str | None = Field(default=None, max_length=300)


_ROUTE_SYSTEM = (
    "Определи, чего хочет пользователь последним сообщением. Варианты intent: "
    "new_tasks — хочет добавить одно или несколько новых дел; chat — разговор, эмоции; "
    "question — вопрос к боту; session_control — начать/поставить на паузу/продолжить/"
    "закончить рабочую сессию (session_action); retro_report — сообщает, что работал "
    "раньше («я работал с 2 до 4»): заполни retro_* (время HH:MM в 24-часовом формате, "
    "retro_day, retro_duration_min, retro_approximate если «примерно»); postpone — хочет "
    "перенести существующую задачу; obstacle — не может начать / что-то мешает; "
    "settings_request — хочет поменять настройки; unclear — непонятно. "
    "task_id — только id из списка открытых задач, иначе null. "
    "Ничего не выполняй и не придумывай задачи."
)
_EXTRACT_SYSTEM = (
    "Выдели из сообщения пользователя самостоятельные дела-кандидаты. Это только "
    "черновики: пользователь подтвердит каждое сам. Не выдумывай дел, которых нет в "
    "сообщении; не превращай размышления, сомнения и пожелания в обязательства — для "
    "них поставь is_reflection_only=true и задай clarifying_question. "
    "Дату НЕ вычисляй: в поле when скопируй слова из сообщения («завтра», «в пятницу», "
    "«на выходных», «25.10»), время — в at («14:00», «вечером»). Если срок — это "
    "дедлайн («к пятнице», «до 25.10»), поставь date_kind=due. priority — только если "
    "пользователь сам сказал о срочности. В uncertainties перечисли, что неясно "
    "(date, time, priority, scope, estimate, intent, goal)."
)

ROUTE_MESSAGE = register_task(
    AITaskSpec(
        name="route_message",
        tier=Tier.FAST,
        output=RouteDecision,
        max_output_tokens=400,
        context_budget_tokens=2500,
        data_categories=frozenset({P.MESSAGE, P.HISTORY, P.TASKS, P.PROFILE}),
        system=_ROUTE_SYSTEM,
        description="Маршрутизация свободного текста: задачи, разговор, сессия, отметка задним числом",
    )
)

EXTRACT_TASKS = register_task(
    AITaskSpec(
        name="extract_tasks",
        tier=Tier.STANDARD,
        output=ExtractedTasks,
        max_output_tokens=1500,
        context_budget_tokens=4000,
        data_categories=frozenset({P.MESSAGE, P.HISTORY, P.TASKS, P.PROFILE}),
        system=_EXTRACT_SYSTEM,
        description="Черновики задач из текста (сценарий A)",
    )
)

_ROUTE_INSTRUCTION = "Классифицируй последнее сообщение пользователя по схеме ответа."
_EXTRACT_INSTRUCTION = (
    "Выдели дела-кандидаты из последнего сообщения пользователя по схеме ответа. "
    "Открытые задачи даны, чтобы не дублировать уже существующие."
)
_CHAT_INSTRUCTION = (
    "Ответь пользователю на последнее сообщение — коротко и по делу, в заданном стиле. "
    "Ты не выполняешь действий и не меняешь данные. Если пользователь просит что-то "
    "изменить, объясни, какой кнопкой или командой это сделать (/tasks, /today, /work, "
    "/retro, /settings), или предложи написать новое дело отдельным сообщением — бот "
    "покажет карточку для подтверждения. Не утверждай, что что-то уже сделал."
)

# --------------------------------------------------------------------------- dates (code, not model)

_PART_OF_DAY = ("с утра", "утром", "днём", "днем", "вечером", "ночью", "после обеда", "в обед")
_WEEKEND = ("на выходных", "в выходные", "на выходные", "в выходных", "выходные")
_VAGUE = (
    "на этой неделе",
    "на следующей неделе",
    "на неделе",
    "в ближайшее время",
    "на днях",
    "скоро",
    "в этом месяце",
)
_BACKLOG = ("когда-нибудь", "когда нибудь", "как-нибудь", "когда будет время", "без срока", "потом")
_IN_DAYS = re.compile(r"через\s+(\d{1,3}|один|два|три|неделю|день)\s*(?:дн\w*|день|недел\w*)?")
_NUM_WORDS = {"один": 1, "день": 1, "два": 2, "три": 3}
_HHMM = re.compile(r"^\s*(\d{1,2})[:.](\d{2})\s*$")
# Deadline prepositions and dative weekday forms ("к пятнице", "до среды").
_FILLER = frozenset({"к", "ко", "до", "не", "позже", "позднее", "числа"})
_WEEKDAY_FORMS = {
    "понедельнику": "понедельник",
    "понедельника": "понедельник",
    "вторнику": "вторник",
    "вторника": "вторник",
    "среде": "среда",
    "среды": "среда",
    "четвергу": "четверг",
    "четверга": "четверг",
    "пятнице": "пятница",
    "пятницы": "пятница",
    "субботе": "суббота",
    "субботы": "суббота",
    "воскресенью": "воскресенье",
    "воскресенья": "воскресенье",
    "завтрашнего": "завтра",
}


@dataclass(frozen=True)
class ResolvedWhen:
    day: date | None = None
    at: time | None = None
    backlog: bool = False
    uncertain_date: bool = False
    uncertain_time: bool = False
    note: str | None = None


def _norm(text: str) -> str:
    t = text.lower().replace("ё", "е").replace(",", " ")
    return " ".join(t.split())


def resolve_when(when: str | None, at: str | None, *, today: date) -> ResolvedWhen | None:
    """Resolve the words of a candidate in local terms. None = not understood."""
    raw = _norm(" ".join(x for x in (when, at) if x))
    if not raw:
        return ResolvedWhen()
    day: date | None = None
    tm: time | None = None
    backlog = uncertain_date = uncertain_time = False
    notes: list[str] = []
    rest = f" {raw} "
    for phrase in _BACKLOG:
        if f" {phrase} " in rest:
            backlog = True
            rest = rest.replace(f" {phrase} ", " ")
    for phrase in _VAGUE:
        if f" {phrase} " in rest:
            uncertain_date = True
            notes.append(f"«{phrase}» — без конкретной даты")
            rest = rest.replace(f" {phrase} ", " ")
    for phrase in _WEEKEND:
        if f" {phrase} " in rest:
            offset = (5 - today.weekday()) % 7 if today.weekday() != 6 else 0
            day = today + timedelta(days=offset)
            uncertain_date = True
            notes.append(f"«{phrase}» → {day.strftime('%d.%m')} (уточни день)")
            rest = rest.replace(f" {phrase} ", " ")
            break
    m = _IN_DAYS.search(rest)
    if m and day is None:
        n_raw = m.group(1)
        n = 7 if n_raw == "неделю" else _NUM_WORDS.get(n_raw) or int(n_raw)
        if "недел" in m.group(0) and n_raw != "неделю":
            n *= 7
        day = today + timedelta(days=n)
        rest = rest[: m.start()] + " " + rest[m.end() :]
    for phrase in _PART_OF_DAY:
        if f" {phrase} " in rest:
            uncertain_time = True
            notes.append(f"время: «{phrase}»")
            rest = rest.replace(f" {phrase} ", " ")
    rest = " ".join(_WEEKDAY_FORMS.get(w, w) for w in rest.split() if w not in _FILLER)
    if rest:
        parsed = parse_when(rest, today=today)
        if parsed is None:
            return None
        if parsed.day is not None:
            if day is not None and parsed.day != day:
                return None
            day = parsed.day
        tm = parsed.at
    return ResolvedWhen(
        day=day,
        at=tm,
        backlog=backlog and day is None,
        uncertain_date=uncertain_date,
        uncertain_time=uncertain_time,
        note="; ".join(notes) or None,
    )


def _parse_hhmm(value: str | None) -> time | None:
    if not value:
        return None
    m = _HHMM.match(value)
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    if hh > 23 or mm > 59:
        return None
    return time(hh, mm)


# --------------------------------------------------------------------------- validators


@dataclass(frozen=True)
class _Now:
    now: datetime
    tz: str | None
    today: date


def extract_validator(clock: _Now) -> Callable[[BaseModel], list[str]]:
    """Business checks on `extract_tasks` output (errors -> gateway repair path)."""

    def check(out: BaseModel) -> list[str]:
        assert isinstance(out, ExtractedTasks)
        errors: list[str] = []
        for i, c in enumerate(out.candidates):
            if not c.title.strip():
                errors.append(f"candidates.{i}.title: пустое название")
            r = resolve_when(c.when, c.at, today=clock.today)
            if r is None:
                errors.append(
                    f"candidates.{i}.when/at: не понял «{c.when or ''} {c.at or ''}». "
                    "Скопируй слова даты из сообщения («завтра», «в пятницу», «25.10») "
                    "или YYYY-MM-DD, время — «HH:MM»; если даты нет — null и 'date' в uncertainties"
                )
                continue
            if r.day is not None and r.day < clock.today:
                errors.append(f"candidates.{i}.when: дата {r.day.isoformat()} уже прошла")
            if r.day is not None and r.day > clock.today + timedelta(days=MAX_FUTURE_DAYS):
                errors.append(f"candidates.{i}.when: дата {r.day.isoformat()} слишком далеко")
        return errors

    return check


def route_validator(open_task_ids: frozenset[int]) -> Callable[[BaseModel], list[str]]:
    def check(out: BaseModel) -> list[str]:
        assert isinstance(out, RouteDecision)
        errors: list[str] = []
        if out.task_id is not None and out.task_id not in open_task_ids:
            errors.append(
                f"task_id: задачи {out.task_id} нет среди открытых — укажи id из списка или null"
            )
        if out.intent == "session_control" and out.session_action is None:
            errors.append("session_action: обязательно для intent=session_control")
        for name in ("retro_start", "retro_end"):
            value = getattr(out, name)
            if value is not None and _parse_hhmm(value) is None:
                errors.append(f"{name}: нужен формат HH:MM (24 часа)")
        if (out.retro_start is None) != (out.retro_end is None):
            errors.append("retro_start/retro_end: укажи оба или ни одного")
        return errors

    return check


# --------------------------------------------------------------------------- helpers


async def _clock(s: AsyncSession, ctx: AppContext) -> _Now:
    tz = await task_service.owner_tz(s)
    now = ctx.clock.now()
    return _Now(now=now, tz=tz, today=local_date(now, tz))


async def _open_task_ids(s: AsyncSession) -> frozenset[int]:
    rows = await s.execute(
        select(Task.id).where(Task.status.in_((str(TaskStatus.TODO), str(TaskStatus.IN_PROGRESS))))
    )
    return frozenset(r[0] for r in rows)


async def _find_by_title(
    s: AsyncSession, model: type[Goal] | type[Project], hint: str
) -> int | None:
    needle = " ".join(hint.casefold().split())
    if not needle:
        return None
    rows = await s.execute(select(model.id, model.title).where(model.status == "active"))
    for rid, title in rows:
        if " ".join(str(title).casefold().split()) == needle:
            return int(rid)
    return None


@dataclass
class DraftWithNotes:
    draft: TaskDraft
    notes: list[str] = field(default_factory=list)


async def candidate_to_draft(s: AsyncSession, c: TaskCandidate, clock: _Now) -> DraftWithNotes:
    """Model candidate -> TaskDraft (dates resolved by code) + card notes."""
    kwargs: dict[str, Any] = {"title": " ".join(c.title.split())[:500]}
    notes: list[str] = []
    if c.description:
        kwargs["description"] = c.description.strip()
    r = resolve_when(c.when, c.at, today=clock.today) or ResolvedWhen(uncertain_date=True)
    uncertain = set(c.uncertainties)
    if r.uncertain_date:
        uncertain.add("date")
    if r.uncertain_time:
        uncertain.add("time")
    day, at = r.day, r.at
    if at is not None and day is None:
        local_now = to_local(clock.now, clock.tz)
        day = clock.today if at > local_now.time() else clock.today + timedelta(days=1)
        uncertain.add("date")
    if day is not None:
        moment = resolve_local(day, at, clock.tz) if at is not None else None
        if c.date_kind == "due":
            if moment is not None:
                kwargs["due_at"] = moment
            else:
                kwargs["due_date"] = day
        elif moment is not None:
            kwargs["planned_start_at"] = moment
        else:
            kwargs["planned_date"] = day
    elif c.is_backlog or r.backlog:
        kwargs["is_backlog"] = True
        notes.append("📥 В бэклог (без даты)")
    if c.estimate_minutes:
        kwargs["estimate_minutes"] = c.estimate_minutes
    if c.priority is not None:
        kwargs["priority"] = c.priority
    for hint, model, key, label in (
        (c.goal_hint, Goal, "goal_id", "цель"),
        (c.project_hint, Project, "project_id", "проект"),
    ):
        if not hint:
            continue
        found = await _find_by_title(s, model, hint)
        if found is not None:
            kwargs[key] = found
            notes.append(f"🎯 {label.capitalize()}: «{hint}»")
        else:
            notes.append(f"🎯 Похоже на {label} «{hint}» — такой не нашёл, не привязываю")
    if r.note:
        notes.append(f"📅 {r.note}")
    if uncertain:
        labels = [UNCERTAINTY_LABELS[u] for u in UNCERTAINTY_LABELS if u in uncertain]
        line = "❓ Неясно: " + ", ".join(labels)
        if c.uncertainty_note:
            line += f" — {c.uncertainty_note.strip()}"
        notes.append(line)
    return DraftWithNotes(TaskDraft(**kwargs), notes)


# --------------------------------------------------------------------------- results

CaptureKind = Literal["proposed", "clarify", "nothing"]
TxHook = Callable[[AsyncSession], Awaitable[None]]


@dataclass
class CaptureResult:
    kind: CaptureKind
    approvals: list[PendingApproval] = field(default_factory=list)
    header: str | None = None
    text: str | None = None  # clarifying question / "nothing found" text
    tz: str | None = None
    ai_call_id: int | None = None
    truncated: bool = False


def ai_configured(ctx: AppContext) -> bool:
    """True when an AI gateway with at least one tier exists (else: stage A silently)."""
    gw = get_gateway(ctx)
    return gw is not None and bool(gw.configured)


# --------------------------------------------------------------------------- routing


async def route_message(ctx: AppContext, text: str, *, update_id: int) -> RouteDecision:
    """Ask the fast model what the owner wants. Raises AIUnavailable."""
    async with ctx.db.transaction() as s:
        open_ids = await _open_task_ids(s)
    blocks = await ContextBuilder(ctx, task="route_message").build(
        message=text[:MAX_TEXT_CHARS], history=6, tasks=30, exclude_update_id=update_id
    )
    res = await run_task(
        ctx,
        "route_message",
        Prompt(instruction=_ROUTE_INSTRUCTION, data=blocks),
        initiator=INITIATOR,
        correlation_id=f"u{update_id}",
        validator=route_validator(open_ids),
    )
    decision = res.as_(RouteDecision)
    if decision.confidence < ROUTE_MIN_CONFIDENCE and decision.intent not in (
        "new_tasks",
        "chat",
        "question",
    ):
        decision = decision.model_copy(update={"intent": "unclear"})
    return decision


def resolve_retro(
    decision: RouteDecision | None, text: str, *, now: datetime, tz: str | None
) -> ParsedInterval | None:
    """Interval of a retro report. The model's explicit HH:MM (it can tell 2 pm
    from 2 am in context) is placed on a day by code; otherwise the text itself
    is parsed by code; a bare duration last."""
    parsed: ParsedInterval | None = None
    if decision is not None and decision.retro_start and decision.retro_end:
        word = {"today": "сегодня", "yesterday": "вчера", "day_before_yesterday": "позавчера"}.get(
            decision.retro_day or "", ""
        )
        parsed = parse_interval(
            f"{word} {decision.retro_start}-{decision.retro_end}", now=now, tz=tz
        )
    if parsed is None:
        parsed = parse_interval(text, now=now, tz=tz)
    if parsed is None and decision is not None and decision.retro_duration_min:
        parsed = ParsedInterval(
            None, None, timedelta(minutes=decision.retro_duration_min), False, True
        )
    if parsed is not None and decision is not None and decision.retro_approximate:
        parsed = ParsedInterval(
            parsed.start, parsed.end, parsed.duration, parsed.explicit, approximate=True
        )
    if parsed is not None and parsed.end is not None and parsed.end > now + timedelta(minutes=1):
        return None
    return parsed


# --------------------------------------------------------------------------- extraction


async def capture_tasks(
    ctx: AppContext,
    text: str,
    *,
    update_id: int,
    in_tx: TxHook | None = None,
    route: RouteDecision | None = None,
) -> CaptureResult:
    """AI extraction -> ONE confirmation card. Raises AIUnavailable (caller falls
    back to `fallback_capture`). `in_tx` runs inside the proposal transaction
    (the handler marks the update processed there)."""
    async with ctx.db.transaction() as s:
        clock = await _clock(s, ctx)
    blocks = await ContextBuilder(ctx, task="extract_tasks").build(
        message=text[:MAX_TEXT_CHARS],
        history=6,
        tasks=40,
        memory=False,
        exclude_update_id=update_id,
    )
    res = await run_task(
        ctx,
        "extract_tasks",
        Prompt(instruction=_EXTRACT_INSTRUCTION, data=blocks),
        initiator=INITIATOR,
        correlation_id=f"u{update_id}",
        validator=extract_validator(clock),
    )
    out = res.as_(ExtractedTasks)
    question = (out.clarifying_question or "").strip() or None
    if not out.candidates:
        async with ctx.db.transaction() as s:
            await record_actions(ctx, res.call_id, [], [])
            if in_tx is not None:
                await in_tx(s)
        if out.is_reflection_only:
            return CaptureResult(
                "clarify",
                text=question
                or "Похоже, это размышления, а не конкретные дела. Оформить что-то из этого задачей?",
                tz=clock.tz,
                ai_call_id=res.call_id,
            )
        return CaptureResult(
            "nothing",
            text=question or "Не нашёл, что можно оформить задачей.",
            tz=clock.tz,
            ai_call_id=res.call_id,
        )
    op_prefix = f"u{update_id}"
    async with ctx.db.transaction() as s:
        prepared = [await candidate_to_draft(s, c, clock) for c in out.candidates]
        cards = await task_service.propose_task_candidates(
            s,
            ctx,
            [p.draft for p in prepared],
            op_prefix=op_prefix,
            origin=Actor.AI,
            ai_call_id=res.call_id,
        )
        by_op = {a.op_key: a for a in cards}
        for i, p in enumerate(prepared):
            row = by_op.get(f"{op_prefix}:cand:{i}")
            if row is not None:
                await approvals.add_diff_notes(s, row.id, p.notes)
        await record_actions(
            ctx,
            res.call_id,
            [actions.dump_action(CreateTask(task=p.draft)) for p in prepared],
            [
                {"op_key": a.op_key, "status": "pending_approval", "approval_id": a.id}
                for a in cards
            ],
        )
        if in_tx is not None:
            await in_tx(s)
    header: str | None = None
    low_confidence = route is not None and route.confidence < ROUTE_MIN_CONFIDENCE
    if out.is_reflection_only or low_confidence:
        header = (
            "Похоже, это скорее размышления, чем дела. Если хочешь — вот черновик; "
            "создам только то, что подтвердишь:"
        )
    if question:
        header = f"{question}\n\n{header or 'Черновик — создам только то, что подтвердишь:'}"
    return CaptureResult(
        "proposed",
        approvals=cards,
        header=header,
        tz=clock.tz,
        ai_call_id=res.call_id,
        truncated=len(out.candidates) > task_service.MAX_CANDIDATES,
    )


async def fallback_capture(
    ctx: AppContext, text: str, *, update_id: int, in_tx: TxHook | None = None
) -> CaptureResult:
    """Stage A without AI: one candidate per line / ';' — still confirmation only."""
    titles = task_service.split_candidates(text)
    async with ctx.db.transaction() as s:
        cards: list[PendingApproval] = []
        if titles:
            cards = await task_service.propose_task_candidates(
                s,
                ctx,
                [TaskDraft(title=t) for t in titles],
                op_prefix=f"u{update_id}",
                origin=Actor.SYSTEM,
            )
        tz = await task_service.owner_tz(s)
        if in_tx is not None:
            await in_tx(s)
    if not cards:
        return CaptureResult("nothing", text="Не нашёл, что можно оформить задачей.", tz=tz)
    header = None
    if len(titles) > task_service.MAX_CANDIDATES:
        header = (
            f"Похоже, тут есть задачи (показаны первые {task_service.MAX_CANDIDATES}). Создать их?"
        )
    return CaptureResult(
        "proposed",
        approvals=cards,
        header=header,
        tz=tz,
        truncated=len(titles) > task_service.MAX_CANDIDATES,
    )


# --------------------------------------------------------------------------- chat


@dataclass(frozen=True)
class ChatAnswer:
    text: str
    call_id: int
    filtered: bool  # the guardrail post-filter replaced the model text


async def chat_answer(ctx: AppContext, text: str, *, update_id: int) -> ChatAnswer:
    """Text-only reply with persona + context. Never executes anything.
    Raises AIUnavailable."""
    blocks = await ContextBuilder(ctx, task="chat_reply").build(
        message=text[:MAX_TEXT_CHARS], exclude_update_id=update_id
    )
    persona = await load_persona(ctx)
    res = await run_task(
        ctx,
        "chat_reply",
        Prompt(instruction=_CHAT_INSTRUCTION, data=blocks, persona=persona),
        initiator=INITIATOR,
        correlation_id=f"u{update_id}",
    )
    reply, filtered = guard_reply(res.text)
    return ChatAnswer(text=reply[:3500], call_id=res.call_id, filtered=filtered)


async def task_title(ctx: AppContext, task_id: int | None) -> str | None:
    if not task_id:
        return None
    async with ctx.db.transaction() as s:
        task = await s.get(Task, task_id)
        return task.title if task is not None else None


__all__ = [
    "AI_DOWN_NOTE",
    "EXTRACT_TASKS",
    "ROUTE_MESSAGE",
    "ROUTE_MIN_CONFIDENCE",
    "CaptureResult",
    "ChatAnswer",
    "ExtractedTasks",
    "ResolvedWhen",
    "RouteDecision",
    "TaskCandidate",
    "ai_configured",
    "candidate_to_draft",
    "capture_tasks",
    "chat_answer",
    "extract_validator",
    "fallback_capture",
    "resolve_retro",
    "resolve_when",
    "route_message",
    "route_validator",
]
