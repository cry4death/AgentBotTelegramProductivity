"""AI gateway: the only door from domain code to any LLM provider.

Domain services call

    result = await ctx.ai.run("extract_tasks", prompt, initiator="services.capture")
    proposal = result.as_(ExtractedTasks)

and get either a validated pydantic object or `AIUnavailable` (then they take
the template branch, hard rule 8). Which provider/model/key answers, retries,
fallbacks, repair, cost and audit are decided here from config/ai.yaml.

Network calls never run inside a DB transaction: `run()`/`transcribe()` raise
AIUsageError when called inside one. Audit rows and credential health are
written in their own short transactions.
"""

from __future__ import annotations

import json
import logging
import secrets
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from procbot.ai import registry as R
from procbot.ai.adapters.anthropic import AnthropicAdapter
from procbot.ai.adapters.base import ProviderAdapter, parse_json_text
from procbot.ai.adapters.fake import FakeAdapter, FakeAI
from procbot.ai.adapters.gemini import GeminiAdapter
from procbot.ai.adapters.openai import OpenAIAdapter
from procbot.ai.adapters.openai_compatible import OpenAICompatibleAdapter
from procbot.ai.audit import AttemptRecord, store_routing_config, write_call
from procbot.ai.config import AIConfig, load_ai_config
from procbot.ai.cost import (
    anomaly_check,
    compute_cost,
    fmt_usd,
    month_spend,
    month_spend_by_credential,
    owner_tz,
)
from procbot.ai.credentials import CredentialRegistry, CredentialState, Notifier
from procbot.ai.errors import AIUnavailable, AIUsageError, ProviderError
from procbot.ai.prompting import Prompt, estimate_tokens, render, repair_instruction
from procbot.ai.registry import AITaskSpec, get_task
from procbot.ai.router import Candidate, Router, lower_tier, next_tier
from procbot.ai.schema import drop_null_defaults, model_schema
from procbot.ai.types import (
    TEXT_TIERS,
    AIResult,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    StructuredStrategy,
    TextOutput,
    Tier,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptResult,
)
from procbot.context import AppContext
from procbot.domain.enums import AIErrorKind, AIOutcome, MessageCategory
from procbot.timeutil import local_date

log = logging.getLogger(__name__)

Validator = Callable[[BaseModel], list[str]]
Sleep = Callable[[float], Awaitable[None]]
RespT = TypeVar("RespT")


async def _asyncio_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def gate_notifier(ctx: AppContext) -> Notifier:
    """One-time owner notices go through the Outbound Gate (SYSTEM_NOTICE)."""

    async def notify(text: str, dedup_key: str) -> None:
        try:
            from procbot.services.gate import deliver  # lazy: built concurrently
        except ImportError:
            log.warning("gate not available; AI notice not sent", extra={"dedup_key": dedup_key})
            return
        try:
            await deliver(
                ctx, category=MessageCategory.SYSTEM_NOTICE, text=text, dedup_key=dedup_key
            )
        except NotImplementedError:
            log.warning("gate not implemented; AI notice not sent", extra={"dedup_key": dedup_key})
        except Exception:
            log.exception("failed to deliver AI notice", extra={"dedup_key": dedup_key})

    return notify


@dataclass
class _CallState:
    attempts: list[AttemptRecord] = field(default_factory=list)
    deadline: float = 0.0
    saw_invalid: bool = False
    saw_refusal: bool = False
    saw_timeout: bool = False
    saw_api_error: bool = False
    last_validation_error: str | None = None


class AIGateway:
    def __init__(
        self,
        ctx: AppContext,
        config: AIConfig | None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        environ: dict[str, str] | None = None,
        notifier: Notifier | None = None,
        sleep: Sleep | None = None,
        fake: FakeAI | None = None,
        config_error: str | None = None,
    ) -> None:
        self.ctx = ctx
        self.config = config or AIConfig()
        self.configured = config is not None and self.config.configured
        self.config_error = config_error
        self.config_sha: str | None = None
        self.notify: Notifier = notifier or gate_notifier(ctx)
        self.sleep: Sleep = sleep or _asyncio_sleep
        self.fake = fake or FakeAI()
        self.client = httpx.AsyncClient(transport=transport)
        self.credentials = CredentialRegistry(
            ctx, self.config, notify=self._notify, environ=environ
        )
        self.router = Router(self.config, self.credentials)
        self._adapters: dict[str, ProviderAdapter] = {}
        self.monotonic: Callable[[], float] = time.monotonic
        # no-AI mode
        self.no_ai_since: datetime | None = None
        self._fail_streak = 0
        self._first_fail_at: datetime | None = None
        self._last_probe_at: datetime | None = None
        self._anomaly_day: date | None = None
        self._budget_notice_month: str | None = None

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> AIGateway:
        if self.config.credentials or self.config.models:
            self.config_sha = await store_routing_config(self.ctx, self.config)
            await self.credentials.sync()
        return self

    async def aclose(self) -> None:
        await self.client.aclose()

    @property
    def no_ai_mode(self) -> bool:
        return self.no_ai_since is not None

    def adapter(self, provider: str) -> ProviderAdapter:
        a = self._adapters.get(provider)
        if a is None:
            if provider == "openai":
                a = OpenAIAdapter(self.client)
            elif provider == "openai_compatible":
                a = OpenAICompatibleAdapter(self.client)
            elif provider == "anthropic":
                a = AnthropicAdapter(self.client)
            elif provider == "gemini":
                a = GeminiAdapter(self.client)
            elif provider == "fake":
                a = FakeAdapter(self.client, self.fake)
            else:
                raise ProviderError(AIErrorKind.UNSUPPORTED, f"unknown provider {provider!r}")
            self._adapters[provider] = a
        return a

    async def _notify(self, text: str, dedup_key: str) -> None:
        if self.ctx.db.in_transaction():  # pragma: no cover - defensive
            log.error("AI notice requested inside a transaction; skipped")
            return
        await self.notify(text, dedup_key)

    def _guard(self) -> None:
        if self.ctx.db.in_transaction():
            raise AIUsageError(
                "AI gateway called inside a DB transaction: commit first, call the AI, "
                "then open a new transaction"
            )

    # ------------------------------------------------------------------ public API

    async def run(
        self,
        task: str,
        prompt: Prompt,
        *,
        initiator: str,
        correlation_id: str | None = None,
        escalate: bool = False,
        escalation_reason: str | None = None,
        validator: Validator | None = None,
    ) -> AIResult:
        """Run an AI task; return a validated result or raise AIUnavailable."""
        self._guard()
        spec = get_task(task)
        if spec.tier == Tier.STT:
            raise AIUsageError(f"task {task!r} is a speech task; use transcribe()")
        undeclared = prompt.categories - spec.data_categories
        if undeclared:
            raise AIUsageError(
                f"task {task!r} may not send data categories {sorted(undeclared)}; "
                "declare them in its AITaskSpec.data_categories"
            )
        return await self._execute(
            spec,
            prompt,
            initiator=initiator,
            correlation_id=correlation_id,
            escalate=escalate,
            escalation_reason=escalation_reason,
            validator=validator,
        )

    async def transcribe(
        self,
        audio: bytes,
        *,
        mime_type: str,
        initiator: str,
        language: str | None = "ru",
        correlation_id: str | None = None,
        duration_seconds: float | None = None,
        filename: str = "voice.ogg",
    ) -> TranscriptResult:
        """Speech to text on the `stt` tier. Raises AIUnavailable."""
        self._guard()
        spec = R.TRANSCRIBE
        categories = frozenset({"audio"})
        started = self.monotonic()
        st = _CallState(deadline=started + self.config.limits.call_timeout_seconds)
        tier = Tier.STT
        try:
            await self._precheck(spec)
        except AIUnavailable as exc:
            exc.call_id = await self._write(
                spec, initiator, correlation_id, tier, None, categories, st, started, exc
            )
            raise
        spend = await self._spend_by_credential()
        sel = self.router.select(
            tier,
            spec,
            categories=categories,
            est_tokens=0,
            now=self.ctx.clock.now(),
            spend_by_credential=spend,
        )
        if not sel.candidates:
            exc = AIUnavailable("no_candidates", self._skipped_text(sel.skipped))
            if self._health_blocked(sel.skipped):
                await self._on_api_failure()
            exc.call_id = await self._write(
                spec, initiator, correlation_id, tier, None, categories, st, started, exc
            )
            raise exc
        lim = self.config.limits
        for cand in sel.candidates:
            if len(st.attempts) >= lim.max_attempts_per_call or self.monotonic() >= st.deadline:
                st.saw_timeout = st.saw_timeout or self.monotonic() >= st.deadline
                break
            cred = await self.credentials.auth(cand.label)
            adapter = self.adapter(cand.provider)
            req = TranscriptionRequest(
                audio=audio, mime_type=mime_type, language=language, filename=filename
            )

            async def do(
                timeout: float,
                req: TranscriptionRequest = req,
                cred: Any = cred,
                adapter: ProviderAdapter = adapter,
                cand: Candidate = cand,
            ) -> TranscriptionResponse:
                req.timeout_seconds = timeout
                return await adapter.transcribe(cred, cand.model, req)

            resp, rec, err = await self._call(cand, st, do, repair=False)
            if resp is None or rec is None:
                if err is not None and err.kind == AIErrorKind.REFUSAL:
                    st.saw_refusal = True
                else:
                    st.saw_api_error = True
                continue
            usage = resp.usage
            if usage.audio_seconds is None and duration_seconds is not None:
                usage.audio_seconds = duration_seconds
            self._fill_usage(rec, cand, usage)
            await self.credentials.record_success(cand.label)
            await self._on_success()
            call_id = await self._write(
                spec,
                initiator,
                correlation_id,
                tier,
                None,
                categories,
                st,
                started,
                None,
                final=rec,
                response_type="transcript",
            )
            return TranscriptResult(
                text=resp.text,
                call_id=call_id,
                model_alias=cand.alias,
                provider=cand.provider,
                credential_label=cand.label,
                language=language,
                cost_usd=rec.cost_usd,
                cost_estimated=rec.cost_estimated,
            )
        exc = AIUnavailable(self._failure_reason(st), "transcription failed")
        if st.saw_api_error and not st.saw_refusal:
            await self._on_api_failure()
        exc.call_id = await self._write(
            spec, initiator, correlation_id, tier, None, categories, st, started, exc
        )
        raise exc

    async def probe(self, *, force: bool = False) -> bool:
        """Tiny request to see whether any provider answers; exits no-AI mode."""
        self._guard()
        now = self.ctx.clock.now()
        interval = timedelta(seconds=self.config.no_ai.probe_interval_seconds)
        if not force and self._last_probe_at is not None and now - self._last_probe_at < interval:
            return False
        self._last_probe_at = now
        try:
            await self._execute(
                R.PROBE,
                Prompt(instruction="Ответь одним словом: ok"),
                initiator="ai.probe",
                correlation_id=None,
                escalate=False,
                escalation_reason=None,
                validator=None,
                probe=True,
            )
        except AIUnavailable:
            return False
        return True

    async def maybe_probe(self) -> bool:
        """Periodic job entry: probe only while in no-AI mode."""
        if not self.no_ai_mode:
            return False
        return await self.probe()

    # ------------------------------------------------------------------ core

    async def _precheck(self, spec: AITaskSpec, *, probe: bool = False) -> bool:
        """Raise AIUnavailable for global reasons; return True if the soft budget is hit."""
        if not self.configured:
            raise AIUnavailable("not_configured", self.config_error or "config/ai.yaml missing")
        if self.no_ai_mode and not probe:
            raise AIUnavailable("no_ai_mode")
        b = self.config.budget
        if b.monthly_hard_usd is None and b.monthly_soft_usd is None:
            return False
        spent = await month_spend(self.ctx)
        if b.monthly_hard_usd is not None and spent >= b.monthly_hard_usd:
            month = self.ctx.clock.now().strftime("%Y-%m")
            if self._budget_notice_month != month:
                self._budget_notice_month = month
                await self._notify(
                    f"💸 Достигнут месячный лимит расходов на ИИ ({fmt_usd(spent)} из "
                    f"{fmt_usd(b.monthly_hard_usd)}). До конца месяца бот работает без ИИ.",
                    f"ai-budget-hard:{month}",
                )
            raise AIUnavailable("budget_hard_limit", fmt_usd(spent))
        return (
            b.monthly_soft_usd is not None
            and spent >= b.monthly_soft_usd
            and b.on_soft_limit == "downgrade_one_tier"
        )

    async def _spend_by_credential(self) -> dict[str, float]:
        if not any(st.spend_limit_usd is not None for st in self.credentials.states.values()):
            return {}
        return await month_spend_by_credential(self.ctx)

    async def _execute(
        self,
        spec: AITaskSpec,
        prompt: Prompt,
        *,
        initiator: str,
        correlation_id: str | None,
        escalate: bool,
        escalation_reason: str | None,
        validator: Validator | None,
        probe: bool = False,
    ) -> AIResult:
        started = self.monotonic()
        st = _CallState(deadline=started + self.config.limits.call_timeout_seconds)
        categories = prompt.categories
        tier, reason = self.router.initial_tier(spec, escalate=escalate)
        if reason is not None:
            reason = escalation_reason or reason
        try:
            soft = await self._precheck(spec, probe=probe)
        except AIUnavailable as exc:
            exc.call_id = await self._write(
                spec, initiator, correlation_id, tier, reason, categories, st, started, exc
            )
            raise
        if soft and tier in TEXT_TIERS and lower_tier(tier) != tier:
            tier = lower_tier(tier)
            reason = "budget_soft_limit"

        schema = model_schema(spec.output) if spec.output is not None else None
        nonce = secrets.token_hex(6)
        base = render(prompt, task_system=spec.system, json_schema=schema, nonce=nonce)
        est = estimate_tokens(base.system, base.user) + (
            estimate_tokens(json.dumps(schema)) if schema else 0
        )
        if spec.context_budget_tokens is not None and est > spec.context_budget_tokens:
            log.warning(
                "AI prompt exceeds task context budget",
                extra={"task": spec.name, "estimated": est, "budget": spec.context_budget_tokens},
            )
        now = self.ctx.clock.now()
        spend = await self._spend_by_credential()
        tiers = list(TEXT_TIERS) if probe else [tier]
        queue: deque[Candidate] = deque()
        skipped: list[tuple[str, str]] = []
        for t in tiers:
            sel = self.router.select(
                t,
                spec,
                categories=categories,
                est_tokens=est,
                now=now,
                spend_by_credential=spend,
                exclude={c.key for c in queue},
            )
            queue.extend(sel.candidates)
            skipped += sel.skipped
        if not queue:
            exc = AIUnavailable("no_candidates", self._skipped_text(skipped))
            if self._health_blocked(skipped):
                await self._on_api_failure()
            exc.call_id = await self._write(
                spec, initiator, correlation_id, tier, reason, categories, st, started, exc
            )
            raise exc

        tried: set[tuple[str, str]] = set()
        escalated: set[Tier] = {tier}
        lim = self.config.limits
        while queue:
            if len(st.attempts) >= lim.max_attempts_per_call:
                break
            if self.monotonic() >= st.deadline:
                st.saw_timeout = True
                break
            cand = queue.popleft()
            if cand.key in tried:
                continue
            tried.add(cand.key)
            kind, output, rec = await self._try_candidate(
                cand, spec, prompt, schema, nonce, st, validator, probe=probe
            )
            if kind == "ok" and output is not None and rec is not None:
                await self._on_success()
                call_id = await self._write(
                    spec,
                    initiator,
                    correlation_id,
                    tier,
                    reason,
                    categories,
                    st,
                    started,
                    None,
                    final=rec,
                    response_type=type(output).__name__,
                )
                return AIResult(
                    output=output,
                    call_id=call_id,
                    task=spec.name,
                    tier=cand.tier.value,
                    model_alias=cand.alias,
                    provider=cand.provider,
                    credential_label=cand.label,
                    tokens_in=sum(a.tokens_in for a in st.attempts),
                    tokens_out=sum(a.tokens_out for a in st.attempts),
                    cost_usd=round(sum(a.cost_usd for a in st.attempts), 8),
                    cost_estimated=any(a.cost_estimated for a in st.attempts),
                )
            if kind == "invalid" and self.config.escalation.validation_failed_twice == "next_tier":
                nt = next_tier(cand.tier)
                if nt is not None and nt not in escalated:
                    escalated.add(nt)
                    up = self.router.select(
                        nt,
                        spec,
                        categories=categories,
                        est_tokens=est,
                        now=self.ctx.clock.now(),
                        spend_by_credential=spend,
                        exclude=tried,
                    )
                    if up.candidates:
                        queue.extendleft(reversed(up.candidates))
                        reason = "validation_failed_twice"
        exc = AIUnavailable(self._failure_reason(st), st.last_validation_error or "")
        if st.saw_api_error and not (st.saw_invalid or st.saw_refusal):
            await self._on_api_failure()
        exc.call_id = await self._write(
            spec, initiator, correlation_id, tier, reason, categories, st, started, exc
        )
        raise exc

    async def _try_candidate(
        self,
        cand: Candidate,
        spec: AITaskSpec,
        prompt: Prompt,
        schema: dict[str, Any] | None,
        nonce: str,
        st: _CallState,
        validator: Validator | None,
        *,
        probe: bool,
    ) -> tuple[str, BaseModel | None, AttemptRecord | None]:
        """("ok"|"invalid"|"refused"|"error", output, final attempt)."""
        strategy = (
            cand.model.structured_output if spec.output is not None else StructuredStrategy.NONE
        )
        rendered = render(
            prompt,
            task_system=spec.system,
            json_schema=schema,
            schema_in_prompt=strategy == StructuredStrategy.JSON_MODE,
            nonce=nonce,
        )
        max_out = spec.max_output_tokens
        if cand.model.max_output_tokens is not None:
            max_out = min(max_out, cand.model.max_output_tokens)
        messages = [ChatMessage("user", rendered.user)]
        cred = await self.credentials.auth(cand.label)
        adapter = self.adapter(cand.provider)
        repair = False
        while True:
            req = CompletionRequest(
                task=spec.name,
                system=rendered.system,
                messages=list(messages),
                max_output_tokens=max_out,
                strategy=strategy,
                json_schema=schema,
                schema_name=spec.output.__name__ if spec.output is not None else "output",
                schema_description=spec.description,
            )

            async def do(timeout: float, req: CompletionRequest = req) -> CompletionResponse:
                req.timeout_seconds = timeout
                return await adapter.complete(cred, cand.model, req)

            resp, rec, err = await self._call(cand, st, do, repair=repair)
            if resp is None or rec is None:
                if err is not None and err.kind == AIErrorKind.REFUSAL:
                    st.saw_refusal = True
                    return "refused", None, rec
                st.saw_api_error = True
                return "error", None, rec
            self._fill_usage(rec, cand, resp.usage)
            await self.credentials.record_success(cand.label)
            if resp.refusal:
                rec.error_kind = AIErrorKind.REFUSAL.value
                st.saw_refusal = True
                log.info("AI refusal", extra={"task": spec.name, "model_alias": cand.alias})
                return "refused", None, rec
            output, errors = self._validate(spec, resp, schema, validator)
            if not errors and output is not None:
                return "ok", output, rec
            rec.error_kind = AIErrorKind.INVALID_OUTPUT.value
            st.last_validation_error = "; ".join(errors)[:2000]
            log.info(
                "AI output invalid",
                extra={"task": spec.name, "model_alias": cand.alias, "repair": repair},
            )
            if repair or probe:
                st.saw_invalid = True
                return "invalid", None, rec
            repair = True
            messages = [
                ChatMessage("user", rendered.user),
                ChatMessage("assistant", resp.raw_for_repair() or "(пустой ответ)"),
                ChatMessage("user", repair_instruction(errors)),
            ]

    async def _call(
        self,
        cand: Candidate,
        st: _CallState,
        fn: Callable[[float], Awaitable[RespT]],
        *,
        repair: bool,
    ) -> tuple[RespT | None, AttemptRecord | None, ProviderError | None]:
        """One logical request with retries on transient errors (timeouts, 5xx,
        network) and exponential backoff. Records every attempt."""
        lim = self.config.limits
        retries = 0
        rec: AttemptRecord | None = None
        while True:
            if len(st.attempts) >= lim.max_attempts_per_call:
                return None, rec, ProviderError(AIErrorKind.UNKNOWN, "attempt limit reached")
            remaining = st.deadline - self.monotonic()
            if remaining <= 0:
                st.saw_timeout = True
                return None, rec, ProviderError(AIErrorKind.TIMEOUT, "call time limit reached")
            timeout = min(cand.model.timeout_seconds or lim.request_timeout_seconds, remaining)
            rec = AttemptRecord(
                attempt_no=len(st.attempts) + 1,
                provider=cand.provider,
                model=cand.model.model,
                credential_label=cand.label,
                model_alias=cand.alias,
                created_at=self.ctx.clock.now(),
                repair=repair,
            )
            st.attempts.append(rec)
            t0 = self.monotonic()
            try:
                resp = await fn(timeout)
            except ProviderError as exc:
                err = exc
            except Exception as exc:  # adapter bug / unexpected payload: never crash domain code
                log.exception(
                    "AI adapter failure",
                    extra={"provider": cand.provider, "model_alias": cand.alias},
                )
                err = ProviderError(AIErrorKind.UNKNOWN, type(exc).__name__)
            else:
                rec.latency_ms = int((self.monotonic() - t0) * 1000)
                return resp, rec, None
            rec.latency_ms = int((self.monotonic() - t0) * 1000)
            rec.error_kind = err.kind.value
            rec.http_status = err.http_status
            log.warning(
                "AI attempt failed",
                extra={
                    "provider": cand.provider,
                    "model_alias": cand.alias,
                    "credential": cand.label,
                    "error_kind": err.kind.value,
                    "http_status": err.http_status,
                },
            )
            if err.transient and retries < lim.retries_per_candidate:
                delay = min(lim.backoff_base_seconds * (2**retries), lim.backoff_max_seconds)
                if self.monotonic() + delay < st.deadline:
                    retries += 1
                    await self.sleep(delay)
                    continue
            if err.kind in (AIErrorKind.RATE_LIMIT, AIErrorKind.AUTH) or err.transient:
                await self.credentials.record_failure(cand.label, err)
            return None, rec, err

    def _fill_usage(self, rec: AttemptRecord, cand: Candidate, usage: Any) -> None:
        rec.tokens_in = usage.tokens_in
        rec.tokens_out = usage.tokens_out
        rec.cost_usd, rec.cost_estimated = compute_cost(cand.model, usage)

    def _validate(
        self,
        spec: AITaskSpec,
        resp: CompletionResponse,
        schema: dict[str, Any] | None,
        validator: Validator | None,
    ) -> tuple[BaseModel | None, list[str]]:
        errors: list[str] = []
        output: BaseModel | None = None
        if spec.output is None:
            text = (resp.text if resp.text is not None else resp.raw_for_repair()).strip()
            if not text:
                return None, ["пустой ответ"]
            output = TextOutput(text=text)
        else:
            raw: Any = resp.data
            if raw is None:
                try:
                    raw = parse_json_text(resp.text or "")
                except ValueError as exc:
                    errors.append(f"ответ не является корректным JSON ({exc.__class__.__name__})")
                    if resp.truncated:
                        errors.append("ответ обрезан по лимиту токенов — ответь короче")
                    return None, errors
            if not isinstance(raw, dict):
                return None, ["ожидался JSON-объект"]
            if schema is not None:
                raw = drop_null_defaults(raw, schema)
            try:
                output = spec.output.model_validate(raw)
            except ValidationError as exc:
                for e in exc.errors()[:20]:
                    loc = ".".join(str(p) for p in e.get("loc", ())) or "(root)"
                    errors.append(f"{loc}: {e.get('msg')}")
                return None, errors
        if validator is not None and output is not None:
            errors = list(validator(output))
            if errors:
                return None, errors
        return output, []

    def _failure_reason(self, st: _CallState) -> str:
        if st.saw_invalid:
            return "invalid_output"
        if st.saw_refusal:
            return "refused"
        if st.saw_timeout:
            return "timeout"
        return "all_failed"

    @staticmethod
    def _health_blocked(skipped: list[tuple[str, str]]) -> bool:
        """Nothing to try because keys are cooling down / disabled (an outage signal)."""
        return any(r in ("credential_cooling_down", "credential_disabled") for _, r in skipped)

    @staticmethod
    def _skipped_text(skipped: list[tuple[str, str]]) -> str:
        return ", ".join(f"{w}={r}" for w, r in skipped[:10])

    async def _write(
        self,
        spec: AITaskSpec,
        initiator: str,
        correlation_id: str | None,
        tier: Tier,
        reason: str | None,
        categories: frozenset[str],
        st: _CallState,
        started: float,
        exc: AIUnavailable | None,
        *,
        final: AttemptRecord | None = None,
        response_type: str | None = None,
    ) -> int:
        if exc is None:
            outcome = AIOutcome.OK
        elif exc.reason == "invalid_output":
            outcome = AIOutcome.INVALID
        elif exc.reason == "refused":
            outcome = AIOutcome.REFUSED
        else:
            outcome = AIOutcome.UNAVAILABLE
        validation_error = st.last_validation_error
        if exc is not None and not validation_error:
            validation_error = exc.reason  # why no answer: not_configured, all_failed, ...
        if final is None and st.attempts:
            final = st.attempts[-1]
        try:
            return await write_call(
                self.ctx,
                task=spec.name,
                initiator=initiator,
                correlation_id=correlation_id,
                tier=tier.value,
                escalation_reason=reason,
                config_sha256=self.config_sha,
                input_categories=sorted(categories),
                outcome=outcome.value,
                attempts=st.attempts,
                latency_ms=int((self.monotonic() - started) * 1000),
                final=final,
                response_type=response_type,
                validation_error=validation_error,
            )
        except Exception:  # audit must never turn an answer into a crash
            log.exception("failed to write ai_calls row", extra={"task": spec.name})
            return 0

    # ------------------------------------------------------------------ no-AI mode

    async def _on_success(self) -> None:
        self._fail_streak = 0
        self._first_fail_at = None
        if self.no_ai_since is not None:
            since = self.no_ai_since
            self.no_ai_since = None
            log.warning("AI available again; leaving no-AI mode")
            await self._notify(
                "✅ ИИ снова доступен, бот вернулся к обычной работе.",
                f"ai-no-ai-recovered:{since:%Y%m%d%H%M%S}",
            )

    async def _on_api_failure(self) -> None:
        now = self.ctx.clock.now()
        self._fail_streak += 1
        if self._first_fail_at is None:
            self._first_fail_at = now
        cfg = self.config.no_ai
        if (
            self.no_ai_since is None
            and self._fail_streak >= cfg.failed_calls_threshold
            and (now - self._first_fail_at).total_seconds() >= cfg.min_failure_seconds
        ):
            self.no_ai_since = now
            self._last_probe_at = now
            log.error("all AI providers failing; entering no-AI mode")
            await self._notify(
                "⚠️ ИИ-провайдеры недоступны, бот временно работает без ИИ: задачи, сессии, "
                "кнопки и напоминания действуют как обычно. Я сам проверю доступность и "
                "сообщу, когда ИИ вернётся.",
                f"ai-no-ai:{now:%Y%m%d%H%M%S}",
            )

    # ------------------------------------------------------------------ costs / keys

    async def check_spend_anomaly(self) -> bool:
        """One SYSTEM_NOTICE per day when today's spend is anomalously high."""
        f = self.config.features
        if f.anomaly_factor is None or not self.configured:
            return False
        flagged, today, avg = await anomaly_check(
            self.ctx, factor=f.anomaly_factor, trailing_days=f.anomaly_trailing_days
        )
        if not flagged:
            return False
        async with self.ctx.db.transaction() as s:
            tz = await owner_tz(s)
        day = local_date(self.ctx.clock.now(), tz)
        if self._anomaly_day == day:
            return False
        self._anomaly_day = day
        await self._notify(
            f"📈 Расходы на ИИ сегодня необычно высокие: {fmt_usd(today)} при среднем "
            f"{fmt_usd(avg)} в день. Подробности — /ai → «Расходы ИИ».",
            f"ai-anomaly:{day.isoformat()}",
        )
        return True

    def credential_by_id(self, credential_id: int) -> CredentialState | None:
        return next((s for s in self.credentials.states.values() if s.id == credential_id), None)

    async def test_credential(self, credential_id: int) -> tuple[bool, str, list[str]]:
        """Check a key: list models (free); fallback to a tiny request."""
        self._guard()
        st = self.credential_by_id(credential_id)
        if st is None:
            return False, "ключ не найден", []
        if not st.has_key:
            return False, "ключ не задан", []
        cred = await self.credentials.auth(st.label)
        adapter = self.adapter(st.provider)
        try:
            models = await adapter.list_models(
                cred, timeout=self.config.limits.request_timeout_seconds
            )
        except ProviderError as exc:
            if exc.kind == AIErrorKind.AUTH:
                await self.credentials.record_failure(st.label, exc, notify=False)
                return False, f"ключ отклонён (HTTP {exc.http_status})", []
            if exc.kind in (AIErrorKind.UNSUPPORTED, AIErrorKind.BAD_REQUEST):
                return await self._test_by_completion(st)
            if exc.kind == AIErrorKind.RATE_LIMIT:
                await self.credentials.record_failure(st.label, exc, notify=False)
            return False, f"ошибка: {exc.kind.value} {exc.http_status or ''}".strip(), []
        except Exception as exc:
            log.exception("key test failed", extra={"credential": st.label})
            return False, f"ошибка: {type(exc).__name__}", []
        await self.credentials.record_success(st.label)
        return True, f"ключ работает, моделей доступно: {len(models)}", models

    async def _test_by_completion(self, st: CredentialState) -> tuple[bool, str, list[str]]:
        alias = next(
            (a for a, m in self.config.models.items() if st.label in self.credentials.pool_for(m)),
            None,
        )
        if alias is None:
            return False, "не удалось проверить: нет модели для этого ключа в ai.yaml", []
        model = self.config.models[alias]
        cand = Candidate(
            alias=alias, model=model, label=st.label, provider=st.provider, tier=Tier.FAST
        )
        call_st = _CallState(deadline=self.monotonic() + self.config.limits.call_timeout_seconds)
        cred = await self.credentials.auth(st.label)
        adapter = self.adapter(st.provider)
        req = CompletionRequest(
            task="probe",
            system="",
            messages=[ChatMessage("user", "Ответь: ok")],
            max_output_tokens=8,
        )

        async def do(timeout: float) -> CompletionResponse:
            req.timeout_seconds = timeout
            return await adapter.complete(cred, model, req)

        started = self.monotonic()
        resp, rec, err = await self._call(cand, call_st, do, repair=False)
        if rec is not None and resp is not None:
            self._fill_usage(rec, cand, resp.usage)
        await self._write(
            R.PROBE,
            "ai_keys.test",
            None,
            Tier.FAST,
            None,
            frozenset(),
            call_st,
            started,
            None if resp is not None else AIUnavailable("all_failed"),
            final=rec,
            response_type="probe",
        )
        if resp is None:
            kind = err.kind.value if err else "unknown"
            return False, f"пробный запрос не прошёл: {kind}", []
        await self.credentials.record_success(st.label)
        return True, f"пробный запрос к «{alias}» прошёл", []

    def tier_table(self) -> dict[str, list[str]]:
        return {t.value: list(self.config.tiers.get(t, [])) for t in Tier}


# --------------------------------------------------------------------------- module helpers


async def build_gateway(
    ctx: AppContext,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    environ: dict[str, str] | None = None,
    notifier: Notifier | None = None,
    sleep: Sleep | None = None,
) -> AIGateway:
    """Load config/ai.yaml (ctx.env.ai_config_path) and start the gateway.

    A missing file -> gateway without candidates: every run raises
    AIUnavailable("not_configured") and the bot works without AI. An invalid
    file is logged and treated the same way (shown in /ai), never a crash.
    The caller sets `ctx.ai = gateway`.
    """
    error: str | None = None
    config: AIConfig | None
    try:
        config = load_ai_config(ctx.env.ai_config_path)
    except Exception as exc:
        log.exception("invalid AI config; running without AI")
        config = None
        error = f"ошибка в {ctx.env.ai_config_path}: {type(exc).__name__}"
    gw = AIGateway(
        ctx,
        config,
        transport=transport,
        environ=environ,
        notifier=notifier,
        sleep=sleep,
        config_error=error,
    )
    try:
        await gw.start()
    except Exception:
        log.exception("AI gateway start failed; running without AI")
        gw.configured = False
        gw.config_error = "не удалось запустить ИИ-шлюз"
    return gw


def get_gateway(ctx: AppContext) -> AIGateway | None:
    return ctx.ai if isinstance(ctx.ai, AIGateway) else None


async def run_task(
    ctx: AppContext,
    task: str,
    prompt: Prompt,
    *,
    initiator: str,
    correlation_id: str | None = None,
    escalate: bool = False,
    escalation_reason: str | None = None,
    validator: Validator | None = None,
) -> AIResult:
    """`ctx.ai.run(...)` that also works when no gateway was built (-> AIUnavailable)."""
    gw = get_gateway(ctx)
    if gw is None:
        if ctx.db.in_transaction():
            raise AIUsageError("AI gateway called inside a DB transaction")
        raise AIUnavailable("not_configured", "no AI gateway in context")
    return await gw.run(
        task,
        prompt,
        initiator=initiator,
        correlation_id=correlation_id,
        escalate=escalate,
        escalation_reason=escalation_reason,
        validator=validator,
    )


def feature_enabled(ctx: AppContext, feature: str) -> bool:
    """Resource-heavy AI features switch (AppConfig.ai_features), e.g.
    "weekly_analysis", "memory_rebuild", "voice". Unknown names -> False."""
    return bool(getattr(ctx.config.ai_features, feature, False))
