# procbot — developer guide

Personal Telegram anti-procrastination bot for ONE owner. Spec: `docs/spec/TZ_v0.9.md`.
Architecture: `docs/architecture.md`. Read both before changing behaviour.

## Commands

```
uv sync                      # install
uv run pytest -q             # tests (must stay green)
uv run ruff check src tests && uv run ruff format src tests
uv run pyright               # type check (0 errors expected)
```

## Hard rules (from the spec — never break)

1. Only the configured owner Telegram ID reaches any handler (`AuthMiddleware`).
2. The AI only *proposes* typed actions (`procbot.domain.actions`). Data changes go
   through `services.actions.ActionExecutor` + the FR-15 permission matrix.
   Free model text is shown, never executed.
3. New standalone tasks from conversation/AI are created ONLY after explicit
   confirmation (pending approval + button / unambiguous "да" on that card).
   Silence is never consent. Open question 36 is unanswered.
4. Bot-initiated messages go ONLY through the Outbound Gate (`services.gate`):
   six initiative categories + timer signals; quiet hours, manual pause, calendar
   busy, active work session and the strict limit of 4 reminders per any sliding
   60 minutes (counted centrally in `outbound_messages`, inside BEGIN IMMEDIATE).
   Silence alone never triggers a message.
5. No regular progress checks during an active work session.
6. Unconfirmed start stays `unconfirmed`; retro entries keep `time_source` and a
   correction journal. Partial result is never "done".
7. Google Calendar: read + insert only after explicit approval. Code that
   updates/deletes external events must not exist.
8. Everything works without AI (`AIUnavailable` -> template branch, no changes by guess).
9. Do not invent answers to spec section 10 open questions: leave the value None
   ("not set") or a clearly marked UNCONFIRMED default in config, and list it in
   `docs/open-questions.md`.

## Code conventions

- Python 3.12+, async everywhere, SQLAlchemy 2 async + aiosqlite, aiogram 3.
- Package layout: `procbot/db` (models, session), `procbot/domain` (enums, actions),
  `procbot/services/*` (business logic, no aiogram imports except keyboards for
  outgoing messages), `procbot/telegram/handlers/*` (thin: parse input, call
  service, reply), `procbot/ai/*` (gateway), `procbot/miniapp/*`.
- Time: always aware UTC datetimes (`ctx.clock.now()`); never `datetime.now()`
  in services. Local rules = LocalTime + owner IANA tz via `procbot.timeutil`.
- Transactions: `async with ctx.db.transaction() as s:` = one unit of work;
  reply "done" only after it exits. NEVER await network (Telegram/AI/Calendar)
  inside a transaction. Nested `transaction()` in the same task joins the outer one.
- Idempotency: Telegram updates are deduped by `update_id`; anything a handler
  creates uses an operation key `f"u{update_id}:..."` or a unique dedup_key;
  repeated button presses must be no-ops (check status before applying).
- Every change to tasks writes a `task_events` row (append-only) with an `actor`.
- Cross-module hooks are registered in `procbot/wiring.py` via each module's
  `register(ctx)`: renderers (`services.outbound.register_renderer`), action
  appliers, periodic jobs.
- Callback data: `procbot/telegram/callbacks.py`, one prefix per module.
- UI text is Russian. Code, comments, identifiers: English.
- Tests: pytest-asyncio auto mode; fixtures in `tests/conftest.py`; doubles in
  `procbot.testing` (FakeMessenger, MockedBotSession, update builders). Freeze time
  with `FrozenClock`; mock AI with the `fake` adapter; never hit the network.
- Each table has one owning service that writes it (see `db/models.py` docstrings).
