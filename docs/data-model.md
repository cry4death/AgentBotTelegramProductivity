# Модель данных и полномочия

Физическая схема — `src/procbot/db/models.py`, миграции — `migrations/` (Alembic,
batch-режим для SQLite). Логическая схема — раздел 5 ТЗ, дополнения — `docs/architecture.md`.

Соглашения:

- все моменты времени — aware UTC (`UTCDateTime`); локальные правила (часы тишины, утро,
  вечер) хранятся как `HH:MM` + IANA-пояс владельца и пересчитываются на каждую дату (DST);
- журналы (`task_events`, `audit_log`, `session_corrections`) только дополняются;
- перечисления хранятся строками (`procbot.domain.enums`);
- **у каждой таблицы один сервис-владелец**, который в неё пишет; остальные только читают;
- секреты (ключи ИИ, OAuth-токены) — только зашифрованные `MASTER_KEY`.

## Таблицы и владельцы

| Группа | Таблицы | Владелец (пишет) | Заметки |
|---|---|---|---|
| Профиль | `owner_profile` (одна строка, id=1), `settings` | `services.profile` (через онбординг/настройки) | Часовой пояс, ритм дня, часы тишины, пауза, характер, тайминги напоминаний, тумблеры памяти/автономности. Каждое изменение — запись в `audit_log` |
| Структура | `goals`, `projects`, `milestones`, `tasks`, `categories`, `contexts` | `services.tasks`, `services.structure` | Подзадачи через `tasks.parent_id`; `importance` без шкалы (вопрос № 4) |
| Журнал задач | `task_events` | `services.tasks` | Каждое изменение задачи с `actor` (user / system / ai) |
| Привычки | `habits`, `habit_occurrences` | HabitService (этап C) | |
| Планы дня | `daily_plans`, `plan_revisions`, `plan_items` | PlanningService | Версии: proposed → approved → revised → closed |
| Работа | `work_sessions`, `session_pauses`, `session_corrections` | `services.sessions` | `time_source` (exact / retro / approximate), статус `unconfirmed` по умолчанию |
| Уведомления | `reminder_schedule`, `outbound_messages` | `services.reminders`, `services.gate` | `outbound_messages` — источник истины для лимита 4/60 (подсчёт внутри `BEGIN IMMEDIATE`) |
| Диалог | `conversation_messages`, `conversation_states` | входной слой (`telegram.middlewares`, `telegram.storage`) | FSM-состояние диалога переживает перезапуск; срок хранения истории — вопрос № 8 |
| Поведение и помощь | `behavior_observations`, `interventions`, `intervention_outcomes` | Analysis / Intervention | |
| Память | `memory_items`, `memory_sources`, `memory_summaries`, `memory_summary_items` | MemoryService | `kind` + `origin`; удаление записи инвалидирует сводки |
| Календарь | `calendar_connections`, `calendar_event_cache`, `calendar_event_links` | CalendarService (этап C) | Только чтение и `insert` после подтверждения |
| Подтверждения | `pending_approvals`, `operation_keys` | `services.approvals` | Типизированный `action_payload`, `diff` для просмотра, `expires_at` |
| ИИ | `ai_credentials`, `ai_routing_config`, `ai_calls`, `ai_call_attempts` | AI Gateway (`procbot.ai`) | В логах и аудите только метка ключа |
| Инфраструктура | `processed_updates`, `job_leases`, `audit_log`, `backup_manifest` | `telegram.middlewares`, `services.leases` / `services.instance_lock`, все сервисы через `services.audit`, `services.backup` | Аренда `poller` защищает от второго экземпляра |

## Матрица полномочий (FR-15)

Источник истины — `procbot.domain.actions.PERMISSIONS`. ИИ только *предлагает* типизированные
действия; применяет их `services.actions.ActionExecutor` по этой матрице. Неизвестный тип
действия по умолчанию требует подтверждения.

| Действие | Полномочие | Ограничение |
|---|---|---|
| Порядок задач (`reorder_tasks`) | автоматически | Не нарушать обязательные сроки и встречи |
| Частота/время напоминаний (`adjust_reminders`) | автоматически | Тишина, пауза, календарь, не более 4 за 60 мин |
| Создание/перестройка подзадач | автоматически | В пределах родительской задачи, без смены её цели |
| Выбор способа помощи | автоматически | Без принуждения к раскрытию личного и без ложных выводов |
| Сохранение гипотезы в памяти | автоматически | Помечается как гипотеза (`ai_inferred`), опровергается пользователем |
| Новая самостоятельная задача | только с подтверждением | Вопрос 36 открыт; молчание — не согласие |
| Перенос утверждённых задач, сроков, смена статуса | только с подтверждением | |
| Запись в Google Calendar | только с подтверждением | Подтверждение конкретной записи |
| Привычки, цели, ключевые этапы | только с подтверждением | |
| Применение плана дня, заполнение свободных окон | только с подтверждением | Только предложение до согласования |
| Изменение настроек профиля (в т.ч. время утра/вечера) | только с подтверждением | Бот может лишь предложить; меняет только пользователь в «⚙️ Настройки» (FR-03) |
| Изменение/удаление внешних событий календаря | запрещено | Кода, который это делает, нет |

Владелец может отключить любое из автоматических действий в «⚙️ Настройки → Память и
автономность» (`owner_profile.autonomy_settings`: `reorder_tasks`, `adjust_reminders`,
`manage_subtasks`, `choose_interventions`; отсутствующий ключ = разрешено).

## Миграции

```sh
uv run procbot migrate                                     # upgrade head
uv run alembic revision --autogenerate -m "что изменилось"  # новая ревизия
```

Тест `tests/test_migrations.py::test_migrations_match_models` поднимает пустую базу до head
и сравнивает её с моделями: любое расхождение моделей и миграций роняет тесты. До первого
релиза начальную ревизию можно пересобрать скриптом `scripts/regen_initial_migration.sh`.
