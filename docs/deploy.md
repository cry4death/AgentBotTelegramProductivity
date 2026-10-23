# Развёртывание

Бот — один процесс в одном контейнере: Telegram long polling, цикл планировщика
напоминаний и маленький HTTP-сервер (`/healthz`). База — SQLite на постоянном томе.

> Работает **ровно один экземпляр**. Два poller'а одного бота конфликтуют в Telegram,
> а лимит «4 напоминания за 60 минут» ведётся по одной базе. Бот сам отказывается
> запускаться второй раз: файловая блокировка `LOCK_PATH` + аренда `poller` в
> таблице `job_leases`. Не масштабируйте сервис в Docker Compose.

## 1. Telegram-токен

1. Откройте [@BotFather](https://t.me/BotFather), команда `/newbot`, задайте имя и username.
2. Скопируйте токен вида `123456789:AA...` в `TELEGRAM_BOT_TOKEN`.
3. Желательно: `/setprivacy` → Enable, `/setjoingroups` → Disable (бот личный).
4. Команды для меню (по желанию): `/setcommands` →
   ```
   start - меню и знакомство
   settings - настройки
   status - состояние бота
   help - справка
   ```

Токен — секрет: только в `.env` на сервере, никогда в git.

## 2. Telegram ID владельца

Бот отвечает только одному пользователю. Узнайте свой числовой ID, например через
[@userinfobot](https://t.me/userinfobot), и запишите в `OWNER_TELEGRAM_ID`.
Остальные получают нейтральный отказ и не видят никаких данных.

## 3. MASTER_KEY

Мастер-ключ (Fernet) шифрует секреты в базе (ключи ИИ, OAuth-токены) и сами бэкапы.

```sh
uv run procbot gen-key                       # локально
docker compose run --rm --no-deps procbot gen-key   # если есть только Docker (нужен .env)
```

Если `.env` ещё нет, можно сгенерировать ключ без проекта:
`python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`.

- Ключ хранится только в окружении; в базу и в бэкапы он не попадает.
- **Сохраните копию вне сервера** (менеджер паролей). Без ключа бэкапы не восстановить.
- Смена ключа сделает существующие зашифрованные значения и бэкапы нечитаемыми; процедуры
  ротации пока нет (см. «Известные ограничения» в README).

## 4. VPS + Docker Compose

Подойдёт небольшой Linux VPS (1 vCPU, 512 МБ–1 ГБ RAM) с Docker Engine и плагином Compose.
Где размещать VPS — открытый вопрос № 10.

```sh
git clone <repo> procbot && cd procbot
cp .env.example .env            # заполните TELEGRAM_BOT_TOKEN, OWNER_TELEGRAM_ID, MASTER_KEY
chmod 600 .env
mkdir -p config
cp config/app.example.yaml config/app.yaml   # по желанию; без файла — умолчания
docker compose up -d --build
docker compose logs -f procbot  # JSON-логи
curl -s http://127.0.0.1:8080/healthz
```

При старте бот: берёт блокировку → применяет миграции Alembic → поднимает восстановление
(просроченные аренды, «зависшие» отправки, пропущенные напоминания сворачиваются в сводку,
активные сессии восстанавливаются) → только потом запускает polling и планировщик.
`SIGTERM` (`docker compose stop`) — корректная остановка до 30 секунд.

Контейнер работает от непривилегированного пользователя (uid 10001), `restart: unless-stopped`,
`HEALTHCHECK` опрашивает `/healthz` (порт 8080, опубликован только на `127.0.0.1`).

### Тома

| Том | Путь в контейнере | Что лежит |
|---|---|---|
| `procbot-data` | `/app/data` | `procbot.db` (+ `-wal`, `-shm`), `procbot.lock` |
| `procbot-backups` | `/app/backups` | `procbot-<UTC>.db.gz.enc` + `.sha256` |
| `./config` (bind, только чтение) | `/app/config` | `app.yaml`, `ai.yaml` (необязательно) |

Права на тома — только у пользователя контейнера (`chmod 700`).

## 5. Бэкапы

Бэкап = SQLite Online Backup API (согласованный снимок без остановки бота) → gzip →
шифрование `MASTER_KEY` → файл + `.sha256` → запись в `backup_manifest`, сразу же
проверка расшифровкой и `PRAGMA integrity_check`. Содержит только базу; секреты в ней уже
зашифрованы, мастер-ключа в бэкапе нет.

- Автоматически: раз в `backup.interval_hours` (UNCONFIRMED: 24 ч; `null` — выключено).
  Срок хранения `backup.retention_days` (по умолчанию `null` — не удалять). Открытые
  вопросы № 8 и 10: периодичность, срок и копии вне сервера не утверждены.
- Вручную: `PROCBOT_DOCKER=1 scripts/backup.sh` (или `docker compose exec procbot procbot backup`).
- Проверить файл: `docker compose exec procbot procbot verify-backup /app/backups/<файл>`.
- Последний бэкап видно в `/status`.

Копия вне сервера (рекомендуется, пока вопрос № 10 не решён) — например:

```sh
docker compose cp procbot:/app/backups ./backups-copy   # затем rsync/scp куда нужно
```

## 6. Восстановление

```sh
PROCBOT_DOCKER=1 scripts/restore.sh /path/on/host/procbot-20261026T030000Z.db.gz.enc
# или имя файла внутри тома бэкапов:
PROCBOT_DOCKER=1 scripts/restore.sh procbot-20261026T030000Z.db.gz.enc
```

Скрипт проверяет бэкап, останавливает сервис, восстанавливает базу (текущая сохраняется
рядом как `procbot.db.pre-restore-<время>`), запускает сервис; при старте применяются
недостающие миграции. Нужен тот же `MASTER_KEY`, что при создании бэкапа.

Локально (без Docker, бот остановлен):

```sh
uv run procbot verify-backup backups/<файл>
uv run procbot restore backups/<файл> --target data/procbot.db --force
```

`restore` отказывается перезаписывать существующую базу без `--force` и отказывается
работать, пока запущенный бот держит блокировку. Восстановление «на чистом окружении»
покрыто автотестом (критерий приёмки 24, `tests/test_backup.py`).

## 7. Обновление

```sh
PROCBOT_DOCKER=1 scripts/backup.sh     # бэкап перед обновлением
git pull
docker compose up -d --build           # миграции применятся при старте
docker compose logs --tail=50 procbot
```

Откат: верните предыдущую версию кода и восстановите бэкап, сделанный перед обновлением
(миграции вниз автоматически не выполняются).

## 8. Локальный запуск без Docker

```sh
uv sync
cp .env.example .env    # заполнить; пути по умолчанию: data/, backups/
uv run procbot migrate
uv run procbot run      # или просто: uv run procbot
```

## 9. Мониторинг

`GET /healthz` → `{"status": "ok", "version": ..., "uptime_seconds": ..., "db": true}`,
HTTP 503 при недоступной базе. Внешнее оповещение о падении — открытый вопрос № 10:
эндпоинт готов, сам бот никуда не сообщает.
