# procbot: one container, one process (poller + scheduler + /healthz).
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Dependencies first (cached layer), strictly from uv.lock.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Application code, migrations and example configs.
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini ./
COPY config/*.example.yaml ./config/
RUN uv sync --frozen --no-dev

# Unprivileged user; data and backups live on volumes owned by it.
RUN useradd --system --uid 10001 --home-dir /app --shell /usr/sbin/nologin procbot \
    && mkdir -p /app/data /app/backups \
    && chown -R procbot:procbot /app/data /app/backups \
    && chmod 700 /app/data /app/backups
USER procbot

ENV PATH="/app/.venv/bin:${PATH}" \
    DATABASE_PATH=/app/data/procbot.db \
    LOCK_PATH=/app/data/procbot.lock \
    BACKUP_DIR=/app/backups \
    APP_CONFIG_PATH=/app/config/app.yaml \
    AI_CONFIG_PATH=/app/config/ai.yaml \
    HTTP_HOST=0.0.0.0 \
    HTTP_PORT=8080

VOLUME ["/app/data", "/app/backups"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4); sys.exit(0 if r.status == 200 else 1)"]

ENTRYPOINT ["procbot"]
CMD ["run"]
