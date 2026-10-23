#!/usr/bin/env sh
# Create an encrypted backup now (thin wrapper over `procbot backup`).
#
#   scripts/backup.sh                 # local checkout: uv run procbot backup
#   PROCBOT_DOCKER=1 scripts/backup.sh  # running Docker Compose service
#
# Safe while the bot runs (SQLite Online Backup API). Prints JSON with the path.
set -eu
cd "$(dirname "$0")/.."
if [ "${PROCBOT_DOCKER:-0}" = "1" ]; then
  exec docker compose exec -T procbot procbot backup "$@"
fi
exec uv run procbot backup "$@"
