#!/usr/bin/env sh
# Restore a backup (thin wrapper over `procbot verify-backup` + `procbot restore`).
#
#   scripts/restore.sh <backup-file> [--target path] [--force]
#       local checkout; the bot must be stopped. --force overwrites the existing
#       database after saving a safety copy next to it (*.pre-restore-*).
#
#   PROCBOT_DOCKER=1 scripts/restore.sh <backup-file>
#       Docker Compose: stops the service, restores into the data volume
#       (with --force + safety copy), starts the service again.
#       <backup-file> is a path on the host, or a file name inside the backups
#       volume (e.g. procbot-20261026T030000Z.db.gz.enc).
#
# The same MASTER_KEY as at backup time must be in .env.
set -eu
cd "$(dirname "$0")/.."
if [ $# -lt 1 ]; then
  sed -n '2,15p' "$0"
  exit 2
fi
file="$1"
shift

if [ "${PROCBOT_DOCKER:-0}" = "1" ]; then
  if [ -f "$file" ]; then
    abs="$(cd "$(dirname "$file")" && pwd)/$(basename "$file")"
    mount="-v ${abs}:/restore/backup:ro"
    inside="/restore/backup"
  else
    mount=""
    inside="/app/backups/$file"
  fi
  # shellcheck disable=SC2086
  docker compose run --rm --no-deps $mount procbot verify-backup "$inside"
  docker compose stop procbot
  # shellcheck disable=SC2086
  docker compose run --rm --no-deps $mount procbot restore "$inside" --force "$@"
  docker compose start procbot
  exit 0
fi

uv run procbot verify-backup "$file"
exec uv run procbot restore "$file" "$@"
