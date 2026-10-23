#!/usr/bin/env sh
# Regenerate migrations/versions/0001_initial.py from the current models.
# Only valid while the project has a single (initial) revision and no deployed
# databases depend on it. After the first release, add new revisions instead:
#   uv run alembic revision --autogenerate -m "describe change"
set -eu
cd "$(dirname "$0")/.."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
rm -f migrations/versions/0001_initial.py
DATABASE_PATH="$tmp/gen.db" uv run alembic revision --autogenerate \
  -m "initial schema" --rev-id 0001_initial >/dev/null
mv migrations/versions/0001_initial_initial_schema.py migrations/versions/0001_initial.py
uv run ruff format migrations >/dev/null
uv run ruff check --fix migrations >/dev/null || true
echo "regenerated migrations/versions/0001_initial.py"
