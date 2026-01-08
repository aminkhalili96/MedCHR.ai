#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "Missing .venv. Run: python -m venv .venv && .venv/bin/pip install -r backend/requirements.txt"
  exit 1
fi

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

"$ROOT_DIR/.venv/bin/python" -m backend.scripts.init_db
"$ROOT_DIR/.venv/bin/uvicorn" app.main:app --app-dir "$ROOT_DIR/backend" --reload
