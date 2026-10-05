#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Virtual-environment Python not found: $PYTHON" >&2
    exit 1
fi
exec "$PYTHON" -m scripts.evaluate_wsd_by_sense_availability "$@"
