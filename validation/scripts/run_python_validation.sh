#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="validation/results/$TS"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/harness.log"

if [[ ! -f ".env" ]]; then
  echo "Missing .env in repo root. Create it first."
  exit 2
fi

echo "Running Agents v2 validation harness..."
echo "Console log: $LOG"

set +e
uv run python validation/scripts/run_agents_v2_validation.py --run-id "$TS" --out-dir "$OUT_DIR" "$@" | tee "$LOG"
ec=${PIPESTATUS[0]}
set -e

echo
echo "Done with exit code: $ec"
echo "Harness log: $LOG"
exit "$ec"
