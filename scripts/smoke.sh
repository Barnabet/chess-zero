#!/usr/bin/env bash
# scripts/smoke.sh — Phase 0 gate: tiny end-to-end run + kill-and-resume drill.
set -euo pipefail
cd "$(dirname "$0")/.."

rm -rf runs/tiny
echo "=== [1/3] tiny training run (6 generations) ==="
timeout 600 python -m chesszero.train configs/tiny.yaml --generations 6

echo "=== [2/3] engine plays from the checkpoint ==="
python scripts/engine_move.py runs/tiny/best configs/tiny.yaml

echo "=== [3/3] kill -9 mid-run, then resume ==="
python -m chesszero.train configs/tiny.yaml --generations 40 &
PID=$!
sleep 45
# the drill is void if the run already finished — fail loudly, don't fake it
kill -0 "$PID" 2>/dev/null || { echo "FAIL: background run ended before kill"; exit 1; }
kill -9 "$PID"
wait "$PID" 2>/dev/null || true
timeout 600 python -m chesszero.train configs/tiny.yaml --generations 12
GENS=$(wc -l < runs/tiny/metrics.jsonl)
echo "metrics rows: $GENS"
test "$GENS" -ge 12
# resume must continue, not restart: a fresh gen-0 row may exist exactly once
test "$(grep -c '"gen": 0,' runs/tiny/metrics.jsonl)" -eq 1
echo "SMOKE OK — Phase 0 gate passed"
