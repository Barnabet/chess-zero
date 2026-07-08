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
kill -9 $PID 2>/dev/null || true
sleep 3
timeout 600 python -m chesszero.train configs/tiny.yaml --generations 12
GENS=$(wc -l < runs/tiny/metrics.jsonl)
echo "metrics rows: $GENS"
test "$GENS" -ge 12
echo "SMOKE OK — Phase 0 gate passed"
