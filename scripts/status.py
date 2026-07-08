#!/usr/bin/env python
"""One-shot status snapshot of a training run, read from metrics.jsonl.

Pure stdlib — no JAX, safe to run next to the trainer (or on a laptop
against a synced run dir).

Usage: python scripts/status.py [runs/v1]
"""
import json
import statistics
import sys
import time
from pathlib import Path


def main():
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/v1")
    path = run / "metrics.jsonl"
    if not path.exists():
        sys.exit(f"no metrics yet at {path}")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        sys.exit(f"{path} is empty")

    last = rows[-1]
    trained = [r for r in rows if "loss" in r]
    gates = [r for r in rows if "gate_score" in r]
    recent = rows[-20:]
    wall_h = (last["ts"] - rows[0]["ts"]) / 3600
    age_min = (time.time() - last["ts"]) / 60

    print(f"ChessZero status — {run}")
    print(f"  last row gen {last['gen']} ({age_min:.1f} min ago)"
          f" | {len(rows)} generations logged over {wall_h:.1f}h")
    print(f"  buffer {last['buffer_size']:,}"
          f" | global step {last['global_step']:,}")

    mps = [r["moves_per_s"] for r in recent if "moves_per_s" in r]
    if mps:
        med = statistics.median(mps)
        print(f"  selfplay {med:.0f} moves/s (median of last {len(mps)})"
              f" -> {med * 3600 / 1e6:.1f}M positions/h")

    if trained:
        t = trained[-1]
        line = (f"  loss {t['loss']:.3f} (pi {t['policy_loss']:.3f}"
                f" wdl {t['wdl_loss']:.3f} ml {t['ml_loss']:.3f})")
        if len(trained) > 50:
            line += f" | 50 gens ago: {trained[-51]['loss']:.3f}"
        print(line)
    else:
        print("  no gradient steps yet (buffer filling)")

    games = sum(r["games"] for r in rows)
    if games:
        draws = sum(r["draws"] for r in rows)
        resigns = sum(r["resigns"] for r in rows)
        lens = [r["avg_len"] for r in recent if r.get("avg_len")]
        line = (f"  games {games:,} total | {100 * draws / games:.0f}% draw"
                f" | {100 * resigns / games:.0f}% resign")
        if lens:
            line += f" | avg len {statistics.mean(lens):.0f} (recent)"
        print(line)

    if gates:
        promoted = sum(1 for r in gates if r.get("promoted"))
        tail = " ".join(
            f"{r['gate_score']:.2f}{'+' if r.get('promoted') else '-'}"
            for r in gates[-8:])
        print(f"  gates {len(gates)} played, {promoted} promoted"
              f" | recent: {tail}")

    alarms = [(r["gen"], r[k]) for r in rows
              for k in ("alarm", "resign_fp_alarm") if k in r]
    if alarms:
        gen, what = alarms[-1]
        print(f"  ALARMS {len(alarms)} rows flagged"
              f" | latest @ gen {gen}: {what}")


if __name__ == "__main__":
    main()
