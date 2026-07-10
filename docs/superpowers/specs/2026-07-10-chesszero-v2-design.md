# ChessZero v2 Design

**Date:** 2026-07-10
**Status:** Approved by Louis (approach A: targeted evolution)
**Prereq reading:** `2026-07-08-chess-zero-design.md` (v1 spec — architecture, pipeline, and
terminology carry over; this doc specifies only what changes).

## Why v2

v1 reached ~+3074 chained self-play Elo in ~46h but external sparring exposed a hard
ceiling: 0/10 vs Negamax3 (3-ply material alpha-beta), 0/10 vs Stockfish 1350, with
occasional 7-move mates as Black — while crushing Random and Greedy 8/8 by checkmate.
Diagnosis (verified against archived checkpoints — gen 1709 loses to Negamax3 *faster*
than gen 2939, so this was a standing ceiling, not a regression):

1. **Tactical blindness.** A 32-sim Gumbel search guided by a policy that has never seen
   a 3-ply material/mate combination cannot discover one. Both self-play sides share the
   blind spot, so blunders into such tactics are never punished — no training signal
   exists. Deeper inference search barely helps (0.5/4 vs Negamax3 at 1.5s/move):
   the policy prior itself is blind.
2. **Zero opening diversity in selfplay.** The lineage plays only its own invented
   repertoire. Gate games temperature-sample 4 plies; selfplay has nothing. As Black the
   net has never faced early queen aggression — hence the miniature mates.
3. **LR schedule was manual and lossy.** Policy-loss improvement per 1000 gens by era:
   0.64 (lr 2e-3), **0.05** (6.6e-4 — a near-dead zone for ~1000 gens), 0.49 (2.2e-4).
   Both cuts were reactive, each triggered by watching gate-failure streaks.
4. **Resign never armed.** FP plateaued at 16–18% against a 5% bar; v1 paid the
   full-length-game tax for its entire life (~40% of games are decided long before they
   end). The v1 trigger (−0.95, 2 consecutive own moves) was also too loose — it
   self-armed at step 20k with an 84% resign rate and contaminated the buffer.
5. **Internal Elo flew blind.** Chained gate Elo inflated ~4× vs external reality and we
   only learned this from a manual sparring run 40 hours in.

## What v2 changes (and nothing else)

Same 6×128 SE-ResNet (bf16 activations), same pgx+mctx pipeline, same 2048 parallel
games, same buffer/batch/gating shape, same pod. From-scratch weights (no warm start —
v1's habits are the disease). Horizon: 2–3 days. v1 keeps training until v2 is smoke-
tested; Louis performs the kill/launch handover himself.

### 1. Stronger search: sims 64/16

`sims_full: 32 → 64`, `sims_cheap: 8 → 16`, `full_search_prob` stays 0.25,
`max_considered_actions` stays 16. Doubles the policy-improvement operator's depth
budget — the direct attack on failure (1).

Cost model (from v1 production data: 32–34 s/gen selfplay at 32/8, ~964 mv/s,
length-independent at steady state): ~65–70 s/gen total, ~550 mv/s, ~2M positions/h,
~4000 generations over 3 days (~500k train steps). `sims_full` stays hot-bumpable:
if ANCHOR results (below) show tactics still lagging late in the run, edit config to
96–128 and checkpoint-restart — the intervention pattern that worked twice in v1.

### 2. Random opening plies in selfplay

On each game-slot reset, play **k ~ Uniform{0..8}** uniformly-random legal moves
(device-side, vmapped) before search-driven play begins. k=0 games keep coverage of the
true initial position; k=8 games land well outside any self-invented repertoire.
Opening plies emit **no training targets** — no search runs on them; the stored game
starts at the post-opening position. Game outcome (value target) is computed as today,
from the final result relative to each stored position's mover.

Config: `selfplay.opening_plies_max: 8` (0 disables, preserving v1 semantics).

### 3. Scheduled learning rate: warmup + cosine to a floor

`make_lr_schedule` becomes: linear warmup 0 → `lr` over `warmup_steps` (500), then
cosine decay from `lr` (2e-3) to `lr * lr_floor_frac` over `lr_decay_steps`, then flat
at the floor. v2 values: `lr_decay_steps: 400_000`, `lr_floor_frac: 0.1` (floor 2e-4 —
v1's measured productive terminal rate). Runs longer than 400k steps continue at the
floor. New TrainConfig fields default to `lr_decay_steps: 0` = constant-after-warmup,
so v1.yaml behavior is unchanged.

### 4. External anchor (ANCHOR line)

Every `anchor_every_generations: 60` (~1h), the trainer spawns a subprocess:

    python scripts/versus_stockfish.py --best-dir <run_dir>/best
        --vs negamax2 negamax3 --games 6 --movetime 0.2

with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.15` (the co-residency pattern validated all
through 2026-07-09/10). The trainer parses the two summary lines, logs
`ANCHOR gen N: negamax2 X/6, negamax3 Y/6`, and writes `anchor_negamax2`,
`anchor_negamax3` into that generation's metrics row. Non-blocking: the subprocess runs
concurrently with the next generations; results are attached when ready. Failures
(crash, timeout 15 min, unparseable output) log a warning and skip — never stall
training. `anchor_every_generations: 0` disables (v1 default).

Success criterion for the whole run lives here: **v2 must trend toward and beat
negamax3 within the run** — the specific wall v1 could never touch.

### 5. Calibrated, self-arming resignation

Trigger tightened: `resign_threshold: 0.98`, `resign_consecutive_moves: 3` (v1:
0.95/2). Shadow holdout stays at 10%. New state machine in the trainer, replacing
`resign_min_train_steps` gatekeeping (field stays, still respected as a hard minimum):

- **Armed** when windowed holdout FP < `resign_arm_fp` (0.05) over the trailing
  `resign_fp_window` (2000) holdout triggers, AND global step ≥ resign_min_train_steps.
- **Disarmed** when windowed FP > `resign_disarm_fp` (0.08) over the same window
  (hysteresis prevents flapping).
- State transitions print one log line each; current state and windowed FP go into
  every metrics row (`resign_armed`, `resign_fp_windowed`).

### 6. mctx max_depth + host perf fixes

- `max_depth=16` passed to `gumbel_muzero_policy` at all three call sites (selfplay,
  gating, engine). Expected win ~10–15% (production data showed steady-state selfplay
  is less depth-dominated than the Phase-1 investigation suggested).
- Low-risk host fixes from the 2026-07-08 investigation: cast observations to fp16 on
  device before host transfer; skip `action_weights` host transfer on cheap steps.
  (mallopt tuning: out of scope — risk over reward.)

## Config file (`configs/v2.yaml`)

```yaml
net: {channels: 128, blocks: 6, se_ratio: 4, precision: bf16}
selfplay:
  num_games: 2048
  sims_full: 64          # v1: 32 — tactical-blindness fix
  sims_cheap: 16         # v1: 8
  full_search_prob: 0.25
  steps_per_generation: 16
  opening_plies_max: 8   # NEW: k ~ U{0..8} random plies per game
  resign_threshold: 0.98         # v1: 0.95
  resign_consecutive_moves: 3    # v1: 2
train:
  batch_size: 1024
  steps_per_generation: 128
  buffer_capacity: 1000000
  min_buffer: 100000
  lr: 0.002
  lr_decay_steps: 400000   # NEW: cosine horizon (~3 days)
  lr_floor_frac: 0.1       # NEW: floor = 2e-4
  resign_min_train_steps: 20000   # hard minimum before auto-arm may trigger
gating: {games: 120, promote_threshold: 0.53}
anchor_every_generations: 60     # NEW
seed: 0
run_dir: runs/v2
checkpoint_every_min: 15.0
gate_every_generations: 30
```

## Testing

- **Unit (CPU, run while v1 owns the GPU):** LR schedule shape (warmup end, cosine
  midpoint, floor clamp, `lr_decay_steps: 0` fallback); opening randomization (post-init
  board has k plies played, all moves were legal, k=0 identity, k distribution covers
  0..8); resign state machine (arm at <5%, hold between 5–8%, disarm at >8%, hard
  step minimum respected); anchor output parser (versus-script summary lines →
  scores, malformed input → skip).
- **Integration smoke (0.15 GPU slice alongside v1):** tiny config (128 games, 4/2
  sims, 2 generations) exercising openings + anchor subprocess + new log lines
  end-to-end.
- **Handover gate:** all tests green → Louis kills v1 (final best archived to GitHub
  first) → `nohup python -m chesszero.train configs/v2.yaml > runs/v2.log 2>&1 &`.

## Ops

Monitoring, checkpoint archiving to `models/` + GitHub push on each promotion, and the
gate/alarm log monitor all carry over unchanged, pointed at `runs/v2`. ANCHOR lines
join the monitor filter. Success = external anchor curve (negamax2 → negamax3 →
SF-1350 sparring) rising within the run; internal Elo is explicitly demoted to a
secondary metric.

## Out of scope

Bigger net (8×192 — wrong for a 2–3 day horizon), search-maximalist sims (128/32 —
game volume matters more), warm-starting from v1, lichess-bot integration, mallopt
tuning, any change to gating shape or buffer sizing (no evidence they're wrong).
