# ChessZero — Pure Self-Play RL Chess Agent — Design Spec

- **Date:** 2026-07-08
- **Status:** Approved by Louis (design sections 1 & 2, this conversation)
- **Goal:** Strongest chess agent achievable from pure self-play RL (no human games, no supervised pretraining) on ~29 hours of a rented RTX 4090, measured against a Stockfish Elo ladder and live humans on Lichess.

## 1. Success criteria

- **Primary metric:** highest Stockfish `UCI_LimitStrength` rung beaten (≥50% score over 40 games), plus a logistic-fit Elo estimate with error bars, tracked against GPU-hours spent.
- **Target:** comfortably beat club-level humans — plausibly 1500–2000 Lichess blitz from this compute budget. Master level is explicitly *not* promised.
- **Secondary:** a live Lichess BOT account playing rated blitz/rapid, runnable on the M2 Air after the pod is gone.
- The pipeline must be fully resumable so a future balance top-up continues the same run.

## 2. Environment

| Resource | Value |
|---|---|
| GPU | RTX 4090, **48GB VRAM** (factory-modded), CC 8.9, 450W |
| CPU / RAM | ~27 vCPUs, 125GB (cgroup limits; `nproc` lies) |
| Storage | `/workspace` = MooseFS network volume — survives pod death |
| Cost | $0.69/hr, ~29 hours on current balance |
| Base image | Python 3.11, CUDA 12.4 driver stack (580.x); we pip-install JAX cuda12 wheels, pgx, mctx, flax, optax, orbax, python-chess |
| Local machine | M2 Air, 8GB — runs `tiny` config smoke tests and the post-pod Lichess bot (CPU JAX) |

## 3. Architecture

Single Python package, three runtime modes: **train** (pod), **eval** (Stockfish ladder + gating), **play** (Lichess bot / local, runs anywhere).

```
chesszero/
├── chesszero/
│   ├── config.py        # one dataclass config; YAML presets
│   ├── net.py           # Flax ResNet-SE: policy + WDL value + moves-left heads
│   ├── selfplay.py      # vectorized self-play: pgx chess env + mctx Gumbel MCTS
│   ├── buffer.py        # replay buffer in host RAM, uniform sampling window
│   ├── train.py         # main loop: selfplay → buffer → SGD → checkpoint → gate
│   ├── bridge.py        # pgx state/action ↔ python-chess FEN/UCI
│   ├── evaluate.py      # gating matches + Stockfish Elo ladder
│   └── engine.py        # inference player: checkpoint + search + time management
├── bot/                 # lichess-bot integration (homemade engine wrapper)
├── configs/             # tiny.yaml (M2 smoke test), v1.yaml (pod run)
├── tests/               # pytest suite (see §9)
└── scripts/             # pod bootstrap, checkpoint sync, ladder report/plots
```

Each module has one job and a narrow interface: `selfplay` emits batches of training examples, `buffer` stores/samples them, `train` owns the loop, `bridge` is the only place pgx and python-chess meet, `engine` is the only consumer of checkpoints outside training.

## 4. Network (`net.py`)

- ResNet with squeeze-excitation blocks. **Start: 6 blocks × 128 channels (~1M params).** Deliberately small — at this compute scale, self-play speed beats capacity. Config-scalable to 8×192 / 10×256 if Phase 1 calibration says the net is the bottleneck.
- **Input:** pgx's AlphaZero-style observation planes (8×8×119: 8-position history + castling/repetition/side-to-move).
- **Outputs:**
  - Policy: 4672-way AlphaZero action encoding (pgx's native encoding), illegal moves masked to −∞ before softmax.
  - Value: 3-way WDL head (win/draw/loss from mover's perspective) — draw-aware value is load-bearing in chess.
  - Moves-left: scalar auxiliary head; improves value calibration and endgame play.
- **Precision:** bf16 activations/inference, fp32 master weights and optimizer state.

## 5. Self-play (`selfplay.py`)

- ~1024 games stepped in parallel on GPU via pgx's chess env (48GB allows 2048 if calibration favors it).
- Search: `mctx.gumbel_muzero_policy` with the real env as dynamics (AlphaZero mode), **32 simulations** per full search. Gumbel root sampling replaces Dirichlet-noise/temperature exploration and stays sound at low sim counts.
- **Playout cap randomization** (KataGo): a random ~25% of moves get the full 32 sims and emit policy targets; the rest get 8-sim cheap searches and emit value/moves-left targets only. More games per GPU-hour without polluting policy targets.
- **Early resignation:** resign when both players' value heads read the position as decided (mover's win prob < 5% for 2 consecutive own-moves). **10% of games are a no-resign holdout** to measure the false-positive rate; alarm if >5% of held-out "resignable" games are not actually lost.
- Training example = (observation, Gumbel-improved policy target, final WDL from mover's perspective, moves-to-end).

## 6. Training loop (`train.py`, `buffer.py`)

- Replay buffer in host RAM: sliding window of recent positions (~1–2M, config), uniform sampling. 125GB RAM makes this trivial.
- Alternate: one self-play generation batch → N gradient steps. N tuned in Phase 1 so sample reuse lands ~4–8× and neither side starves the GPU.
- Loss: policy cross-entropy + WDL cross-entropy + small moves-left term (scaled Huber) + weight decay. Optimizer: AdamW, warmup then constant LR (values are Phase 1 tunables in config, not code).
- **Checkpoints:** Orbax, every 15 min, async-committed to `/workspace` (network volume ⇒ survives pod death). Keep: last 3 + every promoted-best. Resume restores net, optimizer, buffer metadata, and RNG state.

## 7. Evaluation (`evaluate.py`)

**Gating-lite (regression alarm):** every checkpoint, candidate vs current-best, ~120 games batched on GPU at 32-sim search, first 4 plies sampled with temperature for opening diversity (~2 min of GPU time). Promote at ≥53%. On persistent regression (3 consecutive failed gates with declining score), alarm and keep training from latest weights but keep serving/promoting from best.

**Stockfish ladder (the scoreboard):** every ~2h, 40 games/rung vs Stockfish `UCI_LimitStrength` at Elo 1320 (its floor), 1500, 1700, 1900, 2200; Stockfish at ~100ms/move on the idle vCPUs, our side at 32–64 sims on GPU. Report highest rung beaten + fitted Elo. Results append to `ladder.jsonl`; `scripts/ladder_report.py` renders Elo-vs-GPU-hours.

## 8. Lichess bot (`bot/`, `engine.py`)

- Community **lichess-bot** framework + homemade-engine class wrapping `engine.py`.
- `engine.py`: loads a checkpoint, batched Gumbel search, time management = fixed fraction of remaining clock; sim count scales with the time slice.
- During training: bot runs on the pod, sharing the GPU (<5% throughput impact). After: runs on the M2 Air with CPU JAX — a 1M-param net at 32–64 sims is comfortably fast for blitz/rapid.
- Account: fresh Lichess account → API token → BOT upgrade (irreversible per account). Accept standard rated blitz/rapid; decline variants, correspondence, and bullet initially.
- Goes live mid-Phase-2, once the ladder shows it clearing ~1320.

## 9. Testing

- `bridge.py`: round-trip pgx↔python-chess on random games — positions (FEN) and legal-move sets must match move-for-move. This is the highest-risk correctness surface.
- `net.py`: output shapes, illegal-move masking, loss finiteness/gradients.
- `buffer.py`: window eviction, sampling bounds.
- End-to-end smoke: `tiny.yaml` (2 blocks × 32 ch, 8 games, 8 sims) runs selfplay→train→checkpoint→resume→engine-plays-a-move on CPU in minutes. Runs on the M2 and in CI-of-one before any pod hours are spent.
- Kill-and-resume test in Phase 0: SIGKILL mid-run, resume, verify continuity.

## 10. Compute plan

| Phase | Budget | Content | Exit criterion |
|---|---|---|---|
| 0 Bootstrap | ~1h | Install stack; tests green; tiny smoke run; kill-and-resume drill | Loop produces a checkpoint `engine.py` can play from |
| 1 Calibration | ~2h | Throughput matrix: batch 1024/2048 × net 6×128/8×192 × sim budget; pick config maximizing positions/hr at acceptable quality | Config locked into `v1.yaml` |
| 2 The run | ~24h | Long run; ladder probe /2h; bot live mid-run; monitor gates & resign-FP | Balance nearly spent or Elo curve flat |
| 3 Wrap-up | ~1–2h | Final full ladder; export best checkpoint to Mac (+HF); report; **kill pod** | Money stops burning |

## 11. Risks & error handling

| Risk | Mitigation |
|---|---|
| pgx+mctx throughput at our config unproven | Phase 1 exists solely for this; config, not code, absorbs the answer |
| Pod dies mid-run | Orbax checkpoints on network volume; resume drill in Phase 0 |
| Training collapse | Gating alarm + roll-back-to-promoted policy (§7) |
| Resign FPs poison values | 10% no-resign holdout + alarm threshold (§5) |
| Network-volume write latency | Async Orbax commits; buffer never touches disk |
| Lichess API flakiness | lichess-bot reconnect logic; bot process isolated from training |

## 12. Out of scope

- Supervised pretraining on human games (pure RL is the point).
- Opening books and endgame tablebases, in training *and* at play time.
- Multi-GPU / distributed training; spot-instance orchestration.
- Chess variants, bullet time controls (initially).
