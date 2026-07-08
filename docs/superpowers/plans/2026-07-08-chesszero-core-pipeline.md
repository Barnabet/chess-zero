# ChessZero Core Training Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A complete, resumable self-play RL training pipeline for chess (selfplay → replay buffer → SGD → checkpoint → gating) that starts the long run on this pod.

**Architecture:** JAX end-to-end on GPU: pgx's vectorized chess env provides game dynamics, mctx's Gumbel MuZero policy provides search, a Flax ResNet-SE provides policy/WDL-value/moves-left. A thin host-side Python loop owns game bookkeeping (resignation, example emission) and the replay buffer. Spec: `docs/superpowers/specs/2026-07-08-chess-zero-design.md`. This plan covers spec §3–§7 (gating) + §9 + Phase 0/1 tooling; the Stockfish ladder, lichess bot, and sync scripts (§7-ladder, §8) are **Plan 2**, written while the long run is underway.

**Tech Stack:** Python 3.11, JAX 0.10.2 (cuda12), pgx 2.6.0, mctx 0.0.71, flax 0.12.7 (linen), optax 0.2.8, orbax-checkpoint 0.12.1, python-chess 1.11.2, numpy 2.4.6, pytest.

## Global Constraints

- **Pinned versions (already installed on the pod, verified working together):** `jax[cuda12]==0.10.2`, `pgx==2.6.0`, `mctx==0.0.71`, `flax==0.12.7`, `optax==0.2.8`, `orbax-checkpoint==0.12.1`, `chess==1.11.2` (python-chess), `numpy>=2`, `pyyaml>=6`, `pytest>=8`.
- **Verified pgx chess v2 facts** (empirically confirmed on this pod against python-chess over 724 positions / 21k+ moves — do not re-derive, do not "fix"):
  - Action label = `from_square * 73 + plane`. Squares are **FILE-major**: a1=0, a2=1, …, h8=63. python-chess is RANK-major (a1=0, b1=1, …); conversion `(sq % 8) * 8 + (sq // 8)` is an involution (works both directions).
  - Black-to-move positions are rank-flipped: flip squares with `(sq // 8) * 8 + (7 - sq % 8)` before/after encoding.
  - Planes 0–8 = underpromotions: `plane // 3` ∈ {0: Rook, 1: Bishop, 2: Knight}; `plane % 3` ∈ {0: straight, 1: right-capture, 2: left-capture} from mover's view. Planes 9–72 = all other moves incl. queen promotions, via pgx's `TO_PLANE`/`FROM_PLANE` tables.
  - Observation: shape (8, 8, 119) float32, current-player perspective; axis 0 = row with **row 0 = mover's 8th rank** (printed-board order), axis 1 = col = file a..h. Verified: at startpos, mover's pawns occupy row 6, back rank row 7.
  - `pgx.experimental.chess.from_fen(fen)` / `pgx.experimental.chess.to_fen(state)` round-trip FENs exactly and emit no warnings (do NOT use the deprecated `State._from_fen`/`state._to_fen`, which spam DeprecationWarnings; the experimental module is stable at our pinned pgx==2.6.0). `MAX_TERMINATION_STEPS = 512`; truncated games have zero rewards (treat as draw).
  - Stepping an already-terminated state inside search is safe (pgx keeps it terminated, zero rewards); mctx handles terminals via `discount=0`.
- Only `bridge.py` may know pgx↔python-chess encoding facts. Only `buffer.py` owns storage dtypes. All randomness derives from config seed — no wall-clock seeding.
- **GPU sharing:** first lines of `train.py` set `os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.70")` **before importing jax**. Any concurrently-launched second JAX process (eval, engine demo) must run with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.15`. Never launch two default-allocation JAX processes together.
- Params are always fp32; `net.precision: bf16` affects activations only. Use `flax.linen` (not nnx).
- Run all commands from `/workspace/chess-zero`. `pytest` excludes `slow`-marked tests by default (see pyproject).
- The pod bills $0.69/hr — every long-running command in this plan has an explicit timeout or generation bound. Don't invent unbounded runs.
- Commits: conventional style (`feat:`, `test:`, `chore:`), one commit per task minimum, end commit messages with the Claude co-author trailer.

---

### Task 1: Package scaffold + config

**Files:**
- Create: `pyproject.toml`
- Create: `chesszero/__init__.py` (empty)
- Create: `chesszero/config.py`
- Create: `configs/tiny.yaml`
- Create: `configs/v1.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `Config.from_yaml(path) -> Config` with sub-dataclasses `cfg.net: NetConfig`, `cfg.selfplay: SelfplayConfig`, `cfg.train: TrainConfig`, `cfg.gating: GatingConfig`; fields exactly as defined below — every later task reads these names.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from chesszero.config import Config


def test_defaults_roundtrip():
    cfg = Config()
    assert cfg.net.channels == 128 and cfg.net.blocks == 6
    assert cfg.selfplay.sims_full == 32 and cfg.selfplay.sims_cheap == 8
    d = cfg.to_dict()
    cfg2 = Config.from_dict(d)
    assert cfg2 == cfg


def test_tiny_yaml_loads():
    cfg = Config.from_yaml("configs/tiny.yaml")
    assert cfg.net.channels == 32
    assert cfg.selfplay.num_games == 8
    assert cfg.run_dir == "runs/tiny"


def test_v1_yaml_loads():
    cfg = Config.from_yaml("configs/v1.yaml")
    assert cfg.net.channels == 128
    assert cfg.selfplay.num_games >= 1024


def test_unknown_key_rejected():
    import pytest
    with pytest.raises(TypeError):
        Config.from_dict({"nonexistent_field": 1})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'chesszero'`

- [ ] **Step 3: Write pyproject, package, config, YAML presets**

```toml
# pyproject.toml
[project]
name = "chesszero"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "jax==0.10.2",
  "pgx==2.6.0",
  "mctx==0.0.71",
  "flax==0.12.7",
  "optax==0.2.8",
  "orbax-checkpoint==0.12.1",
  "chess==1.11.2",
  "pyyaml>=6",
  "numpy>=2",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["chesszero*"]

[tool.pytest.ini_options]
markers = ["slow: long-running end-to-end tests"]
addopts = "-m 'not slow'"
```

```python
# chesszero/config.py
"""Single source of configuration. YAML presets live in configs/."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class NetConfig:
    channels: int = 128
    blocks: int = 6
    se_ratio: int = 4
    precision: str = "bf16"  # "bf16" | "fp32" — activations; params always fp32


@dataclass
class SelfplayConfig:
    num_games: int = 1024             # parallel game slots on device
    sims_full: int = 32               # full search — emits policy targets
    sims_cheap: int = 8               # cheap search — value/moves-left targets only
    full_search_prob: float = 0.25    # fraction of steps run at sims_full
    max_considered_actions: int = 16  # Gumbel root candidates
    steps_per_generation: int = 16    # env steps (all slots) per generation
    resign_threshold: float = 0.95     # resign when mover E[value] < -threshold…
    resign_consecutive_moves: int = 2  # …on this many consecutive OWN moves
                                       # (per-player counter — values are
                                       # mover-relative and alternate sign, so a
                                       # shared ply counter would never trip)
    resign_holdout_frac: float = 0.10 # games that never resign (FP measurement)


@dataclass
class TrainConfig:
    lr: float = 2e-3
    warmup_steps: int = 500
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    batch_size: int = 1024
    steps_per_generation: int = 64    # gradient steps per generation
    buffer_capacity: int = 1_000_000
    min_buffer: int = 20_000          # no gradient steps below this fill
    policy_weight: float = 1.0
    value_weight: float = 1.0
    moves_left_weight: float = 0.1
    moves_left_scale: float = 50.0    # loss operates on plies / scale
    resign_min_train_steps: int = 2000  # resignation off until net has trained


@dataclass
class GatingConfig:
    games: int = 120
    promote_threshold: float = 0.53
    temperature_plies: int = 4        # opening diversity: sample first N plies
    temperature: float = 1.0


@dataclass
class Config:
    net: NetConfig = field(default_factory=NetConfig)
    selfplay: SelfplayConfig = field(default_factory=SelfplayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    seed: int = 0
    run_dir: str = "runs/dev"
    checkpoint_every_min: float = 15.0
    gate_every_generations: int = 10
    max_generations: int = 1_000_000

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        kwargs = dict(raw)
        subs = {"net": NetConfig, "selfplay": SelfplayConfig,
                "train": TrainConfig, "gating": GatingConfig}
        for name, sub_cls in subs.items():
            if name in kwargs:
                kwargs[name] = sub_cls(**kwargs[name])
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
```

```yaml
# configs/tiny.yaml — M2-Air-sized smoke config; must run end-to-end in minutes on CPU
net: {channels: 32, blocks: 2, se_ratio: 4, precision: fp32}
selfplay:
  num_games: 8
  sims_full: 8
  sims_cheap: 4
  full_search_prob: 0.5
  steps_per_generation: 32
  resign_holdout_frac: 0.5
train:
  batch_size: 64
  steps_per_generation: 4
  buffer_capacity: 10000
  min_buffer: 256
  warmup_steps: 10
  resign_min_train_steps: 20
gating: {games: 8, promote_threshold: 0.5}
seed: 7
run_dir: runs/tiny
checkpoint_every_min: 0.5
gate_every_generations: 5
```

```yaml
# configs/v1.yaml — pod run; batch/sims/net values are Phase 1 calibration outputs,
# these are the pre-calibration defaults from the spec
net: {channels: 128, blocks: 6, se_ratio: 4, precision: bf16}
selfplay:
  num_games: 1024
  sims_full: 32
  sims_cheap: 8
  full_search_prob: 0.25
  steps_per_generation: 16
train:
  batch_size: 1024
  steps_per_generation: 64
  buffer_capacity: 1000000
  min_buffer: 100000
  resign_min_train_steps: 2000
gating: {games: 120, promote_threshold: 0.53}
seed: 0
run_dir: runs/v1
checkpoint_every_min: 15.0
gate_every_generations: 10
```

- [ ] **Step 4: Install editable and run tests**

Run: `pip install -q -e . && pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml chesszero/ configs/ tests/
git commit -m "feat: package scaffold, config dataclasses, tiny/v1 presets"
```

---

### Task 2: bridge.py — pgx ↔ python-chess

**Files:**
- Create: `chesszero/bridge.py`
- Test: `tests/test_bridge.py`

**Interfaces:**
- Consumes: `pgx`, `python-chess` only.
- Produces (used by engine and tests):
  - `move_to_action(move: chess.Move, turn: chess.Color) -> int`
  - `action_to_move(action: int, board: chess.Board) -> chess.Move`
  - `state_from_fen(fen: str) -> pgx State` and `fen_from_state(state) -> str`
  - `ENV = pgx.make("chess")` re-export used nowhere else for FEN work.

The conversion constants below were verified exhaustively on this pod (724 positions, 21k+ moves, both colors, promotions/castling/en-passant). Copy them exactly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bridge.py
import random

import chess
import numpy as np
import pytest

from chesszero import bridge


PROMO_FENS = [
    "8/P6k/8/8/8/8/p6K/8 w - - 0 1",
    "8/P6k/8/8/8/8/p6K/8 b - - 0 1",
    "1n1n3k/2P5/8/8/8/8/2p5/1N1N3K w - - 0 1",
    "1n1n3k/2P5/8/8/8/8/2p5/1N1N3K b - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "8/2p5/8/KP5r/5pPk/8/4P3/6R1 b - g3 0 1",
]


def _assert_position_matches(board: chess.Board):
    state = bridge.state_from_fen(board.fen())
    pgx_legal = set(np.where(np.asarray(state.legal_action_mask))[0].tolist())
    legal = list(board.legal_moves)
    encoded = {bridge.move_to_action(m, board.turn) for m in legal}
    assert encoded == pgx_legal, board.fen()
    decoded = {bridge.action_to_move(a, board) for a in pgx_legal}
    assert decoded == set(legal), board.fen()


def test_fen_roundtrip():
    state = bridge.state_from_fen(chess.STARTING_FEN)
    assert bridge.fen_from_state(state) == chess.STARTING_FEN


def test_fixed_positions_exact():
    for fen in PROMO_FENS:
        _assert_position_matches(chess.Board(fen))


@pytest.mark.slow          # ~70s: pgx from_fen is slow; run via `pytest -m slow`
def test_random_games_exact():
    rng = random.Random(42)
    for _ in range(2):
        board = chess.Board()
        while not board.is_game_over() and board.ply() < 60:
            _assert_position_matches(board)
            board.push(rng.choice(list(board.legal_moves)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bridge.py -v`
Expected: FAIL with `ImportError: cannot import name 'bridge'`

- [ ] **Step 3: Write the implementation**

```python
# chesszero/bridge.py
"""pgx chess <-> python-chess conversion. The ONLY module that knows both encodings.

Verified pgx chess v2 facts (see plan Global Constraints):
- label = from_square * 73 + plane; squares FILE-major (a1=0, a2=1, ..., h8=63)
- black-to-move: rank-flip squares with (sq // 8) * 8 + (7 - sq % 8)
- planes 0-8 underpromotions (plane//3: 0=R 1=B 2=N; plane%3: 0=straight
  1=right-capture 2=left-capture); planes 9-72 via pgx TO_PLANE/FROM_PLANE
"""
from __future__ import annotations

import chess
import numpy as np
import pgx
import pgx._src.games.chess as _cg
import pgx.experimental.chess as _pxc

ENV = pgx.make("chess")

TO_PLANE = np.asarray(_cg.TO_PLANE)      # (64, 64) -> plane
FROM_PLANE = np.asarray(_cg.FROM_PLANE)  # (64, 73) -> to-square
_UNDERPROMO = {chess.ROOK: 0, chess.BISHOP: 1, chess.KNIGHT: 2}
_UNDERPROMO_INV = {v: k for k, v in _UNDERPROMO.items()}
_DIR_IDX = {0: 0, 1: 1, -1: 2}           # file delta -> direction index


def pc_sq_to_pgx(sq: int) -> int:
    """python-chess (rank-major) <-> pgx (file-major); involution."""
    return (sq % 8) * 8 + (sq // 8)


def flip_sq(sq: int) -> int:
    """Mirror ranks (a1<->a8) in pgx numbering."""
    return (sq // 8) * 8 + (7 - (sq % 8))


def move_to_action(move: chess.Move, turn: chess.Color) -> int:
    frm, to = pc_sq_to_pgx(move.from_square), pc_sq_to_pgx(move.to_square)
    if turn == chess.BLACK:
        frm, to = flip_sq(frm), flip_sq(to)
    if move.promotion in _UNDERPROMO:
        plane = _UNDERPROMO[move.promotion] * 3 + _DIR_IDX[(to // 8) - (frm // 8)]
    else:
        plane = int(TO_PLANE[frm, to])
    return int(frm * 73 + plane)


def action_to_move(action: int, board: chess.Board) -> chess.Move:
    frm, plane = action // 73, action % 73
    to = int(FROM_PLANE[frm, plane])
    if board.turn == chess.BLACK:
        frm, to = flip_sq(frm), flip_sq(to)
    frm_pc, to_pc = pc_sq_to_pgx(frm), pc_sq_to_pgx(to)
    promotion = None
    if plane < 9:
        promotion = _UNDERPROMO_INV[plane // 3]
    elif (board.piece_type_at(frm_pc) == chess.PAWN
          and chess.square_rank(to_pc) in (0, 7)):
        promotion = chess.QUEEN
    return chess.Move(frm_pc, to_pc, promotion=promotion)


def state_from_fen(fen: str):
    """pgx State from FEN. History planes are empty — for play, prefer
    stepping the env move-by-move (engine does this)."""
    return _pxc.from_fen(fen)


def fen_from_state(state) -> str:
    return _pxc.to_fen(state)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bridge.py -v && pytest tests/test_bridge.py -m slow -v`
Expected: first command 2 passed / 1 deselected; second command 1 passed in ~1–2 min (pgx `from_fen` is slow; that's fine, it's test-only). Zero warnings in both.

- [ ] **Step 5: Commit**

```bash
git add chesszero/bridge.py tests/test_bridge.py
git commit -m "feat: pgx<->python-chess bridge with exhaustively verified encoding"
```

---

### Task 3: net.py — ResNet-SE with three heads

**Files:**
- Create: `chesszero/net.py`
- Test: `tests/test_net.py`

**Interfaces:**
- Consumes: `NetConfig` from Task 1.
- Produces:
  - `ChessNet(cfg: NetConfig)` — flax linen module; `apply(params, obs)` with `obs: (B, 8, 8, 119)` returns `(policy_logits (B,4672) fp32, wdl_logits (B,3) fp32, moves_left (B,) fp32 ≥ 0)`.
  - `value_from_wdl(wdl_logits) -> (B,)` in [-1, 1] = P(win) − P(loss).
  - Selfplay/train/eval all call exactly these.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_net.py
import jax
import jax.numpy as jnp
import numpy as np

from chesszero.config import NetConfig
from chesszero.net import ChessNet, value_from_wdl


def _make(precision="fp32"):
    cfg = NetConfig(channels=32, blocks=2, se_ratio=4, precision=precision)
    net = ChessNet(cfg)
    obs = jnp.zeros((4, 8, 8, 119), jnp.float32)
    params = net.init(jax.random.PRNGKey(0), obs)
    return net, params, obs


def test_output_shapes_and_dtypes():
    net, params, obs = _make()
    p, w, m = net.apply(params, obs)
    assert p.shape == (4, 4672) and p.dtype == jnp.float32
    assert w.shape == (4, 3) and w.dtype == jnp.float32
    assert m.shape == (4,) and float(m.min()) >= 0.0


def test_bf16_activations_fp32_params():
    net, params, obs = _make("bf16")
    p, w, m = net.apply(params, obs)
    assert p.dtype == jnp.float32  # heads cast back to fp32
    leaves = jax.tree.leaves(params)
    assert all(l.dtype == jnp.float32 for l in leaves)


def test_value_from_wdl_bounds():
    logits = jnp.array([[10.0, 0.0, -10.0], [-10.0, 0.0, 10.0], [0.0, 10.0, 0.0]])
    v = np.asarray(value_from_wdl(logits))
    assert v[0] > 0.99 and v[1] < -0.99 and abs(v[2]) < 0.01


def test_deterministic():
    net, params, obs = _make()
    p1, _, _ = net.apply(params, obs)
    p2, _, _ = net.apply(params, obs)
    assert jnp.array_equal(p1, p2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_net.py -v`
Expected: FAIL with `ModuleNotFoundError`/`ImportError` on `chesszero.net`

- [ ] **Step 3: Write the implementation**

```python
# chesszero/net.py
"""Flax ResNet-SE trunk with policy / WDL-value / moves-left heads.

Policy head spatial mapping (verified obs layout: row 0 = mover's 8th rank,
col = file): pgx label = (file*8 + rank)*73 + plane, so the (row, col) grid
is row-flipped (row -> rank) then transposed (file first) before flattening.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from chesszero.config import NetConfig


def value_from_wdl(wdl_logits):
    p = jax.nn.softmax(wdl_logits, axis=-1)
    return p[..., 0] - p[..., 2]


class SqueezeExcite(nn.Module):
    ratio: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x):
        c = x.shape[-1]
        s = x.mean(axis=(1, 2))
        s = nn.relu(nn.Dense(max(c // self.ratio, 8), dtype=self.dtype)(s))
        s = nn.Dense(2 * c, dtype=self.dtype)(s)
        w, b = jnp.split(s, 2, axis=-1)
        return x * nn.sigmoid(w)[:, None, None, :] + b[:, None, None, :]


class ResBlock(nn.Module):
    channels: int
    se_ratio: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x):
        y = nn.Conv(self.channels, (3, 3), use_bias=False, dtype=self.dtype)(x)
        y = nn.relu(nn.GroupNorm(num_groups=8, dtype=self.dtype)(y))
        y = nn.Conv(self.channels, (3, 3), use_bias=False, dtype=self.dtype)(y)
        y = nn.GroupNorm(num_groups=8, dtype=self.dtype)(y)
        y = SqueezeExcite(self.se_ratio, self.dtype)(y)
        return nn.relu(x + y)


class ChessNet(nn.Module):
    cfg: NetConfig

    @nn.compact
    def __call__(self, obs):
        dtype = jnp.bfloat16 if self.cfg.precision == "bf16" else jnp.float32
        x = obs.astype(dtype)
        x = nn.Conv(self.cfg.channels, (3, 3), use_bias=False, dtype=dtype)(x)
        x = nn.relu(nn.GroupNorm(num_groups=8, dtype=dtype)(x))
        for _ in range(self.cfg.blocks):
            x = ResBlock(self.cfg.channels, self.cfg.se_ratio, dtype)(x)

        # Policy head: (B, row, col, 73) -> pgx label order
        p = nn.Conv(73, (1, 1), dtype=dtype)(x)
        p = jnp.flip(p, axis=1)              # row -> rank (rank 0 at index 0)
        p = jnp.transpose(p, (0, 2, 1, 3))   # (B, file, rank, 73)
        policy_logits = p.reshape(p.shape[0], 4672).astype(jnp.float32)

        v = nn.relu(nn.Conv(8, (1, 1), dtype=dtype)(x))
        v = v.reshape(v.shape[0], -1)
        v = nn.relu(nn.Dense(128, dtype=dtype)(v))
        wdl_logits = nn.Dense(3, dtype=jnp.float32)(v)

        m = nn.relu(nn.Conv(4, (1, 1), dtype=dtype)(x))
        m = m.reshape(m.shape[0], -1)
        m = nn.relu(nn.Dense(64, dtype=dtype)(m))
        moves_left = nn.softplus(nn.Dense(1, dtype=jnp.float32)(m))[..., 0]

        return policy_logits, wdl_logits, moves_left
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_net.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add chesszero/net.py tests/test_net.py
git commit -m "feat: ResNet-SE net with policy/WDL/moves-left heads"
```

---

### Task 4: buffer.py — replay buffer

**Files:**
- Create: `chesszero/buffer.py`
- Test: `tests/test_buffer.py`

**Interfaces:**
- Consumes: numpy only.
- Produces:
  - `ReplayBuffer(capacity: int, seed: int = 0)` with attributes `size`, `total_added`.
  - `add(obs (N,8,8,119) f16, policy (N,4672) f16, has_policy (N,) bool, wdl (N,) int8, moves_left (N,) int16)`.
  - `sample(batch_size) -> dict` with keys `obs` (f32), `policy` (f32), `has_policy` (bool), `wdl` (i32), `moves_left` (f32) — exactly the batch dict Task 6's loss consumes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_buffer.py
import numpy as np

from chesszero.buffer import ReplayBuffer


def _batch(n, fill):
    return (np.full((n, 8, 8, 119), fill, np.float16),
            np.full((n, 4672), 1.0 / 4672, np.float16),
            np.ones(n, bool),
            np.full(n, 1, np.int8),
            np.full(n, 40, np.int16))


def test_add_and_sample():
    buf = ReplayBuffer(capacity=100, seed=0)
    buf.add(*_batch(30, 0.5))
    assert buf.size == 30 and buf.total_added == 30
    s = buf.sample(16)
    assert s["obs"].shape == (16, 8, 8, 119) and s["obs"].dtype == np.float32
    assert s["wdl"].dtype == np.int32 and s["moves_left"].dtype == np.float32
    assert float(s["obs"].max()) == 0.5


def test_ring_wraparound_evicts_oldest():
    buf = ReplayBuffer(capacity=50, seed=0)
    buf.add(*_batch(40, 1.0))
    buf.add(*_batch(40, 2.0))          # wraps: only last 50 remain, all newest first
    assert buf.size == 50 and buf.total_added == 80
    s = buf.sample(256)
    vals = set(np.unique(s["obs"]).tolist())
    assert 2.0 in vals                  # new data present
    # 30 of the 40 old rows were overwritten; old value may remain (10 rows) but
    # buffer must never return anything other than 1.0 / 2.0
    assert vals <= {1.0, 2.0}


def test_sample_only_from_filled():
    buf = ReplayBuffer(capacity=1000, seed=0)
    buf.add(*_batch(10, 3.0))
    s = buf.sample(64)
    assert float(s["obs"].min()) == 3.0  # never samples zero-initialized rows
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_buffer.py -v`
Expected: FAIL with `ModuleNotFoundError: chesszero.buffer`

- [ ] **Step 3: Write the implementation**

```python
# chesszero/buffer.py
"""Replay buffer: preallocated host-RAM ring, uniform sampling.

Storage dtypes are the memory budget: obs f16 (~15GB at 1M positions),
policy f16 (~9GB). Cheap-search examples carry has_policy=False and are
excluded from the policy loss by the trainer.
"""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, 8, 8, 119), np.float16)
        self.policy = np.zeros((capacity, 4672), np.float16)
        self.has_policy = np.zeros(capacity, bool)
        self.wdl = np.zeros(capacity, np.int8)       # 0 win / 1 draw / 2 loss (mover)
        self.moves_left = np.zeros(capacity, np.int16)
        self.size = 0
        self.head = 0
        self.total_added = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, policy, has_policy, wdl, moves_left) -> None:
        n = obs.shape[0]
        assert n <= self.capacity, "single add larger than buffer"
        idx = (self.head + np.arange(n)) % self.capacity
        self.obs[idx] = obs
        self.policy[idx] = policy
        self.has_policy[idx] = has_policy
        self.wdl[idx] = wdl
        self.moves_left[idx] = moves_left
        self.head = int((self.head + n) % self.capacity)
        self.size = int(min(self.size + n, self.capacity))
        self.total_added += int(n)

    def sample(self, batch_size: int) -> dict:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx].astype(np.float32),
            "policy": self.policy[idx].astype(np.float32),
            "has_policy": self.has_policy[idx].copy(),
            "wdl": self.wdl[idx].astype(np.int32),
            "moves_left": self.moves_left[idx].astype(np.float32),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_buffer.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add chesszero/buffer.py tests/test_buffer.py
git commit -m "feat: host-RAM ring replay buffer"
```

---

### Task 5: selfplay.py device half — jitted Gumbel play step

**Files:**
- Create: `chesszero/selfplay.py` (device functions only in this task)
- Test: `tests/test_selfplay_device.py`

**Interfaces:**
- Consumes: `ChessNet`, `value_from_wdl` (Task 3).
- Produces (Task 6/8/9 rely on these exact signatures):
  - `ENV = pgx.make("chess")` (module-level).
  - `net_forward(net, params, obs, legal_mask) -> (masked_logits, value)`.
  - `make_recurrent_fn(net)` — mctx recurrent fn over the real env.
  - `make_play_step(net, num_simulations, max_considered, gumbel_scale)` → jitted `play_step(params, state, reset_mask, key) -> (next_state, record)` where `record` is a dict of device arrays with keys `obs (B,8,8,119)`, `action_weights (B,4672)`, `action (B,)`, `root_value (B,)`, `mover (B,)`, `rewards (B,2)`, `done (B,)`. Slots where `reset_mask` is True are replaced with fresh initial games **before** the search, so the record row belongs to the new game.
  - `init_batch(n, seed) -> state` — vmapped fresh games.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_selfplay_device.py
import jax
import jax.numpy as jnp
import numpy as np

from chesszero.config import NetConfig
from chesszero.net import ChessNet
from chesszero import selfplay


def _setup(n=4):
    net = ChessNet(NetConfig(channels=16, blocks=1, se_ratio=4, precision="fp32"))
    obs = jnp.zeros((1, 8, 8, 119), jnp.float32)
    params = net.init(jax.random.PRNGKey(0), obs)
    state = selfplay.init_batch(n, seed=0)
    step = selfplay.make_play_step(net, num_simulations=4, max_considered=4,
                                   gumbel_scale=1.0)
    return params, state, step


def test_play_step_shapes_and_legality():
    params, state, step = _setup(4)
    key = jax.random.PRNGKey(1)
    mask0 = jnp.zeros(4, bool)
    legal_before = np.asarray(state.legal_action_mask)
    state2, rec = step(params, state, mask0, key)
    a = np.asarray(rec["action"])
    assert all(legal_before[i, a[i]] for i in range(4))       # legal actions
    w = np.asarray(rec["action_weights"])
    assert w.shape == (4, 4672)
    np.testing.assert_allclose(w.sum(-1), 1.0, atol=1e-3)     # proper distribution
    assert not np.asarray(rec["done"]).any()                  # move 1 can't end chess
    assert np.asarray(rec["obs"]).shape == (4, 8, 8, 119)


def test_reset_mask_restarts_slot():
    params, state, step = _setup(4)
    key = jax.random.PRNGKey(2)
    # advance all slots two plies
    state, _ = step(params, state, jnp.zeros(4, bool), key)
    state, _ = step(params, state, jnp.zeros(4, bool), jax.random.PRNGKey(3))
    assert int(np.asarray(state._step_count)[0]) == 2
    # reset slot 0 only; after the step it has exactly 1 ply, others have 3
    mask = jnp.array([True, False, False, False])
    state, rec = step(params, state, mask, jax.random.PRNGKey(4))
    counts = np.asarray(state._step_count)
    assert counts[0] == 1 and counts[1] == 3
    # the recorded obs for slot 0 is the fresh-game obs (pre-step, post-reset):
    fresh = selfplay.init_batch(1, seed=99)
    # startpos observation is identical regardless of key
    np.testing.assert_array_equal(np.asarray(rec["obs"])[0],
                                  np.asarray(fresh.observation)[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_selfplay_device.py -v`
Expected: FAIL with `ModuleNotFoundError: chesszero.selfplay`

- [ ] **Step 3: Write the implementation**

```python
# chesszero/selfplay.py
"""Vectorized self-play: pgx chess env + mctx Gumbel MCTS.

Device half (this file, top): jitted reset-mask -> root inference -> Gumbel
search -> env step, returning one record per slot per step.
Host half (SelfplayWorker, Task 6): game bookkeeping, resignation, and
example emission when games finish.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import mctx
import pgx

from chesszero.net import value_from_wdl

ENV = pgx.make("chess")


def init_batch(n: int, seed: int):
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    return jax.jit(jax.vmap(ENV.init))(keys)


def net_forward(net, params, obs, legal_mask):
    policy_logits, wdl_logits, _ = net.apply(params, obs)
    policy_logits = jnp.where(legal_mask, policy_logits, -1e9)
    return policy_logits, value_from_wdl(wdl_logits)


def make_recurrent_fn(net):
    def recurrent_fn(params, rng_key, action, state):
        del rng_key
        prev_player = state.current_player
        state = jax.vmap(ENV.step)(state, action)
        logits, value = net_forward(net, params, state.observation,
                                    state.legal_action_mask)
        batch_idx = jnp.arange(state.rewards.shape[0])
        reward = state.rewards[batch_idx, prev_player]
        done = state.terminated | state.truncated
        value = jnp.where(done, 0.0, value)
        discount = jnp.where(done, 0.0, -1.0)
        out = mctx.RecurrentFnOutput(reward=reward, discount=discount,
                                     prior_logits=logits, value=value)
        return out, state
    return recurrent_fn


def _reset_where(mask, fresh, state):
    return jax.tree.map(
        lambda f, s: jnp.where(mask.reshape((-1,) + (1,) * (s.ndim - 1)), f, s),
        fresh, state)


def make_play_step(net, num_simulations: int, max_considered: int,
                   gumbel_scale: float):
    recurrent_fn = make_recurrent_fn(net)

    @jax.jit
    def play_step(params, state, reset_mask, key):
        k_init, k_search = jax.random.split(key)
        n = state.current_player.shape[0]
        fresh = jax.vmap(ENV.init)(jax.random.split(k_init, n))
        state = _reset_where(reset_mask, fresh, state)

        logits, value = net_forward(net, params, state.observation,
                                    state.legal_action_mask)
        root = mctx.RootFnOutput(prior_logits=logits, value=value,
                                 embedding=state)
        out = mctx.gumbel_muzero_policy(
            params=params, rng_key=k_search, root=root,
            recurrent_fn=recurrent_fn, num_simulations=num_simulations,
            invalid_actions=~state.legal_action_mask,
            max_num_considered_actions=max_considered,
            gumbel_scale=gumbel_scale)

        next_state = jax.vmap(ENV.step)(state, out.action)
        record = {
            "obs": state.observation,
            "action_weights": out.action_weights,
            "action": out.action,
            "root_value": value,
            "mover": state.current_player,
            "rewards": next_state.rewards,
            "done": next_state.terminated | next_state.truncated,
        }
        return next_state, record

    return play_step
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_selfplay_device.py -v`
Expected: 2 passed (first run compiles; ~1 min total)

- [ ] **Step 5: Commit**

```bash
git add chesszero/selfplay.py tests/test_selfplay_device.py
git commit -m "feat: jitted Gumbel MCTS play step over pgx chess"
```

---

### Task 6: selfplay.py host half — SelfplayWorker

**Files:**
- Modify: `chesszero/selfplay.py` (append)
- Test: `tests/test_selfplay_worker.py`

**Interfaces:**
- Consumes: device half (Task 5), `Config` (Task 1).
- Produces:
  - `Example` dataclass: `obs: np.ndarray (8,8,119) f16`, `policy: np.ndarray|None (4672,) f16`, `wdl: int (0 win/1 draw/2 loss, mover's perspective)`, `moves_left: int` (plies remaining incl. current).
  - `GenStats` dataclass: `games, examples, resigns, draws, holdout_resign_games, holdout_false_positives, sum_game_len` (all int).
  - `SelfplayWorker(net, cfg: Config, seed: int)` with `run_generation(params, allow_resign: bool) -> (list[Example], GenStats)`.
  - `pack_examples(examples) -> (obs, policy, has_policy, wdl, moves_left)` numpy arrays matching `ReplayBuffer.add`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_selfplay_worker.py
import jax
import jax.numpy as jnp
import numpy as np

from chesszero.config import Config, NetConfig, SelfplayConfig
from chesszero.net import ChessNet
from chesszero.selfplay import Example, GenStats, SelfplayWorker, pack_examples


def _tiny_cfg(**sp):
    base = dict(num_games=4, sims_full=4, sims_cheap=2, full_search_prob=1.0,
                max_considered_actions=4, steps_per_generation=6,
                resign_holdout_frac=0.0)
    base.update(sp)
    return Config(net=NetConfig(channels=16, blocks=1, precision="fp32"),
                  selfplay=SelfplayConfig(**base), seed=3)


def _worker(cfg):
    net = ChessNet(cfg.net)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 8, 8, 119)))
    return SelfplayWorker(net, cfg, seed=cfg.seed), params


def test_generation_runs_and_accounts():
    cfg = _tiny_cfg()
    w, params = _worker(cfg)
    examples, stats = w.run_generation(params, allow_resign=False)
    # every recorded ply is either emitted as an example (finished games) or
    # still held in a live slot — nothing lost, nothing duplicated
    held = sum(len(s.obs) for s in w.slots)
    assert held + stats.examples == 4 * 6
    assert len(examples) == stats.examples


def test_flush_emits_correct_targets():
    cfg = _tiny_cfg()
    w, params = _worker(cfg)
    # Inject a synthetic finished 3-ply game into slot 0:
    slot = w.slots[0]
    slot.obs = [np.zeros((8, 8, 119), np.float16)] * 3
    slot.weights = [np.full(4672, 1 / 4672, np.float16), None,
                    np.full(4672, 1 / 4672, np.float16)]
    slot.mover = [0, 1, 0]
    slot.root_value = [0.0, 0.0, 0.0]
    examples, stats = [], GenStats()
    final_rewards = np.array([1.0, -1.0])  # player 0 won
    w._flush(0, final_rewards, examples, stats, resigned=False)
    assert stats.games == 1 and stats.examples == 3
    assert [e.wdl for e in examples] == [0, 2, 0]        # W, L, W (mover view)
    assert [e.moves_left for e in examples] == [3, 2, 1]
    assert examples[1].policy is None                    # cheap-search ply
    assert len(w.slots[0].obs) == 0                      # slot recycled


def _record(movers, values):
    """Synthetic single-step device record for host-logic tests."""
    n = len(movers)
    return {
        "obs": np.zeros((n, 8, 8, 119), np.float16),
        "action_weights": np.full((n, 4672), 1 / 4672, np.float16),
        "action": np.zeros(n, np.int64),
        "root_value": np.asarray(values, np.float32),
        "mover": np.asarray(movers, np.int64),
        "rewards": np.zeros((n, 2), np.float32),
        "done": np.zeros(n, bool),
    }


def test_resign_counter_is_per_player():
    cfg = _tiny_cfg(num_games=1, resign_threshold=0.9, resign_consecutive_moves=2)
    w, _ = _worker(cfg)
    examples, stats = [], GenStats()
    # decided game: player 0 hopeless on own moves, player 1 confident on theirs
    for mover, val in [(0, -0.95), (1, 0.95)]:
        w._process(_record([mover], [val]), True, True, examples, stats)
    assert stats.resigns == 0            # one bad own-move is not enough
    w._process(_record([0], [-0.95]), True, True, examples, stats)
    assert stats.resigns == 1            # 2nd consecutive bad own-move trips
    assert [e.wdl for e in examples] == [2, 0, 2]  # loser L, winner W, loser L


def test_resign_counter_resets_on_recovery():
    cfg = _tiny_cfg(num_games=1, resign_threshold=0.9, resign_consecutive_moves=2)
    w, _ = _worker(cfg)
    examples, stats = [], GenStats()
    seq = [(0, -0.95), (1, 0.95), (0, 0.0), (1, 0.95), (0, -0.95)]
    for mover, val in seq:
        w._process(_record([mover], [val]), True, True, examples, stats)
    assert stats.resigns == 0            # recovery at ply 3 reset player 0's count


def test_resignation_forces_loss():
    cfg = _tiny_cfg(resign_threshold=-1.1,  # -thr = +1.1: every value < 1.1 is "hopeless"
                    resign_consecutive_moves=2, steps_per_generation=4)
    w, params = _worker(cfg)
    examples, stats = w.run_generation(params, allow_resign=True)
    assert stats.resigns >= 1                            # games got adjudicated
    assert stats.games == stats.resigns
    losses = [e for e in examples if e.wdl == 2]
    wins = [e for e in examples if e.wdl == 0]
    assert losses and wins                               # both perspectives present


def test_pack_examples_shapes():
    ex = [Example(np.zeros((8, 8, 119), np.float16),
                  np.full(4672, 1 / 4672, np.float16), 0, 10),
          Example(np.zeros((8, 8, 119), np.float16), None, 1, 5)]
    obs, pol, hasp, wdl, ml = pack_examples(ex)
    assert obs.shape == (2, 8, 8, 119) and pol.shape == (2, 4672)
    assert hasp.tolist() == [True, False]
    assert wdl.tolist() == [0, 1] and ml.tolist() == [10, 5]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_selfplay_worker.py -v`
Expected: FAIL with `ImportError: cannot import name 'SelfplayWorker'`

- [ ] **Step 3: Append the host half to selfplay.py**

```python
# append to chesszero/selfplay.py

import numpy as np
from dataclasses import dataclass, field

from chesszero.config import Config


@dataclass
class Example:
    obs: np.ndarray                 # (8,8,119) float16
    policy: "np.ndarray | None"     # (4672,) float16, None for cheap-search plies
    wdl: int                        # 0 win / 1 draw / 2 loss, mover's perspective
    moves_left: int                 # plies remaining including this one


@dataclass
class GenStats:
    games: int = 0
    examples: int = 0
    resigns: int = 0
    draws: int = 0
    holdout_resign_games: int = 0
    holdout_false_positives: int = 0
    sum_game_len: int = 0


@dataclass
class _Slot:
    obs: list = field(default_factory=list)
    weights: list = field(default_factory=list)      # per-ply f16 array or None
    mover: list = field(default_factory=list)
    root_value: list = field(default_factory=list)
    resign_counts: list = field(default_factory=lambda: [0, 0])  # per player id
    resign_would_have: int = -1                      # 1-based ply, -1 = never
    holdout: bool = False


def pack_examples(examples):
    n = len(examples)
    obs = np.stack([e.obs for e in examples]).astype(np.float16)
    policy = np.zeros((n, 4672), np.float16)
    has_policy = np.zeros(n, bool)
    for j, e in enumerate(examples):
        if e.policy is not None:
            policy[j] = e.policy
            has_policy[j] = True
    wdl = np.array([e.wdl for e in examples], np.int8)
    moves_left = np.array([e.moves_left for e in examples], np.int16)
    return obs, policy, has_policy, wdl, moves_left


class SelfplayWorker:
    """Owns a batch of parallel games + all host-side bookkeeping."""

    def __init__(self, net, cfg: Config, seed: int):
        sp = cfg.selfplay
        self.cfg = cfg
        self.n = sp.num_games
        self.step_full = make_play_step(net, sp.sims_full,
                                        sp.max_considered_actions, 1.0)
        self.step_cheap = make_play_step(net, sp.sims_cheap,
                                         sp.max_considered_actions, 1.0)
        self.key = jax.random.PRNGKey(seed)
        self.np_rng = np.random.default_rng(seed)
        self.state = init_batch(self.n, seed ^ 0x5EED)
        self.reset_mask = np.zeros(self.n, bool)
        self.slots = [self._new_slot() for _ in range(self.n)]

    def _new_slot(self) -> _Slot:
        s = _Slot()
        s.holdout = (self.np_rng.random()
                     < self.cfg.selfplay.resign_holdout_frac)
        return s

    def run_generation(self, params, allow_resign: bool):
        sp = self.cfg.selfplay
        examples: list[Example] = []
        stats = GenStats()
        for _ in range(sp.steps_per_generation):
            self.key, sub = jax.random.split(self.key)
            full = bool(self.np_rng.random() < sp.full_search_prob)
            step_fn = self.step_full if full else self.step_cheap
            self.state, record = step_fn(params, self.state,
                                         jnp.asarray(self.reset_mask), sub)
            self._process(record, full, allow_resign, examples, stats)
        return examples, stats

    def _process(self, record, full, allow_resign, examples, stats):
        sp = self.cfg.selfplay
        obs = np.asarray(record["obs"], np.float16)
        weights = np.asarray(record["action_weights"], np.float16)
        movers = np.asarray(record["mover"])
        values = np.asarray(record["root_value"])
        rewards = np.asarray(record["rewards"])
        done = np.asarray(record["done"])
        new_reset = np.zeros(self.n, bool)
        for i in range(self.n):
            slot = self.slots[i]
            p = int(movers[i])
            slot.obs.append(obs[i])
            slot.weights.append(weights[i] if full else None)
            slot.mover.append(p)
            slot.root_value.append(float(values[i]))
            # per-player counter: values are mover-relative, so only player
            # p's own moves speak to whether p should resign
            if float(values[i]) < -sp.resign_threshold:
                slot.resign_counts[p] += 1
            else:
                slot.resign_counts[p] = 0
            tripped = slot.resign_counts[p] >= sp.resign_consecutive_moves
            if tripped and slot.resign_would_have < 0:
                slot.resign_would_have = len(slot.obs)
            if done[i]:
                self._flush(i, rewards[i], examples, stats, resigned=False)
                new_reset[i] = True
            elif allow_resign and tripped and not slot.holdout:
                fake = np.zeros(2, np.float32)
                fake[p], fake[1 - p] = -1.0, 1.0
                self._flush(i, fake, examples, stats, resigned=True)
                new_reset[i] = True
        self.reset_mask = new_reset

    def _flush(self, i, final_rewards, examples, stats, resigned):
        slot = self.slots[i]
        n_ply = len(slot.obs)
        for t in range(n_ply):
            r = float(final_rewards[slot.mover[t]])
            wdl = 0 if r > 0.5 else (2 if r < -0.5 else 1)
            examples.append(Example(obs=slot.obs[t], policy=slot.weights[t],
                                    wdl=wdl, moves_left=n_ply - t))
        stats.games += 1
        stats.examples += n_ply
        stats.sum_game_len += n_ply
        if resigned:
            stats.resigns += 1
        elif abs(float(final_rewards[0])) < 0.5:
            stats.draws += 1
        if slot.holdout and slot.resign_would_have > 0:
            stats.holdout_resign_games += 1
            mover_at = slot.mover[slot.resign_would_have - 1]
            if float(final_rewards[mover_at]) > -0.5:
                stats.holdout_false_positives += 1
        self.slots[i] = self._new_slot()
```

Note the ordering contract with Task 5: `reset_mask` slots are replaced with fresh games *before* search, so every record row appended here always belongs to the slot's current game, and `_flush` clears the slot in the same host step that sets `reset_mask[i] = True`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_selfplay_worker.py -v`
Expected: 6 passed (~1–2 min, includes compiles)

- [ ] **Step 5: Commit**

```bash
git add chesszero/selfplay.py tests/test_selfplay_worker.py
git commit -m "feat: selfplay worker with resignation, holdout, example emission"
```

---

### Task 7: losses + train step

**Files:**
- Create: `chesszero/train.py` (losses/optimizer part only in this task)
- Test: `tests/test_train_step.py`

**Interfaces:**
- Consumes: `ChessNet` (Task 3), batch dict format (Task 4), `TrainConfig` (Task 1).
- Produces:
  - `make_optimizer(cfg: TrainConfig) -> optax.GradientTransformation` (clip → adamw, warmup-then-constant LR).
  - `make_train_step(net, tx, cfg: TrainConfig)` → jitted `train_step(params, opt_state, batch) -> (params, opt_state, metrics)` where `metrics = {"loss","policy_loss","wdl_loss","ml_loss"}` (fp32 scalars) and `batch` values are jnp arrays as produced by `ReplayBuffer.sample`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_step.py
import jax
import jax.numpy as jnp
import numpy as np

from chesszero.config import NetConfig, TrainConfig
from chesszero.net import ChessNet
from chesszero.train import make_optimizer, make_train_step


def _setup():
    net = ChessNet(NetConfig(channels=16, blocks=1, precision="fp32"))
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 8, 8, 119)))
    cfg = TrainConfig(lr=1e-2, warmup_steps=1, batch_size=8)
    tx = make_optimizer(cfg)
    opt_state = tx.init(params)
    step = make_train_step(net, tx, cfg)
    rng = np.random.default_rng(0)
    pol = rng.random((8, 4672)).astype(np.float32)
    pol /= pol.sum(-1, keepdims=True)
    batch = {
        "obs": jnp.asarray(rng.random((8, 8, 8, 119)), jnp.float32),
        "policy": jnp.asarray(pol),
        "has_policy": jnp.asarray([True] * 6 + [False] * 2),
        "wdl": jnp.asarray(rng.integers(0, 3, 8), jnp.int32),
        "moves_left": jnp.asarray(rng.integers(1, 100, 8), jnp.float32),
    }
    return params, opt_state, step, batch


def test_loss_finite_and_grads_flow():
    params, opt_state, step, batch = _setup()
    params2, opt_state2, m = step(params, opt_state, batch)
    for k in ("loss", "policy_loss", "wdl_loss", "ml_loss"):
        assert np.isfinite(float(m[k])), k
    diffs = jax.tree.map(lambda a, b: float(jnp.abs(a - b).max()),
                         params, params2)
    assert max(jax.tree.leaves(diffs)) > 0  # something actually updated


def test_overfits_fixed_batch():
    params, opt_state, step, batch = _setup()
    first = None
    for i in range(60):
        params, opt_state, m = step(params, opt_state, batch)
        if first is None:
            first = float(m["loss"])
    assert float(m["loss"]) < first * 0.8  # clearly decreasing on a fixed batch
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_step.py -v`
Expected: FAIL with `ModuleNotFoundError: chesszero.train`

- [ ] **Step 3: Write the implementation**

```python
# chesszero/train.py
"""Losses, optimizer, train step (this task); Trainer main loop (Task 9)."""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.70")

from functools import partial

import jax
import jax.numpy as jnp
import optax

from chesszero.config import TrainConfig


def make_optimizer(cfg: TrainConfig) -> optax.GradientTransformation:
    schedule = optax.join_schedules(
        [optax.linear_schedule(0.0, cfg.lr, cfg.warmup_steps),
         optax.constant_schedule(cfg.lr)],
        boundaries=[cfg.warmup_steps])
    return optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(schedule, weight_decay=cfg.weight_decay))


def make_train_step(net, tx, cfg: TrainConfig):
    def loss_fn(params, batch):
        policy_logits, wdl_logits, moves_left = net.apply(params, batch["obs"])
        # Policy CE against the Gumbel-improved distribution; illegal moves are
        # zero in the target. Only full-search rows contribute.
        logp = jax.nn.log_softmax(policy_logits, axis=-1)
        ce = -(batch["policy"] * logp).sum(-1)
        mask = batch["has_policy"].astype(jnp.float32)
        policy_loss = (ce * mask).sum() / jnp.maximum(mask.sum(), 1.0)
        wdl_loss = optax.softmax_cross_entropy_with_integer_labels(
            wdl_logits, batch["wdl"]).mean()
        s = cfg.moves_left_scale
        ml_loss = optax.huber_loss(moves_left / s,
                                   batch["moves_left"] / s, delta=1.0).mean()
        loss = (cfg.policy_weight * policy_loss
                + cfg.value_weight * wdl_loss
                + cfg.moves_left_weight * ml_loss)
        return loss, {"loss": loss, "policy_loss": policy_loss,
                      "wdl_loss": wdl_loss, "ml_loss": ml_loss}

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step(params, opt_state, batch):
        grads, metrics = jax.grad(loss_fn, has_aux=True)(params, batch)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics

    return train_step
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_step.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add chesszero/train.py tests/test_train_step.py
git commit -m "feat: AZ losses (policy CE + WDL CE + moves-left huber) and train step"
```

---

### Task 8: evaluate.py — gating matches

**Files:**
- Create: `chesszero/evaluate.py`
- Test: `tests/test_gating.py`

**Interfaces:**
- Consumes: `net_forward`, `make_recurrent_fn` patterns from Task 5 (re-implemented here for two param sets — do NOT try to reuse `make_play_step`, the two-net selection makes it a different function), `Config` (Task 1).
- Produces: `play_match(net, params_a, params_b, cfg: Config, seed: int = 0) -> float` — score for A in [0, 1], playing `cfg.gating.games` games (half with A as white), deterministic search (`gumbel_scale=0`) except the first `cfg.gating.temperature_plies` plies which are sampled from `action_weights` with `cfg.gating.temperature`. Task 9's Trainer calls exactly this.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gating.py
import jax
import jax.numpy as jnp

from chesszero.config import Config, GatingConfig, NetConfig, SelfplayConfig
from chesszero.net import ChessNet
from chesszero.evaluate import play_match


def test_match_runs_and_scores():
    cfg = Config(net=NetConfig(channels=16, blocks=1, precision="fp32"),
                 selfplay=SelfplayConfig(sims_full=4, max_considered_actions=4),
                 gating=GatingConfig(games=4, temperature_plies=2))
    net = ChessNet(cfg.net)
    obs = jnp.zeros((1, 8, 8, 119))
    pa = net.init(jax.random.PRNGKey(0), obs)
    pb = net.init(jax.random.PRNGKey(1), obs)
    score = play_match(net, pa, pb, cfg, seed=5)
    assert 0.0 <= score <= 1.0


def test_self_match_is_roughly_even():
    # identical params: expect a score near 0.5 (draws + symmetric colors)
    cfg = Config(net=NetConfig(channels=16, blocks=1, precision="fp32"),
                 selfplay=SelfplayConfig(sims_full=4, max_considered_actions=4),
                 gating=GatingConfig(games=8, temperature_plies=4))
    net = ChessNet(cfg.net)
    obs = jnp.zeros((1, 8, 8, 119))
    p = net.init(jax.random.PRNGKey(0), obs)
    score = play_match(net, p, p, cfg, seed=6)
    assert 0.15 <= score <= 0.85  # loose bound: tiny sample, but not degenerate
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gating.py -v`
Expected: FAIL with `ModuleNotFoundError: chesszero.evaluate`

- [ ] **Step 3: Write the implementation**

```python
# chesszero/evaluate.py
"""Gating matches: candidate (A) vs incumbent (B), batched on device.

Both nets run on the full batch every step; per-slot masks pick whose
output drives each game. Cost is 2x inference on gating.games slots for
a couple of minutes — negligible next to selfplay.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import mctx
import numpy as np

from chesszero.config import Config
from chesszero.net import value_from_wdl
from chesszero.selfplay import ENV, init_batch


def _pair_forward(net, pa, pb, is_a, obs, legal_mask):
    la, wa, _ = net.apply(pa, obs)
    lb, wb, _ = net.apply(pb, obs)
    logits = jnp.where(is_a[:, None], la, lb)
    logits = jnp.where(legal_mask, logits, -1e9)
    value = jnp.where(is_a, value_from_wdl(wa), value_from_wdl(wb))
    return logits, value


def _make_versus_step(net, sims, max_considered):
    def recurrent_fn(params, rng_key, action, state):
        del rng_key
        pa, pb, a_player = params["a"], params["b"], params["a_player"]
        prev_player = state.current_player
        state = jax.vmap(ENV.step)(state, action)
        is_a = state.current_player == a_player
        logits, value = _pair_forward(net, pa, pb, is_a,
                                      state.observation,
                                      state.legal_action_mask)
        idx = jnp.arange(state.rewards.shape[0])
        reward = state.rewards[idx, prev_player]
        done = state.terminated | state.truncated
        value = jnp.where(done, 0.0, value)
        discount = jnp.where(done, 0.0, -1.0)
        return mctx.RecurrentFnOutput(reward=reward, discount=discount,
                                      prior_logits=logits, value=value), state

    @jax.jit
    def versus_step(params, state, key, temperature_mask, temperature):
        is_a = state.current_player == params["a_player"]
        logits, value = _pair_forward(net, params["a"], params["b"], is_a,
                                      state.observation,
                                      state.legal_action_mask)
        root = mctx.RootFnOutput(prior_logits=logits, value=value,
                                 embedding=state)
        k_search, k_sample = jax.random.split(key)
        out = mctx.gumbel_muzero_policy(
            params=params, rng_key=k_search, root=root,
            recurrent_fn=recurrent_fn, num_simulations=sims,
            invalid_actions=~state.legal_action_mask,
            max_num_considered_actions=max_considered,
            gumbel_scale=0.0)
        # opening diversity: sample from improved policy on early plies
        w = jnp.maximum(out.action_weights, 1e-9)
        sampled = jax.random.categorical(
            k_sample, jnp.log(w) / temperature, axis=-1)
        action = jnp.where(temperature_mask, sampled, out.action)
        next_state = jax.vmap(ENV.step)(state, action)
        # freeze finished games: keep terminal state, don't step it
        keep = state.terminated | state.truncated
        next_state = jax.tree.map(
            lambda old, new: jnp.where(
                keep.reshape((-1,) + (1,) * (new.ndim - 1)), old, new),
            state, next_state)
        return next_state

    return versus_step


def play_match(net, params_a, params_b, cfg: Config, seed: int = 0) -> float:
    g = cfg.gating
    n = g.games
    state = init_batch(n, seed)
    # which player-id is white in each slot: white moves first
    white_id = np.asarray(state.current_player).copy()
    a_white = np.arange(n) < n // 2
    a_player = jnp.asarray(np.where(a_white, white_id, 1 - white_id))
    params = {"a": params_a, "b": params_b, "a_player": a_player}
    step = _make_versus_step(net, cfg.selfplay.sims_full,
                             cfg.selfplay.max_considered_actions)
    key = jax.random.PRNGKey(seed * 2 + 1)
    ply = 0
    while True:
        done = np.asarray(state.terminated | state.truncated)
        if done.all() or ply >= 512:
            break
        key, sub = jax.random.split(key)
        temp_mask = jnp.full((n,), ply < g.temperature_plies)
        state = step(params, state, sub, temp_mask, g.temperature)
        ply += 1
    rewards = np.asarray(state.rewards)
    a_idx = np.asarray(a_player)
    score = float(np.mean((rewards[np.arange(n), a_idx] + 1.0) / 2.0))
    return score
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gating.py -v`
Expected: 2 passed (~2–4 min: two tiny nets, full games to termination)

- [ ] **Step 5: Commit**

```bash
git add chesszero/evaluate.py tests/test_gating.py
git commit -m "feat: batched two-net gating matches with opening temperature"
```

---

### Task 9: Trainer main loop — checkpointing, resume, gating, metrics

**Files:**
- Modify: `chesszero/train.py` (append Trainer + CLI)
- Test: `tests/test_trainer.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `Trainer(cfg: Config)` with `.run(max_generations: int | None = None)`; state on disk under `cfg.run_dir`: `ckpts/` (orbax CheckpointManager: params, opt_state, best_params, meta), `best/` (orbax StandardCheckpointer: params only — what Engine loads), `metrics.jsonl` (one JSON object per generation).
  - CLI: `python -m chesszero.train <config.yaml> [--generations N]` — N overrides `cfg.max_generations` as an absolute generation target (so a resumed run with `--generations 14` continues from wherever it left off up to gen 14).
  - Meta ints stored as numpy arrays; restore must produce plain ints. RNG streams derive from `cfg.seed` + generation counter — resume is deterministic-ish; the replay buffer intentionally restarts empty on resume (spec §6: metadata only), gated by `min_buffer` refill.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trainer.py
import json
import pathlib

import pytest

from chesszero.config import Config
from chesszero.train import Trainer


@pytest.fixture()
def cfg(tmp_path):
    c = Config.from_yaml("configs/tiny.yaml")
    c.run_dir = str(tmp_path / "run")
    c.checkpoint_every_min = 0.0        # checkpoint every generation
    c.gate_every_generations = 2
    return c


@pytest.mark.slow
def test_train_checkpoint_resume(cfg):
    t = Trainer(cfg)
    t.run(max_generations=3)
    run = pathlib.Path(cfg.run_dir)
    assert (run / "best").exists()
    lines = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(lines) == 3 and lines[-1]["gen"] == 2
    assert "policy_loss" in lines[-1] or lines[-1]["buffer_size"] < cfg.train.min_buffer

    # resume: a NEW Trainer picks up at gen 3, runs to 5
    t2 = Trainer(cfg)
    assert t2.start_generation == 3
    t2.run(max_generations=5)
    lines = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert lines[-1]["gen"] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trainer.py -v -m slow`
Expected: FAIL with `ImportError: cannot import name 'Trainer'`

- [ ] **Step 3: Append Trainer + CLI to train.py**

```python
# append to chesszero/train.py

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from chesszero.buffer import ReplayBuffer
from chesszero.config import Config
from chesszero.evaluate import play_match
from chesszero.net import ChessNet
from chesszero.selfplay import SelfplayWorker, pack_examples


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.net = ChessNet(cfg.net)
        key = jax.random.PRNGKey(cfg.seed)
        dummy = jnp.zeros((1, 8, 8, 119), jnp.float32)
        self.params = self.net.init(key, dummy)
        self.tx = make_optimizer(cfg.train)
        self.opt_state = self.tx.init(self.params)
        self.best_params = self.params
        self.global_step = 0
        self.start_generation = 0
        self.gate_failures = 0
        self._last_saved_gen = -1

        self.mgr = ocp.CheckpointManager(
            (self.run_dir / "ckpts").absolute(),
            options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True))
        self._maybe_restore()
        self._save_best()  # ensure best/ exists from step 0
        self.worker = SelfplayWorker(
            self.net, cfg, seed=cfg.seed + 1000 + self.start_generation)
        self.buffer = ReplayBuffer(cfg.train.buffer_capacity,
                                   seed=cfg.seed + 2000 + self.start_generation)

    # -- persistence ---------------------------------------------------------
    def _payload(self):
        return {"params": self.params, "opt_state": self.opt_state,
                "best_params": self.best_params,
                "meta": {"generation": np.asarray(self.start_generation),
                         "global_step": np.asarray(self.global_step)}}

    def _maybe_restore(self):
        step = self.mgr.latest_step()
        if step is None:
            return
        restored = self.mgr.restore(
            step, args=ocp.args.StandardRestore(self._payload()))
        self.params = restored["params"]
        self.opt_state = restored["opt_state"]
        self.best_params = restored["best_params"]
        self.start_generation = int(restored["meta"]["generation"]) + 1
        self.global_step = int(restored["meta"]["global_step"])
        self._last_saved_gen = int(restored["meta"]["generation"])

    def _save(self, generation: int):
        if generation <= self._last_saved_gen:
            return  # orbax steps must be unique and monotonic
        self.start_generation = generation
        self.mgr.save(generation, args=ocp.args.StandardSave(self._payload()))
        self._last_saved_gen = generation

    def _save_best(self):
        best_dir = (self.run_dir / "best").absolute()
        ocp.StandardCheckpointer().save(
            best_dir, args=ocp.args.StandardSave(self.best_params), force=True)

    # -- main loop -----------------------------------------------------------
    def run(self, max_generations: int | None = None):
        cfg = self.cfg
        target = max_generations if max_generations is not None \
            else cfg.max_generations
        last_ckpt = time.time()
        for gen in range(self.start_generation, target):
            t0 = time.time()
            allow_resign = self.global_step >= cfg.train.resign_min_train_steps
            examples, stats = self.worker.run_generation(self.params,
                                                         allow_resign)
            if examples:
                self.buffer.add(*pack_examples(examples))
            metrics = {}
            if self.buffer.size >= cfg.train.min_buffer:
                if not hasattr(self, "_train_step"):
                    self._train_step = make_train_step(self.net, self.tx,
                                                       cfg.train)
                for _ in range(cfg.train.steps_per_generation):
                    batch = {k: jnp.asarray(v) for k, v in
                             self.buffer.sample(cfg.train.batch_size).items()}
                    self.params, self.opt_state, m = self._train_step(
                        self.params, self.opt_state, batch)
                    self.global_step += 1
                metrics = {k: float(v) for k, v in m.items()}

            row = {"ts": time.time(), "gen": gen,
                   "global_step": self.global_step,
                   "buffer_size": self.buffer.size,
                   "games": stats.games, "resigns": stats.resigns,
                   "draws": stats.draws,
                   "avg_len": (stats.sum_game_len / stats.games
                               if stats.games else None),
                   "holdout_fp": stats.holdout_false_positives,
                   "holdout_n": stats.holdout_resign_games,
                   "gen_seconds": time.time() - t0, **metrics}

            if (gen + 1) % cfg.gate_every_generations == 0 \
                    and self.global_step > 0:
                score = play_match(self.net, self.params, self.best_params,
                                   cfg, seed=cfg.seed + gen)
                row["gate_score"] = score
                if score >= cfg.gating.promote_threshold:
                    self.best_params = self.params
                    self.gate_failures = 0
                    self._save_best()
                else:
                    self.gate_failures += 1
                    if self.gate_failures >= 3:
                        row["alarm"] = "3 consecutive gate failures"

            with (self.run_dir / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")

            if time.time() - last_ckpt >= cfg.checkpoint_every_min * 60:
                self._save(gen)
                last_ckpt = time.time()
        self._save(target - 1)  # no-op if already saved or run was a no-op resume
        self.mgr.wait_until_finished()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--generations", type=int, default=None)
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    Trainer(cfg).run(args.generations)


if __name__ == "__main__":
    main()
```

Careful points the implementer must preserve:
- `_maybe_restore` builds the abstract tree from `self._payload()` — params/opt_state templates must be constructed *before* restore (they are: `__init__` inits them first).
- `_save(gen)` sets `start_generation = gen` so the payload's meta is current; restore adds +1.
- Orbax steps must be unique and monotonically increasing: `_save` keys checkpoints by generation and the `_last_saved_gen` guard makes duplicate or regressive saves (e.g. the trailing `_save` after a no-op resume, or re-running with a smaller `--generations`) a silent no-op. `_maybe_restore` must set `_last_saved_gen` to the restored generation.
- Buffer restarts empty on resume by design (spec §6); `min_buffer` gates training until refilled.

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 900 pytest tests/test_trainer.py -v -m slow`
Expected: 1 passed (tiny config; a few minutes on the 4090)

- [ ] **Step 5: Commit**

```bash
git add chesszero/train.py tests/test_trainer.py
git commit -m "feat: trainer main loop with orbax checkpointing, resume, gating"
```

---

### Task 10: engine.py — checkpoint player

**Files:**
- Create: `chesszero/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `bridge` (Task 2), `ChessNet` (Task 3), `net_forward`/`make_recurrent_fn` (Task 5), `best/` checkpoint format (Task 9).
- Produces (Plan 2's lichess bot wraps exactly this):
  - `Engine(best_dir: str | Path, cfg: Config)`.
  - `.reset(fen: str | None = None)` — new game (startpos default).
  - `.push_uci(uci: str)` — advance internal python-chess board + pgx state together.
  - `.best_move(movetime_s: float) -> chess.Move` — Gumbel search (`gumbel_scale=0`), sims tier chosen from measured search speed; never exceeds `movetime_s` by more than ~2× on a warm engine.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine.py
import chess
import jax
import jax.numpy as jnp
import pytest

from chesszero.config import Config, NetConfig
from chesszero.engine import Engine
from chesszero.net import ChessNet


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    import orbax.checkpoint as ocp
    cfg = Config(net=NetConfig(channels=16, blocks=1, precision="fp32"))
    net = ChessNet(cfg.net)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 8, 8, 119)))
    best = tmp_path_factory.mktemp("ck") / "best"
    ocp.StandardCheckpointer().save(
        best.absolute(), args=ocp.args.StandardSave(params), force=True)
    return Engine(best, cfg)


def test_plays_legal_from_startpos(engine):
    engine.reset()
    mv = engine.best_move(0.2)
    assert mv in chess.Board().legal_moves


def test_follows_game_and_plays_black(engine):
    engine.reset()
    for uci in ["e2e4", "e7e5", "g1f3"]:
        engine.push_uci(uci)
    mv = engine.best_move(0.2)          # black to move
    board = chess.Board()
    for uci in ["e2e4", "e7e5", "g1f3"]:
        board.push_uci(uci)
    assert mv in board.legal_moves


def test_many_random_plies_stay_legal(engine):
    engine.reset()
    board = chess.Board()
    for _ in range(30):
        if board.is_game_over():
            break
        mv = engine.best_move(0.05)
        assert mv in board.legal_moves
        board.push(mv)
        engine.push_uci(mv.uci())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: chesszero.engine`

- [ ] **Step 3: Write the implementation**

```python
# chesszero/engine.py
"""Inference-time player: loads a best/ checkpoint, batch-1 Gumbel search.

Used by tests, the smoke script, and (Plan 2) the lichess bot. When run as
its own process next to training, launch with
XLA_PYTHON_CLIENT_MEM_FRACTION=0.15.
"""
from __future__ import annotations

import time
from pathlib import Path

import chess
import jax
import jax.numpy as jnp
import mctx
import orbax.checkpoint as ocp

from chesszero import bridge
from chesszero.config import Config
from chesszero.net import ChessNet
from chesszero.selfplay import ENV, init_batch, make_recurrent_fn, net_forward

_SIM_TIERS = (16, 32, 64, 128, 256)


class Engine:
    def __init__(self, best_dir: str | Path, cfg: Config):
        self.cfg = cfg
        self.net = ChessNet(cfg.net)
        template = self.net.init(jax.random.PRNGKey(0),
                                 jnp.zeros((1, 8, 8, 119), jnp.float32))
        self.params = ocp.StandardCheckpointer().restore(
            Path(best_dir).absolute(), args=ocp.args.StandardRestore(template))
        self._search_fns: dict[int, callable] = {}
        self.key = jax.random.PRNGKey(0)
        self.sims_per_s: float | None = None
        self.reset()

    def _get_search(self, sims: int):
        if sims not in self._search_fns:
            recurrent_fn = make_recurrent_fn(self.net)

            @jax.jit
            def search(params, state, key):
                logits, value = net_forward(self.net, params,
                                            state.observation,
                                            state.legal_action_mask)
                root = mctx.RootFnOutput(prior_logits=logits, value=value,
                                         embedding=state)
                out = mctx.gumbel_muzero_policy(
                    params=params, rng_key=key, root=root,
                    recurrent_fn=recurrent_fn, num_simulations=sims,
                    invalid_actions=~state.legal_action_mask,
                    max_num_considered_actions=(
                        self.cfg.selfplay.max_considered_actions),
                    gumbel_scale=0.0)
                return out.action

            self._search_fns[sims] = search
        return self._search_fns[sims]

    def reset(self, fen: str | None = None):
        self.board = chess.Board(fen) if fen else chess.Board()
        if fen:
            self.state = jax.tree.map(lambda x: x[None],
                                      bridge.state_from_fen(fen))
        else:
            self.state = init_batch(1, seed=0)

    def push_uci(self, uci: str):
        move = chess.Move.from_uci(uci)
        action = bridge.move_to_action(move, self.board.turn)
        self.state = jax.jit(jax.vmap(ENV.step))(
            self.state, jnp.asarray([action]))
        self.board.push(move)

    def _pick_sims(self, movetime_s: float) -> int:
        if self.sims_per_s is None:
            fn = self._get_search(_SIM_TIERS[0])
            fn(self.params, self.state, self.key)          # compile
            t0 = time.time()
            jax.block_until_ready(fn(self.params, self.state, self.key))
            self.sims_per_s = _SIM_TIERS[0] / max(time.time() - t0, 1e-4)
        best = _SIM_TIERS[0]
        for tier in _SIM_TIERS:
            if tier / self.sims_per_s <= movetime_s * 0.8:
                best = tier
        return best

    def best_move(self, movetime_s: float) -> chess.Move:
        sims = self._pick_sims(movetime_s)
        self.key, sub = jax.random.split(self.key)
        action = int(self._get_search(sims)(self.params, self.state, sub)[0])
        return bridge.action_to_move(action, self.board)
```

Note: `reset(fen=...)` loses history planes (documented bridge limitation) — fine for tests/analysis; game play always goes through `reset()` + `push_uci` so history is exact.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_engine.py -v`
Expected: 3 passed (module-scoped engine fixture keeps compiles to one)

- [ ] **Step 5: Commit**

```bash
git add chesszero/engine.py tests/test_engine.py
git commit -m "feat: checkpoint-loading engine with time-scaled Gumbel search"
```

---

### Task 11: Phase 0 gate — smoke script + kill-and-resume drill

**Files:**
- Create: `scripts/smoke.sh`
- Create: `scripts/engine_move.py`

**Interfaces:**
- Consumes: the full pipeline.
- Produces: the Phase 0 exit criterion, executable as one command.

- [ ] **Step 1: Write the smoke script**

```bash
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
```

```python
# scripts/engine_move.py
"""Load best/ checkpoint, play one engine move from startpos, print it."""
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")

from chesszero.config import Config
from chesszero.engine import Engine

best_dir, config_path = sys.argv[1], sys.argv[2]
cfg = Config.from_yaml(config_path)
engine = Engine(best_dir, cfg)
engine.reset()
move = engine.best_move(0.5)
assert move is not None
print(f"engine move from startpos: {move.uci()}")
```

- [ ] **Step 2: Make executable and run**

Run: `chmod +x scripts/smoke.sh && ./scripts/smoke.sh`
Expected: ends with `SMOKE OK — Phase 0 gate passed`. If the resume run starts from generation 0 (metrics.jsonl gen numbers restart), the Trainer restore path is broken — fix before proceeding.

- [ ] **Step 3: Run the full test suite once**

Run: `pytest -v && pytest tests/test_trainer.py -v -m slow`
Expected: all passed

- [ ] **Step 4: Commit**

```bash
git add scripts/
git commit -m "feat: Phase 0 smoke gate — tiny e2e run, engine move, kill-resume drill"
```

---

### Task 12: Phase 1 tooling — bootstrap + throughput calibration

**Files:**
- Create: `scripts/bootstrap.sh`
- Create: `scripts/calibrate.py`

**Interfaces:**
- Consumes: `SelfplayWorker`, `Config`.
- Produces: `scripts/calibrate.py` prints a throughput table (config → moves/s, est. positions/hour, est. games/hour) used to lock `configs/v1.yaml`. Locking v1 values is a human decision at the end of Phase 1, not part of this task.

- [ ] **Step 1: Write bootstrap script (pod re-creation insurance)**

```bash
#!/usr/bin/env bash
# scripts/bootstrap.sh — recreate the runtime env on a fresh pod.
set -euo pipefail
pip install -q -U "jax[cuda12]==0.10.2" pgx==2.6.0 mctx==0.0.71 flax==0.12.7 \
  optax==0.2.8 orbax-checkpoint==0.12.1 chess==1.11.2 pyyaml pytest
pip install -q -e .
python -c "import jax; assert jax.devices()[0].platform == 'gpu', 'NO GPU'"
echo "bootstrap OK: $(python -c 'import jax; print(jax.devices())')"
```

- [ ] **Step 2: Write the calibration script**

```python
# scripts/calibrate.py
"""Phase 1: throughput matrix. Usage: python scripts/calibrate.py [--quick]

Measures selfplay moves/s for candidate configs; prints positions/hour and
games/hour (assuming ~135 ply/game pre-resignation). Run one config at a
time on an otherwise idle GPU.
"""
import argparse
import time

import jax
import jax.numpy as jnp

from chesszero.config import Config, NetConfig, SelfplayConfig
from chesszero.net import ChessNet
from chesszero.selfplay import SelfplayWorker

CANDIDATES = [
    ("6x128 b1024 s32/8", NetConfig(channels=128, blocks=6),
     dict(num_games=1024, sims_full=32, sims_cheap=8)),
    ("6x128 b2048 s32/8", NetConfig(channels=128, blocks=6),
     dict(num_games=2048, sims_full=32, sims_cheap=8)),
    ("8x192 b1024 s32/8", NetConfig(channels=192, blocks=8),
     dict(num_games=1024, sims_full=32, sims_cheap=8)),
    ("6x128 b1024 s48/12", NetConfig(channels=128, blocks=6),
     dict(num_games=1024, sims_full=48, sims_cheap=12)),
]

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="1 gen instead of 3")
args = ap.parse_args()

print(f"{'config':24} {'moves/s':>10} {'pos/hour':>12} {'games/hour':>11}")
for name, net_cfg, sp_kw in CANDIDATES:
    cfg = Config(net=net_cfg,
                 selfplay=SelfplayConfig(steps_per_generation=16, **sp_kw))
    net = ChessNet(cfg.net)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 8, 8, 119)))
    worker = SelfplayWorker(net, cfg, seed=0)
    worker.run_generation(params, allow_resign=False)      # compile + warmup
    t0 = time.time()
    reps = 1 if args.quick else 3
    for _ in range(reps):
        worker.run_generation(params, allow_resign=False)
    dt = time.time() - t0
    moves = cfg.selfplay.num_games * cfg.selfplay.steps_per_generation * reps
    mps = moves / dt
    print(f"{name:24} {mps:>10.0f} {mps * 3600:>12.0f} {mps * 3600 / 135:>11.0f}")
```

- [ ] **Step 3: Run calibration in quick mode to verify it works**

Run: `chmod +x scripts/bootstrap.sh && timeout 1800 python scripts/calibrate.py --quick`
Expected: a 4-row table with nonzero moves/s for every row. (Full 3-rep run happens as the Phase 1 activity itself, with the results reviewed before locking v1.yaml.)

- [ ] **Step 4: Commit**

```bash
git add scripts/bootstrap.sh scripts/calibrate.py
git commit -m "feat: pod bootstrap + Phase 1 throughput calibration harness"
```

---

## Verification (whole plan)

1. `pytest -v` — all fast tests green.
2. `pytest -m slow -v` — trainer end-to-end green.
3. `./scripts/smoke.sh` — Phase 0 gate: tiny run, engine move, kill-resume all pass.
4. `python scripts/calibrate.py` — Phase 1 numbers in hand → human locks `configs/v1.yaml` → start the long run: `nohup python -m chesszero.train configs/v1.yaml > runs/v1.log 2>&1 &`.

## Explicitly deferred to Plan 2 (written while the long run trains)

- Stockfish ladder (`evaluate.py` ladder half, Stockfish install, Elo fit, `ladder.jsonl` + report script) — spec §7.
- lichess-bot integration (`bot/`) and BOT account setup — spec §8.
- Checkpoint sync to Mac/HF and final report — spec §10 Phase 3.
