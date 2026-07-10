# ChessZero v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the six v2 changes from `docs/superpowers/specs/2026-07-10-chesszero-v2-design.md` — stronger search config, random opening plies, cosine LR, external anchor, self-arming resign, mctx max_depth + host perf — without disturbing the live v1 run.

**Architecture:** All changes extend the existing modules (`config.py`, `train.py`, `selfplay.py`, `evaluate.py`, `engine.py`) plus two new small modules: `chesszero/anchor.py` (subprocess anchor matches) and `chesszero/resign.py` (arm/disarm state machine). Every new config field defaults to v1 behavior.

**Tech Stack:** JAX + pgx + mctx + optax + orbax (installed), pytest.

## Global Constraints

- **The v1 trainer is live on this pod.** Never touch `runs/v1/`, never kill or launch training processes (Louis does that), never use `pkill -f`. Code edits are safe (the running process has already loaded its modules).
- **All new config fields default to exact v1 semantics**: `opening_plies_max: 0` (no random plies), `search_max_depth: 0` (unlimited), `lr_decay_steps: 0` (constant LR after warmup), `anchor_every_generations: 0` (no anchor).
- **Run all tests CPU-only**: prefix every pytest command with `JAX_PLATFORMS=cpu` (the GPU belongs to v1).
- Spec values, verbatim: sims 64/16, openings k~U{0..8}, cosine 2e-3 → floor `lr*0.1` over 400 000 steps, anchor every 60 gens vs negamax2+negamax3 `--games 6 --movetime 0.2` with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.15` and 15-min timeout, resign 0.98 / 3 consecutive, arm < 0.05 / disarm > 0.08 over trailing 2000 holdout triggers, hard minimum `resign_min_train_steps: 20000`.
- Commit after every task; commit messages in the style shown per task.

---

### Task 1: Config fields

**Files:**
- Modify: `chesszero/config.py` (dataclasses `SelfplayConfig`, `TrainConfig`, `Config`)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `SelfplayConfig.opening_plies_max: int`, `SelfplayConfig.search_max_depth: int`, `TrainConfig.lr_decay_steps: int`, `TrainConfig.lr_floor_frac: float`, `TrainConfig.resign_arm_fp: float`, `TrainConfig.resign_disarm_fp: float`, `TrainConfig.resign_fp_window: int`, `Config.anchor_every_generations: int` — every later task reads these exact names.

- [ ] **Step 1: Write the failing test** (append to `tests/test_config.py`)

```python
def test_v2_fields_defaults_and_yaml_override():
    from chesszero.config import Config
    c = Config.from_dict({})
    # defaults must preserve v1 semantics exactly
    assert c.selfplay.opening_plies_max == 0
    assert c.selfplay.search_max_depth == 0
    assert c.train.lr_decay_steps == 0
    assert c.train.lr_floor_frac == 0.1
    assert c.train.resign_arm_fp == 0.05
    assert c.train.resign_disarm_fp == 0.08
    assert c.train.resign_fp_window == 2000
    assert c.anchor_every_generations == 0

    c = Config.from_dict({
        "selfplay": {"opening_plies_max": 8, "search_max_depth": 16},
        "train": {"lr_decay_steps": 400_000},
        "anchor_every_generations": 60})
    assert c.selfplay.opening_plies_max == 8
    assert c.selfplay.search_max_depth == 16
    assert c.train.lr_decay_steps == 400_000
    assert c.anchor_every_generations == 60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/test_config.py::test_v2_fields_defaults_and_yaml_override -v`
Expected: FAIL with `TypeError` / `AttributeError` (unknown fields)

- [ ] **Step 3: Add the fields**

In `SelfplayConfig`, after `resign_holdout_frac`:

```python
    opening_plies_max: int = 0        # k ~ U{0..max} random plies at game reset (0 = off)
    search_max_depth: int = 0         # mctx max_depth for all searches (0 = unlimited)
```

In `TrainConfig`, after `resign_min_train_steps`:

```python
    lr_decay_steps: int = 0           # cosine-decay horizon in steps (0 = constant after warmup)
    lr_floor_frac: float = 0.1        # cosine floor = lr * lr_floor_frac
    resign_arm_fp: float = 0.05       # auto-arm resignation below this windowed holdout FP
    resign_disarm_fp: float = 0.08    # auto-disarm above this (hysteresis)
    resign_fp_window: int = 2000      # trailing holdout triggers in the FP window
```

In `Config`, after `gate_every_generations`:

```python
    anchor_every_generations: int = 0  # spar best vs negamax2/3 every N gens (0 = off)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/test_config.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add chesszero/config.py tests/test_config.py
git commit -m "feat(v2): config fields for openings, max_depth, cosine lr, anchor, resign governor"
```

---

### Task 2: Cosine LR schedule

**Files:**
- Modify: `chesszero/train.py:17-21` (`make_lr_schedule`)
- Test: `tests/test_trainer.py`

**Interfaces:**
- Consumes: `TrainConfig.lr_decay_steps`, `TrainConfig.lr_floor_frac` (Task 1)
- Produces: `make_lr_schedule(cfg: TrainConfig) -> optax.Schedule` — same signature, new behavior when `lr_decay_steps > 0`. Trainer and `make_optimizer` already call it; no call-site changes.

- [ ] **Step 1: Write the failing test** (append to `tests/test_trainer.py`)

```python
def test_make_lr_schedule_cosine_decay_to_floor():
    from chesszero.config import TrainConfig
    from chesszero.train import make_lr_schedule
    cfg = TrainConfig(lr=2e-3, warmup_steps=100,
                      lr_decay_steps=1000, lr_floor_frac=0.1)
    s = make_lr_schedule(cfg)
    assert float(s(0)) == 0.0                                   # warmup start
    assert float(s(100)) == pytest.approx(2e-3)                 # warmup end
    mid = float(s(100 + 500))                                   # cosine midpoint
    assert mid == pytest.approx((2e-3 + 2e-4) / 2, rel=1e-3)
    assert float(s(100 + 1000)) == pytest.approx(2e-4, rel=1e-3)  # floor
    assert float(s(1_000_000)) == pytest.approx(2e-4, rel=1e-3)   # stays at floor
```

The existing `test_make_lr_schedule_warmup_then_constant` guards the
`lr_decay_steps=0` fallback — do not modify it.

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/test_trainer.py::test_make_lr_schedule_cosine_decay_to_floor -v`
Expected: FAIL (schedule is constant after warmup, midpoint assertion breaks)

- [ ] **Step 3: Implement**

Replace `make_lr_schedule` in `chesszero/train.py`:

```python
def make_lr_schedule(cfg: TrainConfig) -> optax.Schedule:
    if cfg.lr_decay_steps > 0:
        main = optax.cosine_decay_schedule(
            cfg.lr, cfg.lr_decay_steps, alpha=cfg.lr_floor_frac)
    else:
        main = optax.constant_schedule(cfg.lr)
    return optax.join_schedules(
        [optax.linear_schedule(0.0, cfg.lr, cfg.warmup_steps), main],
        boundaries=[cfg.warmup_steps])
```

(`optax.join_schedules` passes `step - warmup_steps` to `main`, so the cosine
starts at warmup end — that is what the midpoint assertion checks.)

- [ ] **Step 4: Run tests**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/test_trainer.py -v -k lr_schedule`
Expected: both schedule tests PASS

- [ ] **Step 5: Commit**

```bash
git add chesszero/train.py tests/test_trainer.py
git commit -m "feat(v2): warmup + cosine-to-floor lr schedule (lr_decay_steps)"
```

---

### Task 3: Random opening plies in selfplay

**Files:**
- Modify: `chesszero/selfplay.py` (`make_play_step`, `SelfplayWorker.__init__`)
- Test: `tests/test_selfplay_device.py`

**Interfaces:**
- Consumes: `SelfplayConfig.opening_plies_max` (Task 1)
- Produces: `_random_opening(key, state, max_plies) -> state` (module-level, for tests); `make_play_step(net, num_simulations, max_considered, gumbel_scale, opening_plies_max=0)` — new trailing kwarg, existing callers unaffected.

- [ ] **Step 1: Write the failing test** (append to `tests/test_selfplay_device.py`)

```python
def test_random_opening_plays_k_plies():
    import jax
    import numpy as np
    from chesszero.selfplay import init_batch, _random_opening
    state = init_batch(16, seed=0)
    out = _random_opening(jax.random.PRNGKey(1), state, max_plies=8)
    counts = np.asarray(out._step_count).reshape(-1)
    assert counts.min() >= 0 and counts.max() <= 8
    assert len(set(counts.tolist())) > 2          # k actually varies
    assert not np.asarray(out.terminated).any()   # guard kept games alive
    # k=0 must be the identity
    same = _random_opening(jax.random.PRNGKey(1), state, max_plies=0)
    assert np.asarray(same._step_count).max() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/test_selfplay_device.py::test_random_opening_plays_k_plies -v`
Expected: FAIL with `ImportError: cannot import name '_random_opening'`

- [ ] **Step 3: Implement**

Add to `chesszero/selfplay.py` after `_reset_where` (line 54):

```python
def _random_opening(key, state, max_plies: int):
    """Play k ~ U{0..max_plies} uniformly-random legal moves on each game in
    the batch (v2 opening diversity). Terminated games and games past their
    own k are left untouched. max_plies=0 is the identity."""
    if max_plies == 0:
        return state
    n = state.current_player.shape[0]
    k_count, k_moves = jax.random.split(key)
    k = jax.random.randint(k_count, (n,), 0, max_plies + 1)

    def body(i, carry):
        state, key = carry
        key, sub = jax.random.split(key)
        logits = jnp.where(state.legal_action_mask, 0.0, -jnp.inf)
        action = jax.random.categorical(sub, logits, axis=-1)
        stepped = jax.vmap(ENV.step)(state, action)
        active = (i < k) & ~(state.terminated | state.truncated)
        state = jax.tree.map(
            lambda new, old: jnp.where(
                active.reshape((-1,) + (1,) * (new.ndim - 1)), new, old),
            stepped, state)
        return state, key

    state, _ = jax.lax.fori_loop(0, max_plies, body, (state, k_moves))
    return state
```

In `make_play_step`, change the signature and the reset block:

```python
def make_play_step(net, num_simulations: int, max_considered: int,
                   gumbel_scale: float, opening_plies_max: int = 0):
```

and inside `play_step`, replace the two `fresh` lines:

```python
        k_init, k_open, k_search = jax.random.split(key, 3)
        n = state.current_player.shape[0]
        fresh = jax.vmap(ENV.init)(jax.random.split(k_init, n))
        fresh = _random_opening(k_open, fresh, opening_plies_max)
        state = _reset_where(reset_mask, fresh, state)
```

In `SelfplayWorker.__init__`, pass the knob to both step builders:

```python
        self.step_full = make_play_step(net, sp.sims_full,
                                        sp.max_considered_actions, 1.0,
                                        sp.opening_plies_max)
        self.step_cheap = make_play_step(net, sp.sims_cheap,
                                         sp.max_considered_actions, 1.0,
                                         sp.opening_plies_max)
```

Note: the worker's very first batch (`init_batch` in `__init__`) starts from
the standard position by design — every subsequent reset gets random openings.
Opening plies emit no training records: they happen inside the reset path,
before any search step, so `record` never sees them.

- [ ] **Step 4: Run tests**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/test_selfplay_device.py tests/test_selfplay_worker.py -v`
Expected: all PASS (worker tests confirm the default-0 path is unchanged)

- [ ] **Step 5: Commit**

```bash
git add chesszero/selfplay.py tests/test_selfplay_device.py
git commit -m "feat(v2): k~U{0..8} random opening plies on selfplay game reset"
```

---

### Task 4: mctx max_depth plumbing

**Files:**
- Modify: `chesszero/selfplay.py` (`make_play_step` + `SelfplayWorker.__init__`), `chesszero/evaluate.py` (`_make_versus_step`, `_get_versus_step`, `play_match`), `chesszero/engine.py` (search call)
- Test: `tests/test_selfplay_device.py`

**Interfaces:**
- Consumes: `SelfplayConfig.search_max_depth` (Task 1)
- Produces: `make_play_step(..., opening_plies_max=0, search_max_depth=0)`; `_make_versus_step(net, sims, max_considered, search_max_depth=0)`; engine reads `cfg.selfplay.search_max_depth`. Convention everywhere: `0` → pass `max_depth=None` to mctx.

- [ ] **Step 1: Write the failing test** (append to `tests/test_selfplay_device.py`)

```python
def test_play_step_accepts_search_max_depth():
    import inspect
    from chesszero.selfplay import make_play_step
    from chesszero.evaluate import _make_versus_step
    assert "search_max_depth" in inspect.signature(make_play_step).parameters
    assert "search_max_depth" in inspect.signature(_make_versus_step).parameters
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/test_selfplay_device.py::test_play_step_accepts_search_max_depth -v`
Expected: FAIL (missing parameter)

- [ ] **Step 3: Implement**

`chesszero/selfplay.py` — `make_play_step` signature and the mctx call:

```python
def make_play_step(net, num_simulations: int, max_considered: int,
                   gumbel_scale: float, opening_plies_max: int = 0,
                   search_max_depth: int = 0):
    recurrent_fn = make_recurrent_fn(net)
    max_depth = search_max_depth or None
```

and in the `gumbel_muzero_policy` call add `max_depth=max_depth,` after
`num_simulations=num_simulations,`.

`SelfplayWorker.__init__` — pass `sp.search_max_depth` as the new last arg to
both `make_play_step` calls (after `sp.opening_plies_max`).

`chesszero/evaluate.py` — `_make_versus_step(net, sims, max_considered, search_max_depth=0)`
with `max_depth = search_max_depth or None` at the top and `max_depth=max_depth,`
in its `gumbel_muzero_policy` call. Extend the cache key and getter:

```python
def _get_versus_step(net, sims, max_considered, search_max_depth=0):
    key = (id(net), sims, max_considered, search_max_depth)
    if key not in _VERSUS_CACHE:
        _VERSUS_CACHE[key] = _make_versus_step(net, sims, max_considered,
                                               search_max_depth)
    return _VERSUS_CACHE[key]
```

and in `play_match`:

```python
    step = _get_versus_step(net, cfg.selfplay.sims_full,
                            cfg.selfplay.max_considered_actions,
                            cfg.selfplay.search_max_depth)
```

`chesszero/engine.py` — the `gumbel_muzero_policy` call at line 53 sits in a
closure that already reads `self.cfg` (see `max_num_considered_actions` two
lines below). Add one line to the call, after `num_simulations=sims,`:

```python
                    max_depth=self.cfg.selfplay.search_max_depth or None,
```

- [ ] **Step 4: Run tests**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/test_selfplay_device.py tests/test_gating.py tests/test_engine.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add chesszero/selfplay.py chesszero/evaluate.py chesszero/engine.py tests/test_selfplay_device.py
git commit -m "feat(v2): configurable mctx max_depth at all three search sites"
```

---

### Task 5: Host-transfer perf fixes

**Files:**
- Modify: `chesszero/selfplay.py` (`play_step` record, `SelfplayWorker._process`)
- Test: `tests/test_selfplay_worker.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `record["obs"]` is `float16` on device; `_process` materializes `action_weights` only on full-search steps. `Example` fields unchanged.

- [ ] **Step 1: Write the failing test** (append to `tests/test_selfplay_worker.py`)

```python
def test_record_obs_is_f16_on_device():
    import jax.numpy as jnp
    from chesszero.selfplay import make_play_step, init_batch
    from chesszero.net import ChessNet
    from chesszero.config import NetConfig
    import jax
    net = ChessNet(NetConfig(channels=8, blocks=1))
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 8, 8, 119)))
    step = make_play_step(net, 2, 4, 1.0)
    state = init_batch(2, 0)
    import numpy as np
    _, record = step(params, state, jnp.zeros(2, bool), jax.random.PRNGKey(1))
    assert record["obs"].dtype == jnp.float16
```

(Reuse the existing tiny-net fixture in the file if one exists; the test
above is self-contained either way.)

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/test_selfplay_worker.py::test_record_obs_is_f16_on_device -v`
Expected: FAIL (dtype is float32)

- [ ] **Step 3: Implement**

In `play_step`'s record dict: `"obs": state.observation.astype(jnp.float16),`

In `_process`, replace the two array lines:

```python
        obs = np.asarray(record["obs"])          # already f16 from device
        weights = (np.asarray(record["action_weights"], np.float16)
                   if full else None)             # skip 38MB/step host copy on cheap steps
```

and the slot append line:

```python
            slot.weights.append(weights[i] if full else None)
```

- [ ] **Step 4: Run tests**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/test_selfplay_worker.py tests/test_selfplay_device.py -v`
Expected: all PASS (existing worker tests verify cheap-step policy is `None` and packing still works)

- [ ] **Step 5: Commit**

```bash
git add chesszero/selfplay.py tests/test_selfplay_worker.py
git commit -m "perf(v2): f16 obs on device, skip action_weights transfer on cheap steps"
```

---

### Task 6: Anchor module

**Files:**
- Create: `chesszero/anchor.py`
- Test: `tests/test_anchor.py` (create)

**Interfaces:**
- Consumes: `scripts/versus_stockfish.py` summary-line format: `[Negamax2] score 4.5/6 (75%)`
- Produces: `parse_anchor_output(text: str) -> dict[str, float]` (opponent-name-lowercase → score fraction); `AnchorRunner(best_dir, config_path, games=6, movetime=0.2, timeout_s=900, cmd=None)` with `.running: bool`, `.start() -> None`, `.poll() -> dict | None` (None = still running, `{}` = failed, non-empty dict = results). Task 7's Trainer integration uses exactly these.

- [ ] **Step 1: Write the failing test** (create `tests/test_anchor.py`)

```python
import sys
import time

from chesszero.anchor import AnchorRunner, parse_anchor_output


def test_parse_anchor_output():
    text = ("[Negamax2] game 6/6 as Black: 1-0 -> loss | 30 moves\n"
            "[Negamax2] score 4.5/6 (75%)\n"
            "[Negamax3] score 1/6 (17%)\n")
    assert parse_anchor_output(text) == {"negamax2": 0.75, "negamax3": 1 / 6}
    assert parse_anchor_output("garbage\n") == {}


def test_anchor_runner_roundtrip():
    fake = [sys.executable, "-c",
            "print('[Negamax2] score 3/6 (50%)'); print('[Negamax3] score 0/6 (0%)')"]
    r = AnchorRunner("unused", "unused", cmd=fake)
    r.start()
    assert r.running
    for _ in range(100):
        res = r.poll()
        if res is not None:
            break
        time.sleep(0.05)
    assert res == {"negamax2": 0.5, "negamax3": 0.0}
    assert not r.running


def test_anchor_runner_failure_returns_empty():
    r = AnchorRunner("unused", "unused",
                     cmd=[sys.executable, "-c", "import sys; sys.exit(3)"])
    r.start()
    for _ in range(100):
        res = r.poll()
        if res is not None:
            break
        time.sleep(0.05)
    assert res == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/test_anchor.py -v`
Expected: FAIL with `ModuleNotFoundError: chesszero.anchor`

- [ ] **Step 3: Implement** (create `chesszero/anchor.py`)

```python
# chesszero/anchor.py
"""External anchor: spar the current best vs scripted baselines, off-GPU-budget.

Spawns scripts/versus_stockfish.py as a subprocess in the 0.15 GPU-memory
slice (validated co-residency pattern) and parses its summary lines. Fully
non-blocking: the Trainer polls once per generation."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time

_SCORE_RE = re.compile(r"^\[([\w-]+)\] score ([\d.]+)/(\d+)", re.MULTILINE)


def parse_anchor_output(text: str) -> dict[str, float]:
    return {m.group(1).lower(): float(m.group(2)) / float(m.group(3))
            for m in _SCORE_RE.finditer(text)}


class AnchorRunner:
    def __init__(self, best_dir: str, config_path: str, games: int = 6,
                 movetime: float = 0.2, timeout_s: float = 900.0,
                 cmd: "list[str] | None" = None):
        self.cmd = cmd or [
            sys.executable, "scripts/versus_stockfish.py",
            "--best-dir", str(best_dir), "--config", str(config_path),
            "--vs", "negamax2", "negamax3",
            "--games", str(games), "--movetime", str(movetime)]
        self.timeout_s = timeout_s
        self._proc: "subprocess.Popen | None" = None
        self._started = 0.0

    @property
    def running(self) -> bool:
        return self._proc is not None

    def start(self):
        env = dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION="0.15")
        self._proc = subprocess.Popen(
            self.cmd, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._started = time.time()

    def poll(self) -> "dict[str, float] | None":
        """None while running; {} on failure/timeout; results dict on success."""
        if self._proc is None:
            return None
        if self._proc.poll() is None:
            if time.time() - self._started > self.timeout_s:
                self._proc.kill()
                self._proc.wait()
                self._proc = None
                return {}
            return None
        out = self._proc.stdout.read() if self._proc.stdout else ""
        ok = self._proc.returncode == 0
        self._proc = None
        return parse_anchor_output(out) if ok else {}
```

- [ ] **Step 4: Run tests**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/test_anchor.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add chesszero/anchor.py tests/test_anchor.py
git commit -m "feat(v2): non-blocking external anchor subprocess (negamax2/3 sparring)"
```

---

### Task 7: Resign governor

**Files:**
- Create: `chesszero/resign.py`
- Test: `tests/test_resign.py` (create)

**Interfaces:**
- Consumes: per-generation `stats.holdout_false_positives`, `stats.holdout_resign_games` (existing `GenStats` fields)
- Produces: `ResignGovernor(arm_fp, disarm_fp, window, min_train_steps)` with `.armed: bool` and `.update(fp: int, n: int, global_step: int) -> tuple[bool, float | None, str | None]` returning (armed-after-update, windowed FP or None if window unfilled, transition log message or None). Task 8 uses exactly this.

- [ ] **Step 1: Write the failing test** (create `tests/test_resign.py`)

```python
from chesszero.resign import ResignGovernor


def gov(**kw):
    d = dict(arm_fp=0.05, disarm_fp=0.08, window=100, min_train_steps=10)
    d.update(kw)
    return ResignGovernor(**d)


def test_stays_disarmed_until_window_filled():
    g = gov()
    armed, fp, msg = g.update(fp=0, n=50, global_step=1000)
    assert (armed, fp, msg) == (False, None, None)   # only 50 of 100 triggers seen


def test_arms_below_threshold_and_reports():
    g = gov()
    g.update(fp=1, n=60, global_step=1000)
    armed, fp, msg = g.update(fp=1, n=60, global_step=1000)
    assert armed and fp is not None and fp < 0.05
    assert msg is not None and "armed" in msg.lower()


def test_min_train_steps_blocks_arming():
    g = gov()
    g.update(fp=0, n=60, global_step=5)
    armed, fp, msg = g.update(fp=0, n=60, global_step=5)
    assert not armed and fp is not None and msg is None


def test_hysteresis_holds_then_disarms():
    g = gov()
    g.update(fp=1, n=60, global_step=1000)
    g.update(fp=1, n=60, global_step=1000)          # armed (~1.7% FP)
    armed, fp, _ = g.update(fp=8, n=60, global_step=1000)   # window FP ~6% — hold
    assert armed
    armed, fp, msg = g.update(fp=30, n=60, global_step=1000)  # >8% — disarm
    assert not armed and msg is not None and "disarmed" in msg.lower()


def test_window_trims_old_generations():
    g = gov(window=100)
    g.update(fp=50, n=60, global_step=1000)   # terrible old gen
    g.update(fp=0, n=60, global_step=1000)
    armed, fp, _ = g.update(fp=0, n=60, global_step=1000)
    # window keeps the minimal trailing suffix holding >= 100 triggers:
    # the 50/60 gen fell out -> recent FP is low -> armed
    assert fp < 0.05 and armed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/test_resign.py -v`
Expected: FAIL with `ModuleNotFoundError: chesszero.resign`

- [ ] **Step 3: Implement** (create `chesszero/resign.py`)

```python
# chesszero/resign.py
"""Self-arming resignation governor (v2 spec §5).

Tracks the windowed holdout false-positive rate over the trailing
`window` holdout triggers and arms/disarms resignation with hysteresis.
Not persisted across restarts: a fresh process starts disarmed and
re-arms once the window refills (~1-2h) — safe by default."""
from __future__ import annotations

from collections import deque


class ResignGovernor:
    def __init__(self, arm_fp: float, disarm_fp: float, window: int,
                 min_train_steps: int):
        self.arm_fp = arm_fp
        self.disarm_fp = disarm_fp
        self.window = window
        self.min_train_steps = min_train_steps
        self.armed = False
        self._hist: deque[tuple[int, int]] = deque()   # (fp, n) per generation
        self._fp = 0
        self._n = 0

    def update(self, fp: int, n: int,
               global_step: int) -> "tuple[bool, float | None, str | None]":
        self._hist.append((fp, n))
        self._fp += fp
        self._n += n
        # keep the minimal trailing suffix still holding >= window triggers
        while self._hist and self._n - self._hist[0][1] >= self.window:
            old_fp, old_n = self._hist.popleft()
            self._fp -= old_fp
            self._n -= old_n
        if self._n < self.window:
            return self.armed, None, None
        rate = self._fp / self._n
        msg = None
        if (not self.armed and rate < self.arm_fp
                and global_step >= self.min_train_steps):
            self.armed = True
            msg = (f"RESIGN armed: windowed holdout FP {rate:.1%}"
                   f" < {self.arm_fp:.0%} over last {self._n} triggers")
        elif self.armed and rate > self.disarm_fp:
            self.armed = False
            msg = (f"RESIGN disarmed: windowed holdout FP {rate:.1%}"
                   f" > {self.disarm_fp:.0%}")
        return self.armed, rate, msg
```

- [ ] **Step 4: Run tests**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/test_resign.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add chesszero/resign.py tests/test_resign.py
git commit -m "feat(v2): self-arming resign governor with FP-window hysteresis"
```

---

### Task 8: Trainer integration

**Files:**
- Modify: `chesszero/train.py` (`Trainer.__init__`, `Trainer.run`)
- Test: `tests/test_trainer.py`

**Interfaces:**
- Consumes: `ResignGovernor` (Task 7), `AnchorRunner` (Task 6), config fields (Task 1)
- Produces: metrics rows gain `resign_armed: bool`, `resign_fp_windowed: float|None`, and (on anchor completion) `anchor: dict`; log lines `RESIGN armed/disarmed ...` and `ANCHOR gen N: negamax2 X%, negamax3 Y%`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_trainer.py`; the file's existing `cfg` fixture uses `configs/tiny.yaml`)

```python
def test_trainer_uses_governor_and_spawns_anchor(cfg, monkeypatch):
    import sys
    import chesszero.train as T

    cfg.anchor_every_generations = 1
    spawned = []
    real_init = T.AnchorRunner.__init__

    def fake_init(self, best_dir, config_path, **kw):
        kw["cmd"] = [sys.executable, "-c",
                     "print('[Negamax2] score 3/6 (50%)')"]
        real_init(self, best_dir, config_path, **kw)
        spawned.append(best_dir)

    monkeypatch.setattr(T.AnchorRunner, "__init__", fake_init)
    trainer = T.Trainer(cfg)
    trainer.run(max_generations=3)
    assert spawned, "anchor subprocess was never spawned"
    rows = [__import__("json").loads(l) for l in
            (pathlib.Path(cfg.run_dir) / "metrics.jsonl").read_text().splitlines()]
    assert all("resign_armed" in r for r in rows)
    assert any("anchor" in r for r in rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/test_trainer.py::test_trainer_uses_governor_and_spawns_anchor -v`
Expected: FAIL with `AttributeError: ... no attribute 'AnchorRunner'`

- [ ] **Step 3: Implement**

Imports in `chesszero/train.py` (with the other chesszero imports, line ~68):

```python
from chesszero.anchor import AnchorRunner
from chesszero.resign import ResignGovernor
```

`Trainer.__init__`, after `self._fp_alarm_active = False`:

```python
        t = cfg.train
        self.governor = ResignGovernor(t.resign_arm_fp, t.resign_disarm_fp,
                                       t.resign_fp_window,
                                       t.resign_min_train_steps)
        self.anchor: "AnchorRunner | None" = None
        self._config_path = getattr(cfg, "_source_path", "configs/v1.yaml")
```

(`_source_path`: add one line to `Config.from_yaml` in `chesszero/config.py` —
after building the config object, `cfg._source_path = str(path)` then return.
`Config.from_dict` callers without a file keep the fallback.)

In `Trainer.run`, replace the `allow_resign = ...` line with:

```python
            allow_resign = self.governor.armed
```

After the two `self.holdout_*_total` updates, add:

```python
            armed, fp_windowed, transition = self.governor.update(
                stats.holdout_false_positives, stats.holdout_resign_games,
                self.global_step)
            if transition:
                self._log(transition)
```

In the `row = {...}` dict, add after `"holdout_n"`:

```python
                   "resign_armed": armed,
                   "resign_fp_windowed": fp_windowed,
```

After the gating block (immediately before `with (self.run_dir / "metrics.jsonl")...`):

```python
            if self.anchor is not None:
                res = self.anchor.poll()
                if res is not None:
                    if res:
                        row["anchor"] = res
                        self._log("ANCHOR gen %d: %s" % (gen, ", ".join(
                            f"{k} {v:.0%}" for k, v in sorted(res.items()))))
                    else:
                        self._log("ANCHOR match failed or timed out — skipped")
                    self.anchor = None
            if (cfg.anchor_every_generations
                    and (gen + 1) % cfg.anchor_every_generations == 0
                    and self.anchor is None and self.global_step > 0):
                self.anchor = AnchorRunner(
                    str((self.run_dir / "best").absolute()), self._config_path)
                self.anchor.start()
```

- [ ] **Step 4: Run tests**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/test_trainer.py tests/test_config.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add chesszero/train.py chesszero/config.py tests/test_trainer.py
git commit -m "feat(v2): trainer wires resign governor + anchor subprocess into the loop"
```

---

### Task 9: v2 configs, smoke test, handover

**Files:**
- Create: `configs/v2.yaml`, `configs/v2-smoke.yaml`
- Test: full suite + GPU smoke run

- [ ] **Step 1: Write `configs/v2.yaml`** (verbatim from the spec)

```yaml
net: {channels: 128, blocks: 6, se_ratio: 4, precision: bf16}
selfplay:
  num_games: 2048
  sims_full: 64          # v1: 32 — tactical-blindness fix
  sims_cheap: 16         # v1: 8
  full_search_prob: 0.25
  steps_per_generation: 16
  opening_plies_max: 8   # k ~ U{0..8} random plies per game
  search_max_depth: 16
  resign_threshold: 0.98         # v1: 0.95
  resign_consecutive_moves: 3    # v1: 2
train:
  batch_size: 1024
  steps_per_generation: 128
  buffer_capacity: 1000000
  min_buffer: 100000
  lr: 0.002
  lr_decay_steps: 400000   # cosine horizon (~3 days)
  lr_floor_frac: 0.1       # floor = 2e-4
  resign_min_train_steps: 20000
gating: {games: 120, promote_threshold: 0.53}
anchor_every_generations: 60
seed: 0
run_dir: runs/v2
checkpoint_every_min: 15.0
gate_every_generations: 30
```

- [ ] **Step 2: Write `configs/v2-smoke.yaml`** (tiny end-to-end exercise of every v2 feature)

```yaml
net: {channels: 16, blocks: 2, se_ratio: 4, precision: bf16}
selfplay:
  num_games: 64
  sims_full: 4
  sims_cheap: 2
  full_search_prob: 0.5
  steps_per_generation: 8
  opening_plies_max: 8
  search_max_depth: 16
train:
  batch_size: 64
  steps_per_generation: 4
  buffer_capacity: 20000
  min_buffer: 500
  lr: 0.002
  lr_decay_steps: 1000
  lr_floor_frac: 0.1
  resign_min_train_steps: 100
gating: {games: 8, promote_threshold: 0.53}
anchor_every_generations: 3
seed: 0
run_dir: runs/v2-smoke
checkpoint_every_min: 999.0
gate_every_generations: 5
```

- [ ] **Step 3: Full CPU test suite**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 4: GPU smoke alongside v1** (0.15 slice — same co-residency budget as the sparring script)

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 python -m chesszero.train configs/v2-smoke.yaml --generations 8 2>&1 | tail -30`
Expected: banner shows `sims 4/2`; gen lines appear; an `ANCHOR gen ...` line
(or `ANCHOR match failed`) appears within the 8 generations; `runs/v2-smoke/metrics.jsonl`
rows contain `resign_armed`. Clean exit.

- [ ] **Step 5: Clean up smoke artifacts and commit**

```bash
rm -rf runs/v2-smoke
git add configs/v2.yaml configs/v2-smoke.yaml
git commit -m "feat(v2): v2 run config + smoke config"
```

- [ ] **Step 6: Handover (Louis only — do NOT execute)**

Report readiness. Louis then: archives nothing extra (routine covers it), kills
his v1 trainer by PID from `ps aux | grep chesszero.train` (NOT nvidia-smi PIDs
— container PID namespace differs), and launches:

```bash
nohup python -m chesszero.train configs/v2.yaml > runs/v2.log 2>&1 &
```

Post-launch monitoring switches to `runs/v2.log` with the ANCHOR pattern added
to the monitor filter.
