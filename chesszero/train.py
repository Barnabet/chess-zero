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


def make_lr_schedule(cfg: TrainConfig) -> optax.Schedule:
    if cfg.lr_decay_steps > 0:
        main = optax.cosine_decay_schedule(
            cfg.lr, cfg.lr_decay_steps, alpha=cfg.lr_floor_frac)
    else:
        main = optax.constant_schedule(cfg.lr)
    return optax.join_schedules(
        [optax.linear_schedule(0.0, cfg.lr, cfg.warmup_steps), main],
        boundaries=[cfg.warmup_steps])


def make_optimizer(cfg: TrainConfig) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(make_lr_schedule(cfg), weight_decay=cfg.weight_decay))


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


import argparse
import json
import time
from pathlib import Path

import numpy as np
import orbax.checkpoint as ocp

from chesszero.buffer import ReplayBuffer
from chesszero.config import Config
from chesszero.evaluate import play_match
from chesszero.net import ChessNet
from chesszero.selfplay import SelfplayWorker, pack_examples


def resign_fp_alarm(fp_total: int, n_total: int, *,
                    threshold: float = 0.05, min_n: int = 20) -> "float | None":
    """Spec §5 alarm: fraction of held-out would-resign games that were NOT
    actually lost. Returns the rate when it exceeds threshold with enough
    samples, else None."""
    if n_total >= min_n and fp_total > threshold * n_total:
        return round(fp_total / n_total, 3)
    return None


def _human(n: float) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{int(n)}"


def format_gen_line(row: dict, min_buffer: int) -> str:
    """One terminal line per generation; the row dict is the metrics.jsonl row.
    Two shapes: buffer-fill (no gradient steps yet) and training."""
    if "loss" in row:
        buf = f"buf {_human(row['buffer_size'])}"
        train = (f"loss {row['loss']:.3f} (pi {row['policy_loss']:.3f}"
                 f" wdl {row['wdl_loss']:.3f} ml {row['ml_loss']:.3f})"
                 f" | lr {row['lr']:.1e}")
    else:
        buf = f"buf {_human(row['buffer_size'])}/{_human(min_buffer)} filling"
        train = "no training yet"
    if row["games"]:
        g = (f"{row['games']} games {100 * row['draws'] / row['games']:.0f}% draw"
             f" {100 * row['resigns'] / row['games']:.0f}% resign"
             f" len {row['avg_len']:.0f}")
    else:
        g = "0 games finished"
    return (f"gen {row['gen']:>5} | step {row['global_step']:>6} | {buf}"
            f" | {row['moves_per_s']:.0f} mv/s | {train} | {g}"
            f" | {row['gen_seconds']:.1f}s")


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
        # deep copy — train_step donates params buffers; an alias here would
        # be deleted by the first gradient step and crash the next save/gate
        self.best_params = jax.tree.map(jnp.copy, self.params)
        self.global_step = 0
        self.start_generation = 0
        self.gate_failures = 0
        self.holdout_fp_total = 0
        self.holdout_n_total = 0
        self._last_saved_gen = -1
        self._lr_schedule = make_lr_schedule(cfg.train)
        self._t0 = time.time()
        self._fp_alarm_active = False

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
                         "global_step": np.asarray(self.global_step),
                         "gate_failures": np.asarray(self.gate_failures),
                         "holdout_fp_total": np.asarray(self.holdout_fp_total),
                         "holdout_n_total": np.asarray(self.holdout_n_total)}}

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
        self.gate_failures = int(restored["meta"]["gate_failures"])
        self.holdout_fp_total = int(restored["meta"]["holdout_fp_total"])
        self.holdout_n_total = int(restored["meta"]["holdout_n_total"])
        self._last_saved_gen = int(restored["meta"]["generation"])

    def _save(self, generation: int):
        if generation <= self._last_saved_gen:
            return  # orbax steps must be unique and monotonic
        self.start_generation = generation
        self.mgr.save(generation, args=ocp.args.StandardSave(self._payload()))
        self._last_saved_gen = generation
        self._log(f"checkpoint saved @ gen {generation}")

    def _save_best(self):
        best_dir = (self.run_dir / "best").absolute()
        # orbax 0.12 StandardCheckpointer takes the state positionally and
        # saves asynchronously; the context manager waits before returning.
        with ocp.StandardCheckpointer() as ckptr:
            ckptr.save(best_dir, self.best_params, force=True)

    # -- terminal feedback ---------------------------------------------------
    def _log(self, msg: str):
        up = (time.time() - self._t0) / 3600
        print(f"{time.strftime('%H:%M:%S')} t+{up:5.2f}h  {msg}", flush=True)

    def _log_banner(self):
        cfg = self.cfg
        n_params = sum(x.size for x in jax.tree.leaves(self.params))
        self._log(
            f"ChessZero | net {cfg.net.blocks}x{cfg.net.channels}"
            f" ({_human(n_params)} params)"
            f" | selfplay {cfg.selfplay.num_games} games x"
            f" {cfg.selfplay.steps_per_generation} steps/gen,"
            f" sims {cfg.selfplay.sims_full}/{cfg.selfplay.sims_cheap}"
            f" | train batch {cfg.train.batch_size} x"
            f" {cfg.train.steps_per_generation} steps/gen")
        self._log(
            f"device {jax.devices()[0].device_kind} | run {self.run_dir}"
            f" | buffer min {_human(cfg.train.min_buffer)}"
            f" cap {_human(cfg.train.buffer_capacity)}"
            f" | gate every {cfg.gate_every_generations} gens"
            f" | ckpt every {cfg.checkpoint_every_min:g} min")
        if self.start_generation > 0:
            self._log(f"resumed from checkpoint: gen {self.start_generation},"
                      f" global step {self.global_step}"
                      f" (buffer restarts empty)")

    # -- main loop -----------------------------------------------------------
    def run(self, max_generations: int | None = None):
        cfg = self.cfg
        target = max_generations if max_generations is not None \
            else cfg.max_generations
        self._log_banner()
        last_ckpt = time.time()
        for gen in range(self.start_generation, target):
            t0 = time.time()
            allow_resign = self.global_step >= cfg.train.resign_min_train_steps
            examples, stats = self.worker.run_generation(self.params,
                                                         allow_resign)
            sp_seconds = time.time() - t0
            self.holdout_fp_total += stats.holdout_false_positives
            self.holdout_n_total += stats.holdout_resign_games
            if examples:
                self.buffer.add(*pack_examples(examples))
            metrics = {}
            if self.buffer.size >= cfg.train.min_buffer:
                if not hasattr(self, "_train_step"):
                    self._train_step = make_train_step(self.net, self.tx,
                                                       cfg.train)
                step_metrics = []
                for _ in range(cfg.train.steps_per_generation):
                    batch = {k: jnp.asarray(v) for k, v in
                             self.buffer.sample(cfg.train.batch_size).items()}
                    self.params, self.opt_state, m = self._train_step(
                        self.params, self.opt_state, batch)
                    self.global_step += 1
                    step_metrics.append(m)
                metrics = {k: float(np.mean([sm[k] for sm in step_metrics]))
                           for k in step_metrics[0]}
                metrics["lr"] = float(self._lr_schedule(self.global_step))

            moves = cfg.selfplay.num_games * cfg.selfplay.steps_per_generation
            row = {"ts": time.time(), "gen": gen,
                   "global_step": self.global_step,
                   "buffer_size": self.buffer.size,
                   "games": stats.games, "resigns": stats.resigns,
                   "draws": stats.draws,
                   "avg_len": (stats.sum_game_len / stats.games
                               if stats.games else None),
                   "holdout_fp": stats.holdout_false_positives,
                   "holdout_n": stats.holdout_resign_games,
                   "sp_seconds": sp_seconds,
                   "moves_per_s": moves / sp_seconds,
                   "gen_seconds": time.time() - t0, **metrics}
            fp_rate = resign_fp_alarm(self.holdout_fp_total,
                                      self.holdout_n_total)
            if fp_rate is not None:
                row["resign_fp_alarm"] = fp_rate
            self._log(format_gen_line(row, cfg.train.min_buffer))
            if fp_rate is not None and not self._fp_alarm_active:
                self._log(f"ALARM resign false-positive rate"
                          f" {100 * fp_rate:.1f}% (>5% of"
                          f" {self.holdout_n_total} held-out resign games)"
                          f" — resign threshold may be unsafe")
            self._fp_alarm_active = fp_rate is not None

            if (gen + 1) % cfg.gate_every_generations == 0 \
                    and self.global_step > 0:
                gt0 = time.time()
                score = play_match(self.net, self.params, self.best_params,
                                   cfg, seed=cfg.seed + gen)
                row["gate_score"] = score
                promoted = score >= cfg.gating.promote_threshold
                row["promoted"] = promoted
                if promoted:
                    self.best_params = jax.tree.map(jnp.copy, self.params)
                    self.gate_failures = 0
                    self._save_best()
                    self._log(f"GATE gen {gen}: challenger {score:.3f}"
                              f" vs best -> PROMOTED, new best saved"
                              f" ({time.time() - gt0:.0f}s)")
                else:
                    self.gate_failures += 1
                    self._log(f"GATE gen {gen}: challenger {score:.3f}"
                              f" vs best -> kept best"
                              f" (fail {self.gate_failures}/3,"
                              f" {time.time() - gt0:.0f}s)")
                    if self.gate_failures >= 3:
                        row["alarm"] = "3 consecutive gate failures"
                        self._log("ALARM 3 consecutive gate failures —"
                                  " net is not improving; inspect losses"
                                  " before letting the run continue")

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
