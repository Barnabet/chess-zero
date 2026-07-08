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
                         "global_step": np.asarray(self.global_step),
                         "gate_failures": np.asarray(self.gate_failures)}}

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
        self._last_saved_gen = int(restored["meta"]["generation"])

    def _save(self, generation: int):
        if generation <= self._last_saved_gen:
            return  # orbax steps must be unique and monotonic
        self.start_generation = generation
        self.mgr.save(generation, args=ocp.args.StandardSave(self._payload()))
        self._last_saved_gen = generation

    def _save_best(self):
        best_dir = (self.run_dir / "best").absolute()
        # orbax 0.12 StandardCheckpointer takes the state positionally and
        # saves asynchronously; the context manager waits before returning.
        with ocp.StandardCheckpointer() as ckptr:
            ckptr.save(best_dir, self.best_params, force=True)

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
                    self.best_params = jax.tree.map(jnp.copy, self.params)
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
