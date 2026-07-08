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
