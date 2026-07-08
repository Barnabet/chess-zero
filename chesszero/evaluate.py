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
