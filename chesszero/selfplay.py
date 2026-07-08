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
