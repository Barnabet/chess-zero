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
    resign_count: int = 0
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
            slot.obs.append(obs[i])
            slot.weights.append(weights[i] if full else None)
            slot.mover.append(int(movers[i]))
            slot.root_value.append(float(values[i]))
            if float(values[i]) < -sp.resign_threshold:
                slot.resign_count += 1
            else:
                slot.resign_count = 0
            tripped = slot.resign_count >= sp.resign_consecutive_plies
            if tripped and slot.resign_would_have < 0:
                slot.resign_would_have = len(slot.obs)
            if done[i]:
                self._flush(i, rewards[i], examples, stats, resigned=False)
                new_reset[i] = True
            elif allow_resign and tripped and not slot.holdout:
                loser = int(movers[i])
                fake = np.zeros(2, np.float32)
                fake[loser], fake[1 - loser] = -1.0, 1.0
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
