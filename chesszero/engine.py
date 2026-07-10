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
        # orbax 0.12 StandardCheckpointer: positional target, no args= kwarg
        self.params = ocp.StandardCheckpointer().restore(
            Path(best_dir).absolute(), template)
        self._search_fns: dict[int, callable] = {}
        self._step = jax.jit(jax.vmap(ENV.step))  # jit once, not per push_uci
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
                    max_depth=self.cfg.selfplay.search_max_depth or None,
                    invalid_actions=~state.legal_action_mask,
                    max_num_considered_actions=(
                        self.cfg.selfplay.max_considered_actions),
                    gumbel_scale=0.0)
                return out.action

            self._search_fns[sims] = search
        return self._search_fns[sims]

    def reset(self, fen: str | None = None):
        """New game. With `fen`, pgx history planes start empty — analysis
        only; for real play use reset() + push_uci so history stays exact."""
        self.board = chess.Board(fen) if fen else chess.Board()
        if fen:
            self.state = jax.tree.map(lambda x: x[None],
                                      bridge.state_from_fen(fen))
        else:
            self.state = init_batch(1, seed=0)

    def push_uci(self, uci: str):
        move = chess.Move.from_uci(uci)
        action = bridge.move_to_action(move, self.board.turn)
        self.state = self._step(self.state, jnp.asarray([action]))
        self.board.push(move)

    def _pick_sims(self, movetime_s: float) -> int:
        if self.sims_per_s is None:
            fn = self._get_search(_SIM_TIERS[0])
            jax.block_until_ready(
                fn(self.params, self.state, self.key))     # compile & sync
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
