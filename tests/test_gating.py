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
