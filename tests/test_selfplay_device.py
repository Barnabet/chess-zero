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
