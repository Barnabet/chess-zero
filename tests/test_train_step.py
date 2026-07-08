# tests/test_train_step.py
import jax
import jax.numpy as jnp
import numpy as np

from chesszero.config import NetConfig, TrainConfig
from chesszero.net import ChessNet
from chesszero.train import make_optimizer, make_train_step


def _setup():
    net = ChessNet(NetConfig(channels=16, blocks=1, precision="fp32"))
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 8, 8, 119)))
    cfg = TrainConfig(lr=1e-2, warmup_steps=1, batch_size=8)
    tx = make_optimizer(cfg)
    opt_state = tx.init(params)
    step = make_train_step(net, tx, cfg)
    rng = np.random.default_rng(0)
    pol = rng.random((8, 4672)).astype(np.float32)
    pol /= pol.sum(-1, keepdims=True)
    batch = {
        "obs": jnp.asarray(rng.random((8, 8, 8, 119)), jnp.float32),
        "policy": jnp.asarray(pol),
        "has_policy": jnp.asarray([True] * 6 + [False] * 2),
        "wdl": jnp.asarray(rng.integers(0, 3, 8), jnp.int32),
        "moves_left": jnp.asarray(rng.integers(1, 100, 8), jnp.float32),
    }
    return params, opt_state, step, batch


def test_loss_finite_and_grads_flow():
    params, opt_state, step, batch = _setup()
    # train_step donates params/opt_state buffers, so diff against a copy.
    params0 = jax.tree.map(jnp.copy, params)
    params2, opt_state2, m = step(params, opt_state, batch)
    for k in ("loss", "policy_loss", "wdl_loss", "ml_loss"):
        assert np.isfinite(float(m[k])), k
    # The first update is a no-op (lr warms up from 0 at optimizer count 0),
    # so take a second step before checking that parameters actually move.
    params2, opt_state2, m = step(params2, opt_state2, batch)
    diffs = jax.tree.map(lambda a, b: float(jnp.abs(a - b).max()),
                         params0, params2)
    assert max(jax.tree.leaves(diffs)) > 0  # something actually updated


def test_overfits_fixed_batch():
    params, opt_state, step, batch = _setup()
    # Policy CE is lower-bounded by the entropy of the dense random targets
    # (~8.26 nats here), so measure decrease on the reducible excess above it.
    pol = np.asarray(batch["policy"])
    mask = np.asarray(batch["has_policy"])
    ent = float(-(pol * np.log(pol)).sum(-1)[mask].mean())
    first = None
    for i in range(60):
        params, opt_state, m = step(params, opt_state, batch)
        if first is None:
            first = float(m["loss"])
    # clearly decreasing on a fixed batch
    assert float(m["loss"]) - ent < (first - ent) * 0.8
