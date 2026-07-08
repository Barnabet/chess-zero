import jax
import jax.numpy as jnp
import numpy as np

from chesszero.config import NetConfig
from chesszero.net import ChessNet, value_from_wdl


def _make(precision="fp32"):
    cfg = NetConfig(channels=32, blocks=2, se_ratio=4, precision=precision)
    net = ChessNet(cfg)
    obs = jnp.zeros((4, 8, 8, 119), jnp.float32)
    params = net.init(jax.random.PRNGKey(0), obs)
    return net, params, obs


def test_output_shapes_and_dtypes():
    net, params, obs = _make()
    p, w, m = net.apply(params, obs)
    assert p.shape == (4, 4672) and p.dtype == jnp.float32
    assert w.shape == (4, 3) and w.dtype == jnp.float32
    assert m.shape == (4,) and float(m.min()) >= 0.0


def test_bf16_activations_fp32_params():
    net, params, obs = _make("bf16")
    p, w, m = net.apply(params, obs)
    assert p.dtype == jnp.float32  # heads cast back to fp32
    leaves = jax.tree.leaves(params)
    assert all(l.dtype == jnp.float32 for l in leaves)


def test_value_from_wdl_bounds():
    logits = jnp.array([[10.0, 0.0, -10.0], [-10.0, 0.0, 10.0], [0.0, 10.0, 0.0]])
    v = np.asarray(value_from_wdl(logits))
    assert v[0] > 0.99 and v[1] < -0.99 and abs(v[2]) < 0.01


def test_deterministic():
    net, params, obs = _make()
    p1, _, _ = net.apply(params, obs)
    p2, _, _ = net.apply(params, obs)
    assert jnp.array_equal(p1, p2)
