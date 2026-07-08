# tests/test_buffer.py
import numpy as np

from chesszero.buffer import ReplayBuffer


def _batch(n, fill):
    return (np.full((n, 8, 8, 119), fill, np.float16),
            np.full((n, 4672), 1.0 / 4672, np.float16),
            np.ones(n, bool),
            np.full(n, 1, np.int8),
            np.full(n, 40, np.int16))


def test_add_and_sample():
    buf = ReplayBuffer(capacity=100, seed=0)
    buf.add(*_batch(30, 0.5))
    assert buf.size == 30 and buf.total_added == 30
    s = buf.sample(16)
    assert s["obs"].shape == (16, 8, 8, 119) and s["obs"].dtype == np.float32
    assert s["wdl"].dtype == np.int32 and s["moves_left"].dtype == np.float32
    assert float(s["obs"].max()) == 0.5


def test_ring_wraparound_evicts_oldest():
    buf = ReplayBuffer(capacity=50, seed=0)
    buf.add(*_batch(40, 1.0))
    buf.add(*_batch(40, 2.0))          # wraps: only last 50 remain, all newest first
    assert buf.size == 50 and buf.total_added == 80
    s = buf.sample(256)
    vals = set(np.unique(s["obs"]).tolist())
    assert 2.0 in vals                  # new data present
    # 30 of the 40 old rows were overwritten; old value may remain (10 rows) but
    # buffer must never return anything other than 1.0 / 2.0
    assert vals <= {1.0, 2.0}


def test_sample_only_from_filled():
    buf = ReplayBuffer(capacity=1000, seed=0)
    buf.add(*_batch(10, 3.0))
    s = buf.sample(64)
    assert float(s["obs"].min()) == 3.0  # never samples zero-initialized rows
