# tests/test_selfplay_worker.py
import jax
import jax.numpy as jnp
import numpy as np

from chesszero.config import Config, NetConfig, SelfplayConfig
from chesszero.net import ChessNet
from chesszero.selfplay import Example, GenStats, SelfplayWorker, pack_examples


def _tiny_cfg(**sp):
    base = dict(num_games=4, sims_full=4, sims_cheap=2, full_search_prob=1.0,
                max_considered_actions=4, steps_per_generation=6,
                resign_holdout_frac=0.0)
    base.update(sp)
    return Config(net=NetConfig(channels=16, blocks=1, precision="fp32"),
                  selfplay=SelfplayConfig(**base), seed=3)


def _worker(cfg):
    net = ChessNet(cfg.net)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 8, 8, 119)))
    return SelfplayWorker(net, cfg, seed=cfg.seed), params


def test_generation_runs_and_accounts():
    cfg = _tiny_cfg()
    w, params = _worker(cfg)
    examples, stats = w.run_generation(params, allow_resign=False)
    # every recorded ply is either emitted as an example (finished games) or
    # still held in a live slot — nothing lost, nothing duplicated
    held = sum(len(s.obs) for s in w.slots)
    assert held + stats.examples == 4 * 6
    assert len(examples) == stats.examples


def test_flush_emits_correct_targets():
    cfg = _tiny_cfg()
    w, params = _worker(cfg)
    # Inject a synthetic finished 3-ply game into slot 0:
    slot = w.slots[0]
    slot.obs = [np.zeros((8, 8, 119), np.float16)] * 3
    slot.weights = [np.full(4672, 1 / 4672, np.float16), None,
                    np.full(4672, 1 / 4672, np.float16)]
    slot.mover = [0, 1, 0]
    slot.root_value = [0.0, 0.0, 0.0]
    examples, stats = [], GenStats()
    final_rewards = np.array([1.0, -1.0])  # player 0 won
    w._flush(0, final_rewards, examples, stats, resigned=False)
    assert stats.games == 1 and stats.examples == 3
    assert [e.wdl for e in examples] == [0, 2, 0]        # W, L, W (mover view)
    assert [e.moves_left for e in examples] == [3, 2, 1]
    assert examples[1].policy is None                    # cheap-search ply
    assert len(w.slots[0].obs) == 0                      # slot recycled


def _record(movers, values):
    """Synthetic single-step device record for host-logic tests."""
    n = len(movers)
    return {
        "obs": np.zeros((n, 8, 8, 119), np.float16),
        "action_weights": np.full((n, 4672), 1 / 4672, np.float16),
        "action": np.zeros(n, np.int64),
        "root_value": np.asarray(values, np.float32),
        "mover": np.asarray(movers, np.int64),
        "rewards": np.zeros((n, 2), np.float32),
        "done": np.zeros(n, bool),
    }


def test_resign_counter_is_per_player():
    cfg = _tiny_cfg(num_games=1, resign_threshold=0.9, resign_consecutive_moves=2)
    w, _ = _worker(cfg)
    examples, stats = [], GenStats()
    # decided game: player 0 hopeless on own moves, player 1 confident on theirs
    for mover, val in [(0, -0.95), (1, 0.95)]:
        w._process(_record([mover], [val]), True, True, examples, stats)
    assert stats.resigns == 0            # one bad own-move is not enough
    w._process(_record([0], [-0.95]), True, True, examples, stats)
    assert stats.resigns == 1            # 2nd consecutive bad own-move trips
    assert [e.wdl for e in examples] == [2, 0, 2]  # loser L, winner W, loser L


def test_resign_counter_resets_on_recovery():
    cfg = _tiny_cfg(num_games=1, resign_threshold=0.9, resign_consecutive_moves=2)
    w, _ = _worker(cfg)
    examples, stats = [], GenStats()
    seq = [(0, -0.95), (1, 0.95), (0, 0.0), (1, 0.95), (0, -0.95)]
    for mover, val in seq:
        w._process(_record([mover], [val]), True, True, examples, stats)
    assert stats.resigns == 0            # recovery at ply 3 reset player 0's count


def test_resignation_forces_loss():
    cfg = _tiny_cfg(resign_threshold=-1.1,  # -thr = +1.1: every value < 1.1 is "hopeless"
                    resign_consecutive_moves=2, steps_per_generation=4)
    w, params = _worker(cfg)
    examples, stats = w.run_generation(params, allow_resign=True)
    assert stats.resigns >= 1                            # games got adjudicated
    assert stats.games == stats.resigns
    losses = [e for e in examples if e.wdl == 2]
    wins = [e for e in examples if e.wdl == 0]
    assert losses and wins                               # both perspectives present


def test_pack_examples_shapes():
    ex = [Example(np.zeros((8, 8, 119), np.float16),
                  np.full(4672, 1 / 4672, np.float16), 0, 10),
          Example(np.zeros((8, 8, 119), np.float16), None, 1, 5)]
    obs, pol, hasp, wdl, ml = pack_examples(ex)
    assert obs.shape == (2, 8, 8, 119) and pol.shape == (2, 4672)
    assert hasp.tolist() == [True, False]
    assert wdl.tolist() == [0, 1] and ml.tolist() == [10, 5]
