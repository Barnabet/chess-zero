# scripts/calibrate.py
"""Phase 1: throughput matrix. Usage: python scripts/calibrate.py [--quick]

Measures selfplay moves/s for candidate configs; prints positions/hour and
games/hour (assuming ~135 ply/game pre-resignation). Run one config at a
time on an otherwise idle GPU.
"""
import argparse
import time

import jax
import jax.numpy as jnp

from chesszero.config import Config, NetConfig, SelfplayConfig
from chesszero.net import ChessNet
from chesszero.selfplay import SelfplayWorker

CANDIDATES = [
    ("6x128 b1024 s32/8", NetConfig(channels=128, blocks=6),
     dict(num_games=1024, sims_full=32, sims_cheap=8)),
    ("6x128 b2048 s32/8", NetConfig(channels=128, blocks=6),
     dict(num_games=2048, sims_full=32, sims_cheap=8)),
    ("8x192 b1024 s32/8", NetConfig(channels=192, blocks=8),
     dict(num_games=1024, sims_full=32, sims_cheap=8)),
    ("6x128 b1024 s48/12", NetConfig(channels=128, blocks=6),
     dict(num_games=1024, sims_full=48, sims_cheap=12)),
]

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="1 gen instead of 3")
args = ap.parse_args()

print(f"{'config':24} {'moves/s':>10} {'pos/hour':>12} {'games/hour':>11}")
for name, net_cfg, sp_kw in CANDIDATES:
    cfg = Config(net=net_cfg,
                 selfplay=SelfplayConfig(steps_per_generation=16, **sp_kw))
    net = ChessNet(cfg.net)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 8, 8, 119)))
    worker = SelfplayWorker(net, cfg, seed=0)
    worker.run_generation(params, allow_resign=False)      # compile + warmup
    t0 = time.time()
    reps = 1 if args.quick else 3
    for _ in range(reps):
        worker.run_generation(params, allow_resign=False)
    dt = time.time() - t0
    moves = cfg.selfplay.num_games * cfg.selfplay.steps_per_generation * reps
    mps = moves / dt
    print(f"{name:24} {mps:>10.0f} {mps * 3600:>12.0f} {mps * 3600 / 135:>11.0f}")
