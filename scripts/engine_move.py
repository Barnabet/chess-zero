# scripts/engine_move.py
"""Load best/ checkpoint, play one engine move from startpos, print it."""
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")

from chesszero.config import Config
from chesszero.engine import Engine

best_dir, config_path = sys.argv[1], sys.argv[2]
cfg = Config.from_yaml(config_path)
engine = Engine(best_dir, cfg)
engine.reset()
move = engine.best_move(0.5)
assert move is not None
print(f"engine move from startpos: {move.uci()}")
