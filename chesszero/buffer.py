# chesszero/buffer.py
"""Replay buffer: preallocated host-RAM ring, uniform sampling.

Storage dtypes are the memory budget: obs f16 (~15GB at 1M positions),
policy f16 (~9GB). Cheap-search examples carry has_policy=False and are
excluded from the policy loss by the trainer.
"""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, 8, 8, 119), np.float16)
        self.policy = np.zeros((capacity, 4672), np.float16)
        self.has_policy = np.zeros(capacity, bool)
        self.wdl = np.zeros(capacity, np.int8)       # 0 win / 1 draw / 2 loss (mover)
        self.moves_left = np.zeros(capacity, np.int16)
        self.size = 0
        self.head = 0
        self.total_added = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, policy, has_policy, wdl, moves_left) -> None:
        n = obs.shape[0]
        assert n <= self.capacity, "single add larger than buffer"
        idx = (self.head + np.arange(n)) % self.capacity
        self.obs[idx] = obs
        self.policy[idx] = policy
        self.has_policy[idx] = has_policy
        self.wdl[idx] = wdl
        self.moves_left[idx] = moves_left
        self.head = int((self.head + n) % self.capacity)
        self.size = int(min(self.size + n, self.capacity))
        self.total_added += int(n)

    def sample(self, batch_size: int) -> dict:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx].astype(np.float32),
            "policy": self.policy[idx].astype(np.float32),
            "has_policy": self.has_policy[idx].copy(),
            "wdl": self.wdl[idx].astype(np.int32),
            "moves_left": self.moves_left[idx].astype(np.float32),
        }
