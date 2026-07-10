# chesszero/resign.py
"""Self-arming resignation governor (v2 spec §5).

Tracks the windowed holdout false-positive rate over the trailing
`window` holdout triggers and arms/disarms resignation with hysteresis.
Not persisted across restarts: a fresh process starts disarmed and
re-arms once the window refills (~1-2h) — safe by default."""
from __future__ import annotations

from collections import deque


class ResignGovernor:
    def __init__(self, arm_fp: float, disarm_fp: float, window: int,
                 min_train_steps: int):
        self.arm_fp = arm_fp
        self.disarm_fp = disarm_fp
        self.window = window
        self.min_train_steps = min_train_steps
        self.armed = False
        self._hist: deque[tuple[int, int]] = deque()   # (fp, n) per generation
        self._fp = 0
        self._n = 0

    def update(self, fp: int, n: int,
               global_step: int) -> "tuple[bool, float | None, str | None]":
        self._hist.append((fp, n))
        self._fp += fp
        self._n += n
        # keep the minimal trailing suffix still holding >= window triggers
        while self._hist and self._n - self._hist[0][1] >= self.window:
            old_fp, old_n = self._hist.popleft()
            self._fp -= old_fp
            self._n -= old_n
        if self._n < self.window:
            return self.armed, None, None
        rate = self._fp / self._n
        msg = None
        if (not self.armed and rate < self.arm_fp
                and global_step >= self.min_train_steps):
            self.armed = True
            msg = (f"RESIGN armed: windowed holdout FP {rate:.1%}"
                   f" < {self.arm_fp:.0%} over last {self._n} triggers")
        elif self.armed and rate > self.disarm_fp:
            self.armed = False
            msg = (f"RESIGN disarmed: windowed holdout FP {rate:.1%}"
                   f" > {self.disarm_fp:.0%}")
        return self.armed, rate, msg
