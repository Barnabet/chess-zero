# chesszero/anchor.py
"""External anchor: spar the current best vs scripted baselines, off-GPU-budget.

Spawns scripts/versus_stockfish.py as a subprocess in the 0.15 GPU-memory
slice (validated co-residency pattern) and parses its summary lines. Fully
non-blocking: the Trainer polls once per generation."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

_SCORE_RE = re.compile(r"^\[([\w-]+)\] score ([\d.]+)/(\d+)", re.MULTILINE)


def parse_anchor_output(text: str) -> dict[str, float]:
    return {m.group(1).lower(): float(m.group(2)) / float(m.group(3))
            for m in _SCORE_RE.finditer(text)}


class AnchorRunner:
    def __init__(self, best_dir: str, config_path: str, games: int = 6,
                 movetime: float = 0.2, timeout_s: float = 900.0,
                 cmd: "list[str] | None" = None):
        script = Path(__file__).resolve().parents[1] \
            / "scripts" / "versus_stockfish.py"
        self.cmd = cmd or [
            sys.executable, str(script),
            "--best-dir", str(best_dir), "--config", str(config_path),
            "--vs", "negamax2", "negamax3",
            "--games", str(games), "--movetime", str(movetime),
            "--sims", "32"]  # pinned so anchor scores compare across the run
        self.timeout_s = timeout_s
        self._proc: "subprocess.Popen | None" = None
        self._started = 0.0

    @property
    def running(self) -> bool:
        return self._proc is not None

    def start(self):
        env = dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION="0.15")
        self._proc = subprocess.Popen(
            self.cmd, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._started = time.time()

    def poll(self) -> "dict[str, float] | None":
        """None while running; {} on failure/timeout; results dict on success."""
        if self._proc is None:
            return None
        if self._proc.poll() is None:
            if time.time() - self._started > self.timeout_s:
                self._proc.kill()
                self._proc.wait()
                self._proc = None
                return {}
            return None
        out = self._proc.stdout.read() if self._proc.stdout else ""
        ok = self._proc.returncode == 0
        self._proc = None
        return parse_anchor_output(out) if ok else {}
