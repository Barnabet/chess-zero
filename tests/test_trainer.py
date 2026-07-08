# tests/test_trainer.py
import json
import pathlib

import pytest

from chesszero.config import Config
from chesszero.train import Trainer


@pytest.fixture()
def cfg(tmp_path):
    c = Config.from_yaml("configs/tiny.yaml")
    c.run_dir = str(tmp_path / "run")
    c.checkpoint_every_min = 0.0        # checkpoint every generation
    c.gate_every_generations = 2
    return c


def test_resign_fp_alarm_thresholds():
    from chesszero.train import resign_fp_alarm
    assert resign_fp_alarm(0, 0) is None      # no data
    assert resign_fp_alarm(5, 19) is None     # below min_n
    assert resign_fp_alarm(1, 20) is None     # exactly 5%: not exceeded
    assert resign_fp_alarm(2, 20) == 0.1      # tripped


@pytest.mark.slow
def test_train_checkpoint_resume(cfg):
    t = Trainer(cfg)
    t.run(max_generations=3)
    run = pathlib.Path(cfg.run_dir)
    assert (run / "best").exists()
    lines = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(lines) == 3 and lines[-1]["gen"] == 2
    assert "policy_loss" in lines[-1] or lines[-1]["buffer_size"] < cfg.train.min_buffer

    # resume: a NEW Trainer picks up at gen 3, runs to 5
    t2 = Trainer(cfg)
    assert t2.start_generation == 3
    t2.run(max_generations=5)
    lines = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert lines[-1]["gen"] == 4 and len(lines) == 5


@pytest.mark.slow
def test_training_gating_checkpoint_after_donation(cfg):
    """Force the training path (buffer pre-seeded past min_buffer) so the
    donation-sensitive best_params/checkpoint/gating code actually executes.
    Catches aliasing of donated buffers, which the organic 3-gen run cannot
    reach (games don't finish that fast)."""
    import numpy as np

    from chesszero.selfplay import Example, pack_examples

    cfg.gate_every_generations = 2      # gate fires at gen 1
    t = Trainer(cfg)
    exs = [Example(np.zeros((8, 8, 119), np.float16),
                   np.full(4672, 1 / 4672, np.float16) if i % 2 == 0 else None,
                   i % 3, 10)
           for i in range(cfg.train.min_buffer + 64)]
    t.buffer.add(*pack_examples(exs))
    t.run(max_generations=2)            # trains, checkpoints, gates — no crash
    run = pathlib.Path(cfg.run_dir)
    lines = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert "policy_loss" in lines[-1]                 # training fired
    assert any("gate_score" in l for l in lines)      # gating fired
    t2 = Trainer(cfg)                                 # checkpoint is loadable
    assert t2.start_generation == 2 and t2.global_step > 0
