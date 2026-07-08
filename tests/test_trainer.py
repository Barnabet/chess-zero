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
    assert lines[-1]["gen"] == 4
