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


def test_make_lr_schedule_warmup_then_constant():
    from chesszero.config import TrainConfig
    from chesszero.train import make_lr_schedule
    s = make_lr_schedule(TrainConfig(lr=1e-3, warmup_steps=100))
    assert float(s(0)) == 0.0
    assert float(s(50)) == pytest.approx(5e-4)
    assert float(s(100)) == pytest.approx(1e-3)
    assert float(s(10_000)) == pytest.approx(1e-3)


def test_format_gen_line_fill_and_train_shapes():
    from chesszero.train import format_gen_line
    fill = {"gen": 3, "global_step": 0, "buffer_size": 45_000, "games": 0,
            "resigns": 0, "draws": 0, "avg_len": None,
            "moves_per_s": 1042.0, "gen_seconds": 31.4}
    line = format_gen_line(fill, min_buffer=100_000)
    assert "45.0k/100.0k filling" in line
    assert "0 games finished" in line and "1042 mv/s" in line

    train = dict(fill, gen=42, global_step=5376, buffer_size=481_000,
                 games=512, draws=160, resigns=61, avg_len=87.2,
                 loss=2.314, policy_loss=1.802, wdl_loss=0.401,
                 ml_loss=0.111, lr=2e-4)
    line = format_gen_line(train, min_buffer=100_000)
    assert "loss 2.314" in line and "pi 1.802" in line
    assert "31% draw" in line and "12% resign" in line
    assert "lr 2.0e-04" in line and "filling" not in line


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
