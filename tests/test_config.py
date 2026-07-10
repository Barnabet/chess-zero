from chesszero.config import Config


def test_defaults_roundtrip():
    cfg = Config()
    assert cfg.net.channels == 128 and cfg.net.blocks == 6
    assert cfg.selfplay.sims_full == 32 and cfg.selfplay.sims_cheap == 8
    d = cfg.to_dict()
    cfg2 = Config.from_dict(d)
    assert cfg2 == cfg


def test_tiny_yaml_loads():
    cfg = Config.from_yaml("configs/tiny.yaml")
    assert cfg.net.channels == 32
    assert cfg.selfplay.num_games == 8
    assert cfg.run_dir == "runs/tiny"


def test_v1_yaml_loads():
    cfg = Config.from_yaml("configs/v1.yaml")
    assert cfg.net.channels == 128
    assert cfg.selfplay.num_games >= 1024


def test_unknown_key_rejected():
    import pytest
    with pytest.raises(TypeError):
        Config.from_dict({"nonexistent_field": 1})


def test_v2_fields_defaults_and_yaml_override():
    from chesszero.config import Config
    c = Config.from_dict({})
    # defaults must preserve v1 semantics exactly
    assert c.selfplay.opening_plies_max == 0
    assert c.selfplay.search_max_depth == 0
    assert c.train.lr_decay_steps == 0
    assert c.train.lr_floor_frac == 0.1
    assert c.train.resign_arm_fp == 0.05
    assert c.train.resign_disarm_fp == 0.08
    assert c.train.resign_fp_window == 2000
    assert c.anchor_every_generations == 0

    c = Config.from_dict({
        "selfplay": {"opening_plies_max": 8, "search_max_depth": 16},
        "train": {"lr_decay_steps": 400_000},
        "anchor_every_generations": 60})
    assert c.selfplay.opening_plies_max == 8
    assert c.selfplay.search_max_depth == 16
    assert c.train.lr_decay_steps == 400_000
    assert c.anchor_every_generations == 60
