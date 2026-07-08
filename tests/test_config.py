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
