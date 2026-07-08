"""Single source of configuration. YAML presets live in configs/."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class NetConfig:
    channels: int = 128
    blocks: int = 6
    se_ratio: int = 4
    precision: str = "bf16"  # "bf16" | "fp32" — activations; params always fp32


@dataclass
class SelfplayConfig:
    num_games: int = 1024             # parallel game slots on device
    sims_full: int = 32               # full search — emits policy targets
    sims_cheap: int = 8               # cheap search — value/moves-left targets only
    full_search_prob: float = 0.25    # fraction of steps run at sims_full
    max_considered_actions: int = 16  # Gumbel root candidates
    steps_per_generation: int = 16    # env steps (all slots) per generation
    resign_threshold: float = 0.95    # resign when mover E[value] < -threshold…
    resign_consecutive_plies: int = 4 # …for this many consecutive plies
    resign_holdout_frac: float = 0.10 # games that never resign (FP measurement)


@dataclass
class TrainConfig:
    lr: float = 2e-3
    warmup_steps: int = 500
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    batch_size: int = 1024
    steps_per_generation: int = 64    # gradient steps per generation
    buffer_capacity: int = 1_000_000
    min_buffer: int = 20_000          # no gradient steps below this fill
    policy_weight: float = 1.0
    value_weight: float = 1.0
    moves_left_weight: float = 0.1
    moves_left_scale: float = 50.0    # loss operates on plies / scale
    resign_min_train_steps: int = 2000  # resignation off until net has trained


@dataclass
class GatingConfig:
    games: int = 120
    promote_threshold: float = 0.53
    temperature_plies: int = 4        # opening diversity: sample first N plies
    temperature: float = 1.0


@dataclass
class Config:
    net: NetConfig = field(default_factory=NetConfig)
    selfplay: SelfplayConfig = field(default_factory=SelfplayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    seed: int = 0
    run_dir: str = "runs/dev"
    checkpoint_every_min: float = 15.0
    gate_every_generations: int = 10
    max_generations: int = 1_000_000

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        kwargs = dict(raw)
        subs = {"net": NetConfig, "selfplay": SelfplayConfig,
                "train": TrainConfig, "gating": GatingConfig}
        for name, sub_cls in subs.items():
            if name in kwargs:
                kwargs[name] = sub_cls(**kwargs[name])
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
