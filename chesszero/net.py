"""Flax ResNet-SE trunk with policy / WDL-value / moves-left heads.

Policy head spatial mapping (verified obs layout: row 0 = mover's 8th rank,
col = file): pgx label = (file*8 + rank)*73 + plane, so the (row, col) grid
is row-flipped (row -> rank) then transposed (file first) before flattening.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from chesszero.config import NetConfig


def value_from_wdl(wdl_logits):
    p = jax.nn.softmax(wdl_logits, axis=-1)
    return p[..., 0] - p[..., 2]


class SqueezeExcite(nn.Module):
    ratio: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x):
        c = x.shape[-1]
        s = x.mean(axis=(1, 2))
        s = nn.relu(nn.Dense(max(c // self.ratio, 8), dtype=self.dtype)(s))
        s = nn.Dense(2 * c, dtype=self.dtype)(s)
        w, b = jnp.split(s, 2, axis=-1)
        return x * nn.sigmoid(w)[:, None, None, :] + b[:, None, None, :]


class ResBlock(nn.Module):
    channels: int
    se_ratio: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x):
        y = nn.Conv(self.channels, (3, 3), use_bias=False, dtype=self.dtype)(x)
        y = nn.relu(nn.GroupNorm(num_groups=8, dtype=self.dtype)(y))
        y = nn.Conv(self.channels, (3, 3), use_bias=False, dtype=self.dtype)(y)
        y = nn.GroupNorm(num_groups=8, dtype=self.dtype)(y)
        y = SqueezeExcite(self.se_ratio, self.dtype)(y)
        return nn.relu(x + y)


class ChessNet(nn.Module):
    cfg: NetConfig

    @nn.compact
    def __call__(self, obs):
        dtype = jnp.bfloat16 if self.cfg.precision == "bf16" else jnp.float32
        x = obs.astype(dtype)
        x = nn.Conv(self.cfg.channels, (3, 3), use_bias=False, dtype=dtype)(x)
        x = nn.relu(nn.GroupNorm(num_groups=8, dtype=dtype)(x))
        for _ in range(self.cfg.blocks):
            x = ResBlock(self.cfg.channels, self.cfg.se_ratio, dtype)(x)

        # Policy head: (B, row, col, 73) -> pgx label order
        p = nn.Conv(73, (1, 1), dtype=dtype)(x)
        p = jnp.flip(p, axis=1)              # row -> rank (rank 0 at index 0)
        p = jnp.transpose(p, (0, 2, 1, 3))   # (B, file, rank, 73)
        policy_logits = p.reshape(p.shape[0], 4672).astype(jnp.float32)

        v = nn.relu(nn.Conv(8, (1, 1), dtype=dtype)(x))
        v = v.reshape(v.shape[0], -1)
        v = nn.relu(nn.Dense(128, dtype=dtype)(v))
        wdl_logits = nn.Dense(3, dtype=jnp.float32)(v)

        m = nn.relu(nn.Conv(4, (1, 1), dtype=dtype)(x))
        m = m.reshape(m.shape[0], -1)
        m = nn.relu(nn.Dense(64, dtype=dtype)(m))
        moves_left = nn.softplus(nn.Dense(1, dtype=jnp.float32)(m))[..., 0]

        return policy_logits, wdl_logits, moves_left
