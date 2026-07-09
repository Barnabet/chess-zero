# ChessZero released checkpoints

Orbax checkpoints of promoted "best" networks from the v1 run (6 blocks x 128 channels, bf16).
Load with `chesszero.engine.Engine(<dir>, cfg)` using `configs/v1.yaml`.

| checkpoint | date | internal Elo vs init | notes |
|---|---|---|---|
| best-gen1709 | 2026-07-09 | ~+1563 | LR 6.6e-4 era; resign disabled |
| best-gen1739 | 2026-07-09 | ~+1589 | gate 0.538 vs gen 1709 |
| best-gen1769 | 2026-07-09 | ~+1618 | gate 0.542 vs gen 1739 |
| best-gen1829 | 2026-07-09 | ~+1686 | gate 0.596 vs gen 1769 |
| best-gen1859 | 2026-07-09 | ~+1744 | gate 0.583 vs gen 1829 |
| best-gen1889 | 2026-07-09 | ~+1773 | gate 0.542 vs gen 1859 |
| best-gen1979 | 2026-07-09 | ~+1796 | gate 0.533 vs gen 1889 |

Internal Elo is chained self-play gate Elo — inflated vs external opponents
(see project notes); use versus_stockfish.py for real-world anchoring.
