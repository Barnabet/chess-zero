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
| best-gen2129 | 2026-07-09 | ~+1825 | gate 0.542 vs gen 1979; first gate after 2nd lr cut (2.2e-4) |
| best-gen2159 | 2026-07-09 | ~+1875 | gate 0.571 vs gen 2129 |
| best-gen2189 | 2026-07-09 | ~+1907 | gate 0.546 vs gen 2159 |
| best-gen2369 | 2026-07-09 | ~+2033 | gate 0.575 vs gen 2309 (gens 2279/2309 promoted but not archived) |
| best-gen2459 | 2026-07-10 | ~+2153 | gate 0.667 vs gen 2369 |
| best-gen2519 | 2026-07-10 | ~+2205 | gate 0.575 vs gen 2459 |
| best-gen2549 | 2026-07-10 | ~+2273 | gate 0.596 vs gen 2519 |
| best-gen2609 | 2026-07-10 | ~+2334 | gate 0.587 vs gen 2549 |
| best-gen2639 | 2026-07-10 | ~+2374 | gate 0.558 vs gen 2609 |
| best-gen2669 | 2026-07-10 | ~+2421 | gate 0.567 vs gen 2639 |
| best-gen2729 | 2026-07-10 | ~+2491 | gate 0.600 vs gen 2669 |
| best-gen2759 | 2026-07-10 | ~+2635 | gate 0.696 vs gen 2729 |
| best-gen2789 | 2026-07-10 | ~+2782 | gate 0.700 vs gen 2759 |
| best-gen2849 | 2026-07-10 | ~+2820 | gate 0.554 vs gen 2789 |
| best-gen2879 | 2026-07-10 | ~+2934 | gate 0.658 vs gen 2849 |
| best-gen2909 | 2026-07-10 | ~+2963 | gate 0.542 vs gen 2879 |
| best-gen2939 | 2026-07-10 | ~+3024 | gate 0.587 vs gen 2909 |
| best-gen3029 | 2026-07-10 | ~+3074 | gate 0.571 vs gen 2939 |
| best-gen3089 | 2026-07-10 | ~+3142 | gate 0.596 vs gen 3029 |
| best-gen3119 | 2026-07-10 | ~+3222 | gate 0.613 vs gen 3089 |
| best-gen3179 | 2026-07-10 | ~+3397 | gate 0.733 vs gen 3119 |
| best-gen3239 | 2026-07-10 | ~+3429 | gate 0.546 vs gen 3179 |
| best-gen3269 | 2026-07-10 | ~+3467 | gate 0.554 vs gen 3239 |

Internal Elo is chained self-play gate Elo — inflated vs external opponents
(see project notes); use versus_stockfish.py for real-world anchoring.

## v2 run

Same 6x128 net; sims 64/16, random opening plies, cosine LR, resign governor.
Load with `configs/v2.yaml`. Internal Elo chained from v2's own random init.

| checkpoint | date | internal Elo vs init | notes |
|---|---|---|---|
| v2-gen0029 | 2026-07-10 | ~+280 | gate 0.833 vs random init; first promotion |
| v2-gen0059 | 2026-07-10 | ~+404 | gate 0.671 vs gen 29 |
| v2-gen0089 | 2026-07-10 | ~+444 | gate 0.558 vs gen 59 |
| v2-gen0149 | 2026-07-10 | ~+533 | gate 0.625 vs gen 89 (gen 119 gate failed 0.496) |
| v2-gen0209 | 2026-07-10 | ~+559 | gate 0.538 vs gen 149 (gen 179 gate failed 0.483) |
| v2-gen0239 | 2026-07-10 | ~+582 | gate 0.533 vs gen 209 |
| v2-gen0269 | 2026-07-10 | ~+614 | gate 0.546 vs gen 239 |
| v2-gen0329 | 2026-07-10 | ~+649 | gate 0.550 vs gen 269 (gen 299 gate failed 0.500) |
| v2-gen0359 | 2026-07-11 | ~+672 | gate 0.533 vs gen 329 |
| v2-gen0419 | 2026-07-11 | ~+710 | gate 0.554 vs gen 359 (gen 389 gate failed 0.504) |
| v2-gen0509 | 2026-07-11 | ~+760 | gate 0.571 vs gen 419 (gens 449/479 gates failed 0.521/0.529) |
| v2-gen0599 | 2026-07-11 | ~+830 | gate 0.600 vs gen 509 (gens 539/569 gates failed 0.454/0.492) |
| v2-gen0719 | 2026-07-11 | ~+910 | gate 0.613 vs gen 599 (gens 629/659/689 gates failed 0.429/0.492/0.458) |
