#!/usr/bin/env bash
# scripts/bootstrap.sh — recreate the runtime env on a fresh pod.
set -euo pipefail
pip install -q -U "jax[cuda12]==0.10.2" pgx==2.6.0 mctx==0.0.71 flax==0.12.7 \
  optax==0.2.8 orbax-checkpoint==0.12.1 chess==1.11.2 pyyaml pytest
pip install -q -e .
python -c "import jax; assert jax.devices()[0].platform == 'gpu', 'NO GPU'"
echo "bootstrap OK: $(python -c 'import jax; print(jax.devices())')"
