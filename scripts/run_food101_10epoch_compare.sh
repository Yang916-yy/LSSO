#!/usr/bin/env bash
set -euo pipefail

cd "${LSSO_ROOT:-$HOME/LSSO}"
source "${LSSO_VENV:-/mnt/d/LSSO/.venv/bin/activate}"

common=(
  --dataset food101
  --data-dir "$HOME/LSSO/data"
  --run-dir "$HOME/LSSO/runs"
  --image-size 224
  --patch-size 16
  --dim 192
  --depth 4
  --num-heads 3
  --epochs 10
  --batch-size 64
  --num-workers 4
  --device cuda
  --amp
)

PYTHONUNBUFFERED=1 python train_cifar.py "${common[@]}" \
  --mixer mha \
  2>&1 | tee "$HOME/LSSO/runs/food101_224_mha_10epoch.console.log"

PYTHONUNBUFFERED=1 python train_cifar.py "${common[@]}" \
  --mixer lsso \
  --rank 16 \
  --gamma-max 0.3 \
  --theta-gamma-init -4 \
  2>&1 | tee "$HOME/LSSO/runs/food101_224_lsso_r16_10epoch.console.log"
