#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if [[ "${LSSO_FOOD101_LOGGING:-0}" != "1" ]]; then
  mkdir -p runs/food101_vit_b16_80ep
  exec env LSSO_FOOD101_LOGGING=1 bash "$0" \
    > runs/food101_vit_b16_80ep/train.log 2>&1
fi

exec .venv/bin/python experiments/cv_vit_rrlsso_cifar100.py \
  --dataset food101 \
  --data-dir /mnt/d/LSSO-data/torchvision \
  --out-dir runs/food101_vit_b16_80ep \
  --models mha rrlsso \
  --epochs 80 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --image-size 224 \
  --patch-size 16 \
  --rank 32 \
  --gamma-max 1.2 \
  --theta-gamma-init 0.5 \
  --length-normalize \
  --length-reference 1.0 \
  --lr 5e-4 \
  --min-lr 1e-5 \
  --warmup-epochs 5 \
  --weight-decay 0.05 \
  --label-smoothing 0.1 \
  --grad-clip 1.0 \
  --dtype bf16 \
  --num-workers 8 \
  --seed 1234 \
  --save-checkpoints
