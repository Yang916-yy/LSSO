#!/usr/bin/env bash
set -euo pipefail

cd /root/LSSO
source /mnt/d/LSSO/.venv/bin/activate
mkdir -p runs

PYTHONUNBUFFERED=1 python train_cifar.py \
  --dataset food101 \
  --data-dir /root/LSSO/data \
  --run-dir /root/LSSO/runs \
  --mixer mha \
  --image-size 224 \
  --patch-size 16 \
  --dim 192 \
  --depth 4 \
  --num-heads 3 \
  --epochs 10 \
  --batch-size 256 \
  --num-workers 8 \
  --device cuda \
  --amp \
  --no-pin-memory \
  2>&1 | tee runs/food101_224_resizecrop_bs256_w8_mha_10epoch.console.log
