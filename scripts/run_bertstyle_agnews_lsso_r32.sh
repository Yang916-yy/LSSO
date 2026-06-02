#!/usr/bin/env bash
set -euo pipefail

cd /root/LSSO
source /mnt/d/LSSO/.venv/bin/activate
mkdir -p runs

PYTHONUNBUFFERED=1 python train_bertstyle.py \
  --dataset ag_news \
  --data-dir /root/LSSO/data \
  --run-dir /root/LSSO/runs \
  --mixer lsso \
  --rank 32 \
  --gamma-max 0.3 \
  --theta-gamma-init -4 \
  --dim 256 \
  --depth 6 \
  --num-heads 4 \
  --max-len 256 \
  --max-vocab 50000 \
  --epochs 5 \
  --batch-size 128 \
  --num-workers 2 \
  --device cuda \
  --amp \
  2>&1 | tee runs/bertstyle_agnews_lsso_r32_d256_L6_len256_5epoch.console.log
