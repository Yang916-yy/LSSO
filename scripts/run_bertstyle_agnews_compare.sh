#!/usr/bin/env bash
set -euo pipefail

cd /root/LSSO
source /mnt/d/LSSO/.venv/bin/activate
mkdir -p runs

common=(
  --dataset ag_news
  --data-dir /root/LSSO/data
  --run-dir /root/LSSO/runs
  --dim 256
  --depth 6
  --num-heads 4
  --max-len 256
  --max-vocab 50000
  --epochs 5
  --batch-size 128
  --num-workers 2
  --device cuda
  --amp
)

PYTHONUNBUFFERED=1 python train_bertstyle.py "${common[@]}" \
  --mixer mha \
  2>&1 | tee runs/bertstyle_agnews_mha_d256_L6_len256_5epoch.console.log

PYTHONUNBUFFERED=1 python train_bertstyle.py "${common[@]}" \
  --mixer lsso \
  --rank 16 \
  --gamma-max 0.3 \
  --theta-gamma-init -4 \
  2>&1 | tee runs/bertstyle_agnews_lsso_r16_d256_L6_len256_5epoch.console.log
