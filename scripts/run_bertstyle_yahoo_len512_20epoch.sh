#!/usr/bin/env bash
set -euo pipefail

cd /root/LSSO
source /mnt/d/LSSO/.venv/bin/activate
mkdir -p runs

common=(
  --dataset yahoo_answers
  --data-dir /root/LSSO/data
  --run-dir /root/LSSO/runs
  --dim 256
  --depth 6
  --num-heads 4
  --max-len 512
  --max-vocab 50000
  --max-train-examples 200000
  --max-eval-examples 20000
  --epochs 20
  --batch-size 64
  --num-workers 2
  --device cuda
  --amp
)

PYTHONUNBUFFERED=1 python train_bertstyle.py "${common[@]}" \
  --mixer mha \
  2>&1 | tee runs/bertstyle_yahoo_mha_d256_L6_len512_20epoch.console.log

PYTHONUNBUFFERED=1 python train_bertstyle.py "${common[@]}" \
  --mixer lsso \
  --rank 16 \
  --gamma-max 0.3 \
  --theta-gamma-init -4 \
  2>&1 | tee runs/bertstyle_yahoo_lsso_r16_d256_L6_len512_20epoch.console.log

PYTHONUNBUFFERED=1 python train_bertstyle.py "${common[@]}" \
  --mixer lsso \
  --rank 32 \
  --gamma-max 0.3 \
  --theta-gamma-init -4 \
  2>&1 | tee runs/bertstyle_yahoo_lsso_r32_d256_L6_len512_20epoch.console.log
