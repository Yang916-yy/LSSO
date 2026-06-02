#!/usr/bin/env bash
set -euo pipefail

cd /root/LSSO
source /mnt/d/LSSO/.venv/bin/activate
mkdir -p runs

common=(
  --dataset cifar100
  --epochs 60
  --batch-size 256
  --dim 96
  --depth 3
  --num-heads 3
  --num-workers 2
  --amp
)

python train_cifar.py "${common[@]}" --mixer mha \
  2>&1 | tee runs/cifar100_mha_d96_L3_60epoch.console.log

python train_cifar.py "${common[@]}" --mixer lsso --rank 16 --gamma-max 0.3 --theta-gamma-init -4 \
  2>&1 | tee runs/cifar100_lsso_r16_d96_L3_60epoch.console.log

python train_cifar.py "${common[@]}" --mixer lsso --rank 32 --gamma-max 0.3 --theta-gamma-init -4 \
  2>&1 | tee runs/cifar100_lsso_r32_d96_L3_60epoch.console.log

python train_cifar.py "${common[@]}" --mixer lsso-no-global --rank 16 --gamma-max 0.3 --theta-gamma-init -4 \
  2>&1 | tee runs/cifar100_lsso_no_global_d96_L3_60epoch.console.log
