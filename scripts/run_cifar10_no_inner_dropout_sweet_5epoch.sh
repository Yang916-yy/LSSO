#!/usr/bin/env bash
set -euo pipefail

cd /root/LSSO
source /mnt/d/LSSO/.venv/bin/activate
mkdir -p runs

common=(
  --dataset cifar10
  --mixer lsso
  --epochs 5
  --batch-size 256
  --dim 96
  --depth 3
  --num-heads 3
  --gamma-max 0.3
  --theta-gamma-init -4
  --num-workers 2
  --amp
)

python train_cifar.py "${common[@]}" --rank 16
python train_cifar.py "${common[@]}" --rank 32
