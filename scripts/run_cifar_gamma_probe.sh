#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-cifar10}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MODEL_DIM="${MODEL_DIM:-96}"
DEPTH="${DEPTH:-3}"
HEADS="${HEADS:-3}"
RANK="${RANK:-32}"

COMMON_ARGS=(
  --dataset "$DATASET"
  --mixer lsso
  --rank "$RANK"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --dim "$MODEL_DIM"
  --depth "$DEPTH"
  --num-heads "$HEADS"
  --num-workers 2
  --amp
)

python train_cifar.py "${COMMON_ARGS[@]}" --gamma-max 0.1 --theta-gamma-init -4
python train_cifar.py "${COMMON_ARGS[@]}" --gamma-max 0.3 --theta-gamma-init -6
python train_cifar.py "${COMMON_ARGS[@]}" --gamma-max 0.3 --theta-gamma-init -4
