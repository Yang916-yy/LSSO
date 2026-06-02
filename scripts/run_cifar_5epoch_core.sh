#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-cifar10}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MODEL_DIM="${MODEL_DIM:-96}"
DEPTH="${DEPTH:-3}"
HEADS="${HEADS:-3}"

COMMON_ARGS=(
  --dataset "$DATASET"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --dim "$MODEL_DIM"
  --depth "$DEPTH"
  --num-heads "$HEADS"
  --num-workers 2
  --amp
)

python train_cifar.py --mixer mha "${COMMON_ARGS[@]}"
python train_cifar.py --mixer lsso --rank 16 "${COMMON_ARGS[@]}"
python train_cifar.py --mixer lsso --rank 32 "${COMMON_ARGS[@]}"
python train_cifar.py --mixer lsso-no-global --rank 16 "${COMMON_ARGS[@]}"
