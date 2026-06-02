#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-cifar10}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MODEL_DIM="${MODEL_DIM:-96}"
DEPTH="${DEPTH:-3}"
HEADS="${HEADS:-3}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-20}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-20}"

COMMON_ARGS=(
  --dataset "$DATASET"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --dim "$MODEL_DIM"
  --depth "$DEPTH"
  --num-heads "$HEADS"
  --num-workers 2
  --max-train-batches "$MAX_TRAIN_BATCHES"
  --max-eval-batches "$MAX_EVAL_BATCHES"
  --amp
)

python train_cifar.py --mixer mha "${COMMON_ARGS[@]}"

for RANK in 4 8 16 32; do
  python train_cifar.py --mixer lsso --rank "$RANK" "${COMMON_ARGS[@]}"
done

python train_cifar.py --mixer lsso-no-global --rank 16 "${COMMON_ARGS[@]}"
