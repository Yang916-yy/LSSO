#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-cifar10}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MODEL_DIM="${MODEL_DIM:-192}"
DEPTH="${DEPTH:-6}"
HEADS="${HEADS:-3}"

python train_cifar.py \
  --dataset "$DATASET" \
  --mixer mha \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --dim "$MODEL_DIM" \
  --depth "$DEPTH" \
  --num-heads "$HEADS"

for RANK in 4 8 16 32; do
  python train_cifar.py \
    --dataset "$DATASET" \
    --mixer lsso \
    --rank "$RANK" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --dim "$MODEL_DIM" \
    --depth "$DEPTH" \
    --num-heads "$HEADS"
done

python train_cifar.py \
  --dataset "$DATASET" \
  --mixer lsso-no-global \
  --rank 16 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --dim "$MODEL_DIM" \
  --depth "$DEPTH" \
  --num-heads "$HEADS"
