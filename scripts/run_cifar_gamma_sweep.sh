#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-cifar10}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MODEL_DIM="${MODEL_DIM:-96}"
DEPTH="${DEPTH:-3}"
HEADS="${HEADS:-3}"
RANK="${RANK:-32}"

GAMMA_VALUES=(${GAMMA_VALUES:-0.05 0.1 0.2 0.3 0.5 0.8})
THETA_VALUES=(${THETA_VALUES:--5 -4 -3})

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

for GAMMA_MAX in "${GAMMA_VALUES[@]}"; do
  for THETA_INIT in "${THETA_VALUES[@]}"; do
    echo "sweep gamma_max=${GAMMA_MAX} theta_gamma_init=${THETA_INIT}" >&2
    python train_cifar.py \
      "${COMMON_ARGS[@]}" \
      --gamma-max "$GAMMA_MAX" \
      --theta-gamma-init "$THETA_INIT"
  done
done
