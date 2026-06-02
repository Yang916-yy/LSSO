#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-cifar10}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MODEL_DIM="${MODEL_DIM:-96}"
DEPTH="${DEPTH:-3}"
HEADS="${HEADS:-3}"
GAMMA_MAX="${GAMMA_MAX:-0.3}"
THETA_GAMMA_INIT="${THETA_GAMMA_INIT:--4}"
RANK_VALUES=(${RANK_VALUES:-16 32})

COMMON_ARGS=(
  --dataset "$DATASET"
  --mixer lsso
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --dim "$MODEL_DIM"
  --depth "$DEPTH"
  --num-heads "$HEADS"
  --gamma-max "$GAMMA_MAX"
  --theta-gamma-init "$THETA_GAMMA_INIT"
  --num-workers 2
  --amp
)

for RANK in "${RANK_VALUES[@]}"; do
  echo "long run rank=${RANK} epochs=${EPOCHS} gamma_max=${GAMMA_MAX} theta_gamma_init=${THETA_GAMMA_INIT}" >&2
  python train_cifar.py --rank "$RANK" "${COMMON_ARGS[@]}"
done
