#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MODEL_DIM="${MODEL_DIM:-128}"
DEPTH="${DEPTH:-3}"
HEADS="${HEADS:-4}"
MAX_LEN="${MAX_LEN:-128}"
GAMMA_MAX="${GAMMA_MAX:-0.3}"
THETA_GAMMA_INIT="${THETA_GAMMA_INIT:--4}"

COMMON_ARGS=(
  --dataset ag_news
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --dim "$MODEL_DIM"
  --depth "$DEPTH"
  --num-heads "$HEADS"
  --max-len "$MAX_LEN"
  --num-workers 2
  --amp
)

python train_text.py --mixer mha "${COMMON_ARGS[@]}"
python train_text.py \
  --mixer lsso \
  --rank 16 \
  --gamma-max "$GAMMA_MAX" \
  --theta-gamma-init "$THETA_GAMMA_INIT" \
  "${COMMON_ARGS[@]}"
python train_text.py \
  --mixer lsso \
  --rank 32 \
  --gamma-max "$GAMMA_MAX" \
  --theta-gamma-init "$THETA_GAMMA_INIT" \
  "${COMMON_ARGS[@]}"
