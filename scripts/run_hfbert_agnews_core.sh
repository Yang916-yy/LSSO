#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-bert-base-uncased}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LEN="${MAX_LEN:-256}"
RANK="${RANK:-16}"
GAMMA_MAX="${GAMMA_MAX:-0.3}"
THETA_GAMMA_INIT="${THETA_GAMMA_INIT:--4}"

COMMON_ARGS=(
  --dataset ag_news
  --model-name "$MODEL_NAME"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --max-len "$MAX_LEN"
  --amp
)

python train_hf_bert.py --mixer mha "${COMMON_ARGS[@]}"
python train_hf_bert.py \
  --mixer lsso \
  --rank "$RANK" \
  --gamma-max "$GAMMA_MAX" \
  --theta-gamma-init "$THETA_GAMMA_INIT" \
  "${COMMON_ARGS[@]}"
