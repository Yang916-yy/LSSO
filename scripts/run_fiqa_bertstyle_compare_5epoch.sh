#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

COMMON_ARGS=(
  --dataset fiqa
  --tokenizer-name bert-base-uncased
  --dim 256
  --depth 4
  --num-heads 4
  --epochs 5
  --batch-size 128
  --eval-batch-size 256
  --max-query-len 64
  --max-doc-len 256
  --max-train-pairs 0
  --max-eval-queries 0
  --max-corpus-docs 20000
  --num-workers 0
  --amp
  --offline
  --local-files-only
)

./.venv/bin/python train_bertstyle_retrieval.py \
  "${COMMON_ARGS[@]}" \
  --mixer lsso \
  --rank 16

./.venv/bin/python train_bertstyle_retrieval.py \
  "${COMMON_ARGS[@]}" \
  --mixer mha \
  --rank 16
