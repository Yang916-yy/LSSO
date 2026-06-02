#!/usr/bin/env bash
set -euo pipefail

python train_hf_bert.py \
  --dataset ag_news \
  --model-name bert-base-uncased \
  --tokenizer-name bert-base-uncased \
  --mixer lsso \
  --rank 16 \
  --max-len 128 \
  --epochs 1 \
  --batch-size 8 \
  --num-workers 0 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  --local-files-only \
  --amp
