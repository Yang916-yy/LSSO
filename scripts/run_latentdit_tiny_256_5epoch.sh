#!/usr/bin/env bash
set -euo pipefail

cd /root/LSSO
source /mnt/d/LSSO/.venv/bin/activate

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

DATA_ROOT="/mnt/c/Users/chaoy/Desktop/newexp/newexp/data/tiny-imagenet-200"
STAMP="$(date +%Y%m%d-%H%M%S)"

COMMON=(
  --data-root "$DATA_ROOT"
  --local-files-only
  --image-size 256
  --patch-size 2
  --hidden-size 96
  --depth 2
  --heads 6
  --batch-size 32
  --workers 8
  --epochs 5
  --val-batches 16
  --amp
  --seed 1
)

python train_latent_dit_tiny.py \
  "${COMMON[@]}" \
  --mixer lsso \
  --rank 16 \
  > "runs/${STAMP}_latentdit_tiny256_lsso_r16_5epoch.out" \
  2> "runs/${STAMP}_latentdit_tiny256_lsso_r16_5epoch.err"

python train_latent_dit_tiny.py \
  "${COMMON[@]}" \
  --mixer mha \
  --rank 16 \
  > "runs/${STAMP}_latentdit_tiny256_mha_5epoch.out" \
  2> "runs/${STAMP}_latentdit_tiny256_mha_5epoch.err"
