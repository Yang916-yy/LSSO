#!/usr/bin/env bash
set -euo pipefail

cd /root/LSSO
source /mnt/d/LSSO/.venv/bin/activate

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

CACHE_DIR="/root/LSSO/data/food101-sdvae256-latents"
STAMP="$(date +%Y%m%d-%H%M%S)"

COMMON=(
  --latent-cache "$CACHE_DIR"
  --image-size 256
  --patch-size 2
  --hidden-size 192
  --depth 8
  --heads 6
  --batch-size 128
  --workers 0
  --epochs 50
  --val-batches 40
  --amp
  --ema
  --ema-decay 0.999
  --seed 1
)

python train_latent_dit_tiny.py \
  "${COMMON[@]}" \
  --mixer lsso \
  --rank 16 \
  > "runs/${STAMP}_latentdit_food101_256_cached_lsso_r16_d192_L8_50epoch_ema.out" \
  2> "runs/${STAMP}_latentdit_food101_256_cached_lsso_r16_d192_L8_50epoch_ema.err"

python train_latent_dit_tiny.py \
  "${COMMON[@]}" \
  --mixer mha \
  --rank 16 \
  > "runs/${STAMP}_latentdit_food101_256_cached_mha_d192_L8_50epoch_ema.out" \
  2> "runs/${STAMP}_latentdit_food101_256_cached_mha_d192_L8_50epoch_ema.err"
