#!/usr/bin/env bash
set -euo pipefail

cd /root/LSSO
source /mnt/d/LSSO/.venv/bin/activate

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

DATA_ROOT="/root/LSSO/data/food-101"
CACHE_DIR="/root/LSSO/data/food101-sdvae256-latents"
STAMP="$(date +%Y%m%d-%H%M%S)"

if [ ! -f "${CACHE_DIR}/meta.json" ]; then
  python cache_tiny_latents.py \
    --dataset food101 \
    --data-root "$DATA_ROOT" \
    --out-dir "$CACHE_DIR" \
    --local-files-only \
    --image-size 256 \
    --batch-size 64 \
    --workers 8 \
    --shard-size 4096 \
    --amp \
    > "runs/${STAMP}_cache_food101_sdvae256_latents.out" \
    2> "runs/${STAMP}_cache_food101_sdvae256_latents.err"
fi

COMMON=(
  --latent-cache "$CACHE_DIR"
  --image-size 256
  --patch-size 2
  --hidden-size 96
  --depth 2
  --heads 6
  --batch-size 256
  --workers 0
  --epochs 5
  --val-batches 40
  --amp
  --seed 1
)

python train_latent_dit_tiny.py \
  "${COMMON[@]}" \
  --mixer lsso \
  --rank 16 \
  > "runs/${STAMP}_latentdit_food101_256_cached_lsso_r16_5epoch.out" \
  2> "runs/${STAMP}_latentdit_food101_256_cached_lsso_r16_5epoch.err"

python train_latent_dit_tiny.py \
  "${COMMON[@]}" \
  --mixer mha \
  --rank 16 \
  > "runs/${STAMP}_latentdit_food101_256_cached_mha_5epoch.out" \
  2> "runs/${STAMP}_latentdit_food101_256_cached_mha_5epoch.err"
