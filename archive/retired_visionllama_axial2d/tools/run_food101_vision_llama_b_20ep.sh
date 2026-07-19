#!/usr/bin/env bash
# Historical unsupported launcher; retained only for provenance.
set -euo pipefail

cd /root/LSSO
mkdir -p runs/food101_vision_llama_b_20ep

exec .venv/bin/python experiments/cv_vit_rrlsso_cifar100.py \
  --backbone vision-llama-b \
  --models mha lsso rrlsso \
  --dataset food101 \
  --data-dir /mnt/d/LSSO-data/torchvision \
  --data-archive /mnt/d/LSSO-data/food-101.tar.gz \
  --out-dir runs/food101_vision_llama_b_20ep \
  --epochs 20 \
  --batch-size 64 \
  --grad-accum-steps 2 \
  --eval-batch-size 128 \
  --image-size 224 \
  --patch-size 16 \
  --rank 32 \
  --lr 5e-4 \
  --min-lr 1e-5 \
  --warmup-epochs 2 \
  --weight-decay 0.05 \
  --label-smoothing 0.1 \
  --grad-clip 1.0 \
  --dtype bf16 \
  --num-workers 8 \
  --seed 1234 \
  --auto-resume
