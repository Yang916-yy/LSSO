#!/usr/bin/env bash
# Historical unsupported launcher; retained only for provenance.
set -euo pipefail

cd /root/LSSO
mkdir -p runs/cifar100_vision_llama_s_20ep

exec .venv/bin/python experiments/cv_vit_rrlsso_cifar100.py \
  --backbone vision-llama-s \
  --models mha lsso rrlsso \
  --dataset cifar100 \
  --data-dir /root/LSSO-data/cifar100 \
  --data-archive /mnt/d/LSSO-data/cifar-100-python.tar.gz \
  --out-dir runs/cifar100_vision_llama_s_20ep \
  --epochs 20 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --image-size 32 \
  --patch-size 4 \
  --rank 32 \
  --lr 5e-4 \
  --min-lr 1e-5 \
  --warmup-epochs 2 \
  --weight-decay 0.05 \
  --label-smoothing 0.1 \
  --grad-clip 1.0 \
  --dtype bf16 \
  --num-workers 4 \
  --seed 1234
