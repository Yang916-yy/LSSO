#!/usr/bin/env bash
set -euo pipefail

cd /root/LSSO
source /mnt/d/LSSO/.venv/bin/activate
mkdir -p runs

EPOCHS=60 \
BATCH_SIZE=256 \
MODEL_DIM=96 \
DEPTH=3 \
HEADS=3 \
GAMMA_MAX=0.3 \
THETA_GAMMA_INIT=-4 \
RANK_VALUES="16 32" \
bash scripts/run_cifar_long_lsso.sh cifar10 \
  2>&1 | tee runs/cifar10_lsso_no_inner_dropout_long60.console.log
