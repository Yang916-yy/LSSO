#!/usr/bin/env bash
set -euo pipefail

cd /root/LSSO
source /mnt/d/LSSO/.venv/bin/activate
mkdir -p runs

EPOCHS=5
BATCH_SIZE=256
MODEL_DIM=96
DEPTH=3
HEADS=3

EPOCHS="$EPOCHS" \
BATCH_SIZE="$BATCH_SIZE" \
MODEL_DIM="$MODEL_DIM" \
DEPTH="$DEPTH" \
HEADS="$HEADS" \
bash scripts/run_cifar_5epoch_core.sh cifar10 \
  2>&1 | tee runs/cifar10_core_no_inner_dropout_5epoch.console.log
