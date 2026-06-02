#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_DIR="${RUN_DIR:-runs/cifar10_pytorch_vs_custom_3epoch_20260602}"
mkdir -p "$RUN_DIR/console"

COMMON=(
  --dataset cifar10
  --data-dir data
  --run-dir "$RUN_DIR"
  --epochs 3
  --batch-size 256
  --num-workers 8
  --dim 96
  --depth 3
  --num-heads 6
  --patch-size 2
  --image-size 32
  --lr 3e-4
  --weight-decay 0.05
  --amp
  --seed 17
  --mixer lsso
  --rank 32
)

run_one() {
  local name="$1"
  local custom_flag="$2"
  local log="$RUN_DIR/console/${name}.console.log"
  echo "START $name $custom_flag $(date -Iseconds)" | tee "$log"
  local start
  start=$(date +%s)
  python train_cifar.py "${COMMON[@]}" "$custom_flag" 2>&1 | tee -a "$log"
  local end
  end=$(date +%s)
  echo "ELAPSED_SECONDS=$((end - start))" | tee -a "$log"
}

run_one "lsso_r32_custom" "--use-custom-backward"
run_one "lsso_r32_pytorch" "--no-use-custom-backward"
