#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_DIR="runs/package_train_sanity_20260602"
mkdir -p "$RUN_DIR/console"

COMMON=(
  --dataset cifar10
  --data-dir data
  --run-dir "$RUN_DIR"
  --epochs 1
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
  --seed 7
)

for spec in "mha 16" "lsso 16" "lsso 32"; do
  read -r mixer rank <<<"$spec"
  log="$RUN_DIR/console/${mixer}_r${rank}.console.log"
  echo "START mixer=$mixer rank=$rank $(date -Iseconds)" | tee "$log"
  start=$(date +%s)
  python train_cifar.py "${COMMON[@]}" --mixer "$mixer" --rank "$rank" 2>&1 | tee -a "$log"
  end=$(date +%s)
  echo "ELAPSED_SECONDS=$((end - start))" | tee -a "$log"
done
