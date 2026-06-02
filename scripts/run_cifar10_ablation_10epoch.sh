#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_DIR="${RUN_DIR:-runs/cifar10_ablation_10epoch_20260602}"
mkdir -p "$RUN_DIR/console"

COMMON=(
  --dataset cifar10
  --data-dir data
  --run-dir "$RUN_DIR"
  --epochs 10
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
  --seed 11
)

run_one() {
  local name="$1"
  local mixer="$2"
  local rank="$3"
  local done_file="$RUN_DIR/${name}.done"
  local log="$RUN_DIR/console/${name}.console.log"

  if [[ -f "$done_file" ]]; then
    echo "SKIP $name already done" | tee -a "$RUN_DIR/master.log"
    return
  fi

  echo "START $name mixer=$mixer rank=$rank $(date -Iseconds)" | tee -a "$RUN_DIR/master.log" "$log"
  local start
  start=$(date +%s)
  python train_cifar.py "${COMMON[@]}" --mixer "$mixer" --rank "$rank" 2>&1 | tee -a "$log"
  local end
  end=$(date +%s)
  echo "ELAPSED_SECONDS=$((end - start))" | tee -a "$RUN_DIR/master.log" "$log"
  touch "$done_file"
  echo "DONE $name $(date -Iseconds)" | tee -a "$RUN_DIR/master.log" "$log"
}

run_one "mha_seed11" "mha" 16
run_one "lsso_r16_seed11" "lsso" 16
run_one "lsso_r32_seed11" "lsso" 32
run_one "lsso_r32_no_global_seed11" "lsso-no-global" 32
