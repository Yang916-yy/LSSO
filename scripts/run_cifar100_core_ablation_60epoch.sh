#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/LSSO
source .venv/bin/activate

RUN_DIR="runs/cifar100_core_ablation_60epoch_20260602"
mkdir -p "$RUN_DIR/console"

common=(
  --dataset cifar100
  --run-dir "$RUN_DIR"
  --epochs 60
  --batch-size 256
  --dim 96
  --depth 3
  --num-heads 3
  --num-workers 2
  --amp
  --gamma-max 0.3
  --theta-gamma-init -4
  --seed 1
)

run_one() {
  local label="$1"
  shift
  local log="$RUN_DIR/console/${label}.console.log"
  echo "START ${label} $(date -Iseconds)" | tee "$log"
  python train_cifar.py "${common[@]}" "$@" 2>&1 | tee -a "$log"
  echo "DONE ${label} $(date -Iseconds)" | tee -a "$log"
}

run_one "lsso_r32_no_global" --mixer lsso-no-global --rank 32
run_one "lsso_r32_fixed_mu_gamma" --mixer lsso --rank 32 --fixed-mu-gamma
run_one "lsso_r32_no_u_rms_norm" --mixer lsso --rank 32 --no-u-rms-norm
