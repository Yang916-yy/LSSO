#!/usr/bin/env bash
set -euo pipefail

cd /root/LSSO
source /mnt/d/LSSO/.venv/bin/activate
mkdir -p runs

common=(
  --dataset cifar100
  --mixer lsso
  --epochs 60
  --batch-size 256
  --dim 96
  --depth 3
  --gamma-max 0.3
  --theta-gamma-init -4
  --num-workers 2
  --amp
)

# Compare against existing h3-r32. These variants test whether more heads with
# smaller per-head rank can keep accuracy while reducing r^2 solve work.
configs=(
  "6 16"
  "12 8"
  "4 16"
  "8 8"
  "12 4"
)

for cfg in "${configs[@]}"; do
  read -r heads rank <<< "$cfg"
  echo "cifar100 head-rank sweep heads=${heads} rank=${rank}" >&2
  python train_cifar.py "${common[@]}" \
    --num-heads "$heads" \
    --rank "$rank" \
    2>&1 | tee "runs/cifar100_lsso_h${heads}_r${rank}_d96_L3_60epoch.console.log"
done
