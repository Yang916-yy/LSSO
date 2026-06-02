#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/LSSO

RUN_DIR="${RUN_DIR:-runs/retrieval_ablation_seed1}"
mkdir -p "$RUN_DIR/console"

export SEEDS="${SEEDS:-1}"
export EPOCHS="${EPOCHS:-20}"

nohup bash scripts/run_retrieval_ablation_seed1.sh \
  > "$RUN_DIR/master.log" \
  2>&1 \
  < /dev/null &

pid="$!"
echo "$pid" > "$RUN_DIR/master.pid"
echo "$pid"
