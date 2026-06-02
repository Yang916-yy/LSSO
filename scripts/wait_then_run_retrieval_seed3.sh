#!/usr/bin/env bash
set -u

cd /mnt/d/LSSO

WAIT_PID="${WAIT_PID:-}"
if [[ -n "$WAIT_PID" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
fi

env \
  PYTHONUNBUFFERED=1 \
  EPOCHS="${EPOCHS:-20}" \
  SEEDS=3 \
  RUN_DIR="${RUN_DIR:-runs/retrieval_paper_main_seeds23}" \
  CONSOLE_DIR="${CONSOLE_DIR:-runs/retrieval_paper_main_seeds23/console}" \
  BATCH_SIZE="${BATCH_SIZE:-64}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}" \
  MAX_TRAIN_PAIRS="${MAX_TRAIN_PAIRS:-50000}" \
  MAX_EVAL_QUERIES="${MAX_EVAL_QUERIES:-1000}" \
  bash scripts/run_retrieval_paper_main_seeds.sh
