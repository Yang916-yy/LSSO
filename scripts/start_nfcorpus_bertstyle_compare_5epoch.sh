#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p runs/retrieval_jobs
LOG="runs/retrieval_jobs/bertstyle_nfcorpus_d256_L4_lsso_mha_5ep.log"
PID_FILE="runs/retrieval_jobs/bertstyle_nfcorpus_d256_L4_lsso_mha_5ep.pid"

nohup bash scripts/run_nfcorpus_bertstyle_compare_5epoch.sh > "$LOG" 2>&1 < /dev/null &
echo "$!" > "$PID_FILE"
echo "pid=$(cat "$PID_FILE")"
echo "log=$LOG"
