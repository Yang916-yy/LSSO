#!/usr/bin/env bash
set -u

cd /mnt/d/LSSO
source /mnt/d/LSSO/.venv/bin/activate 2>/dev/null || true

RUN_DIR="${RUN_DIR:-runs/rank_pruning_main_seed1}"
CONSOLE_DIR="${CONSOLE_DIR:-${RUN_DIR}/console}"
KEEP_RANKS="${KEEP_RANKS:-0,24,16,12,8,4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
AMP_FLAG="${AMP_FLAG:---amp}"
RESUME_FLAG="${RESUME_FLAG:---resume}"

mkdir -p "$RUN_DIR" "$CONSOLE_DIR"

run_one() {
  local name="$1"
  local ckpt="$2"
  local console="${CONSOLE_DIR}/${name}.console.log"

  echo "===== START ${name} $(date -Is) ====="
  PYTHONPATH=/mnt/d/LSSO python eval_retrieval_rank_pruning.py \
    --checkpoint "$ckpt" \
    --run-dir "$RUN_DIR" \
    --keep-ranks "$KEEP_RANKS" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    $RESUME_FLAG \
    $AMP_FLAG \
    2>&1 | tee "$console"
  local status="${PIPESTATUS[0]}"
  echo "===== END ${name} status=${status} $(date -Is) ====="
  return "$status"
}

run_one fiqa_lsso_r32 \
  runs/retrieval_main_20ep_seed1/20260531-175727_bertstyle_retr_fiqa_lsso_r32_g0.3_tgi-4.0_d256_L8_h8_lend512_s1.pt || true

run_one nfcorpus_lsso_r32 \
  runs/retrieval_main_missing_20ep_seed1/20260531-211546_bertstyle_retr_nfcorpus_lsso_r32_g0.3_tgi-4.0_d256_L8_h8_lend512_s1.pt || true

run_one scifact_lsso_r32 \
  runs/retrieval_main_missing_20ep_seed1/20260531-215230_bertstyle_retr_scifact_lsso_r32_g0.3_tgi-4.0_d256_L8_h8_lend512_s1.pt || true
