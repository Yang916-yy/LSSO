#!/usr/bin/env bash
set -u

cd /mnt/d/LSSO
source /mnt/d/LSSO/.venv/bin/activate

EPOCHS="${EPOCHS:-20}"
SEEDS="${SEEDS:-1}"
SEEDS="${SEEDS//,/ }"
DATASETS="${DATASETS:-fiqa scifact}"
DATASETS="${DATASETS//,/ }"
RUN_DIR="${RUN_DIR:-runs/retrieval_ablation_seed1}"
CONSOLE_DIR="${CONSOLE_DIR:-${RUN_DIR}/console}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
MAX_EVAL_QUERIES="${MAX_EVAL_QUERIES:-1000}"
MAX_TRAIN_PAIRS="${MAX_TRAIN_PAIRS:-50000}"
DEVICE="${DEVICE:-cuda}"
AMP_FLAG="${AMP_FLAG:---amp}"

mkdir -p "$RUN_DIR" "$CONSOLE_DIR"

COMMON_ARGS=(
  --run-dir "$RUN_DIR"
  --dim 256
  --depth 8
  --num-heads 8
  --max-query-len 64
  --max-doc-len 512
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --lr 3e-4
  --weight-decay 0.01
  --warmup-ratio 0.05
  --temperature 0.05
  --max-train-pairs "$MAX_TRAIN_PAIRS"
  --max-eval-queries "$MAX_EVAL_QUERIES"
  --max-corpus-docs 0
  --candidate-negatives 0
  --num-workers 0
  --device "$DEVICE"
  $AMP_FLAG
)

run_one() {
  local dataset="$1"
  local variant="$2"
  local mixer="$3"
  local rank="$4"
  local seed="$5"
  shift 5
  local label="${dataset}_${variant}_seed${seed}"
  local done="${RUN_DIR}/${label}.done"
  local console="${CONSOLE_DIR}/${label}.console.log"

  if [[ -f "$done" ]]; then
    echo "===== SKIP ${label}: done marker exists ====="
    return 0
  fi

  echo "===== START ${label} $(date -Is) ====="
  PYTHONPATH=. python train_bertstyle_retrieval.py \
    --dataset "$dataset" \
    --mixer "$mixer" \
    --rank "$rank" \
    --seed "$seed" \
    "${COMMON_ARGS[@]}" \
    "$@" \
    2>&1 | tee "$console"
  local status="${PIPESTATUS[0]}"
  echo "===== END ${label} status=${status} $(date -Is) ====="
  if [[ "$status" == "0" ]]; then
    date -Is > "$done"
  fi
  return "$status"
}

for seed in $SEEDS; do
  for dataset in $DATASETS; do
    run_one "$dataset" "lsso_r32_no_global" "lsso-no-global" 32 "$seed" || true
    run_one "$dataset" "lsso_r32_fixed_mu_gamma" "lsso" 32 "$seed" --fixed-mu-gamma || true
    run_one "$dataset" "lsso_r32_no_u_rms_norm" "lsso" 32 "$seed" --no-u-rms-norm || true
    run_one "$dataset" "lsso_r8_full" "lsso" 8 "$seed" || true
    run_one "$dataset" "lsso_r4_full" "lsso" 4 "$seed" || true
  done
done
