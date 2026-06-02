#!/usr/bin/env bash
set -u

cd /mnt/d/LSSO
source /mnt/d/LSSO/.venv/bin/activate 2>/dev/null || true

EPOCHS="${EPOCHS:-10}"
SEEDS="${SEEDS:-1}"
RUN_DIR="${RUN_DIR:-runs}"
CONSOLE_DIR="${CONSOLE_DIR:-runs/retrieval_main}"
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

if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("mamba_ssm") else 1)
PY
then
  BIMAMBA_AVAILABLE=1
else
  BIMAMBA_AVAILABLE=0
fi

run_one() {
  local dataset="$1"
  local mixer="$2"
  local rank="$3"
  local seed="$4"
  local label="${dataset}_${mixer}_r${rank}_seed${seed}"
  local console="${CONSOLE_DIR}/${label}.console.log"

  echo "===== START ${label} $(date -Is) ====="
  python train_bertstyle_retrieval.py \
    --dataset "$dataset" \
    --mixer "$mixer" \
    --rank "$rank" \
    --seed "$seed" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$console"
  local status="${PIPESTATUS[0]}"
  echo "===== END ${label} status=${status} $(date -Is) ====="
  return "$status"
}

for seed in $SEEDS; do
  for dataset in fiqa nfcorpus scifact; do
    run_one "$dataset" mha 16 "$seed" || true
    run_one "$dataset" nystrom 16 "$seed" || true
    if [[ "$BIMAMBA_AVAILABLE" == "1" ]]; then
      run_one "$dataset" bimamba 16 "$seed" || true
    else
      echo "===== SKIP ${dataset}_bimamba_seed${seed}: official mamba_ssm is not installed =====" \
        | tee "${CONSOLE_DIR}/${dataset}_bimamba_seed${seed}.console.log"
    fi
    run_one "$dataset" lsso 16 "$seed" || true
    run_one "$dataset" lsso 32 "$seed" || true
  done
done
