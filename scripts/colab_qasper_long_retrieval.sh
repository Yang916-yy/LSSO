#!/usr/bin/env bash
set -uo pipefail

# Colab/A100-heavy QASPER long-document retrieval runner.
#
# This is for experiments that are inconvenient on the local 16GB GPU:
# QASPER evidence/chunk retrieval with doc_len 1024/2048, multiple seeds.
#
# Colab usage:
#   !git clone <your-lsso-repo-url> /content/LSSO
#   %cd /content/LSSO
#   !SEEDS="1 2 3" DOC_LENS="1024" bash scripts/colab_qasper_long_retrieval.sh
#
# Longer-context extension:
#   !SEEDS="1 2 3" DOC_LENS="1024 2048" DEPTH=8 HEADS=8 BATCH_SIZE=8 GRAD_ACCUM=4 bash scripts/colab_qasper_long_retrieval.sh

REPO_DIR="${REPO_DIR:-$(pwd)}"
if [[ -z "${OUT_DIR:-}" ]]; then
  if [[ -d "/content/drive/MyDrive" ]]; then
    OUT_DIR="/content/drive/MyDrive/lsso_qasper_long_runs"
  else
    OUT_DIR="/content/lsso_qasper_long_runs"
  fi
fi
CACHE_DIR="${CACHE_DIR:-/content/lsso_qasper_cache}"
DONE_DIR="${DONE_DIR:-${OUT_DIR}/done}"
CONSOLE_DIR="${CONSOLE_DIR:-${OUT_DIR}/console}"

SEEDS="${SEEDS:-1 2 3}"
MODELS="${MODELS:-mha lsso16 lsso32}"
DOC_LENS="${DOC_LENS:-1024}"

DIM="${DIM:-256}"
DEPTH="${DEPTH:-8}"
HEADS="${HEADS:-8}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_QUERY_LEN="${MAX_QUERY_LEN:-96}"
CHUNK_WORDS="${CHUNK_WORDS:-420}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-80}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
TEMPERATURE="${TEMPERATURE:-0.05}"
MAX_TRAIN_PAIRS="${MAX_TRAIN_PAIRS:-0}"
MAX_EVAL_QUERIES="${MAX_EVAL_QUERIES:-0}"
MAX_EVAL_CORPUS="${MAX_EVAL_CORPUS:-0}"
NUM_WORKERS="${NUM_WORKERS:-2}"
GAMMA_MAX="${GAMMA_MAX:-0.3}"
THETA_GAMMA_INIT="${THETA_GAMMA_INIT:--4.0}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"

cd "${REPO_DIR}"
mkdir -p "${OUT_DIR}" "${CACHE_DIR}" "${DONE_DIR}" "${CONSOLE_DIR}"

echo "===== Colab QASPER long retrieval ====="
date -Is
pwd
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  python -m pip install -q -U pip
  python -m pip install -q -U datasets transformers accelerate tqdm numpy
fi

model_args() {
  local model="$1"
  case "${model}" in
    mha)
      echo "--mixer mha --rank 16"
      ;;
    lsso16)
      echo "--mixer lsso --rank 16"
      ;;
    lsso32)
      echo "--mixer lsso --rank 32"
      ;;
    *)
      echo "ERROR: unknown model ${model}; supported: mha lsso16 lsso32" >&2
      return 1
      ;;
  esac
}

run_one() {
  local model="$1"
  local seed="$2"
  local doc_len="$3"
  local label="qasper_${model}_dim${DIM}_L${DEPTH}_h${HEADS}_doc${doc_len}_seed${seed}"
  local done_file="${DONE_DIR}/${label}.done"
  local status_file="${DONE_DIR}/${label}.status"
  local console="${CONSOLE_DIR}/${label}.console.log"

  if [[ -f "${done_file}" ]]; then
    echo "===== SKIP done ${label} ====="
    return 0
  fi

  local extra
  extra="$(model_args "${model}")" || return 1

  echo "===== START ${label} $(date -Is) =====" | tee "${console}"
  set +e
  python scripts/kaggle_qasper_lsso_retrieval.py \
    ${extra} \
    --dim "${DIM}" \
    --depth "${DEPTH}" \
    --num-heads "${HEADS}" \
    --max-query-len "${MAX_QUERY_LEN}" \
    --max-doc-len "${doc_len}" \
    --chunk-words "${CHUNK_WORDS}" \
    --chunk-overlap "${CHUNK_OVERLAP}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --temperature "${TEMPERATURE}" \
    --max-train-pairs "${MAX_TRAIN_PAIRS}" \
    --max-eval-queries "${MAX_EVAL_QUERIES}" \
    --max-eval-corpus "${MAX_EVAL_CORPUS}" \
    --num-workers "${NUM_WORKERS}" \
    --gamma-max "${GAMMA_MAX}" \
    --theta-gamma-init "${THETA_GAMMA_INIT}" \
    --out-dir "${OUT_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    --amp \
    --no-compile \
    2>&1 | tee -a "${console}"
  local status="${PIPESTATUS[0]}"
  set +e

  echo "===== END ${label} status=${status} $(date -Is) =====" | tee -a "${console}"
  echo "${status}" > "${status_file}"
  if [[ "${status}" == "0" ]]; then
    touch "${done_file}"
  fi
}

for doc_len in ${DOC_LENS}; do
  for seed in ${SEEDS}; do
    for model in ${MODELS}; do
      run_one "${model}" "${seed}" "${doc_len}"
    done
  done
done

echo "===== Logs ====="
find "${OUT_DIR}" -maxdepth 1 -name "*.jsonl" -print | sort
echo "outputs: ${OUT_DIR}"
