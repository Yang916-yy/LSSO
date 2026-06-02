#!/usr/bin/env bash
set -uo pipefail

# Colab retrieval main-table runner.
#
# Typical Colab usage:
#   !git clone <your-lsso-repo-url> /content/LSSO
#   %cd /content/LSSO
#   !bash scripts/colab_retrieval_paper.sh
#
# Useful overrides:
#   SEEDS="1 2 3" DATASETS="fiqa nfcorpus scifact" EPOCHS=10 bash scripts/colab_retrieval_paper.sh
#   MODELS="mha nystrom lsso16 lsso32" bash scripts/colab_retrieval_paper.sh

REPO_DIR="${REPO_DIR:-$(pwd)}"
if [[ -z "${RUN_DIR:-}" ]]; then
  if [[ -d "/content/drive/MyDrive" ]]; then
    RUN_DIR="/content/drive/MyDrive/lsso_paper_runs"
  else
    RUN_DIR="/content/lsso_paper_runs"
  fi
fi
DONE_DIR="${DONE_DIR:-${RUN_DIR}/done}"
CONSOLE_DIR="${CONSOLE_DIR:-${RUN_DIR}/console}"

SEEDS="${SEEDS:-1 2 3}"
DATASETS="${DATASETS:-fiqa nfcorpus scifact}"
MODELS="${MODELS:-mha nystrom bimamba lsso16 lsso32}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
MAX_TRAIN_PAIRS="${MAX_TRAIN_PAIRS:-50000}"
MAX_EVAL_QUERIES="${MAX_EVAL_QUERIES:-1000}"
MAX_QUERY_LEN="${MAX_QUERY_LEN:-64}"
MAX_DOC_LEN="${MAX_DOC_LEN:-512}"
DIM="${DIM:-256}"
DEPTH="${DEPTH:-8}"
HEADS="${HEADS:-8}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
TEMPERATURE="${TEMPERATURE:-0.05}"
NUM_WORKERS="${NUM_WORKERS:-2}"
GAMMA_MAX="${GAMMA_MAX:-0.3}"
THETA_GAMMA_INIT="${THETA_GAMMA_INIT:--4.0}"

INSTALL_DEPS="${INSTALL_DEPS:-1}"
INSTALL_MAMBA="${INSTALL_MAMBA:-1}"
AMP_FLAG="${AMP_FLAG:---amp}"
DEVICE="${DEVICE:-cuda}"

cd "${REPO_DIR}"
mkdir -p "${RUN_DIR}" "${DONE_DIR}" "${CONSOLE_DIR}"

echo "===== Colab LSSO retrieval paper run ====="
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
  python -m pip install -q -U transformers datasets accelerate evaluate scikit-learn tqdm numpy
  python -m pip install -q torchvision
fi

if [[ "${INSTALL_MAMBA}" == "1" ]]; then
  echo "===== Try installing official mamba-ssm ====="
  python -m pip install -q -U packaging ninja einops
  python -m pip install -q causal-conv1d mamba-ssm --no-build-isolation || \
    echo "WARNING: official mamba-ssm install failed; BiMamba jobs will be skipped."
fi

if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("mamba_ssm") else 1)
PY
then
  BIMAMBA_AVAILABLE=1
else
  BIMAMBA_AVAILABLE=0
fi

common_args=(
  --run-dir "${RUN_DIR}"
  --tokenizer-name bert-base-uncased
  --dim "${DIM}"
  --depth "${DEPTH}"
  --num-heads "${HEADS}"
  --max-query-len "${MAX_QUERY_LEN}"
  --max-doc-len "${MAX_DOC_LEN}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --warmup-ratio "${WARMUP_RATIO}"
  --grad-clip "${GRAD_CLIP}"
  --temperature "${TEMPERATURE}"
  --max-train-pairs "${MAX_TRAIN_PAIRS}"
  --max-eval-queries "${MAX_EVAL_QUERIES}"
  --max-corpus-docs 0
  --candidate-negatives 0
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --no-offline
  --no-local-files-only
  ${AMP_FLAG}
)

model_to_args() {
  local model="$1"
  case "${model}" in
    mha)
      echo "--mixer mha --rank 16"
      ;;
    performer)
      echo "--mixer performer --rank 16"
      ;;
    nystrom)
      echo "--mixer nystrom --rank 16"
      ;;
    bimamba)
      echo "--mixer bimamba --rank 16"
      ;;
    lsso16)
      echo "--mixer lsso --rank 16 --gamma-max ${GAMMA_MAX} --theta-gamma-init ${THETA_GAMMA_INIT}"
      ;;
    lsso32)
      echo "--mixer lsso --rank 32 --gamma-max ${GAMMA_MAX} --theta-gamma-init ${THETA_GAMMA_INIT}"
      ;;
    *)
      echo "ERROR: unknown model ${model}" >&2
      return 1
      ;;
  esac
}

run_one() {
  local dataset="$1"
  local model="$2"
  local seed="$3"
  local label="${dataset}_${model}_dim${DIM}_L${DEPTH}_h${HEADS}_q${MAX_QUERY_LEN}_doc${MAX_DOC_LEN}_seed${seed}"
  local done_file="${DONE_DIR}/${label}.done"
  local console="${CONSOLE_DIR}/${label}.console.log"
  local status_file="${DONE_DIR}/${label}.status"

  if [[ -f "${done_file}" ]]; then
    echo "===== SKIP done ${label} ====="
    return 0
  fi
  if [[ "${model}" == "bimamba" && "${BIMAMBA_AVAILABLE}" != "1" ]]; then
    echo "===== SKIP ${label}: official mamba_ssm is unavailable =====" | tee "${console}"
    echo "skipped_mamba_unavailable" > "${status_file}"
    return 0
  fi

  local extra
  extra="$(model_to_args "${model}")" || return 1

  echo "===== START ${label} $(date -Is) =====" | tee "${console}"
  set +e
  python train_bertstyle_retrieval.py \
    --dataset "${dataset}" \
    --seed "${seed}" \
    ${extra} \
    "${common_args[@]}" \
    2>&1 | tee -a "${console}"
  local status="${PIPESTATUS[0]}"
  set +e

  echo "===== END ${label} status=${status} $(date -Is) =====" | tee -a "${console}"
  echo "${status}" > "${status_file}"
  if [[ "${status}" == "0" ]]; then
    touch "${done_file}"
  fi
  return 0
}

for seed in ${SEEDS}; do
  for dataset in ${DATASETS}; do
    for model in ${MODELS}; do
      run_one "${dataset}" "${model}" "${seed}"
      python scripts/summarize_retrieval_runs.py "${RUN_DIR}"/*.jsonl > "${RUN_DIR}/summary.tsv" 2>/dev/null || true
    done
  done
done

python scripts/summarize_retrieval_runs.py "${RUN_DIR}"/*.jsonl > "${RUN_DIR}/summary.tsv" 2>/dev/null || true
echo "===== Summary ====="
cat "${RUN_DIR}/summary.tsv" 2>/dev/null || true
echo "outputs: ${RUN_DIR}"
