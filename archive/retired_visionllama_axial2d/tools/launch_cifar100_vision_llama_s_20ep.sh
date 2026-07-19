#!/usr/bin/env bash
# Historical unsupported launcher; retained only for provenance.
set -euo pipefail

run_dir=/root/LSSO/runs/cifar100_vision_llama_s_20ep
mkdir -p "${run_dir}"
nohup bash /root/LSSO/tools/run_cifar100_vision_llama_s_20ep.sh \
  >"${run_dir}/train.log" 2>&1 &
pid=$!
echo "${pid}" >"${run_dir}/launcher.pid"
echo "${pid}"
