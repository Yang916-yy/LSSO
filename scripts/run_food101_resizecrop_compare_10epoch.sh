#!/usr/bin/env bash
set -euo pipefail

cd /root/LSSO

bash scripts/run_food101_mha_10epoch.sh
bash scripts/run_food101_lsso_r16_10epoch.sh
