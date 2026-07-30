#!/usr/bin/env bash

# Queue the 2-GPU 20..1000 tokens/movement Frogger run.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/frogger_range_qwen3_4b_rl_stream_2_gpu.sh"

mkdir -p "${REPO_ROOT}/.cache/slurm-out"
cd "${REPO_ROOT}"

JOB_ID="$(sbatch --parsable "${RUN_SCRIPT}")"
JOB_NUMBER="${JOB_ID%%;*}"
echo "Queued Frogger speed-range run as Slurm job ${JOB_ID}."
echo "Log: ${REPO_ROOT}/.cache/slurm-out/frogger-range-${JOB_NUMBER}.out"
