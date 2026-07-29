#!/usr/bin/env bash
#SBATCH --job-name=frogger-range-20-1000
#SBATCH --partition=VZHNG
#SBATCH --nodes=1
#SBATCH --nodelist=watgpu1103
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:vzhnggpu:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=72:00:00
#SBATCH --output=.cache/slurm-out/frogger-range-%j.out

# Train streaming Frogger over a per-prompt 20..1000 tokens/movement range.
# Speeds are sampled log-uniformly and stored in the dataset metadata, so all
# eight GRPO candidates for one prompt share the exact same seed and clock.
# Evaluation uses the same 64 seeds at 1000, 900, ..., 100, 50, and 20
# tokens/movement and logs one reward series per speed.

set -euo pipefail

MICROMAMBA_BIN="$(command -v micromamba || true)"
if [[ -z "${MICROMAMBA_BIN}" ]]; then
    for candidate in \
        "${MAMBA_ROOT_PREFIX:-$HOME/micromamba}/bin/micromamba" \
        "$HOME/.local/bin/micromamba" \
        "$HOME/micromamba/bin/micromamba"; do
        if [[ -x "${candidate}" ]]; then
            MICROMAMBA_BIN="${candidate}"
            break
        fi
    done
fi
if [[ -z "${MICROMAMBA_BIN}" ]]; then
    echo "ERROR: micromamba binary not found" >&2
    exit 1
fi
eval "$("${MICROMAMBA_BIN}" shell hook --shell bash)"
micromamba activate slime

# A full-node allocation makes this cleanup local to this user's exclusive job.
pkill -9 -x sglang || true
ray stop --force || true
pkill -9 -x ray || true
pkill -9 -x python || true
pkill -9 -x python3 || true
sleep 3

set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/slime/examples/realtime" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
    SCRIPT_DIR="${REPO_ROOT}/slime/examples/realtime"
else
    REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
fi
cd "${REPO_ROOT}"

export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$HOME}"
export PYTHONUNBUFFERED=1
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib/stubs:$CONDA_PREFIX/lib/stubs:${LIBRARY_PATH:-}"
export LDFLAGS="-L$CONDA_PREFIX/targets/x86_64-linux/lib/stubs -L$CONDA_PREFIX/lib/stubs ${LDFLAGS:-}"

NVLINK_COUNT="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)"
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

source "${REPO_ROOT}/slime/scripts/models/qwen3-4B.sh"

RUN_DIR="${REPO_ROOT}/.cache/frogger-range-20-1000-stream"
TRAIN_DATA="${ARTIFACT_ROOT}/obstacles-seeds/train_realtime_frogger_stream_20_1000_loguniform.jsonl"
EVAL_DATA="${ARTIFACT_ROOT}/obstacles-seeds/eval_realtime_frogger_stream_range.jsonl"
EVAL_CONFIG="${SCRIPT_DIR}/frogger_range_eval.yaml"
mkdir -p "${RUN_DIR}" "$(dirname -- "${TRAIN_DATA}")"
echo "Artifacts (checkpoints + rollout dumps) -> ${RUN_DIR}"

ensure_dataset() {
    local path="$1"
    local expected_rows="$2"
    shift 2
    if [[ -f "${path}" && "$(wc -l < "${path}")" -eq "${expected_rows}" ]]; then
        echo "Using existing ${expected_rows}-row dataset: ${path}"
        return
    fi
    local temporary="${path}.tmp.${SLURM_JOB_ID:-$$}"
    PYTHONPATH="${REPO_ROOT}/real-time" python3 \
        "${SCRIPT_DIR}/obstacles_data_preprocess.py" \
        --out "${temporary}" \
        --train-size "${expected_rows}" \
        "$@"
    mv -f "${temporary}" "${path}"
}

# Log-uniform cadence sampling gives meaningful coverage to both the 20-token
# fast end and the 1000-token slow end of this 50x range.
ensure_dataset "${TRAIN_DATA}" 20000 \
    --env realtime_frogger_stream \
    --seed 1234 \
    --tokens-per-movement-range 20 1000 \
    --tokens-per-movement-sampling log-uniform

# All eval dataset entries reuse these exact seeds; frogger_range_eval.yaml
# overrides only tokens_per_movement, making cross-speed comparisons paired.
ensure_dataset "${EVAL_DATA}" 64 \
    --env realtime_frogger_stream \
    --seed 999

CKPT_ARGS=(
    --hf-checkpoint "${ARTIFACT_ROOT}/Qwen/Qwen3-4B"
    --ref-load "${ARTIFACT_ROOT}/Qwen/Qwen3-4B_torch_dist"
    --save "${RUN_DIR}/checkpoints/"
    --save-interval 10
    --save-retain-interval 100
    --rotary-base 1000000
)
if [[ -f "${RUN_DIR}/checkpoints/latest_checkpointed_iteration.txt" ]]; then
    echo "Found checkpoint tracker; resuming from ${RUN_DIR}/checkpoints/"
    CKPT_ARGS+=(--load "${RUN_DIR}/checkpoints/")
fi

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_DATA}"
    --input-key prompt
    --label-key seed
    --reward-key score
    --num-rollout 300
    # 64 distinct prompts/clocks per rollout gives the mixed-speed policy update
    # twice the cadence coverage of the fixed-speed launchers. Keeping eight
    # candidates per prompt preserves a useful GRPO baseline for binary rewards.
    --rollout-batch-size 64
    --n-samples-per-prompt 8
    --rollout-max-response-len 16384
    --rollout-temperature 1
    --save-debug-rollout-data "${RUN_DIR}/rollout_dumps/{rollout_id}.pt"
    --global-batch-size 512
    --balance-data
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 17408
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project slime-obstacles
    --wandb-group qwen3-4B-realtime-frogger-stream-range-20-1000
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

CUSTOM_ARGS=(
    --custom-generate-function-path generate_with_obstacles.generate_streaming
    --custom-rm-path generate_with_obstacles.reward_func
)

EVAL_ARGS=(
    # Twelve speed slices make eval expensive, especially at 20 tokens/movement.
    # A paired 64-seed panel every 25 rollouts keeps the signal useful and bounded.
    --eval-interval 25
    --eval-config "${EVAL_CONFIG}"
    --eval-reward-key score
)

# Ray's gRPC stack can abort with the cluster's very high default descriptor limit.
if [[ "$(ulimit -n)" -gt 65535 ]]; then
    ulimit -n 65535
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
DASH_PORT="${DASH_PORT:-8273}"
AGENT_PORT="${AGENT_PORT:-52373}"
RAY_CPUS="${SLURM_CPUS_PER_TASK:-80}"
ray start \
    --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-cpus "${RAY_CPUS}" \
    --num-gpus 4 \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${DASH_PORT}" \
    --dashboard-agent-listen-port="${AGENT_PORT}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${REPO_ROOT}/Megatron-LM:${SCRIPT_DIR}:${REPO_ROOT}/slime:${REPO_ROOT}/real-time\",
    \"ARTIFACT_ROOT\": \"${ARTIFACT_ROOT}\",
    \"TICK_RATE\": \"600\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"LIBRARY_PATH\": \"${LIBRARY_PATH}\",
    \"LDFLAGS\": \"${LDFLAGS}\"
  }
}"

ray job submit --address="http://127.0.0.1:${DASH_PORT}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${REPO_ROOT}/slime/train.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}" \
    "${EVAL_ARGS[@]}"
