#!/bin/bash

# Resume version of obstacles_qwen3_4b_rl_stream_2_gpu.sh: continues RL training on
# the STREAMING real-time Frogger env (env tag `realtime_frogger_stream`) from a
# previous run's checkpoints.
#
# Usage:
#   bash obstacles_qwen3_4b_rl_stream_2_gpu_resume.sh <run-id-or-checkpoint-dir>
# where the argument is either the RUN_ID (uuid) of a previous run (resolved to
# <repo root>/.cache/<uuid>/checkpoints) or an explicit path to a checkpoints dir.
# Alternatively set LOAD_DIR in the environment.
#
# The only differences from obstacles_qwen3_4b_rl_stream_2_gpu.sh are:
#   * --load points at the previous run's checkpoints (slime resumes the rollout
#     step from the checkpoint, so it continues where the old run stopped)
#   * --num-rollout is raised 100 -> 400 to train past the original schedule
#   * --override-opt-param-scheduler (see comment in CKPT_ARGS)
# A fresh RUN_DIR is still created: new checkpoints and rollout dumps go to the
# new run's directory, leaving the old run untouched.

eval "$("${MAMBA_ROOT_PREFIX:-$HOME/micromamba}/bin/micromamba" shell hook --shell bash)"
micromamba activate slime
# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

ARTIFACT_ROOT=${ARTIFACT_ROOT:-$HOME}

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1

# Help flashinfer's JIT compiler find libcuda: the conda toolchain is sandboxed and
# does not search system lib dirs, and the env's lib64/stubs path does not exist.
# Point the linker at the existing conda stub dirs so -lcuda resolves.
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib/stubs:$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH"
export LDFLAGS="-L$CONDA_PREFIX/targets/x86_64-linux/lib/stubs -L$CONDA_PREFIX/lib/stubs $LDFLAGS"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Repo root that contains slime/, real-time/, Megatron-LM/ (this script lives at
# slime/examples/realtime/), used to put the obstacles `environment` package on
# PYTHONPATH.
# When submitted via sbatch, BASH_SOURCE[0] resolves to a SLURM-managed temp copy of
# the script, making SCRIPT_DIR wrong. Use SLURM_SUBMIT_DIR (the directory where sbatch
# was invoked, i.e. the repo root) to recompute both paths in that case.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
    SCRIPT_DIR="${REPO_ROOT}/slime/examples/realtime"
else
    REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
fi
source "slime/scripts/models/qwen3-4B.sh"

# Resolve the checkpoint dir to resume from: $1 (RUN_ID or path) or $LOAD_DIR.
LOAD_DIR="${1:-${LOAD_DIR:-}}"
if [ -z "${LOAD_DIR}" ]; then
    echo "ERROR: pass the previous RUN_ID or a checkpoints dir as \$1 (or set LOAD_DIR)." >&2
    echo "Usage: bash $0 <run-id-or-checkpoint-dir>" >&2
    exit 1
fi
# A bare RUN_ID resolves to the standard per-run artifact layout.
if [ ! -d "${LOAD_DIR}" ]; then
    LOAD_DIR="${REPO_ROOT}/.cache/${LOAD_DIR}/checkpoints"
fi
if [ ! -d "${LOAD_DIR}" ]; then
    echo "ERROR: checkpoint dir not found: ${LOAD_DIR}" >&2
    exit 1
fi
echo "Resuming from: ${LOAD_DIR}"

# Per-run artifact directory: a fresh UUID is generated each time the run starts,
# and every local artifact (model checkpoints + rollout/completion dumps -- i.e.
# everything except the wandb logs) is written under <repo root>/.cache/<uuid>/.
# Anchored to REPO_ROOT so it lands in the same place regardless of cwd. This
# keeps runs isolated and easy to clean up.
RUN_ID="$(uuidgen 2>/dev/null || python3 -c 'import uuid; print(uuid.uuid4())')"
RUN_DIR="${REPO_ROOT}/.cache/${RUN_ID}"
mkdir -p "${RUN_DIR}"
echo "RUN_ID: ${RUN_ID}"
echo "Artifacts (checkpoints + rollout dumps) -> ${RUN_DIR}"

CKPT_ARGS=(
   --hf-checkpoint ${ARTIFACT_ROOT}/Qwen/Qwen3-4B
   --ref-load ${ARTIFACT_ROOT}/Qwen/Qwen3-4B_torch_dist
   --load ${LOAD_DIR}
   --save ${RUN_DIR}/checkpoints/
   --save-interval 100
   --rotary-base 1000000
   # We bump --num-rollout below to train past the original schedule, which
   # changes the derived lr_decay_steps and would otherwise fail the scheduler's
   # checkpoint-consistency assert. Override it to use the new schedule values
   # (LR is constant here anyway; step counting still resumes from num_steps).
   --override-opt-param-scheduler
)

# Build the train seed set (once) with the streaming env tag:
#   PYTHONPATH=./real-time python3 slime/examples/realtime/obstacles_data_preprocess.py \
#       --env realtime_frogger_stream --train-size 20000 --seed 1234 \
#       --out $HOME/obstacles-seeds/train_realtime_frogger_stream.jsonl
ROLLOUT_ARGS=(
   # Seed dataset produced by obstacles_data_preprocess.py. Each row is
   # {"prompt": <streaming game rules>, "seed": <int>, "metadata": {"env": "realtime_frogger_stream"}};
   # the seed arrives on sample.label and the env is reconstructed from it at rollout time.
   --prompt-data ${ARTIFACT_ROOT}/obstacles-seeds/train_realtime_frogger_stream.jsonl
   --input-key prompt
   --label-key seed
   # Single env here, so batch balancing/round-robin is a no-op; leaving shuffle off
   # matches the other obstacles scripts (see obstacles_data_preprocess.py).
   --reward-key score
   # Raised from 100 so the resumed run keeps training past the original schedule
   # (requires --override-opt-param-scheduler above).
   --num-rollout 400
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 16384
   --rollout-temperature 1

   # Dump every rollout's samples (decoded completions + prompt/tokens/loss_mask/
   # metadata via Sample.to_dict) to a per-step .pt file for inspection during
   # training. {rollout_id} is filled in by slime, not bash.
   --save-debug-rollout-data ${RUN_DIR}/rollout_dumps/{rollout_id}.pt

   --global-batch-size 256
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

   # --micro-batch-size 1
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
   --wandb-group qwen3-4B-realtime-frogger-stream
   #--wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_obstacles.generate_streaming
   --custom-rm-path generate_with_obstacles.reward_func
)

# Held-out eval seed set, built the same way as the train set but with a different
# --seed so the seeds don't overlap:
#   PYTHONPATH=./real-time python3 slime/examples/realtime/obstacles_data_preprocess.py \
#       --env realtime_frogger_stream --train-size 256 --seed 999 \
#       --out $HOME/obstacles-seeds/eval_realtime_frogger_stream.jsonl
EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data realtime_frogger_stream $HOME/obstacles-seeds/eval_realtime_frogger_stream.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 16384
   --eval-temperature 0.7
   --eval-top-p 0.95
)

# Cap the open-file limit: the default (1048576) triggers a raylet SIGABRT crash
# ("Too many open files") in gRPC/boost-asio, which kills the dashboard job agent
# and makes `ray job submit` fail with a 500 / ServerDisconnectedError.
ulimit -n 65535

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 2 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution.
# ${REPO_ROOT}/real-time puts the obstacles `environment` package on PYTHONPATH
# for the ray rollout workers; ${SCRIPT_DIR} makes generate_with_obstacles importable.
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"./Megatron-LM/:${SCRIPT_DIR}:./slime:${REPO_ROOT}/real-time\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"LIBRARY_PATH\": \"${LIBRARY_PATH}\",
    \"LDFLAGS\": \"${LDFLAGS}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 slime/train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${EVAL_ARGS[@]}
