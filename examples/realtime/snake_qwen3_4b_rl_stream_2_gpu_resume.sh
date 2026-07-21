#!/bin/bash
#SBATCH --output=.cache/slurm-out/slurm-%j.out

# Resume version of snake_qwen3_4b_rl_stream_2_gpu.sh: continues RL training on
# the STREAMING real-time Snake env (env tag `realtime_snake_stream`) from a
# previous run's checkpoints.
#
# Usage:
#   bash snake_qwen3_4b_rl_stream_2_gpu_resume.sh <run-id-or-checkpoint-dir>
# where the argument is either the RUN_ID (uuid) of a previous run (resolved to
# <repo root>/.cache/<uuid>/checkpoints) or an explicit path to a checkpoints dir.
# Alternatively set LOAD_DIR in the environment.
#
# The only differences from snake_qwen3_4b_rl_stream_2_gpu.sh are:
#   * --load points at the previous run's checkpoints (slime resumes the rollout
#     step from the checkpoint, so it continues where the old run stopped)
#   * --num-rollout is raised 100 -> 400 to train past the original schedule
#   * --override-opt-param-scheduler (see comment in CKPT_ARGS)
# No new RUN_DIR is created: the run continues in the directory it loads from,
# so new checkpoints and rollout dumps accumulate there. This makes the script
# safe to requeue with the same argument -- each restart resumes from the latest
# checkpoint saved in that directory.

# Locate the micromamba binary: MAMBA_ROOT_PREFIX is the *env* root and does not
# necessarily contain the binary (here it lives in ~/.local/bin), so probe PATH
# first and fall back to the common install locations. Activation must succeed --
# everything below (python, ray, $CONDA_PREFIX-based LIBRARY_PATH) needs the env --
# so fail loudly instead of silently continuing with system python.
MICROMAMBA_BIN="$(command -v micromamba || true)"
if [ -z "${MICROMAMBA_BIN}" ]; then
    for cand in "${MAMBA_ROOT_PREFIX:-$HOME/micromamba}/bin/micromamba" "$HOME/.local/bin/micromamba" "$HOME/micromamba/bin/micromamba"; do
        if [ -x "${cand}" ]; then MICROMAMBA_BIN="${cand}"; break; fi
    done
fi
if [ -z "${MICROMAMBA_BIN}" ]; then
    echo "ERROR: micromamba binary not found (checked PATH, \$MAMBA_ROOT_PREFIX/bin, ~/.local/bin, ~/micromamba/bin)" >&2
    exit 1
fi
eval "$("${MICROMAMBA_BIN}" shell hook --shell bash)"
micromamba activate slime || { echo "ERROR: failed to activate micromamba env 'slime'" >&2; exit 1; }
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

# Continue in the same run directory we are loading from: RUN_DIR is the parent
# of the checkpoints dir, so new checkpoints and rollout dumps accumulate in the
# original run's directory instead of a fresh one. Because --save and --load are
# the same dir, a requeued/restarted job re-resolves this same directory and
# Megatron's latest_checkpointed_iteration.txt tracker (updated on every save)
# makes it resume from the newest checkpoint rather than the one this script was
# first pointed at.
RUN_DIR="$(cd -- "${LOAD_DIR}/.." &>/dev/null && pwd)"
RUN_ID="$(basename "${RUN_DIR}")"
echo "RUN_ID: ${RUN_ID}"
echo "Artifacts (checkpoints + rollout dumps) -> ${RUN_DIR}"

CKPT_ARGS=(
   --hf-checkpoint ${ARTIFACT_ROOT}/Qwen/Qwen3-4B
   --ref-load ${ARTIFACT_ROOT}/Qwen/Qwen3-4B_torch_dist
   --load ${LOAD_DIR}
   # Save back into the load dir: keeps everything in one run dir and lets a
   # requeued job pick up from the latest checkpoint via the tracker file.
   --save ${LOAD_DIR}
   # Checkpoint every 5 steps, but keep permanently only every 50th step: on
   # each save Megatron deletes the previous checkpoint unless it falls on the
   # retain interval, so the newest checkpoint always exists (cheap requeue
   # recovery) without accumulating one every 5 steps. slime numbers saves with
   # 0-based rollout ids, so retained checkpoints are iterations 49, 99, 149,
   # ... (the retention test is (iteration+1) % 50 == 0 -- see the slime patch
   # in Megatron-LM/megatron/training/checkpointing.py; upstream's plain
   # modulo would never match slime's numbering and delete everything).
   --save-interval 5
   --save-retain-interval 50
   --rotary-base 1000000
   # We bump --num-rollout below to train past the original schedule, which
   # changes the derived lr_decay_steps and would otherwise fail the scheduler's
   # checkpoint-consistency assert. Override it to use the new schedule values
   # (LR is constant here anyway; step counting still resumes from num_steps).
   --override-opt-param-scheduler
)

# Build the train seed set (once) with the streaming env tag:
#   PYTHONPATH=./real-time python3 slime/examples/realtime/obstacles_data_preprocess.py \
#       --env realtime_snake_stream --train-size 20000 --seed 1234 \
#       --out $HOME/obstacles-seeds/train_realtime_snake_stream.jsonl
ROLLOUT_ARGS=(
   # Seed dataset produced by obstacles_data_preprocess.py. Each row is
   # {"prompt": <streaming game rules>, "seed": <int>, "metadata": {"env": "realtime_snake_stream"}};
   # the seed arrives on sample.label and the env is reconstructed from it at rollout time.
   --prompt-data ${ARTIFACT_ROOT}/obstacles-seeds/train_realtime_snake_stream.jsonl
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
   --wandb-group qwen3-4B-realtime-snake-stream
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
#       --env realtime_snake_stream --train-size 256 --seed 999 \
#       --out $HOME/obstacles-seeds/eval_realtime_snake_stream.jsonl
EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data realtime_snake_stream $HOME/obstacles-seeds/eval_realtime_snake_stream.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 16384
   --eval-temperature 0.7
   --eval-top-p 0.95
)

# Cap the open-file limit: the default (1048576) triggers a raylet SIGABRT crash
# ("Too many open files") in gRPC/boost-asio, which kills the dashboard job agent
# and makes `ray job submit` fail with a 500 / ServerDisconnectedError. Only
# lower it: since ~2026-07-11 jobs get a 51200 hard limit (cluster-wide, even
# with --propagate=NONE), and raising past the hard limit fails.
if [ "$(ulimit -n)" -gt 65535 ]; then ulimit -n 65535; fi

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# Dashboard port is overridable: on shared nodes another user's Ray cluster may
# already own the default 8265, and `ray job submit` would silently target theirs.
DASH_PORT=${DASH_PORT:-8270}
# The job agent binds its own HTTP port (default 52365) — also shared per-node, and
# a collision leaves the dashboard with no agent ("No available agent to submit job").
AGENT_PORT=${AGENT_PORT:-52370}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 2 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=${DASH_PORT} --dashboard-agent-listen-port=${AGENT_PORT}

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

ray job submit --address="http://127.0.0.1:${DASH_PORT}" \
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
