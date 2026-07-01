"""Build a seed dataset for the obstacles environments.

Each example is a (system prompt, seed, env) triple. The seed fully determines the
grid the agent will face (obstacle layout, start position), so "initializing the
environment" is all that is needed to materialize a problem instance.

The seed is the crucial field: it is carried in the ``seed`` column and wired into
slime via ``--label-key seed`` so it arrives on ``sample.label``. At rollout time
``generate_with_obstacles.generate`` reconstructs the *exact* same environment from
that seed (``<ToolEnv>.reset(seed=...)``) and keeps that single instance live across
the whole tool-calling loop, so every move acts on the real grid the seed describes.
The constant ``prompt`` column carries the game rules via ``--input-key prompt``, and
the ``metadata.env`` field selects which tool-env to instantiate.

Multi-env mixing: rows are always written **stratified round-robin** across the
requested ``--env`` values (A, B, A, B, ...). slime serves each rollout step a
*contiguous* slice of the dataset (rollout-batch-size prompts) and we disable
``--rollout-shuffle`` in the training script, so every batch contains an exactly
equal number of each env — provided ``rollout-batch-size`` is a multiple of the
number of envs. A single ``--env`` is just the degenerate (no-op interleave) case.

Run from the directory that contains ``slime/`` with the real-time environment on
the path::

    PYTHONPATH=./real-time python3 slime/examples/realtime/obstacles_data_preprocess.py \
        --env clear_obstacles static_obstacles

Requires the ``environment`` package (``real-time/environment``) to be importable.
"""

import argparse
import json
import os
import random

from environment.clear_obstacles import CLEAR_SYSTEM_PROMPT
from environment.frogger import FROGGER_SYSTEM_PROMPT
from environment.static_obstacles_grpo import SYSTEM_PROMPT as STATIC_SYSTEM_PROMPT

# Default system prompt per environment. The keys must match ENV_REGISTRY in
# generate_with_obstacles.py: the `env` value written below arrives on
# sample.metadata and selects the tool-env at rollout time.
ENV_SYSTEM_PROMPTS = {
    "clear_obstacles": CLEAR_SYSTEM_PROMPT,
    "static_obstacles": STATIC_SYSTEM_PROMPT,
    "frogger": FROGGER_SYSTEM_PROMPT,
}


def _env_rows(n: int, rng: random.Random, env: str) -> list[dict]:
    """Build ``n`` rows for a single env, each with an independent random seed."""
    system_prompt = ENV_SYSTEM_PROMPTS[env]
    return [
        {
            "prompt": system_prompt,
            "seed": rng.randint(0, 2**31 - 1),
            # Tags the row with its environment; loaded verbatim into
            # sample.metadata and read by generate_with_obstacles.generate.
            "metadata": {"env": env},
        }
        for _ in range(n)
    ]


def build_rows(n: int, rng: random.Random, envs: list[str]) -> list[dict]:
    """Build ``n`` rows split equally across ``envs`` and stratified round-robin.

    Each env gets ``n // len(envs)`` rows (with independently drawn seeds), then the
    per-env lists are interleaved A, B, A, B, ... so that every contiguous window of
    ``len(envs)`` rows holds exactly one of each. With ``--rollout-shuffle`` off, this
    makes each rollout batch exactly balanced across envs.
    """
    num_envs = len(envs)
    if n % num_envs != 0:
        raise ValueError(
            f"--train-size ({n}) must be divisible by the number of envs ({num_envs}) "
            f"for an exactly balanced split."
        )
    per_env = n // num_envs
    # Draw each env's seeds independently; this also acts as the per-env shuffle.
    env_rows = [_env_rows(per_env, rng, env) for env in envs]
    # Round-robin interleave: rows[i*num_envs + j] belongs to envs[j].
    return [env_rows[j][i] for i in range(per_env) for j in range(num_envs)]


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} examples -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.expanduser("~/obstacles-seeds/train.jsonl"))
    parser.add_argument("--train-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--env",
        nargs="+",
        choices=sorted(ENV_SYSTEM_PROMPTS),
        default=["clear_obstacles"],
        metavar="ENV",
        help=(
            "One or more environments to mix. Rows are split equally and interleaved "
            "round-robin so each rollout batch is balanced (keep rollout-batch-size a "
            "multiple of the env count and --rollout-shuffle off)."
        ),
    )
    args = parser.parse_args()

    if len(set(args.env)) != len(args.env):
        parser.error(f"--env contains duplicates: {args.env}")

    rng = random.Random(args.seed)
    write_jsonl(args.out, build_rows(args.train_size, rng, args.env))


if __name__ == "__main__":
    main()
