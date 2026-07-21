"""Slime rollout for the clear-obstacles grid game.

This replaces the Python-code-interpreter tool environment with the
``ClearObstaclesToolEnv`` from ``real-time/environment``. The model plays a grid
game: it must move ``F`` to the GOAL row while avoiding obstacles, by emitting
``move_up`` / ``move_down`` / ``move_left`` / ``move_right`` tool calls.

Key design points:
  * The seed for each episode arrives on ``sample.label`` (wired via
    ``--label-key seed``). ``generate`` reconstructs the *exact* grid from that
    seed with ``ClearObstaclesToolEnv.reset(seed=...)`` and keeps that single
    env instance live for the whole tool-calling loop, so the game is fully
    interactive — each move acts on the real, evolving grid.
  * The terminal reward is whatever ``ClearObstaclesToolEnv`` reports
    (``1.0`` win / ``0.0`` loss, per ``ClearObstaclesEnvironment.is_complete``).
    It is stashed on ``sample.metadata`` during the rollout and read back out by
    ``reward_func``.

Plumbing (token-by-token logprob handling, context-length clamping, abort/length
handling) mirrors ``generate_with_retool.py``.
"""

import json
import re
from typing import Any

try:
    from jinja2 import Template
except ImportError as e:
    raise ImportError("Jinja2 is required. Please install it with: pip install jinja2") from e

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# The obstacles environments live in real-time/environment; they must be on
# PYTHONPATH (the training script adds ./real-time).
from environment.clear_obstacles import CLEAR_SYSTEM_PROMPT, ClearObstaclesToolEnv
from environment.frogger import FROGGER_SYSTEM_PROMPT, FroggerToolEnv
from environment.realtime_frogger import (
    REALTIME_FROGGER_STREAM_SYSTEM_PROMPT,
    REALTIME_FROGGER_SYSTEM_PROMPT,
    RealtimeFroggerToolEnv,
)
from environment.realtime_snake import (
    REALTIME_SNAKE_STREAM_SYSTEM_PROMPT,
    REALTIME_SNAKE_SYSTEM_PROMPT,
    RealtimeSnake2400ToolEnv,
    RealtimeSnakeToolEnv,
)
from environment.snake import SNAKE_SYSTEM_PROMPT, SnakeToolEnv
from environment.static_obstacles_grpo import SYSTEM_PROMPT as STATIC_SYSTEM_PROMPT
from environment.static_obstacles_grpo import StaticObstaclesToolEnv

# Selectable environments, keyed by the ``env`` field carried on
# ``sample.metadata`` (populated by obstacles_data_preprocess.py). Each entry
# pairs the tool-env class with the default system prompt to fall back on when a
# dataset row does not carry its own. To add a task here, register its ToolEnv
# wrapper (must expose reset(seed=...), move_*, .done, .reward, .env.won) and a
# default prompt — nothing else in this file needs to change.
#
# Real-time envs (``token_aware = True``) instead expose move_*(tokens_elapsed);
# step_environment passes the number of tokens the model produced this turn.
ENV_REGISTRY: dict[str, tuple[type, str]] = {
    "clear_obstacles": (ClearObstaclesToolEnv, CLEAR_SYSTEM_PROMPT),
    "static_obstacles": (StaticObstaclesToolEnv, STATIC_SYSTEM_PROMPT),
    "frogger": (FroggerToolEnv, FROGGER_SYSTEM_PROMPT),
    "realtime_frogger": (RealtimeFroggerToolEnv, REALTIME_FROGGER_SYSTEM_PROMPT),
    # Same tool-env, but driven by the streaming rollout (generate_streaming): the
    # grid is force-fed into the model's reasoning every movement window. Use with
    # --custom-generate-function-path generate_with_obstacles.generate_streaming.
    "realtime_frogger_stream": (RealtimeFroggerToolEnv, REALTIME_FROGGER_STREAM_SYSTEM_PROMPT),
    # Synchronous snake (Frogger-style: one tool call = one time step; the move
    # turns the snake, then it slides one cell).
    "snake": (SnakeToolEnv, SNAKE_SYSTEM_PROMPT),
    "realtime_snake": (RealtimeSnakeToolEnv, REALTIME_SNAKE_SYSTEM_PROMPT),
    # Same tool-env, but driven by the streaming rollout (generate_streaming): the
    # grid is force-fed into the model's reasoning every movement window. Use with
    # --custom-generate-function-path generate_with_obstacles.generate_streaming.
    "realtime_snake_stream": (RealtimeSnakeToolEnv, REALTIME_SNAKE_STREAM_SYSTEM_PROMPT),
    # Same streaming env at an easier clock (2400 tokens per movement window, the
    # edge of the zero-shot plateau) for training with usable reward signal.
    "realtime_snake_stream_2400": (RealtimeSnake2400ToolEnv, REALTIME_SNAKE_STREAM_SYSTEM_PROMPT),
}
# Used when a sample carries no ``env`` (keeps old single-task datasets working).
DEFAULT_ENV = "clear_obstacles"

# Any registered tool-env. The wrappers are duck-typed (no shared base class),
# so this is a plain union of the registered classes for annotation purposes.
ToolEnv = (
    ClearObstaclesToolEnv
    | StaticObstaclesToolEnv
    | FroggerToolEnv
    | RealtimeFroggerToolEnv
    | SnakeToolEnv
    | RealtimeSnakeToolEnv
    | RealtimeSnake2400ToolEnv
)

# Max number of moves (tool calls) before we cut the episode off.
MAX_TURNS = 64

# Streaming rollout (generate_streaming): hard cap on movement windows (think-chunks
# + moves) per episode, so a model that never emits a move still terminates. The
# context-length guard usually trips first; this is just a backstop.
STREAM_MAX_WINDOWS = 512

# The four move tools exposed to the model.
MOVES = ("move_up", "move_down", "move_left", "move_right")


def _move_tool_specs(descriptions: tuple[tuple[str, str], ...]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
        for name, desc in descriptions
    ]


TOOL_SPECS: list[dict[str, Any]] = _move_tool_specs(
    (
        ("move_up", "Move F up one row (row - 1, toward the GOAL)."),
        ("move_down", "Move F down one row (row + 1)."),
        ("move_left", "Move F left one column (col - 1)."),
        ("move_right", "Move F right one column (col + 1)."),
    )
)

# Same four moves, worded for snake: move_* sets the snake's heading rather than
# stepping the character one cell. No apostrophes here: the Jinja tojson filter
# HTML-escapes them to ' in the rendered prompt.
SNAKE_TOOL_SPECS: list[dict[str, Any]] = _move_tool_specs(
    (
        ("move_up", "Steer the snake up (heading row - 1)."),
        ("move_down", "Steer the snake down (heading row + 1)."),
        ("move_left", "Steer the snake left (heading col - 1)."),
        ("move_right", "Steer the snake right (heading col + 1)."),
    )
)

# Synchronous snake: the move turns the snake and it slides one cell immediately.
SNAKE_SYNC_TOOL_SPECS: list[dict[str, Any]] = _move_tool_specs(
    (
        ("move_up", "Turn the snake up (row - 1) and slide one cell."),
        ("move_down", "Turn the snake down (row + 1) and slide one cell."),
        ("move_left", "Turn the snake left (col - 1) and slide one cell."),
        ("move_right", "Turn the snake right (col + 1) and slide one cell."),
    )
)

# Env-specific tool-spec wording; envs not listed here use TOOL_SPECS, keeping the
# rendered prompts of the pre-existing envs byte-identical.
ENV_TOOL_SPECS: dict[str, list[dict[str, Any]]] = {
    "snake": SNAKE_SYNC_TOOL_SPECS,
    "realtime_snake": SNAKE_TOOL_SPECS,
    "realtime_snake_stream": SNAKE_TOOL_SPECS,
    "realtime_snake_stream_2400": SNAKE_TOOL_SPECS,
}

INVALID_ACTION_MSG = (
    "Invalid action: no valid tool call was found. Emit exactly one tool call, e.g.\n"
    "<tool_call>\n{\"name\": \"move_up\", \"arguments\": {}}\n</tool_call>\n"
    "where the name is one of move_up, move_down, move_left, move_right."
)

# Jinja2 template for tool-enabled conversations (Qwen-style). Only the initial
# prompt is rendered through this; subsequent moves and observations are appended
# to the token stream verbatim.
TOOL_TEMPLATE = """<|im_start|>system
{%- if messages[0]['role'] == 'system' %}
{{- messages[0]['content'] }}
{%- else %}
You are a helpful assistant.
{%- endif %}
{%- if tools %}
# Tools

You may call one function at a time to move your character in the game.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{{- tool | tojson }}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{%- endif %}
<|im_end|>
{%- for message in messages %}
{%- if message['role'] == 'user' %}
<|im_start|>user
{{- message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{- message['content'] }}<|im_end|>
{%- endif %}
{%- endfor %}
<|im_start|>assistant
"""


def format_initial_prompt(
    system_prompt: str, initial_obs: str, tool_specs: list[dict[str, Any]] = TOOL_SPECS
) -> str:
    """Render the opening prompt: game rules (system) + starting grid (user)."""
    template = Template(TOOL_TEMPLATE)
    messages = [
        {"role": "system", "content": system_prompt or CLEAR_SYSTEM_PROMPT},
        {"role": "user", "content": initial_obs},
    ]
    return template.render(messages=messages, tools=tool_specs)


def parse_action(prediction: str) -> str | None:
    """Extract the move name from the last <tool_call> in the prediction.

    Returns one of MOVES, or None if no valid move tool call is found.
    """
    matches = list(re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", prediction, re.DOTALL))
    if not matches:
        return None
    json_str = matches[-1].group(1).replace("\n", "\\n")
    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None
    name = data.get("name")
    return name if name in MOVES else None


def postprocess_responses(resp: str) -> str:
    """Trim the response to the end of its last complete <tool_call> block."""
    matches = list(re.finditer(r"<tool_call>\s*\{.*?\}\s*</tool_call>", resp, re.DOTALL))
    if matches:
        return resp[: matches[-1].end()]
    return resp


def tool_response_turn(content: str, open_assistant: bool) -> str:
    """Wrap env feedback as a native Qwen3 tool result.

    The grid (or an error message) is emitted as a user turn containing a
    <tool_response> block, matching how Qwen3 was trained on tool use. It is
    injected verbatim into the token stream (loss_mask=0) right after the
    assistant's <|im_end|>. When open_assistant is True we also emit the next
    <|im_start|>assistant header so the model starts a fresh turn (and, by Qwen3
    default, opens a <think> block); we omit it on the terminal turn.
    """
    turn = f"\n<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
    if open_assistant:
        turn += "<|im_start|>assistant\n"
    return turn


def inline_observation(content: str, close_turn: bool) -> str:
    """Splice a live grid into the model's OPEN assistant turn (streaming rollout).

    Unlike tool_response_turn (which closes the assistant turn and starts a native
    <tool_response> user turn), this injects an <observation> block *inside* the
    assistant's current reasoning without emitting <|im_end|>, so the model keeps
    thinking with a refreshed view of the world. Injected verbatim with loss_mask=0.
    close_turn=True appends <|im_end|> to end the episode (e.g. a car ran the frog
    over mid-think), since no further generation follows.
    """
    turn = f"\n<observation>\n{content}\n</observation>\n"
    if close_turn:
        turn += "<|im_end|>\n"
    return turn


def step_environment(env: ToolEnv, prediction: str, tokens_elapsed: int) -> tuple[str, bool]:
    """Apply the model's move to the live env and return (tool_response_content, done).

    Returns the raw grid render (or INVALID_ACTION_MSG); the turn markers are added
    by tool_response_turn at the call site, not here. ``tokens_elapsed`` is the number
    of tokens the model produced this turn; real-time envs (``token_aware``) use it to
    advance their world, static envs ignore it.
    """
    token_aware = getattr(env, "token_aware", False)

    action = parse_action(prediction)
    if action is None:
        if not token_aware:
            return INVALID_ACTION_MSG, False
        # Real-time env: the tokens were still spent, so advance the cars (which can
        # run the frog over) even though no move is applied. If that ends the game,
        # surface the GAME OVER text; otherwise show the error plus the updated grid.
        obs = env.advance(tokens_elapsed)
        if env.done:
            return obs, True
        return f"{INVALID_ACTION_MSG}\n{obs}", False

    move = getattr(env, action)
    obs = move(tokens_elapsed) if token_aware else move()
    return obs, env.done


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Custom generation function: play the obstacles game via move tool calls."""
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    # Retried samples arrive with stale rollout state from the first attempt;
    # clear it so this generation starts clean (see generate_with_retool.py).
    sample.rollout_log_probs = None
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = None

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Pick the environment for this sample. The dataset tags each row with an
    # `env` on sample.metadata (see obstacles_data_preprocess.py); rows without
    # one fall back to DEFAULT_ENV so single-task datasets keep working.
    meta = sample.metadata or {}
    env_name = meta.get("env", DEFAULT_ENV)
    try:
        env_cls, default_system_prompt = ENV_REGISTRY[env_name]
    except KeyError:
        raise ValueError(f"Unknown env '{env_name}'; registered: {sorted(ENV_REGISTRY)}")

    # Reconstruct the exact grid from the dataset seed and keep this env instance
    # live across every move below.
    seed = int(sample.label) if sample.label is not None else None
    env = env_cls()
    initial_obs = env.reset(seed=seed) if seed is not None else env.reset()

    # Carry the system prompt from the dataset (falls back to the env's default).
    system_prompt = sample.prompt if isinstance(sample.prompt, str) and sample.prompt else default_system_prompt
    prompt = format_initial_prompt(system_prompt, initial_obs, ENV_TOOL_SPECS.get(env_name, TOOL_SPECS))

    prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids = []
    loss_masks = []
    move_count = 0

    if args.rollout_max_context_len is not None:
        max_context_length = args.rollout_max_context_len
    else:
        max_context_length = args.context_parallel_size * args.max_tokens_per_gpu

    for turn in range(MAX_TURNS):
        total_length = len(prompt_tokens_ids) + len(response_token_ids)
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break

        # Clamp per-turn max_new_tokens to the remaining context budget.
        remaining_budget = max_context_length - total_length
        per_turn_sampling_params = dict(sampling_params)
        per_turn_sampling_params["max_new_tokens"] = min(
            sampling_params.get("max_new_tokens", remaining_budget),
            remaining_budget,
        )

        current_token_ids = prompt_tokens_ids + response_token_ids
        payload = {
            "input_ids": current_token_ids,
            "sampling_params": per_turn_sampling_params,
            "return_logprob": True,
        }

        output = await post(url, payload)

        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        if "output_token_logprobs" in output["meta_info"]:
            cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_response = state.tokenizer.decode(cur_response_token_ids)
            cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            if sample.rollout_log_probs is None:
                sample.rollout_log_probs = []
            sample.rollout_log_probs += cur_log_probs
        else:
            # No per-token logprobs -> cannot keep rollout_log_probs in sync;
            # abort so the group is retried instead of poisoning the trainer.
            sample.status = Sample.Status.ABORTED
            return sample

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_masks += [1] * len(cur_response_token_ids)

        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        # Tokens produced this turn drive real-time envs (cars move while the model
        # thinks); static envs ignore the count.
        obs_content, done = step_environment(env, cur_response, len(cur_response_token_ids))
        if parse_action(cur_response) is not None:
            move_count += 1
        if done:
            # Final tool result (GAME OVER grid) as a closing user turn; no new
            # assistant header since the episode is over.
            next_obs = tool_response_turn(obs_content, open_assistant=False)
            obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
            response += next_obs
            response_token_ids += obs_tokens_ids
            loss_masks += [0] * len(obs_tokens_ids)
            if sample.rollout_log_probs is not None:
                sample.rollout_log_probs += [0.0] * len(obs_tokens_ids)
            sample.status = Sample.Status.COMPLETED
            break

        # Native tool turn + a fresh assistant header so the model reasons and
        # moves again on the next iteration.
        next_obs = tool_response_turn(obs_content, open_assistant=True)
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_masks += [0] * len(obs_tokens_ids)

        if sample.rollout_log_probs is not None:
            sample.rollout_log_probs += [0.0] * len(obs_tokens_ids)
            assert len(response_token_ids) == len(sample.rollout_log_probs), (
                f"Token/logp length mismatch at turn {turn}: "
                f"{len(response_token_ids)} tokens vs {len(sample.rollout_log_probs)} logps"
            )

        # Observation is appended verbatim and can push us past the budget; trim
        # the tail so the final sample fits the training budget exactly.
        overflow = len(prompt_tokens_ids) + len(response_token_ids) - max_context_length
        if overflow > 0:
            response_token_ids = response_token_ids[:-overflow]
            loss_masks = loss_masks[:-overflow]
            if sample.rollout_log_probs is not None:
                sample.rollout_log_probs = sample.rollout_log_probs[:-overflow]
            response = state.tokenizer.decode(response_token_ids)
            sample.status = Sample.Status.TRUNCATED
            break

    # Terminal reward from the env (1.0 win / 0.0 otherwise). Stash it for
    # reward_func, which runs as a separate call on this same sample.
    sample.metadata = dict(sample.metadata or {})
    sample.metadata["env_reward"] = float(env.reward)
    sample.metadata["env_done"] = bool(env.done)
    sample.metadata["env_won"] = bool(env.env.won) if env.env is not None else False
    sample.metadata["move_count"] = move_count

    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks

    # If we exited the loop without setting a terminal status above, classify by
    # the last finish_reason.
    if sample.status == Sample.Status.PENDING:
        match output["meta_info"]["finish_reason"]["type"]:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "abort":
                sample.status = Sample.Status.ABORTED
            case "stop":
                sample.status = Sample.Status.COMPLETED

    return sample


async def generate_streaming(args, sample: Sample, sampling_params) -> Sample:
    """Real-time *streaming* rollout for the realtime_frogger_stream env.

    The world runs on a fixed token cadence: every ``tokens_per_movement`` tokens the
    model produces is one movement window == one car tick. This differs from
    ``generate`` (which advances the whole turn's tokens at once, only when the model
    acts) in two ways:

      * Force-feed: whenever a full window elapses while the model is still thinking
        (it hit the per-chunk token cap without emitting a move), the cars advance one
        cell and the current grid is spliced into the model's OPEN assistant turn as a
        masked ``<observation>`` block. The model keeps reasoning with fresh state. A
        stationary frog can be run over here.
      * On a move, the remainder of the window elapses FIRST (cars advance one more
        cell), THEN the move is applied and the resulting grid is returned as a native
        ``<tool_response>`` -- cars-first order, same as ``env.act`` / the
        non-streaming rollout.

    Every injected token (observations and tool responses) carries loss_mask=0, so
    only the model's own reasoning/move tokens are trained on. A move always consumes
    a full window (it rounds up to the window boundary), so the cars advance exactly
    one cell per window regardless of when in the window the move was emitted.
    """
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    # Clear any stale state from a retried attempt (mirrors generate).
    sample.rollout_log_probs = None
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = None

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    meta = sample.metadata or {}
    env_name = meta.get("env", "realtime_frogger_stream")
    try:
        env_cls, default_system_prompt = ENV_REGISTRY[env_name]
    except KeyError:
        raise ValueError(f"Unknown env '{env_name}'; registered: {sorted(ENV_REGISTRY)}")
    if not getattr(env_cls, "token_aware", False):
        raise ValueError(
            f"generate_streaming requires a token-aware env; '{env_name}' is not. "
            "Use a realtime_* env (e.g. realtime_frogger_stream)."
        )

    seed = int(sample.label) if sample.label is not None else None
    env = env_cls()
    initial_obs = env.reset(seed=seed) if seed is not None else env.reset()

    # One movement window == this many tokens == one car tick. Read off the env so it
    # always matches the env's own tokens_per_movement (injections land on ticks).
    window = env.tokens_per_movement

    system_prompt = sample.prompt if isinstance(sample.prompt, str) and sample.prompt else default_system_prompt
    prompt = format_initial_prompt(system_prompt, initial_obs, ENV_TOOL_SPECS.get(env_name, TOOL_SPECS))

    prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids = []
    loss_masks = []
    move_count = 0
    injection_count = 0

    if args.rollout_max_context_len is not None:
        max_context_length = args.rollout_max_context_len
    else:
        max_context_length = args.context_parallel_size * args.max_tokens_per_gpu

    output = None
    for _window_idx in range(STREAM_MAX_WINDOWS):
        total_length = len(prompt_tokens_ids) + len(response_token_ids)
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break

        # Generate at most one window of tokens (or whatever context budget remains).
        # Hitting `window` exactly == a full time-tick elapsed mid-think; hitting a
        # smaller budget cap == out of room (truncation).
        remaining_budget = max_context_length - total_length
        chunk_cap = min(window, remaining_budget)
        per_turn_sampling_params = dict(sampling_params)
        per_turn_sampling_params["max_new_tokens"] = chunk_cap

        payload = {
            "input_ids": prompt_tokens_ids + response_token_ids,
            "sampling_params": per_turn_sampling_params,
            "return_logprob": True,
        }
        output = await post(url, payload)

        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        if "output_token_logprobs" not in output["meta_info"]:
            # No per-token logprobs -> cannot keep rollout_log_probs in sync; abort so
            # the group is retried instead of poisoning the trainer.
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        cur_response = state.tokenizer.decode(cur_response_token_ids)
        cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs += cur_log_probs

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_masks += [1] * len(cur_response_token_ids)

        finish = output["meta_info"]["finish_reason"]["type"]
        action = parse_action(cur_response)
        hit_full_window = finish == "length" and chunk_cap == window

        # Decide what to append (masked env feedback) and whether the episode ends.
        if action is not None:
            # MOVE: the rest of the window elapses (cars +1), THEN the move is applied
            # (cars-first order, same as env.act / the non-streaming rollout).
            obs = getattr(env, action)(window)
            move_count += 1
            terminal = env.done
            next_text = tool_response_turn(obs, open_assistant=not terminal)
        elif hit_full_window:
            # FORCE-FEED: a window elapsed mid-think. Cars +1 (may run over the
            # stationary frog); splice the grid into the still-open assistant turn.
            obs = env.advance(window)
            injection_count += 1
            terminal = env.done
            next_text = inline_observation(obs, close_turn=terminal)
        elif finish == "length":
            # Hit the context budget (chunk_cap < window), not a real window boundary.
            sample.status = Sample.Status.TRUNCATED
            break
        elif finish == "stop":
            # Model ended its turn without a valid move; the window still elapses.
            obs = env.advance(window)
            terminal = env.done
            msg = obs if terminal else f"{INVALID_ACTION_MSG}\n{obs}"
            next_text = tool_response_turn(msg, open_assistant=not terminal)
        else:
            # Unexpected finish reason; stop defensively.
            break

        obs_tokens_ids = state.tokenizer(next_text, add_special_tokens=False)["input_ids"]
        response += next_text
        response_token_ids += obs_tokens_ids
        loss_masks += [0] * len(obs_tokens_ids)
        if sample.rollout_log_probs is not None:
            sample.rollout_log_probs += [0.0] * len(obs_tokens_ids)
            assert len(response_token_ids) == len(sample.rollout_log_probs), (
                f"Token/logp length mismatch at window {_window_idx}: "
                f"{len(response_token_ids)} tokens vs {len(sample.rollout_log_probs)} logps"
            )

        if terminal:
            sample.status = Sample.Status.COMPLETED
            break

        # Injected feedback is appended verbatim and can push us past the budget;
        # trim the tail so the final sample fits the training budget exactly.
        overflow = len(prompt_tokens_ids) + len(response_token_ids) - max_context_length
        if overflow > 0:
            response_token_ids = response_token_ids[:-overflow]
            loss_masks = loss_masks[:-overflow]
            if sample.rollout_log_probs is not None:
                sample.rollout_log_probs = sample.rollout_log_probs[:-overflow]
            response = state.tokenizer.decode(response_token_ids)
            sample.status = Sample.Status.TRUNCATED
            break

    # Terminal reward from the env (1.0 win / 0.0 otherwise), stashed for reward_func.
    sample.metadata = dict(sample.metadata or {})
    sample.metadata["env_reward"] = float(env.reward)
    sample.metadata["env_done"] = bool(env.done)
    sample.metadata["env_won"] = bool(env.env.won) if env.env is not None else False
    sample.metadata["move_count"] = move_count
    sample.metadata["injection_count"] = injection_count

    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks

    if sample.status == Sample.Status.PENDING:
        match (output["meta_info"]["finish_reason"]["type"] if output is not None else "stop"):
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "abort":
                sample.status = Sample.Status.ABORTED
            case _:
                sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample, **kwargs):
    """Return the obstacles env reward stashed during generation."""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    metadata = sample.metadata or {}
    return {
        "score": float(metadata.get("env_reward", 0.0)),
        "won": bool(metadata.get("env_won", False)),
        "move_count": int(metadata.get("move_count", 0)),
    }
