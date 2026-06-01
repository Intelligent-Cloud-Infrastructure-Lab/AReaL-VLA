"""
Reward functions for VLA robot RL training.

SimpleVLA-RL uses binary 0/1 outcome rewards extracted directly from the
simulator's `info["success"]` flag.  This module exposes thin wrappers around
that logic so they plug into AReaL's AsyncRewardWrapper pattern.

Design note
-----------
In text-based AReaL workflows, reward_fn takes (prompt, completion, …) as args.
For robot workflows, the reward is already computed inside arun_episode (the
environment returns it via info["success"]).  Therefore these functions serve
two roles:

  1. As an *optional* secondary reward signal (e.g. a shaped per-step reward
     that could be added on top of the binary outcome reward in the future).

  2. As a compatibility layer if you want to extend VLARobotWorkflow to use
     AReaL's AsyncRewardWrapper for logging, batching, or LLM-as-judge scoring.

For most users, the reward functions below can be ignored — `binary_outcome_reward`
is called automatically inside VLARobotWorkflow.arun_episode.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Outcome reward (main reward signal, called inside the workflow)
# ---------------------------------------------------------------------------


def binary_outcome_reward(
    info: dict[str, Any],
    *,
    success_key: str = "success",
    success_reward: float = 1.0,
    failure_reward: float = 0.0,
) -> float:
    """
    Extract a binary 0/1 reward from an environment info dict.

    Parameters
    ----------
    info:
        The dict returned by env.step().
    success_key:
        Key in info that holds the boolean success signal.
        LIBERO uses "success"; RoboTwin 2.0 uses "task_success".
    success_reward / failure_reward:
        Values to return.  Defaults to 1.0 / 0.0 (SimpleVLA-RL style).

    Returns
    -------
    float: success_reward if the task succeeded, else failure_reward.
    """
    succeeded = bool(info.get(success_key, False))
    return success_reward if succeeded else failure_reward


# ---------------------------------------------------------------------------
# LIBERO-specific reward helpers
# ---------------------------------------------------------------------------


def libero_reward_fn(
    info: dict[str, Any],
    task_name: str | None = None,
) -> float:
    """
    Binary reward for LIBERO environments.

    LIBERO's step() returns info["success"] == True when the task is complete.
    This wrapper is provided for clarity; it's equivalent to calling
    binary_outcome_reward with success_key="success".
    """
    return binary_outcome_reward(info, success_key="success")


def libero_shaped_reward_fn(
    info: dict[str, Any],
    task_name: str | None = None,
    *,
    completion_bonus: float = 1.0,
    progress_scale: float = 0.0,
) -> float:
    """
    Optional shaped reward for LIBERO.

    Most LIBERO tasks expose a "progress" float in [0, 1] alongside "success".
    Set progress_scale > 0 to add a dense shaping term.  SimpleVLA-RL does NOT
    use shaping — this is provided for ablation studies.

    Note: Dense shaping can speed up early learning but may bias the policy
    toward sub-optimal behaviours.  Start with the binary outcome reward.
    """
    success = bool(info.get("success", False))
    if success:
        return completion_bonus

    progress = float(info.get("progress", 0.0))
    return progress_scale * progress


# ---------------------------------------------------------------------------
# RoboTwin-specific reward helpers
# ---------------------------------------------------------------------------


def robottwin_reward_fn(
    info: dict[str, Any],
    task_name: str | None = None,
) -> float:
    """
    Binary reward for RoboTwin 1.0 / 2.0 environments.

    RoboTwin 2.0 uses "task_success" in the info dict.
    RoboTwin 1.0 uses "success".  We check both for compatibility.
    """
    # RoboTwin 2.0 key first, then fallback to generic "success"
    success = bool(info.get("task_success", info.get("success", False)))
    return 1.0 if success else 0.0


# ---------------------------------------------------------------------------
# Registry: look up reward function by benchmark name
# ---------------------------------------------------------------------------

_REWARD_REGISTRY: dict[str, Any] = {
    "libero_object": libero_reward_fn,
    "libero_spatial": libero_reward_fn,
    "libero_goal": libero_reward_fn,
    "libero_long": libero_reward_fn,
    "robottwin": robottwin_reward_fn,
    "robottwin_1": robottwin_reward_fn,
    "robottwin_2": robottwin_reward_fn,
}


def get_reward_fn(benchmark: str):
    """
    Look up the reward function for a given benchmark name.

    Raises ValueError for unknown benchmarks.
    """
    key = benchmark.lower()
    if key not in _REWARD_REGISTRY:
        raise ValueError(
            f"Unknown benchmark '{benchmark}'. "
            f"Available: {sorted(_REWARD_REGISTRY)}"
        )
    return _REWARD_REGISTRY[key]


# ---------------------------------------------------------------------------
# Token-level reward alignment (called by VLARobotWorkflow internally)
# ---------------------------------------------------------------------------


def align_reward_to_tokens(
    binary_reward: float,
    finish_step: int,
    episode_steps: int,
    action_chunk_len: int,
    total_action_tokens: int,
) -> list[float]:
    """
    Convert a scalar trajectory reward to a per-action-token reward list.

    This implements the core SimpleVLA-RL reward alignment trick:

        reward[action_token_idx] = binary_reward  if action_token_idx <= finish_step * action_chunk_len
        reward[action_token_idx] = 0              if action_token_idx >  finish_step * action_chunk_len

    The function returns a flat list of length `total_action_tokens`.
    Prompt tokens are NOT included here; the workflow interleaves zeros
    for prompt tokens automatically.

    Parameters
    ----------
    binary_reward:
        0.0 or 1.0.
    finish_step:
        The 1-indexed environment step at which success was first detected.
        If the episode failed, pass episode_steps (all tokens are labelled 0).
    episode_steps:
        Total number of environment steps executed.
    action_chunk_len:
        Number of action tokens per step.
    total_action_tokens:
        Should equal episode_steps * action_chunk_len (but may differ if
        the episode was truncated early).

    Returns
    -------
    list[float]: per-action-token rewards.
    """
    # Token cutoff: the last valid action token index (1-based)
    cutoff = finish_step * action_chunk_len  # inclusive upper bound

    rewards: list[float] = []
    for i in range(1, total_action_tokens + 1):  # 1-indexed
        rewards.append(binary_reward if i <= cutoff else 0.0)
    return rewards
