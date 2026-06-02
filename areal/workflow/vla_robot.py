"""
VLARobotWorkflow — AReaL rollout workflow for Vision-Language-Action (VLA) models.

This module ports the core ideas from SimpleVLA-RL (PRIME-RL/SimpleVLA-RL) into
AReaL's RolloutWorkflow abstraction.  The key algorithmic contributions preserved are:

  1. Embodied rollout: LIBERO/RoboTwin env.reset → render → generate → step loop
  2. Sparse-reward → token-level alignment (finish_step × action_chunk_len)
  3. Post-success loss masking (zero gradient past the success boundary)
  4. Mixed-success group batching (handled by GRPO's group-advantage normalization)

Usage
-----
Slot VLARobotWorkflow into the same position that RLVRWorkflow occupies in the
standard AReaL math example:

    workflow = VLARobotWorkflow(
        env_factory=make_libero_env,
        action_decoder=openvla_action_decoder,
        instruction_tokenizer=vla_instruction_tokenizer,
        gconfig=config.gconfig,
        action_chunk_len=7,
        max_episode_steps=300,
        dump_dir=os.path.join(log_path, "robot_episodes"),
    )
    batch = actor.prepare_batch(train_dataloader, workflow=workflow, ...)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from areal.api.workflow_api import RolloutWorkflow
from areal.utils.data import concat_padded_tensors
from areal.utils.logging import getLogger

logger = getLogger("VLARobotWorkflow")


# ---------------------------------------------------------------------------
# Data structures for VLA-specific requests / responses
# ---------------------------------------------------------------------------


@dataclass
class VLAStepRequest:
    """
    Per-timestep request passed to the inference engine.

    Unlike text-only generation, each step carries a raw RGB image observation
    alongside the tokenised instruction.  The image is either:
      (a) pre-encoded to image tokens (for tokeniser-based VLMs), or
      (b) passed as raw numpy pixels (for models using a vision encoder inline).
    """

    # Tokenised language instruction (constant across all steps of an episode)
    instruction_ids: list[int]

    # Current RGB observation: shape (H, W, 3), dtype uint8
    image: np.ndarray

    # How many action tokens to generate (= action_chunk_len)
    max_new_tokens: int = 7

    # Optional: pre-encoded image token IDs (used when the model tokenises images)
    image_token_ids: list[int] | None = None

    # Raw instruction text forwarded to VLAInferenceServer so it can tokenise
    # using its own processor (transformers-openvla-oft fork in SimpleVLA env).
    # Leave None when using VLALocalEngine in the same environment.
    instruction_text: str | None = None


@dataclass
class VLAStepResponse:
    """
    Result from one VLA generation step.

    Fields mirror AReaL's ModelResponse so that the workflow can build the
    standard AReaL trajectory tensors.
    """

    # Full prompt tokens fed to the model (image tokens + instruction tokens)
    input_tokens: list[int]

    # Generated action token IDs (length == action_chunk_len)
    output_tokens: list[int]

    # Per-token log-probabilities for output_tokens
    output_logprobs: list[float]

    # Weight version at the time of generation (for off-policy staleness tracking)
    output_versions: list[int]

    # Pre-decoded continuous action returned by VLAInferenceServer.
    # When set, VLARobotWorkflow uses this for env.step() directly instead of
    # calling action_decoder(output_tokens) — the server already decoded using
    # the model's norm_stats in the SimpleVLA environment.
    # None when using VLALocalEngine (decoding happens locally).
    decoded_action: "np.ndarray | None" = None

    @property
    def input_len(self) -> int:
        return len(self.input_tokens)

    @property
    def output_len(self) -> int:
        return len(self.output_tokens)


# ---------------------------------------------------------------------------
# Abstract interface that the VLA inference engine must implement
# ---------------------------------------------------------------------------


class VLAEngine(ABC):
    """
    Minimal inference interface required by VLARobotWorkflow.

    The default AReaL engines (RemoteSGLangEngine, RemoteVLLMEngine) satisfy this
    interface when the VLA model is served via SGLang / vLLM.  For VLA models not yet
    supported by those backends, use VLALocalEngine (areal/engine/vla_local_engine.py).
    """

    @abstractmethod
    async def agenerate(self, req: VLAStepRequest) -> VLAStepResponse:
        """Run one VLA forward pass and return action tokens + logprobs."""

    @abstractmethod
    def get_version(self) -> int:
        """Return the current weight version (used for off-policy staleness)."""


# ---------------------------------------------------------------------------
# Trajectory helper
# ---------------------------------------------------------------------------


@dataclass
class _EpisodeBuffer:
    """
    Accumulates per-step data during an episode, then assembles AReaL tensors.
    """

    # Flat lists grown at each step
    input_ids: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    versions: list[int] = field(default_factory=list)

    # Action token counter (used for post-success boundary)
    total_action_tokens: int = 0

    def append_step(
        self,
        resp: VLAStepResponse,
        step_is_post_success: bool,
    ) -> None:
        """
        Append one step's tokens to the buffer.

        Prompt tokens get loss_mask=0.
        Action tokens get loss_mask=1, EXCEPT when post-success (→ 0).
        """
        # Prompt tokens
        self.input_ids.extend(resp.input_tokens)
        self.loss_mask.extend([0] * resp.input_len)
        self.logprobs.extend([0.0] * resp.input_len)
        self.versions.extend([-1] * resp.input_len)

        # Action tokens
        action_mask_value = 0 if step_is_post_success else 1
        self.input_ids.extend(resp.output_tokens)
        self.loss_mask.extend([action_mask_value] * resp.output_len)
        self.logprobs.extend(resp.output_logprobs)
        self.versions.extend(resp.output_versions)

        if not step_is_post_success:
            self.total_action_tokens += resp.output_len

    def build_tensors(self, reward: float) -> dict[str, torch.Tensor]:
        """Convert the buffer to AReaL's standard per-trajectory tensor dict."""
        seq_len = len(self.input_ids)
        result = {
            "input_ids": torch.tensor(self.input_ids, dtype=torch.int32),
            "loss_mask": torch.tensor(self.loss_mask, dtype=torch.int32),
            "logprobs": torch.tensor(self.logprobs, dtype=torch.float32),
            "versions": torch.tensor(self.versions, dtype=torch.int32),
            "attention_mask": torch.ones(seq_len, dtype=torch.bool),
            "rewards": torch.tensor([reward], dtype=torch.float32),
        }
        # AReaL expects a leading batch dimension of 1
        return {k: v.unsqueeze(0) for k, v in result.items()}


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


class VLARobotWorkflow(RolloutWorkflow):
    """
    AReaL RolloutWorkflow for VLA robot training.

    One call to arun_episode() corresponds to one full robot episode:
      - environment initialised from the task spec in `data`
      - image rendered at each step, passed to the inference engine
      - action tokens decoded to continuous robot action and stepped
      - binary 0/1 reward aligned to token positions
      - post-success tokens masked from the loss

    Parameters
    ----------
    env_factory:
        Callable(task_name, seed) → gym-like environment.  The environment must
        expose reset(), render(mode='rgb_array'), step(action), and return an
        `info` dict with a boolean "success" key.
    action_decoder:
        Callable(token_ids: list[int]) → np.ndarray of shape (action_dim,).
        Converts the VLA model's discrete action token IDs back to continuous
        joint angles / end-effector deltas.
    instruction_tokenizer:
        Callable(instruction_str: str) → list[int].  Converts the task's natural
        language instruction into token IDs for the VLA prompt.
    image_tokenizer:
        Optional callable(np.ndarray image) → list[int].  If the VLA model
        encodes images as discrete token IDs, supply this.  If None, raw numpy
        images are passed directly in VLAStepRequest.image.
    gconfig:
        AReaL GenerationHyperparameters (temperature, top-p, …).  Only
        `max_new_tokens` is forwarded to the VLA engine per step.
    action_chunk_len:
        Number of action tokens generated per environment step.  Must match
        the VLA model's action tokenisation.  Default 7 (OpenVLA-OFT / LIBERO).
    max_episode_steps:
        Maximum number of environment steps before the episode is cut off.
    rollout_stat_scope:
        Name prefix for AReaL's stats_tracker metrics.
    dump_dir:
        If set, saves episode summaries (reward, steps, success) as JSON files
        for debugging.  Mirrors SimpleVLA-RL's video-saving logic.
    """

    def __init__(
        self,
        env_factory: Callable[[str, int], Any],
        action_decoder: Callable[[list[int]], np.ndarray],
        instruction_tokenizer: Callable[[str], list[int]],
        gconfig: Any,  # areal.api.cli_args.GenerationHyperparameters
        image_tokenizer: Callable[[np.ndarray], list[int]] | None = None,
        action_chunk_len: int = 7,
        max_episode_steps: int = 300,
        rollout_stat_scope: str = "robot-rollout",
        dump_dir: str | None = None,
    ) -> None:
        self.env_factory = env_factory
        self.action_decoder = action_decoder
        self.instruction_tokenizer = instruction_tokenizer
        self.image_tokenizer = image_tokenizer
        self.gconfig = gconfig
        self.action_chunk_len = action_chunk_len
        self.max_episode_steps = max_episode_steps
        self.rollout_stat_scope = rollout_stat_scope
        self.dump_dir = dump_dir

        if dump_dir is not None:
            os.makedirs(dump_dir, exist_ok=True)

        logger.info(
            f"VLARobotWorkflow initialised: action_chunk_len={action_chunk_len}, "
            f"max_episode_steps={max_episode_steps}"
        )

    # ------------------------------------------------------------------
    # RolloutWorkflow entry point
    # ------------------------------------------------------------------

    async def arun_episode(
        self,
        engine: VLAEngine,
        data: dict[str, Any],
    ) -> dict[str, torch.Tensor] | None:
        """
        Execute one full robot episode and return an AReaL-format trajectory.

        This is the single method that AReaL's WorkflowExecutor calls
        (via AsyncTaskRunner) for every item that comes off the dataloader.

        Parameters
        ----------
        engine:
            The VLA inference engine.  Must implement VLAEngine (or AReaL's
            InferenceEngine + the VLA-specific fields).
        data:
            A single row from RobotTaskDataset:
                task_name   : str
                instruction : str
                benchmark   : str
                seed        : int

        Returns
        -------
        AReaL trajectory dict with keys:
            input_ids, loss_mask, logprobs, versions, attention_mask, rewards.
        Each tensor has shape [1, seq_len] (leading batch dim = 1).
        Returns None if the episode setup fails (skipped by WorkflowExecutor).
        """
        task_name = data["task_name"]
        instruction = data["instruction"]
        seed = int(data.get("seed", 0))
        version = engine.get_version()

        # ------------------------------------------------------------------
        # 1. Tokenise instruction (once per episode — same prefix every step)
        # ------------------------------------------------------------------
        instruction_ids: list[int] = self.instruction_tokenizer(instruction)

        # ------------------------------------------------------------------
        # 2. Initialise environment
        # ------------------------------------------------------------------
        loop = asyncio.get_event_loop()
        try:
            env = await loop.run_in_executor(
                None, self.env_factory, task_name, seed
            )
            await loop.run_in_executor(None, env.reset)
        except Exception as exc:
            logger.warning(f"[VLARobotWorkflow] env init failed for {task_name}: {exc}")
            return None

        # ------------------------------------------------------------------
        # 3. Episode rollout loop
        # ------------------------------------------------------------------
        buf = _EpisodeBuffer()
        success = False
        finish_step: int = self.max_episode_steps  # pessimistic default
        episode_steps: int = 0
        t_start = time.monotonic()

        for step_idx in range(self.max_episode_steps):
            # 3a. Render current observation --------------------------------
            try:
                image: np.ndarray = await loop.run_in_executor(
                    None, lambda: env.render(mode="rgb_array")
                )
            except Exception as exc:
                logger.warning(f"[VLARobotWorkflow] render failed at step {step_idx}: {exc}")
                break

            # 3b. Build VLA request ----------------------------------------
            image_token_ids: list[int] | None = None
            if self.image_tokenizer is not None:
                image_token_ids = await loop.run_in_executor(
                    None, self.image_tokenizer, image
                )

            req = VLAStepRequest(
                instruction_ids=instruction_ids,
                image=image,
                image_token_ids=image_token_ids,
                max_new_tokens=self.action_chunk_len,
                instruction_text=instruction,  # for VLAInferenceServer
            )

            # 3c. Generate action tokens (async, does not block training) ---
            try:
                resp: VLAStepResponse = await engine.agenerate(req)
            except Exception as exc:
                logger.warning(f"[VLARobotWorkflow] engine.agenerate failed: {exc}")
                break

            # 3d. Decode action tokens → continuous action -----------------
            # VLAInferenceServer pre-decodes actions using model norm_stats in
            # the SimpleVLA env and returns decoded_action directly.
            # VLALocalEngine (same-env) leaves decoded_action=None, so we fall
            # back to the local action_decoder.
            try:
                if resp.decoded_action is not None:
                    action: np.ndarray = resp.decoded_action
                else:
                    action = await loop.run_in_executor(
                        None, self.action_decoder, resp.output_tokens
                    )
            except Exception as exc:
                logger.warning(f"[VLARobotWorkflow] action decode failed: {exc}")
                break

            # 3e. Step environment -----------------------------------------
            try:
                step_result = await loop.run_in_executor(
                    None, env.step, action
                )
                # Support both (obs, rew, done, info) and (obs, rew, term, trunc, info)
                if len(step_result) == 5:
                    obs, _rew, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, _rew, done, info = step_result
            except Exception as exc:
                logger.warning(f"[VLARobotWorkflow] env.step failed: {exc}")
                break

            episode_steps = step_idx + 1

            # 3f. Detect task success ---------------------------------------
            #
            # The info dict from LIBERO/RoboTwin contains a boolean "success"
            # flag.  We record the first step at which success is detected.
            step_succeeded = bool(info.get("success", False))
            if step_succeeded and not success:
                success = True
                finish_step = episode_steps   # 1-indexed: steps 1..finish_step are valid

            # 3g. Determine whether this step's action tokens are post-success
            #
            # Once we record success=True, all subsequent steps' tokens are
            # masked out.  The current step (where success==True for the first
            # time) is STILL included in the loss — it's the pivot step.
            step_is_post_success = success and (step_idx + 1) > finish_step

            # 3h. Buffer this step's tokens --------------------------------
            buf.append_step(resp, step_is_post_success=step_is_post_success)

            if done:
                break

        # ------------------------------------------------------------------
        # 4. Cleanup environment
        # ------------------------------------------------------------------
        try:
            await loop.run_in_executor(None, env.close)
        except Exception:
            pass

        elapsed = time.monotonic() - t_start

        # ------------------------------------------------------------------
        # 5. Compute binary reward
        #
        # SimpleVLA-RL uses outcome-level 0/1 rewards.  No shaping needed.
        # ------------------------------------------------------------------
        binary_reward = 1.0 if success else 0.0

        # ------------------------------------------------------------------
        # 6. Emit stats for AReaL's stats_tracker / wandb
        # ------------------------------------------------------------------
        self._emit_stats(
            reward=binary_reward,
            success=success,
            episode_steps=episode_steps,
            finish_step=finish_step,
            elapsed=elapsed,
            task_name=task_name,
        )

        # ------------------------------------------------------------------
        # 7. Handle degenerate episodes
        #
        # If we collected zero steps (immediate env failure), return None so
        # WorkflowExecutor can skip this sample.
        # ------------------------------------------------------------------
        if episode_steps == 0 or len(buf.input_ids) == 0:
            logger.warning(
                f"[VLARobotWorkflow] episode for {task_name} collected 0 steps — skipping"
            )
            return None

        # ------------------------------------------------------------------
        # 8. Assemble AReaL trajectory tensors
        # ------------------------------------------------------------------
        trajectory = buf.build_tensors(reward=binary_reward)

        # ------------------------------------------------------------------
        # 9. Optional: dump episode summary for debugging
        # ------------------------------------------------------------------
        if self.dump_dir is not None:
            self._dump_episode(
                task_name=task_name,
                seed=seed,
                version=version,
                success=success,
                finish_step=finish_step,
                episode_steps=episode_steps,
                reward=binary_reward,
            )

        return trajectory

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit_stats(
        self,
        reward: float,
        success: bool,
        episode_steps: int,
        finish_step: int,
        elapsed: float,
        task_name: str,
    ) -> None:
        """Push per-episode metrics to AReaL's stats_tracker."""
        try:
            # Lazy import so the module loads even when stats_tracker is unavailable
            from areal.utils.stats_tracker import stats_tracker

            tracker = stats_tracker.get(self.rollout_stat_scope)
            tracker.scalar(reward=reward)
            tracker.scalar(success_rate=float(success))
            tracker.scalar(episode_steps=episode_steps)
            tracker.scalar(episode_wall_time=elapsed)
            if success:
                tracker.scalar(finish_step=finish_step)
        except Exception:
            # stats_tracker may not be initialised in unit-test contexts
            pass

    def _dump_episode(
        self,
        task_name: str,
        seed: int,
        version: int,
        success: bool,
        finish_step: int,
        episode_steps: int,
        reward: float,
    ) -> None:
        """Write a lightweight JSON summary of the episode for post-hoc analysis."""
        import json

        summary = {
            "task_name": task_name,
            "seed": seed,
            "version": version,
            "success": success,
            "finish_step": finish_step,
            "episode_steps": episode_steps,
            "reward": reward,
        }
        fname = (
            f"{task_name.replace('/', '_')}__"
            f"v{version:06d}__"
            f"{'ok' if success else 'fail'}.json"
        )
        fpath = os.path.join(self.dump_dir, fname)
        try:
            with open(fpath, "w") as fh:
                json.dump(summary, fh, indent=2)
        except OSError as exc:
            logger.warning(f"[VLARobotWorkflow] could not write dump: {exc}")


# ---------------------------------------------------------------------------
# Convenience batch collation for GRPO mixed-success groups
# ---------------------------------------------------------------------------


def collate_robot_trajectories(
    trajectories: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """
    Pad and stack a list of per-episode trajectory dicts into a single batch.

    This is a thin wrapper around AReaL's concat_padded_tensors.  It is called
    automatically by WorkflowExecutor but is exposed here for testing.

    The function intentionally filters out None entries (failed episodes) so
    that a bad environment initialisation does not crash the whole batch.
    """
    valid = [t for t in trajectories if t is not None]
    if not valid:
        raise RuntimeError(
            "collate_robot_trajectories: all trajectories in the batch are None. "
            "Check environment setup."
        )
    return concat_padded_tensors(valid)
