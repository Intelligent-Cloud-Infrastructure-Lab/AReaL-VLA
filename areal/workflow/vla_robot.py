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

    PPOTrainer instantiates this class with workflow_kwargs as **kwargs,
    then calls arun_episode(engine, data) for each training sample.

    The InferenceEngine passed by AReaL is IGNORED — we use our own
    VLAInferenceClient (ZMQ) to talk to vla_inference_server.py running
    in the simplevla conda environment.

    Parameters (passed as workflow_kwargs from libero_rl.py)
    ---------------------------------------------------------
    server_address : ZMQ address of the inference server
    benchmark      : LIBERO benchmark name (used as unnorm_key fallback)
    action_chunks_len : env steps per VLA generation call (default 8)
    max_episode_steps : max LIBERO steps per episode (default 512)
    unnorm_key     : action unnormalisation key for generate_action_verl
    """

    def __init__(self, **kwargs):
        self.server_address   = kwargs.get("server_address", "tcp://localhost:5556")
        self.benchmark        = kwargs.get("benchmark", "libero_spatial")
        self.action_chunks_len = kwargs.get("action_chunks_len", 8)
        self.max_episode_steps = kwargs.get("max_episode_steps", 512)
        self.unnorm_key       = kwargs.get("unnorm_key", "libero_spatial_no_noops")
        self.dump_dir         = kwargs.get("dump_dir", None)
        self.rollout_stat_scope = kwargs.get("rollout_stat_scope", "robot-rollout")

        # ZMQ client — connected lazily on first arun_episode call
        self._client = None
        self._client_lock = None  # asyncio.Lock, created on first use

        if self.dump_dir is not None:
            os.makedirs(self.dump_dir, exist_ok=True)

        logger.info(
            f"VLARobotWorkflow: server={self.server_address}  "
            f"benchmark={self.benchmark}  "
            f"action_chunks_len={self.action_chunks_len}  "
            f"max_episode_steps={self.max_episode_steps}"
        )

    async def _get_client(self):
        """Lazy-connect the ZMQ client (safe for concurrent coroutines)."""
        if self._client_lock is None:
            self._client_lock = asyncio.Lock()
        async with self._client_lock:
            if self._client is None:
                from areal.engine.vla_inference_client import VLAInferenceClient
                self._client = VLAInferenceClient(
                    server_address=self.server_address,
                    ping_timeout=60.0,
                    request_timeout=600.0,
                )
                await self._client.connect()
        return self._client

    def _make_env(self, task_key: str, trial_id: int):
        """
        Create and reset a LIBERO environment.
        task_key format: "libero_spatial/3"  (benchmark/task_idx)
        trial_id: index into task_suite.get_task_init_states(task_idx)
        """
        import numpy as np
        from libero.libero import benchmark as libero_benchmark  # type: ignore
        from libero.libero import get_libero_path               # type: ignore
        from libero.libero.envs import OffScreenRenderEnv        # type: ignore

        suite_name, task_id_str = task_key.rsplit("/", 1)
        task_id = int(task_id_str)

        bm = libero_benchmark.get_benchmark_dict()[suite_name]()
        task = bm.get_task(task_id)
        initial_states = bm.get_task_init_states(task_id)
        initial_state = initial_states[trial_id]

        import os as _os
        task_bddl_file = _os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file,
        )
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=256,
            camera_widths=256,
        )
        env.reset()
        obs = env.set_init_state(initial_state)

        # Physics settle (num_steps_wait=10 from shell script)
        dummy = np.zeros(7, dtype=np.float32)
        for _ in range(10):
            obs, _, _, _ = env.step(dummy)

        return env, obs, task.language

    async def arun_episode(
        self,
        engine: Any,   # AReaL InferenceEngine — ignored, we use ZMQ client
        data: dict[str, Any],
    ) -> dict[str, torch.Tensor] | None:
        """
        Run one full LIBERO episode and return an AReaL-format trajectory.

        Parameters
        ----------
        engine : AReaL InferenceEngine (ignored — we use VLAInferenceClient via ZMQ)
        data   : dict from RobotTaskDataset:
                   task_name   : "libero_spatial/3"  (suite/task_idx)
                   instruction : str
                   benchmark   : str
                   seed        : int (trial_id into initial_states[])
        """
        task_key    = data["task_name"]
        instruction = data["instruction"]
        trial_id    = int(data.get("seed", 0))

        # 1. Get ZMQ client (lazy connect on first call)
        try:
            client = await self._get_client()
        except Exception as exc:
            logger.warning(f"[VLARobotWorkflow] ZMQ client connect failed: {exc}")
            return None

        # 2. Create LIBERO environment
        loop = asyncio.get_event_loop()
        try:
            env, obs, instr = await loop.run_in_executor(
                None, self._make_env, task_key, trial_id
            )
            instruction = instr  # use task's own language instruction
        except Exception as exc:
            logger.warning(f"[VLARobotWorkflow] env init failed {task_key}: {exc}")
            return None

        # 3. Episode loop
        buf = _EpisodeBuffer()
        success = False
        finish_step = self.max_episode_steps
        episode_steps = 0
        t_start = time.monotonic()
        done = False

        while episode_steps < self.max_episode_steps and not done:
            # Render image
            try:
                image = obs.get("agentview_image")
                if image is None:
                    image = await loop.run_in_executor(
                        None, lambda: env.render(mode="rgb_array")
                    )
                image = np.ascontiguousarray(image[::-1, ::-1])
            except Exception as exc:
                logger.warning(f"[VLARobotWorkflow] render failed: {exc}")
                break

            # Request action chunk from ZMQ server
            req = VLAStepRequest(
                instruction_ids=[],
                image=image,
                max_new_tokens=self.action_chunks_len,
                instruction_text=instruction,
            )
            try:
                resp: VLAStepResponse = await client.agenerate(req)
            except Exception as exc:
                logger.warning(f"[VLARobotWorkflow] agenerate failed: {exc}")
                break

            action = resp.decoded_action  # continuous action, pre-decoded by server
            if action is None:
                logger.warning("[VLARobotWorkflow] decoded_action is None — skipping")
                break

            # Step action_chunks_len environment steps
            step_success = False
            try:
                for chunk_step in range(self.action_chunks_len):
                    if episode_steps >= self.max_episode_steps:
                        done = True
                        break
                    a = action[min(chunk_step, len(action) - 1)]
                    obs, _, done, info = await loop.run_in_executor(None, env.step, a)
                    episode_steps += 1
                    if info.get("success", False):
                        step_success = True
                        break
                    if done:
                        break
            except Exception as exc:
                logger.warning(f"[VLARobotWorkflow] env.step failed: {exc}")
                break

            if step_success and not success:
                success = True
                finish_step = episode_steps

            step_is_post_success = success and episode_steps > finish_step
            buf.append_step(resp, step_is_post_success=step_is_post_success)

        # 4. Cleanup
        try:
            await loop.run_in_executor(None, env.close)
        except Exception:
            pass

        elapsed = time.monotonic() - t_start
        binary_reward = 1.0 if success else 0.0

        self._emit_stats(
            reward=binary_reward, success=success,
            episode_steps=episode_steps, finish_step=finish_step,
            elapsed=elapsed, task_name=task_key,
        )

        if episode_steps == 0 or len(buf.input_ids) == 0:
            logger.warning(f"[VLARobotWorkflow] 0 steps for {task_key} — skipping")
            return None

        trajectory = buf.build_tensors(reward=binary_reward)

        if self.dump_dir is not None:
            self._dump_episode(
                task_name=task_key, seed=trial_id,
                version=client.get_version(), success=success,
                finish_step=finish_step, episode_steps=episode_steps,
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
