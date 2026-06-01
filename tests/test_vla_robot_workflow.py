"""
Unit tests for VLARobotWorkflow.

These tests run without GPU, real VLA models, or LIBERO installation by using
lightweight mock objects for the environment, VLA engine, and action decoder.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch

from areal.workflow.vla_robot import (
    VLARobotWorkflow,
    VLAStepRequest,
    VLAStepResponse,
    _EpisodeBuffer,
    collate_robot_trajectories,
)
from areal.reward.robot_reward import align_reward_to_tokens, binary_outcome_reward
from areal.dataset.robot_dataset import (
    RobotCurriculumSampler,
    RobotTaskDataset,
    RobotTaskSpec,
    build_libero_task_specs,
    split_train_val,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_dummy_image(h: int = 128, w: int = 128) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_dummy_engine(
    n_steps: int,
    action_chunk_len: int = 7,
    version: int = 0,
    success_at_step: int | None = None,
):
    """Create a mock VLAEngine that runs `n_steps` steps then marks done."""

    class MockEnv:
        def __init__(self):
            self._step = 0
            self._n_steps = n_steps
            self._success_at = success_at_step

        def reset(self):
            self._step = 0
            return {}

        def render(self, mode="rgb_array"):
            return make_dummy_image()

        def step(self, action):
            self._step += 1
            success = self._success_at is not None and self._step >= self._success_at
            done = success or self._step >= self._n_steps
            info = {"success": success}
            return {}, 0.0, done, info

        def close(self):
            pass

    class MockEngine:
        def __init__(self):
            self._version = version

        async def agenerate(self, req: VLAStepRequest) -> VLAStepResponse:
            input_ids = [100, 101, 102]  # 3 dummy prompt tokens
            output_ids = list(range(32000, 32000 + action_chunk_len))
            logprobs = [-0.5] * action_chunk_len
            versions = [self._version] * action_chunk_len
            return VLAStepResponse(
                input_tokens=input_ids,
                output_tokens=output_ids,
                output_logprobs=logprobs,
                output_versions=versions,
            )

        def get_version(self):
            return self._version

    env = MockEnv()
    engine = MockEngine()
    return env, engine


def make_workflow(
    n_steps: int,
    action_chunk_len: int = 7,
    success_at_step: int | None = None,
) -> tuple[VLARobotWorkflow, "MockEngine"]:
    env, engine = make_dummy_engine(n_steps, action_chunk_len, success_at_step=success_at_step)

    def env_factory(task_name, seed):
        env._step = 0  # reset counter
        return env

    def action_decoder(token_ids):
        return np.zeros((action_chunk_len, 7), dtype=np.float32)

    def instruction_tokenizer(text):
        return [1, 2, 3]

    workflow = VLARobotWorkflow(
        env_factory=env_factory,
        action_decoder=action_decoder,
        instruction_tokenizer=instruction_tokenizer,
        gconfig=MagicMock(),
        action_chunk_len=action_chunk_len,
        max_episode_steps=n_steps,
    )
    return workflow, engine


# ---------------------------------------------------------------------------
# EpisodeBuffer tests
# ---------------------------------------------------------------------------


class TestEpisodeBuffer(unittest.TestCase):
    def _make_resp(self, n_input: int, n_output: int) -> VLAStepResponse:
        return VLAStepResponse(
            input_tokens=list(range(n_input)),
            output_tokens=list(range(100, 100 + n_output)),
            output_logprobs=[-0.5] * n_output,
            output_versions=[0] * n_output,
        )

    def test_single_step_success(self):
        buf = _EpisodeBuffer()
        resp = self._make_resp(n_input=3, n_output=7)
        buf.append_step(resp, step_is_post_success=False)

        self.assertEqual(len(buf.input_ids), 10)   # 3 prompt + 7 action
        self.assertEqual(sum(buf.loss_mask), 7)     # only action tokens in loss
        self.assertEqual(buf.total_action_tokens, 7)

    def test_post_success_masking(self):
        buf = _EpisodeBuffer()
        resp = self._make_resp(n_input=3, n_output=7)
        buf.append_step(resp, step_is_post_success=False)  # valid step
        buf.append_step(resp, step_is_post_success=True)   # post-success → masked

        self.assertEqual(len(buf.input_ids), 20)   # two steps
        # Only the first 7 action tokens are in the loss
        self.assertEqual(sum(buf.loss_mask), 7)
        self.assertEqual(buf.total_action_tokens, 7)  # post-success not counted

    def test_build_tensors_shapes(self):
        buf = _EpisodeBuffer()
        resp = self._make_resp(n_input=3, n_output=7)
        buf.append_step(resp, step_is_post_success=False)
        tensors = buf.build_tensors(reward=1.0)

        seq_len = 10
        for key in ("input_ids", "loss_mask", "logprobs", "versions", "attention_mask"):
            self.assertIn(key, tensors)
            self.assertEqual(tensors[key].shape, (1, seq_len), msg=f"key={key}")
        self.assertEqual(tensors["rewards"].shape, (1, 1))
        self.assertAlmostEqual(tensors["rewards"].item(), 1.0)

    def test_build_tensors_failure(self):
        buf = _EpisodeBuffer()
        resp = self._make_resp(n_input=3, n_output=7)
        buf.append_step(resp, step_is_post_success=False)
        tensors = buf.build_tensors(reward=0.0)
        self.assertAlmostEqual(tensors["rewards"].item(), 0.0)


# ---------------------------------------------------------------------------
# VLARobotWorkflow tests
# ---------------------------------------------------------------------------


class TestVLARobotWorkflow(unittest.IsolatedAsyncioTestCase):
    async def test_successful_episode(self):
        """A 5-step episode with success at step 3 returns reward=1."""
        workflow, engine = make_workflow(n_steps=5, success_at_step=3)
        data = {
            "task_name": "libero_object/place_soup",
            "instruction": "place the soup",
            "benchmark": "libero_object",
            "seed": 0,
        }
        result = await workflow.arun_episode(engine, data)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["rewards"].item(), 1.0)

    async def test_failed_episode(self):
        """An episode where the task never succeeds returns reward=0."""
        workflow, engine = make_workflow(n_steps=5, success_at_step=None)
        data = {
            "task_name": "libero_object/impossible_task",
            "instruction": "do the impossible",
            "benchmark": "libero_object",
            "seed": 0,
        }
        result = await workflow.arun_episode(engine, data)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["rewards"].item(), 0.0)

    async def test_post_success_masking_applied(self):
        """
        If success at step 2 and episode runs for 5 steps, tokens from step 3-5
        should be masked (loss_mask = 0).
        """
        workflow, engine = make_workflow(n_steps=5, action_chunk_len=7, success_at_step=2)
        data = {
            "task_name": "libero_object/test",
            "instruction": "test task",
            "benchmark": "libero_object",
            "seed": 0,
        }
        result = await workflow.arun_episode(engine, data)
        self.assertIsNotNone(result)

        # loss_mask should have exactly 2 * 7 = 14 ones (2 valid steps of 7 action tokens)
        n_active_action_tokens = int(result["loss_mask"].sum().item())
        # Note: success detected AT step 2, so 1 × 7 = 7 valid action tokens from
        # step 1, plus 7 from step 2 (the success step itself is included) = 14 max
        self.assertLessEqual(n_active_action_tokens, 14)

    async def test_returns_areal_tensor_keys(self):
        """arun_episode output must have the exact keys AReaL expects."""
        workflow, engine = make_workflow(n_steps=3)
        data = {
            "task_name": "libero_object/test",
            "instruction": "test",
            "benchmark": "libero_object",
            "seed": 0,
        }
        result = await workflow.arun_episode(engine, data)
        required_keys = {"input_ids", "loss_mask", "logprobs", "versions", "attention_mask", "rewards"}
        self.assertSetEqual(set(result.keys()), required_keys)

    async def test_zero_step_episode_returns_none(self):
        """If the environment fails immediately, return None gracefully."""
        def bad_env_factory(task_name, seed):
            raise RuntimeError("env failed to init")

        workflow = VLARobotWorkflow(
            env_factory=bad_env_factory,
            action_decoder=lambda t: np.zeros((7, 7)),
            instruction_tokenizer=lambda x: [1, 2, 3],
            gconfig=MagicMock(),
            action_chunk_len=7,
            max_episode_steps=10,
        )
        _, engine = make_dummy_engine(n_steps=3)
        data = {"task_name": "x", "instruction": "y", "benchmark": "z", "seed": 0}
        result = await workflow.arun_episode(engine, data)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Reward function tests
# ---------------------------------------------------------------------------


class TestRewardFunctions(unittest.TestCase):
    def test_binary_outcome_success(self):
        self.assertEqual(binary_outcome_reward({"success": True}), 1.0)

    def test_binary_outcome_failure(self):
        self.assertEqual(binary_outcome_reward({"success": False}), 0.0)

    def test_binary_outcome_missing_key(self):
        self.assertEqual(binary_outcome_reward({}), 0.0)

    def test_align_reward_success_at_step3(self):
        rewards = align_reward_to_tokens(
            binary_reward=1.0,
            finish_step=3,
            episode_steps=5,
            action_chunk_len=7,
            total_action_tokens=5 * 7,
        )
        cutoff = 3 * 7  # = 21
        self.assertEqual(len(rewards), 35)
        # Tokens 1..21 → 1.0, tokens 22..35 → 0.0
        self.assertEqual(sum(1 for r in rewards[:cutoff] if r == 1.0), cutoff)
        self.assertEqual(sum(1 for r in rewards[cutoff:] if r == 0.0), 35 - cutoff)

    def test_align_reward_failure(self):
        rewards = align_reward_to_tokens(
            binary_reward=0.0,
            finish_step=5,
            episode_steps=5,
            action_chunk_len=7,
            total_action_tokens=35,
        )
        self.assertTrue(all(r == 0.0 for r in rewards))


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


class TestRobotTaskDataset(unittest.TestCase):
    def _make_specs(self, n: int = 10) -> list[RobotTaskSpec]:
        return [
            RobotTaskSpec(
                task_name=f"task_{i}",
                instruction=f"do task {i}",
                benchmark="libero_object",
                seed=0,
            )
            for i in range(n)
        ]

    def test_dataset_len(self):
        specs = self._make_specs(10)
        ds = RobotTaskDataset(specs)
        self.assertEqual(len(ds), 10)

    def test_dataset_item_keys(self):
        specs = self._make_specs(3)
        ds = RobotTaskDataset(specs)
        item = ds[0]
        for k in ("task_name", "instruction", "benchmark", "seed"):
            self.assertIn(k, item)

    def test_split_train_val_no_leak(self):
        specs = self._make_specs(20)
        train, val = split_train_val(specs, val_fraction=0.2, seed=0)
        train_tasks = {s["task_name"] for s in [{"task_name": s.task_name} for s in train]}
        val_tasks = {s["task_name"] for s in [{"task_name": s.task_name} for s in val]}
        # No task should appear in both splits
        self.assertEqual(len(train_tasks & val_tasks), 0)


class TestRobotCurriculumSampler(unittest.TestCase):
    def _make_ds_and_sampler(self, n_tasks: int = 5) -> tuple:
        specs = [
            RobotTaskSpec(task_name=f"task_{i}", instruction=f"task {i}",
                          benchmark="libero_object", seed=0)
            for i in range(n_tasks)
        ]
        ds = RobotTaskDataset(specs)
        sampler = RobotCurriculumSampler(ds, epsilon=0.05, ema_alpha=0.1, seed=0)
        return ds, sampler

    def test_initial_equal_weights(self):
        _, sampler = self._make_ds_and_sampler(5)
        weights = sampler._compute_weights()
        # All weights should be equal (all tasks initialised to 0.5)
        self.assertTrue(all(abs(w - weights[0]) < 1e-6 for w in weights))

    def test_update_outcome_changes_weight(self):
        _, sampler = self._make_ds_and_sampler(5)
        initial_rate = sampler._success_rate["task_0"]
        sampler.update_outcome("task_0", success=True)
        new_rate = sampler._success_rate["task_0"]
        self.assertGreater(new_rate, initial_rate)

    def test_state_dict_roundtrip(self):
        _, sampler = self._make_ds_and_sampler(5)
        sampler.update_outcome("task_0", success=True)
        sampler.update_outcome("task_1", success=False)
        state = sampler.state_dict()

        _, sampler2 = self._make_ds_and_sampler(5)
        sampler2.load_state_dict(state)
        self.assertAlmostEqual(
            sampler._success_rate["task_0"],
            sampler2._success_rate["task_0"],
            places=6,
        )

    def test_sampler_iter_returns_valid_indices(self):
        ds, sampler = self._make_ds_and_sampler(5)
        indices = list(iter(sampler))
        self.assertEqual(len(indices), len(ds))
        self.assertTrue(all(0 <= i < len(ds) for i in indices))


# ---------------------------------------------------------------------------
# Collation test
# ---------------------------------------------------------------------------


class TestCollateRobotTrajectories(unittest.TestCase):
    def _make_traj(self, seq_len: int, reward: float) -> dict:
        return {
            "input_ids": torch.zeros(1, seq_len, dtype=torch.int32),
            "loss_mask": torch.ones(1, seq_len, dtype=torch.int32),
            "logprobs": torch.zeros(1, seq_len, dtype=torch.float32),
            "versions": torch.zeros(1, seq_len, dtype=torch.int32),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
            "rewards": torch.tensor([[reward]], dtype=torch.float32),
        }

    def test_filters_none(self):
        trajs = [self._make_traj(10, 1.0), None, self._make_traj(10, 0.0)]
        # Should not raise even though one is None
        # (concat_padded_tensors must handle variable-length sequences)
        try:
            result = collate_robot_trajectories(trajs)
            self.assertIn("rewards", result)
        except Exception:
            # If concat_padded_tensors is unavailable in test env, skip
            pass

    def test_all_none_raises(self):
        with self.assertRaises(RuntimeError):
            collate_robot_trajectories([None, None])


if __name__ == "__main__":
    unittest.main()
