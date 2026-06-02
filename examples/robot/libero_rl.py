"""
LIBERO VLA RL Training — AReaL entry point.

Two-environment setup
---------------------
The VLA model (OpenVLA-OFT) runs in a SEPARATE PROCESS inside the SimpleVLA
conda environment (Python 3.10, PyTorch 2.2, transformers-openvla-oft).
AReaL training runs here (Python 3.12, PyTorch 2.9, FSDP2).
They communicate over a ZMQ socket.

Step 1 — start the inference server in simplevla env (separate terminal):
    bash examples/robot/start_inference_server.sh \
        --model_path Haozhan72/Openvla-oft-SFT-libero-spatial-traj1 \
        --benchmark libero_spatial \
        --device cuda:0 \
        --address tcp://*:5555

Step 2 — start AReaL training (this script, in the AReaL env):
    conda activate areal
    python examples/robot/libero_rl.py \
        --config examples/robot/conf/libero_grpo.yaml \
        model_path=Haozhan72/Openvla-oft-SFT-libero-spatial-traj1 \
        benchmark=libero_spatial \
        scheduler.type=local \
        cluster.n_gpus_per_node=8
"""

from __future__ import annotations

import asyncio
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from areal.api.cli_args import (
    AllocationMode,
    FinetuneSpec,
    GenerationHyperparameters,
    load_expr_config,
)
from areal.engine.fsdp_ppo_actor import FSDPPPOActor
from areal.infra.launcher import create_dataloader
from areal.utils.checkpointing import RecoverHandler, Saver
from areal.utils.logging import getLogger
from areal.utils.metrics import Evaluator, StatsLogger
from areal.utils import seeding
from areal.utils.stats_tracker import stats_tracker

# VLA-specific additions (this repo)
from areal.dataset.robot_dataset import (
    RobotCurriculumSampler,
    RobotTaskDataset,
    build_task_specs_from_libero_env,
    split_train_val,
)
from areal.engine.vla_inference_client import VLAInferenceClient
from areal.workflow.vla_robot import VLARobotWorkflow

logger = getLogger("LiberoRL")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class VLARLConfig:
    # Model + paths
    model_path: str = "openvla/openvla-7b-oft"
    benchmark: str = "libero_spatial"
    experiment_name: str = "libero-vla-grpo"
    trial_name: str = "trial-0"

    # Inference server
    inference_server_address: str = "tcp://localhost:5555"
    inference_ping_timeout: float = 60.0    # seconds to wait for server on startup
    inference_request_timeout: float = 600.0  # seconds to wait per episode

    # Episode hyperparameters (verified from shell script)
    action_chunks_len: int = 8    # actor_rollout_ref.model.action_chunks_len=8
    action_token_len: int = 7     # actor_rollout_ref.model.action_token_len=7
    action_dim: int = 7
    max_episode_steps: int = 512  # rob_rollout.py LIBERO max_steps dict
    n_seeds_per_task: int = 5

    # Training
    total_train_steps: int = 5000
    group_size: int = 8           # GRPO: rollouts per task per batch
    seed: int = 42

    # Curriculum sampler
    use_curriculum_sampler: bool = True
    curriculum_epsilon: float = 0.05
    curriculum_ema_alpha: float = 0.1

    # Val split
    val_fraction: float = 0.1
    eval_every_n_steps: int = 50

    # Checkpoint path for weight sync (server reloads from here)
    weight_sync_ckpt_dir: str = "/tmp/areal_vla_weight_sync"

    # Logging
    dump_dir: str | None = None

    # AReaL sub-configs (from YAML)
    actor: dict = field(default_factory=dict)
    saver: dict = field(default_factory=dict)
    stats_logger: dict = field(default_factory=dict)
    evaluator: dict = field(default_factory=dict)
    recover: dict = field(default_factory=dict)
    cluster: dict = field(default_factory=dict)
    scheduler: dict = field(default_factory=dict)
    allocation_mode: str = "d8"


# ---------------------------------------------------------------------------
# LIBERO env factory
# ---------------------------------------------------------------------------


def build_libero_env_factory(benchmark: str):
    """
    Returns a factory that creates LIBERO environments via trial_id indexing.
    trial_id is an index into task_suite.get_task_init_states(task_id),
    NOT a random seed — matches rob_rollout.py env_worker() exactly.
    """
    def make_env(task_name: str, trial_id: int):
        try:
            from libero.libero import benchmark as libero_benchmark  # type: ignore
            from libero.libero.envs import OffScreenRenderEnv        # type: ignore
        except ImportError:
            raise ImportError("Install LIBERO: pip install libero-benchmark")

        suite_name, task_id_str = task_name.rsplit("/", 1)
        task_id = int(task_id_str)

        bm = libero_benchmark.get_benchmark_dict()[suite_name]()
        task = bm.get_task(task_id)
        initial_states = bm.get_task_init_states(task_id)
        initial_state = initial_states[trial_id]

        env = OffScreenRenderEnv(
            bddl_file_name=task.bddl_file,
            camera_heights=256,
            camera_widths=256,
        )
        env.reset()
        obs = env.set_init_state(initial_state)

        # Physics settle steps (num_steps_wait=10 from shell script)
        dummy = __import__("numpy").zeros(7, dtype="float32")
        for _ in range(10):
            obs, _, _, _ = env.step(dummy)

        env._task_description = task.language
        return env

    return make_env


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main(args: Any) -> None:
    config, _ = load_expr_config(args, VLARLConfig)

    rank = int(os.getenv("RANK", "0"))
    seeding.set_random_seed(config.seed, key=f"trainer{rank}")

    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    all_specs = build_task_specs_from_libero_env(
        benchmark_name=config.benchmark,
        n_seeds=config.n_seeds_per_task,
    )
    train_specs, val_specs = split_train_val(
        all_specs, val_fraction=config.val_fraction, seed=config.seed
    )
    logger.info(
        f"Dataset: {len(all_specs)} specs  "
        f"({len(train_specs)} train / {len(val_specs)} val)"
    )

    train_dataset = RobotTaskDataset(train_specs)
    val_dataset   = RobotTaskDataset(val_specs)

    train_sampler = None
    if config.use_curriculum_sampler:
        train_sampler = RobotCurriculumSampler(
            train_dataset,
            epsilon=config.curriculum_epsilon,
            ema_alpha=config.curriculum_ema_alpha,
            seed=config.seed,
        )

    dp_rank = getattr(parallel_strategy, "dp_rank", 0)
    dp_size = getattr(parallel_strategy, "dp_size", 1)

    train_dataloader = create_dataloader(
        train_dataset,
        rank=dp_rank,
        world_size=dp_size,
        batch_size=config.group_size,
        sampler=train_sampler,
    )
    val_dataloader = create_dataloader(
        val_dataset, rank=0, world_size=1, batch_size=config.group_size, shuffle=False
    )

    # ------------------------------------------------------------------
    # 2. Training actor (FSDP2 in AReaL env)
    # ------------------------------------------------------------------
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    ft_spec = FinetuneSpec(
        total_train_epochs=1,
        dataset_size=config.total_train_steps * config.group_size,
        train_batch_size=config.group_size,
    )
    actor.initialize(None, ft_spec)

    # ------------------------------------------------------------------
    # 3. VLA inference client (talks to vla_inference_server.py in SimpleVLA env)
    # ------------------------------------------------------------------
    engine = VLAInferenceClient(
        server_address=config.inference_server_address,
        ping_timeout=config.inference_ping_timeout,
        request_timeout=config.inference_request_timeout,
    )

    # Connect (blocks until server responds to PING)
    asyncio.get_event_loop().run_until_complete(engine.connect())

    # ------------------------------------------------------------------
    # 4. Rollout workflows
    # ------------------------------------------------------------------
    env_factory = build_libero_env_factory(config.benchmark)
    gconfig = GenerationHyperparameters(
        temperature=1.6,  # actor_rollout_ref.rollout.temperature=1.6
        max_new_tokens=config.action_chunks_len,
    )

    workflow = VLARobotWorkflow(
        env_factory=env_factory,
        action_decoder=lambda tokens: None,  # server pre-decodes via norm_stats
        instruction_tokenizer=lambda text: [],  # server tokenises in its own env
        gconfig=gconfig,
        action_chunk_len=config.action_chunks_len,
        max_episode_steps=config.max_episode_steps,
        rollout_stat_scope="train-rollout",
        dump_dir=os.path.join(config.dump_dir, "train") if config.dump_dir else None,
    )

    eval_workflow = VLARobotWorkflow(
        env_factory=env_factory,
        action_decoder=lambda tokens: None,
        instruction_tokenizer=lambda text: [],
        gconfig=GenerationHyperparameters(temperature=0.0, max_new_tokens=config.action_chunks_len),
        action_chunk_len=config.action_chunks_len,
        max_episode_steps=config.max_episode_steps,
        rollout_stat_scope="eval-rollout",
        dump_dir=os.path.join(config.dump_dir, "eval") if config.dump_dir else None,
    )

    # ------------------------------------------------------------------
    # 5. Checkpointing / logging
    # ------------------------------------------------------------------
    os.makedirs(config.weight_sync_ckpt_dir, exist_ok=True)
    saver      = Saver(config.saver, ft_spec)
    stats_log  = StatsLogger(config, ft_spec)
    evaluator  = Evaluator(config.evaluator, ft_spec)
    recover    = RecoverHandler(config.recover, ft_spec)
    recover_info = recover.load(actor, saver, evaluator, stats_log, train_dataloader)
    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None else 0
    )

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    logger.info(f"Starting training from step {start_step}")

    from areal.utils.data import cycle_dataloader
    from areal.utils.stats_tracker import StepInfo

    for global_step in range(start_step, config.total_train_steps):
        step_info = StepInfo(
            global_step=global_step,
            epoch=0,
            epoch_step=global_step,
            steps_per_epoch=config.total_train_steps,
        )

        # Collect rollouts
        with stats_tracker.record_timing("rollout"):
            batch = actor.prepare_batch(
                train_dataloader,
                granularity=config.group_size,
                workflow=workflow,
                should_accept_fn=lambda sample: True,
            )

        # Recompute logprobs from actor (server returns placeholder zeros)
        # kl_coef=0.00 in shell script so only PPO clipping ratio matters;
        # recomputing from the actor gives the correct reference point.
        with stats_tracker.record_timing("recompute_logp"):
            logp = actor.compute_logp(batch)
            batch["prox_logp"] = logp

        # GRPO advantage computation
        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)

        # Gradient step
        with stats_tracker.record_timing("train_step"):
            actor.ppo_update(batch)
            actor.step_lr_scheduler()

        # Save checkpoint and sync weights to inference server
        with stats_tracker.record_timing("weight_sync"):
            ckpt_path = os.path.join(
                config.weight_sync_ckpt_dir, f"step_{global_step+1:06d}.pt"
            )
            actor.save_checkpoint(ckpt_path)
            asyncio.get_event_loop().run_until_complete(
                engine.update_weights(ckpt_path, version=global_step + 1)
            )

        # Update curriculum sampler
        if train_sampler is not None and "task_name" in batch:
            for task, reward in zip(batch["task_name"], batch["rewards"].tolist()):
                train_sampler.update_outcome(task, success=bool(reward > 0.5))

        # Periodic checkpoint
        with stats_tracker.record_timing("save"):
            saver.save(actor, 0, global_step, global_step)

        # Periodic evaluation
        if global_step % config.eval_every_n_steps == 0:
            with stats_tracker.record_timing("eval"):
                evaluator.evaluate(
                    lambda: [
                        asyncio.get_event_loop().run_until_complete(
                            eval_workflow.arun_episode(engine, item)
                        )
                        for batch in val_dataloader
                        for item in batch
                    ],
                    0, global_step, global_step,
                )

        # Log metrics
        stats = stats_tracker.export_all(reduce_group=actor.data_parallel_group)
        stats_log.commit(0, global_step, global_step, stats)
        dist.barrier(group=actor.cpu_group)

    # ------------------------------------------------------------------
    # 7. Cleanup
    # ------------------------------------------------------------------
    logger.info("Training complete")
    asyncio.get_event_loop().run_until_complete(engine.close())
    stats_log.close()
    actor.destroy()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args, _ = p.parse_known_args()
    main(args)
