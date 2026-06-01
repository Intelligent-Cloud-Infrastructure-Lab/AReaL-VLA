"""
LIBERO VLA RL Training — AReaL entry point.

This script is the AReaL equivalent of SimpleVLA-RL's
  examples/run_openvla_oft_rl_libero.sh → verl/trainer/main_ppo.py

It wires together:
  - RobotTaskDataset  (areal/dataset/robot_dataset.py)
  - VLARobotWorkflow  (areal/workflow/vla_robot.py)
  - VLALocalEngine    (areal/engine/vla_local_engine.py)
  - AReaL's FSDPPPOActor + RemoteSGLangEngine training loop

Usage (single node, 8 GPUs)
----------------------------
    python examples/robot/libero_rl.py \
        --config examples/robot/conf/libero_grpo.yaml \
        model.path=/path/to/openvla_oft_sft \
        benchmark=libero_object \
        scheduler.type=local

Multi-node (2 × 8 GPUs)
------------------------
    python examples/robot/libero_rl.py \
        --config examples/robot/conf/libero_grpo.yaml \
        model.path=/path/to/openvla_oft_sft \
        cluster.n_nodes=2 \
        cluster.n_gpus_per_node=8 \
        cluster.fileroot=/shared/nfs/path \
        scheduler.type=ray
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

# AReaL imports — standard across all AReaL training scripts
from areal.api.cli_args import (
    AllocationMode,
    FinetuneSpec,
    GenerationHyperparameters,
    load_expr_config,
)
from areal.engine.fsdp_ppo_actor import FSDPPPOActor
from areal.engine.remote_sglang import RemoteSGLangEngine
from areal.engine.weight_update import WeightUpdateMeta
from areal.infra.launcher import create_dataloader
from areal.reward.robot_reward import get_reward_fn
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
from areal.engine.vla_local_engine import (
    OpenVLAOFTModel,
    VLALocalEngine,
    make_openvla_action_decoder,
)
from areal.workflow.vla_robot import VLARobotWorkflow

logger = getLogger("LiberoRL")


# ---------------------------------------------------------------------------
# Config dataclass (extends standard AReaL config with VLA-specific fields)
# ---------------------------------------------------------------------------


@dataclass
class VLARLConfig:
    """
    Top-level config for VLA robot RL training.

    Hydra populates this from the YAML file (examples/robot/conf/libero_grpo.yaml)
    and CLI overrides.
    """

    # Paths
    model_path: str = "openvla/openvla-7b-oft"
    benchmark: str = "libero_object"
    experiment_name: str = "libero-vla-grpo"
    trial_name: str = "trial-0"

    # Episode hyperparameters
    action_chunk_len: int = 7        # OpenVLA-OFT default
    action_dim: int = 7              # 7-DOF robot arm
    max_episode_steps: int = 300     # LIBERO default horizon
    n_seeds_per_task: int = 5        # seeds for environment diversity

    # Training
    total_train_steps: int = 5000
    group_size: int = 8              # GRPO group size (= n rollouts per task)
    seed: int = 42

    # Dynamic sampling curriculum
    use_curriculum_sampler: bool = True
    curriculum_epsilon: float = 0.05
    curriculum_ema_alpha: float = 0.1

    # Val fraction (fraction of tasks held out for evaluation)
    val_fraction: float = 0.1
    eval_every_n_steps: int = 50

    # Generation hyperparameters (temperature, top-p, …)
    temperature: float = 1.0
    eval_temperature: float = 0.0   # greedy decoding at eval

    # Logging
    dump_dir: str | None = None      # set to a path to save episode summaries

    # AReaL sub-configs (populated from YAML)
    actor: dict = field(default_factory=dict)
    rollout: dict = field(default_factory=dict)
    saver: dict = field(default_factory=dict)
    stats_logger: dict = field(default_factory=dict)
    evaluator: dict = field(default_factory=dict)
    recover: dict = field(default_factory=dict)
    cluster: dict = field(default_factory=dict)
    scheduler: dict = field(default_factory=dict)
    allocation_mode: str = "d8"     # e.g. "d8" = 8-way data parallel


# ---------------------------------------------------------------------------
# VLA model / engine factory helpers
# ---------------------------------------------------------------------------


def build_vla_engine(config: VLARLConfig, device: str = "cuda") -> VLALocalEngine:
    """
    Load the VLA model and wrap it in VLALocalEngine.

    For OpenVLA-OFT, we load via HuggingFace transformers.  Adapt this function
    for other VLA models (π0, RoboFlamingo, …) by swapping the model class.
    """
    from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore[import]

    logger.info(f"Loading VLA model from {config.model_path}")
    model = AutoModelForVision2Seq.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(config.model_path, trust_remote_code=True)

    vla_model = OpenVLAOFTModel(
        model=model,
        processor=processor,
        action_chunk_len=config.action_chunk_len,
        action_dim=config.action_dim,
        device=device,
    )
    engine = VLALocalEngine(
        model=vla_model,
        action_chunk_len=config.action_chunk_len,
        temperature=config.temperature,
    )
    return engine


def build_libero_env_factory(benchmark: str):
    """Return a closure that creates LIBERO environments."""

    def make_env(task_name: str, seed: int):
        """Create and initialise a LIBERO environment."""
        try:
            import libero.libero.envs.bddl_utils as bu  # type: ignore[import]
            from libero.libero.envs import OffScreenRenderEnv  # type: ignore[import]

            task_suite_name, task_name_short = task_name.split("/", 1)
            env = OffScreenRenderEnv(
                task_name=task_name_short,
                task_suite_name=task_suite_name or benchmark,
            )
            env.seed(seed)
            env.reset()
            return env
        except ImportError:
            raise ImportError(
                "LIBERO is not installed.  Run: pip install libero-benchmark"
            )

    return make_env


def build_instruction_tokenizer(model_path: str):
    """Return a tokenizer callable for language instructions."""
    from transformers import AutoTokenizer  # type: ignore[import]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def tokenize(instruction: str) -> list[int]:
        return tokenizer.encode(instruction, add_special_tokens=False)

    return tokenize


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main(args: Any) -> None:
    config, _ = load_expr_config(args, VLARLConfig)

    rank = int(os.getenv("RANK", "0"))
    seeding.set_random_seed(config.seed, key=f"trainer{rank}")

    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None, "allocation_mode must define a train strategy"

    # ------------------------------------------------------------------
    # 1. Build task dataset (train + val splits)
    # ------------------------------------------------------------------
    all_specs = build_task_specs_from_libero_env(
        benchmark_name=config.benchmark,
        n_seeds=config.n_seeds_per_task,
    )
    train_specs, val_specs = split_train_val(
        all_specs,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )

    logger.info(
        f"Dataset: {len(all_specs)} total specs "
        f"({len(train_specs)} train, {len(val_specs)} val)"
    )

    train_dataset = RobotTaskDataset(train_specs)
    val_dataset = RobotTaskDataset(val_specs)

    # ------------------------------------------------------------------
    # 2. Build dataloaders (with optional curriculum sampler)
    # ------------------------------------------------------------------
    train_sampler: Any = None
    if config.use_curriculum_sampler:
        train_sampler = RobotCurriculumSampler(
            train_dataset,
            epsilon=config.curriculum_epsilon,
            ema_alpha=config.curriculum_ema_alpha,
            seed=config.seed,
        )
        logger.info("Curriculum sampler enabled (dynamic task weighting)")

    train_dataloader = create_dataloader(
        train_dataset,
        rank=parallel_strategy.dp_rank if hasattr(parallel_strategy, "dp_rank") else 0,
        world_size=parallel_strategy.dp_size,
        batch_size=config.group_size,  # GRPO: group_size rollouts per batch
        sampler=train_sampler,
    )
    val_dataloader = create_dataloader(
        val_dataset,
        rank=0,
        world_size=1,
        batch_size=config.group_size,
        shuffle=False,
    )

    # ------------------------------------------------------------------
    # 3. Initialise training actor (FSDP PPO/GRPO)
    # ------------------------------------------------------------------
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    ft_spec = FinetuneSpec(
        total_train_epochs=1,
        dataset_size=config.total_train_steps * config.group_size,
        train_batch_size=config.group_size,
    )

    # ------------------------------------------------------------------
    # 4. Build VLA inference engine (local or remote)
    #
    # For production, replace VLALocalEngine with a SGLang-based engine once
    # the VLA model is supported by SGLang.
    # ------------------------------------------------------------------
    device = f"cuda:{rank % torch.cuda.device_count()}"
    vla_engine = build_vla_engine(config, device=device)

    env_factory = build_libero_env_factory(config.benchmark)
    instruction_tokenizer = build_instruction_tokenizer(config.model_path)
    action_decoder = make_openvla_action_decoder(
        action_dim=config.action_dim,
        action_chunk_len=config.action_chunk_len,
    )

    # ------------------------------------------------------------------
    # 5. Build rollout workflows (train + eval)
    # ------------------------------------------------------------------
    dump_dir = config.dump_dir or None
    gconfig = GenerationHyperparameters(
        temperature=config.temperature,
        max_new_tokens=config.action_chunk_len,
    )

    workflow = VLARobotWorkflow(
        env_factory=env_factory,
        action_decoder=action_decoder,
        instruction_tokenizer=instruction_tokenizer,
        gconfig=gconfig,
        action_chunk_len=config.action_chunk_len,
        max_episode_steps=config.max_episode_steps,
        rollout_stat_scope="train-rollout",
        dump_dir=os.path.join(dump_dir, "train") if dump_dir else None,
    )

    eval_workflow = VLARobotWorkflow(
        env_factory=env_factory,
        action_decoder=action_decoder,
        instruction_tokenizer=instruction_tokenizer,
        gconfig=GenerationHyperparameters(
            temperature=config.eval_temperature,
            max_new_tokens=config.action_chunk_len,
        ),
        action_chunk_len=config.action_chunk_len,
        max_episode_steps=config.max_episode_steps,
        rollout_stat_scope="eval-rollout",
        dump_dir=os.path.join(dump_dir, "eval") if dump_dir else None,
    )

    # ------------------------------------------------------------------
    # 6. Connect actor → inference engine (weight sync channel)
    # ------------------------------------------------------------------
    actor.initialize(None, ft_spec)
    # For VLALocalEngine: weight updates are pushed via vla_engine.set_weights()
    # AReaL's standard weight_update_meta is only needed for SGLang engines.
    # weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)
    # actor.connect_engine(vla_engine, weight_update_meta)

    # ------------------------------------------------------------------
    # 7. Setup checkpointing, logging, evaluation
    # ------------------------------------------------------------------
    saver = Saver(config.saver, ft_spec)
    stats_log = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)
    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(actor, saver, evaluator, stats_log, train_dataloader)
    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )

    # ------------------------------------------------------------------
    # 8. Main training loop
    #
    # This mirrors examples/math/gsm8k_rl.py but with robot-specific hooks.
    # ------------------------------------------------------------------
    logger.info(f"Starting VLA robot RL training from step {start_step}")

    from areal.utils.data import cycle_dataloader
    from areal.utils.stats_tracker import StepInfo

    total_steps = config.total_train_steps
    data_gen = cycle_dataloader(train_dataloader)

    for global_step in range(start_step, total_steps):
        step_info = StepInfo(
            global_step=global_step,
            epoch=0,
            epoch_step=global_step,
            steps_per_epoch=total_steps,
        )

        # ---- Rollout: collect a group of robot episode trajectories ----
        with stats_tracker.record_timing("rollout"):
            batch = actor.prepare_batch(
                train_dataloader,
                granularity=config.group_size,
                workflow=workflow,
                should_accept_fn=lambda sample: True,
            )

        # ---- Optional: recompute logprobs with current actor weights ----
        if getattr(config.actor, "recompute_logprob", False):
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp

        # ---- Compute GRPO / PPO advantages ----
        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)

        # ---- Gradient step ----
        with stats_tracker.record_timing("train_step"):
            actor.ppo_update(batch)
            actor.step_lr_scheduler()

        # ---- Push new weights to VLA inference engine ----
        with stats_tracker.record_timing("update_weights"):
            # For VLALocalEngine, extract the state dict and push
            new_state_dict = actor.state_dict()
            vla_engine.set_weights(new_state_dict)
            vla_engine.set_version(global_step + 1)

        # ---- Optional: update curriculum sampler ----
        if train_sampler is not None and "task_name" in batch:
            for task, reward in zip(batch["task_name"], batch["rewards"].tolist()):
                train_sampler.update_outcome(task, success=bool(reward > 0.5))

        # ---- Save checkpoint ----
        with stats_tracker.record_timing("save"):
            saver.save(actor, 0, global_step, global_step)

        # ---- Evaluation ----
        if global_step % config.eval_every_n_steps == 0:
            with stats_tracker.record_timing("eval"):

                def evaluate_fn():
                    for data in val_dataloader:
                        for item in data:
                            vla_engine.submit(item, eval_workflow)

                evaluator.evaluate(evaluate_fn, 0, global_step, global_step)

        # ---- Log metrics ----
        stats = stats_tracker.export_all(reduce_group=actor.data_parallel_group)
        stats_log.commit(0, global_step, global_step, stats)

        dist.barrier(group=actor.cpu_group)

    # ------------------------------------------------------------------
    # 9. Cleanup
    # ------------------------------------------------------------------
    logger.info("Training complete")
    stats_log.close()
    vla_engine.destroy()
    actor.destroy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLA robot RL training on LIBERO")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args, overrides = parser.parse_known_args()
    main(args)
