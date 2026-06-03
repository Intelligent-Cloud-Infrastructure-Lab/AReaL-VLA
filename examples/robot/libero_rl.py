"""
LIBERO VLA RL Training — AReaL entry point.

Mirrors examples/math/gsm8k_rl.py exactly, substituting:
  - RobotTaskDataset   instead of get_custom_dataset
  - VLARobotWorkflow   instead of MathAgent

Two-environment setup
---------------------
The VLA model runs in a SEPARATE PROCESS (simplevla env, PyTorch 2.2).
AReaL trains here (AReaL venv, PyTorch 2.9 + FSDP2).
They communicate over ZMQ — VLARobotWorkflow connects to the ZMQ server
inside arun_episode() and ignores the InferenceEngine AReaL passes.

Step 1 — start inference server (separate terminal, simplevla env):
    bash examples/robot/start_inference_server.sh \
        --model_path Haozhan72/Openvla-oft-SFT-libero-spatial-traj1 \
        --benchmark libero_spatial \
        --device cuda:0 \
        --address tcp://*:5556

Step 2 — start training (AReaL venv):
    source .venv/bin/activate
    python examples/robot/libero_rl.py \
        examples/robot/conf/libero_grpo.yaml \
        inference_server_address=tcp://localhost:5556 \
        benchmark=libero_spatial
"""

import sys
from dataclasses import dataclass, field

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config

from areal.dataset.robot_dataset import (
    RobotTaskDataset,
    build_task_specs_from_libero_env,
    split_train_val,
)


@dataclass
class VLAGRPOConfig(GRPOConfig):
    """GRPOConfig extended with VLA-specific fields."""
    # Benchmark / environment
    benchmark: str = "libero_spatial"
    n_seeds_per_task: int = 5
    val_fraction: float = 0.1
    seed: int = 42

    # Inference server (simplevla env ZMQ process)
    inference_server_address: str = "tcp://localhost:5556"

    # Episode hyperparameters — verified from shell script
    action_chunks_len: int = 8     # actor_rollout_ref.model.action_chunks_len=8
    max_episode_steps: int = 512   # rob_rollout.py LIBERO max_steps dict
    unnorm_key: str = "libero_spatial_no_noops"


def main(args):
    config, _ = load_expr_config(args, VLAGRPOConfig)

    all_specs = build_task_specs_from_libero_env(
        benchmark_name=config.benchmark,
        n_seeds=config.n_seeds_per_task,
    )
    train_specs, val_specs = split_train_val(
        all_specs,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    train_dataset = RobotTaskDataset(train_specs)
    valid_dataset = RobotTaskDataset(val_specs)

    workflow_kwargs = dict(
        server_address=config.inference_server_address,
        benchmark=config.benchmark,
        action_chunks_len=config.action_chunks_len,
        max_episode_steps=config.max_episode_steps,
        unnorm_key=config.unnorm_key,
    )
    # Eval uses greedy decoding (do_sample=False) — same kwargs for now
    eval_workflow_kwargs = workflow_kwargs.copy()

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.vla_robot.VLARobotWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.vla_robot.VLARobotWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
