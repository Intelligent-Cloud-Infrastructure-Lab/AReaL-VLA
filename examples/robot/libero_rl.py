"""
LIBERO VLA RL Training.

Mirrors examples/math/gsm8k_rl.py exactly — only the dataset
and workflow differ.

Step 1 — start inference server (simplevla env, separate terminal):
    bash examples/robot/start_inference_server.sh \
        --model_path Haozhan72/Openvla-oft-SFT-libero-spatial-traj1 \
        --benchmark libero_spatial \
        --device cuda:0 \
        --address tcp://*:5556

Step 2 — start training (AReaL venv):
    source .venv/bin/activate
    python examples/robot/libero_rl.py \
        examples/robot/conf/libero_grpo.yaml
"""

import sys

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config

from areal.dataset.robot_dataset import (
    RobotTaskDataset,
    build_task_specs_from_libero_env,
    split_train_val,
)


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)

    # VLA-specific fields come from the YAML (accessed via getattr with defaults)
    benchmark      = getattr(config, "benchmark", "libero_spatial")
    n_seeds        = getattr(config, "n_seeds_per_task", 5)
    val_fraction   = getattr(config, "val_fraction", 0.1)
    seed           = getattr(config, "seed", 42)
    server_address = getattr(config, "inference_server_address", "tcp://localhost:5556")
    action_chunks  = getattr(config, "action_chunks_len", 8)
    max_steps      = getattr(config, "max_episode_steps", 512)
    unnorm_key     = getattr(config, "unnorm_key", "libero_spatial_no_noops")

    all_specs = build_task_specs_from_libero_env(
        benchmark_name=benchmark,
        n_seeds=n_seeds,
    )
    train_specs, val_specs = split_train_val(
        all_specs,
        val_fraction=val_fraction,
        seed=seed,
    )
    train_dataset = RobotTaskDataset(train_specs)
    valid_dataset = RobotTaskDataset(val_specs)

    workflow_kwargs = dict(
        server_address=server_address,
        benchmark=benchmark,
        action_chunks_len=action_chunks,
        max_episode_steps=max_steps,
        unnorm_key=unnorm_key,
    )
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
