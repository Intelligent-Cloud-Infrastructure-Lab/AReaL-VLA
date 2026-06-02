"""
LIBERO evaluation using the AReaL-VLA stack.

Runs the identical rollout path as RL training (VLARobotWorkflow +
VLALocalEngine + RobotTaskDataset) but with no actor, no gradient steps,
and no AReaL infrastructure.  This lets you verify that our rollout
implementation reproduces the same success rates as SimpleVLA-RL's
evaluation before trusting any training results.

Usage
-----
    # Evaluate the SFT baseline (should match SimpleVLA-RL Table 1 SFT numbers)
    python examples/robot/eval_libero.py \
        --model_path Haozhan72/Openvla-oft-SFT-libero-spatial-traj1 \
        --benchmark libero_spatial \
        --n_episodes_per_task 20 \
        --n_seeds 4

    # Evaluate an RL checkpoint
    python examples/robot/eval_libero.py \
        --model_path /path/to/rl_checkpoint \
        --benchmark libero_long \
        --n_episodes_per_task 20

    # Quick smoke-test (3 episodes per task, greedy decoding)
    python examples/robot/eval_libero.py \
        --model_path Haozhan72/Openvla-oft-SFT-libero-spatial-traj1 \
        --benchmark libero_spatial \
        --n_episodes_per_task 3 \
        --temperature 0.0

Outputs
-------
Per-task success rates + overall mean printed to stdout, and optionally
saved to a JSON file (--output_path).  Format matches SimpleVLA-RL's
evaluation table so numbers are directly comparable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

# ── AReaL-VLA modules (same ones used in libero_rl.py) ──────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from areal.dataset.robot_dataset import (
    RobotTaskDataset,
    RobotTaskSpec,
    build_task_specs_from_libero_env,
)
from areal.engine.vla_local_engine import (
    OpenVLAOFTModel,
    VLALocalEngine,
    make_openvla_action_decoder,
)
from areal.workflow.vla_robot import VLARobotWorkflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_libero")


# ─── Config ──────────────────────────────────────────────────────────────────


@dataclass
class EvalConfig:
    model_path: str
    benchmark: str
    n_episodes_per_task: int
    n_seeds: int
    temperature: float
    action_chunk_len: int
    action_dim: int
    max_episode_steps: int
    device: str
    output_path: str | None
    tasks: list[str] | None       # restrict to specific task names if given


# ─── LIBERO env factory ───────────────────────────────────────────────────────


def build_libero_env_factory(benchmark: str):
    """
    Same factory used in libero_rl.py so the eval environment is identical
    to the training environment.
    """
    def make_env(task_name: str, seed: int):
        try:
            from libero.libero.envs import OffScreenRenderEnv  # type: ignore
        except ImportError:
            raise ImportError("Install LIBERO: pip install libero-benchmark")

        # task_name is stored as the full LIBERO benchmark task name
        env = OffScreenRenderEnv(
            task_name=task_name,
            task_suite_name=benchmark,
        )
        env.seed(seed)
        env.reset()
        return env

    return make_env


# ─── Engine + workflow factory ────────────────────────────────────────────────


def build_engine(config: EvalConfig) -> VLALocalEngine:
    """
    Load the VLA model from HuggingFace (or a local path) and wrap it in
    VLALocalEngine — the same engine class used during RL training.
    """
    logger.info(f"Loading model from: {config.model_path}")
    hf_model = AutoModelForVision2Seq.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(config.device)
    hf_model.eval()

    processor = AutoProcessor.from_pretrained(
        config.model_path,
        trust_remote_code=True,
    )

    vla_model = OpenVLAOFTModel(
        model=hf_model,
        processor=processor,
        action_chunk_len=config.action_chunk_len,
        action_dim=config.action_dim,
        device=config.device,
    )

    engine = VLALocalEngine(
        model=vla_model,
        action_chunk_len=config.action_chunk_len,
        temperature=config.temperature,  # 0.0 = greedy, matches SimpleVLA-RL eval
    )
    logger.info("Engine ready")
    return engine


def build_workflow(config: EvalConfig) -> VLARobotWorkflow:
    """
    Build VLARobotWorkflow with the same parameters as training so the rollout
    logic is byte-for-byte identical.  The only difference from libero_rl.py
    is temperature=0.0 (greedy) which is what SimpleVLA-RL uses at eval time.
    """
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, trust_remote_code=True
    )

    def instruction_tokenizer(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    action_decoder = make_openvla_action_decoder(
        action_dim=config.action_dim,
        action_chunk_len=config.action_chunk_len,
    )

    # GenerationHyperparameters stub — only max_new_tokens and temperature
    # are forwarded to agenerate(), which VLALocalEngine reads from its own
    # constructor args (temperature), so this is just for interface compat.
    class _GConfig:
        max_new_tokens = config.action_chunk_len
        temperature = config.temperature

    return VLARobotWorkflow(
        env_factory=build_libero_env_factory(config.benchmark),
        action_decoder=action_decoder,
        instruction_tokenizer=instruction_tokenizer,
        gconfig=_GConfig(),
        action_chunk_len=config.action_chunk_len,
        max_episode_steps=config.max_episode_steps,
        rollout_stat_scope="eval",
    )


# ─── Evaluation loop ──────────────────────────────────────────────────────────


async def run_eval(config: EvalConfig) -> dict[str, Any]:
    """
    Run evaluation using the same VLARobotWorkflow.arun_episode() that
    RL training uses.  Results are therefore directly comparable.

    Returns a dict of {task_name: {"success_rate": float, "n_episodes": int}}.
    """
    # ── Build task specs ──────────────────────────────────────────────────────
    all_specs = build_task_specs_from_libero_env(
        benchmark_name=config.benchmark,
        n_seeds=config.n_seeds,
    )

    # Optionally restrict to specific tasks
    if config.tasks:
        all_specs = [s for s in all_specs if s.task_name in config.tasks]
        if not all_specs:
            raise ValueError(
                f"None of the requested tasks found in benchmark '{config.benchmark}'. "
                f"Check --tasks spelling."
            )

    # Deduplicate by task name, then create n_episodes_per_task specs
    unique_tasks: dict[str, RobotTaskSpec] = {}
    for spec in all_specs:
        if spec.task_name not in unique_tasks:
            unique_tasks[spec.task_name] = spec

    # Build eval specs: n_episodes_per_task seeds per task, starting from seed 100
    # (offset from training seeds 0..n_seeds-1 to avoid data leakage)
    eval_specs: list[RobotTaskSpec] = []
    for task_name, base_spec in unique_tasks.items():
        for ep_idx in range(config.n_episodes_per_task):
            eval_specs.append(RobotTaskSpec(
                task_name=base_spec.task_name,
                instruction=base_spec.instruction,
                benchmark=base_spec.benchmark,
                seed=100 + ep_idx,   # eval seeds offset from training seeds
            ))

    logger.info(
        f"Evaluating {len(unique_tasks)} tasks × {config.n_episodes_per_task} episodes "
        f"= {len(eval_specs)} total episodes"
    )

    # ── Build engine + workflow (same classes as training) ────────────────────
    engine = build_engine(config)
    workflow = build_workflow(config)

    # ── Run episodes ──────────────────────────────────────────────────────────
    results: dict[str, list[float]] = defaultdict(list)  # task → [reward, ...]
    t_start = time.monotonic()

    for i, spec in enumerate(eval_specs):
        data = {
            "task_name":  spec.task_name,
            "instruction": spec.instruction,
            "benchmark":  spec.benchmark,
            "seed":       spec.seed,
        }

        logger.info(
            f"[{i+1}/{len(eval_specs)}]  "
            f"{spec.task_name}  seed={spec.seed}"
        )

        # arun_episode is the same coroutine called during RL training
        trajectory = await workflow.arun_episode(engine, data)

        if trajectory is None:
            logger.warning(f"  → episode returned None (env failure), counting as 0")
            reward = 0.0
        else:
            reward = float(trajectory["rewards"].item())

        results[spec.task_name].append(reward)
        logger.info(f"  → {'SUCCESS ✓' if reward > 0.5 else 'fail ✗'}")

    elapsed = time.monotonic() - t_start
    engine.destroy()

    # ── Aggregate ─────────────────────────────────────────────────────────────
    per_task: dict[str, dict] = {}
    for task_name, rewards in results.items():
        per_task[task_name] = {
            "success_rate": float(np.mean(rewards)),
            "n_success":    int(sum(r > 0.5 for r in rewards)),
            "n_episodes":   len(rewards),
            "rewards":      rewards,
        }

    all_rewards = [r for rlist in results.values() for r in rlist]
    overall = float(np.mean(all_rewards))

    summary = {
        "model_path":   config.model_path,
        "benchmark":    config.benchmark,
        "n_episodes_per_task": config.n_episodes_per_task,
        "temperature":  config.temperature,
        "overall_success_rate": overall,
        "per_task":     per_task,
        "elapsed_s":    round(elapsed, 1),
    }
    return summary


# ─── Pretty-print ─────────────────────────────────────────────────────────────


def print_results(summary: dict[str, Any]) -> None:
    """
    Print results in the same format as SimpleVLA-RL's evaluation table so
    you can directly compare numbers.
    """
    print()
    print("=" * 72)
    print(f"  Model    : {summary['model_path']}")
    print(f"  Benchmark: {summary['benchmark']}")
    print(f"  Episodes : {summary['n_episodes_per_task']} per task  "
          f"(temperature={summary['temperature']})")
    print("=" * 72)
    print(f"  {'Task':<55}  {'Success':>7}")
    print(f"  {'-'*55}  {'-'*7}")

    per_task = summary["per_task"]
    for task_name in sorted(per_task):
        sr = per_task[task_name]["success_rate"]
        n_s = per_task[task_name]["n_success"]
        n_ep = per_task[task_name]["n_episodes"]
        # Truncate long task names for readability
        display = task_name if len(task_name) <= 55 else task_name[:52] + "..."
        print(f"  {display:<55}  {sr*100:6.1f}%  ({n_s}/{n_ep})")

    print(f"  {'-'*55}  {'-'*7}")
    overall = summary["overall_success_rate"]
    print(f"  {'OVERALL MEAN':<55}  {overall*100:6.1f}%")
    print("=" * 72)
    print(f"  Total wall time: {summary['elapsed_s']:.0f}s")
    print()

    # Cross-reference hint
    simplevla_refs = {
        "libero_object":  "SimpleVLA-RL Table 5: SFT=54.9, RL=98.7",
        "libero_spatial": "SimpleVLA-RL Table 5: SFT=63.6, RL=98.2",
        "libero_goal":    "SimpleVLA-RL Table 5: SFT=59.6, RL=98.8",
        "libero_long":    "SimpleVLA-RL Table 5: SFT=17.3, RL=91.7",
    }
    ref = simplevla_refs.get(summary["benchmark"])
    if ref:
        print(f"  Reference: {ref}")
        print(f"  Your result: {overall*100:.1f}%")
        print()


# ─── Entry point ──────────────────────────────────────────────────────────────


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(
        description="Evaluate a VLA checkpoint using the AReaL-VLA rollout stack"
    )
    p.add_argument(
        "--model_path", required=True,
        help="HuggingFace model ID or local path. "
             "e.g. Haozhan72/Openvla-oft-SFT-libero-spatial-traj1"
    )
    p.add_argument(
        "--benchmark", default="libero_spatial",
        choices=["libero_object", "libero_spatial", "libero_goal", "libero_long"],
        help="LIBERO benchmark suite to evaluate on"
    )
    p.add_argument(
        "--n_episodes_per_task", type=int, default=20,
        help="Episodes per task (SimpleVLA-RL uses 20)"
    )
    p.add_argument(
        "--n_seeds", type=int, default=5,
        help="Seeds used to enumerate tasks from the benchmark (used only to "
             "discover task names; eval seeds are always offset to 100+)"
    )
    p.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature. 0.0 = greedy (default, matches SimpleVLA-RL eval)"
    )
    p.add_argument(
        "--action_chunk_len", type=int, default=7,
        help="Action tokens per environment step (OpenVLA-OFT default: 7)"
    )
    p.add_argument(
        "--action_dim", type=int, default=7,
        help="Robot action dimensionality (LIBERO default: 7)"
    )
    p.add_argument(
        "--max_episode_steps", type=int, default=300,
        help="Maximum environment steps per episode"
    )
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available)"
    )
    p.add_argument(
        "--output_path", default=None,
        help="Optional path to save results as JSON"
    )
    p.add_argument(
        "--tasks", nargs="+", default=None,
        help="Optional: restrict evaluation to specific task names"
    )
    args = p.parse_args()

    return EvalConfig(
        model_path=args.model_path,
        benchmark=args.benchmark,
        n_episodes_per_task=args.n_episodes_per_task,
        n_seeds=args.n_seeds,
        temperature=args.temperature,
        action_chunk_len=args.action_chunk_len,
        action_dim=args.action_dim,
        max_episode_steps=args.max_episode_steps,
        device=args.device,
        output_path=args.output_path,
        tasks=args.tasks,
    )


def main() -> None:
    config = parse_args()

    logger.info(f"Device: {config.device}")
    if config.device.startswith("cuda"):
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    summary = asyncio.run(run_eval(config))
    print_results(summary)

    if config.output_path:
        os.makedirs(os.path.dirname(config.output_path) or ".", exist_ok=True)
        # Remove raw reward lists from JSON (keep it readable)
        json_summary = {
            k: v for k, v in summary.items() if k != "per_task"
        }
        json_summary["per_task"] = {
            task: {k: v for k, v in d.items() if k != "rewards"}
            for task, d in summary["per_task"].items()
        }
        with open(config.output_path, "w") as f:
            json.dump(json_summary, f, indent=2)
        logger.info(f"Results saved to {config.output_path}")


if __name__ == "__main__":
    main()
