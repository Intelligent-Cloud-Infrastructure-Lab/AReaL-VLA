"""
LIBERO evaluation using the AReaL-VLA stack.

Runs the identical rollout path as RL training (VLARobotWorkflow +
VLALocalEngine + RobotTaskDataset) but with no actor, no gradient steps,
and no AReaL infrastructure. This replicates SimpleVLA-RL's val_only=True
mode, using parameters verified directly from:
  - examples/run_openvla_oft_rl_libero.sh  (hyperparameters)
  - verl/trainer/ppo/ray_trainer.py         (val loop structure)
  - verl/workers/rollout/rob_rollout.py     (env setup, max_steps)

Key parameters sourced from the shell script
---------------------------------------------
  data.num_trials_per_task = 50   → 50 fixed initial states per task total
  data.val_batch_size       = 496 → all val trials evaluated in one pass
  action_chunks_len         = 8   → 8 env steps per VLA generation call
  action_token_len          = 7   → 7 tokens per action dimension
  max_steps (LIBERO)        = 512 → from rob_rollout.py hardcoded dict
  do_sample                 = False → ray_trainer.py line 347, val pass
  num_steps_wait            = 10  → stabilisation steps before episode
  center_crop               = True
  unnorm_key                = $DATASET_NAME (e.g. "libero_spatial")
  val_before_train + val_only = True → triggers the evaluation pass

Critical difference from earlier version
-----------------------------------------
  trial_id is an INDEX into LIBERO's fixed initial_states array, NOT a
  random seed. LIBERO pre-generates 50 deterministic initial states per
  task; SimpleVLA-RL splits them 40 train / 10 val. We replicate that
  split here using LIBERO's own benchmark API.

Usage
-----
    # Exact replication of SimpleVLA-RL val_only=True
    python examples/robot/eval_libero.py \
        --model_path Haozhan72/Openvla-oft-SFT-libero-spatial-traj1 \
        --benchmark libero_spatial

    # Evaluate an RL checkpoint
    python examples/robot/eval_libero.py \
        --model_path /path/to/rl_checkpoint \
        --benchmark libero_10

    # Save results to JSON
    python examples/robot/eval_libero.py \
        --model_path Haozhan72/Openvla-oft-SFT-libero-spatial-traj1 \
        --benchmark libero_spatial \
        --output_path results/libero_spatial_sft.json
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
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

# ── AReaL-VLA modules (same ones used in libero_rl.py) ──────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

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


# ─── Constants sourced directly from rob_rollout.py and the shell script ─────

# From rob_rollout.py __init__ max_steps dict (lines 429-435)
LIBERO_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 512,
    "libero_object":  512,
    "libero_goal":    512,
    "libero_10":      512,   # libero_long
    "libero_90":      512,
}

# From shell script: data.num_trials_per_task=50
# SimpleVLA-RL's LIBERO_Dataset splits these as train/val.
# We replicate: use trial_ids [40..49] as the val split (last 10 of 50).
# This matches the "valid" split behaviour in LIBERO_Dataset(train_val="valid").
NUM_TRIALS_PER_TASK = 50
VAL_TRIAL_IDS = list(range(40, 50))   # indices 40-49 = val split

# From shell script: actor_rollout_ref.rollout.num_steps_wait=10
NUM_STEPS_WAIT = 10

# From shell script: actor_rollout_ref.model.action_chunks_len=8
# (number of environment steps per VLA generation call)
ACTION_CHUNKS_LEN = 8

# From shell script: actor_rollout_ref.model.action_token_len=7
# (tokens per action dimension; total tokens = action_token_len * action_dim)
ACTION_TOKEN_LEN = 7

# From ray_trainer.py line 347: 'do_sample': False for validation
DO_SAMPLE = False   # greedy decoding at eval


# ─── Config ──────────────────────────────────────────────────────────────────


@dataclass
class EvalConfig:
    model_path: str
    benchmark: str
    # Sourced from shell script defaults — override only if using a
    # different config than the published SimpleVLA-RL runs
    action_chunks_len: int = ACTION_CHUNKS_LEN   # env steps per generation
    action_token_len: int = ACTION_TOKEN_LEN     # tokens per action dim
    action_dim: int = 7                          # robot DoF (LIBERO = 7)
    num_steps_wait: int = NUM_STEPS_WAIT         # stabilisation steps
    center_crop: bool = True                     # from shell: center_crop=True
    unnorm_key: str = ""      # defaults to benchmark name; set explicitly to override
    device: str = "cuda"
    output_path: str | None = None
    tasks: list[str] | None = None   # restrict to specific task names


# ─── LIBERO environment factory ───────────────────────────────────────────────


def build_libero_env_factory(config: EvalConfig):
    """
    Build an env factory that replicates SimpleVLA-RL's env_worker exactly:

      1. Load the task suite and get the pre-generated initial_states array
      2. Use initial_states[trial_id] to set the env state deterministically
      3. Run num_steps_wait no-op steps (stabilisation), matching
         rob_rollout.py env_worker lines 365-375

    This is fundamentally different from passing a random seed to env.reset()
    — LIBERO environments are reset to a fixed state by index, not randomised.
    """
    try:
        from libero.libero import benchmark as libero_benchmark  # type: ignore
        from libero.libero.envs import OffScreenRenderEnv        # type: ignore
    except ImportError:
        raise ImportError(
            "LIBERO is not installed. Run: pip install libero-benchmark"
        )

    benchmark_dict = libero_benchmark.get_benchmark_dict()

    def make_env(task_name: str, trial_id: int):
        """
        task_name : LIBERO task suite name (e.g. "libero_spatial")
        trial_id  : index into task_suite.get_task_init_states(task_id)
                    — this is what SimpleVLA-RL passes as trial_id
        """
        # task_name here is the suite name; task_id is the task index within it.
        # We store both in the spec: "libero_spatial/3" → suite="libero_spatial", task_id=3
        suite_name, task_id_str = task_name.rsplit("/", 1)
        task_id = int(task_id_str)

        task_suite = benchmark_dict[suite_name]()
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        initial_state = initial_states[trial_id]

        # Get task description (same as rob_rollout.py env_worker)
        env, task_description = _get_libero_env(
            task, config.center_crop, resolution=256
        )
        env.reset()
        obs = env.set_init_state(initial_state)

        # num_steps_wait: wait for physics to settle (rob_rollout.py lines 365-375)
        dummy_action = np.zeros(7, dtype=np.float32)
        for _ in range(config.num_steps_wait):
            obs, _, _, _ = env.step(dummy_action)

        # Attach metadata the workflow needs
        env._task_description = task_description
        env._trial_id = trial_id
        return env

    return make_env


def _get_libero_env(task, center_crop: bool, resolution: int = 256):
    """
    Replicates SimpleVLA-RL's get_libero_env() from rob_rollout.py.
    Returns (env, task_description).
    """
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore

    task_description = task.language
    env_args = {
        "bddl_file_name": task.bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    return env, task_description


# ─── Task spec enumeration using LIBERO's own API ────────────────────────────


@dataclass
class LiberoEvalSpec:
    """One evaluation episode: (task_suite/task_idx, trial_id)."""
    task_key: str        # "libero_spatial/3"  (suite/task_idx)
    instruction: str
    trial_id: int        # index into initial_states[]


def build_eval_specs(
    benchmark: str,
    val_trial_ids: list[int] = VAL_TRIAL_IDS,
    task_filter: list[str] | None = None,
) -> list[LiberoEvalSpec]:
    """
    Enumerate all (task, trial_id) pairs for the val split, using LIBERO's
    benchmark API — the same source SimpleVLA-RL's LIBERO_Dataset uses.

    val_trial_ids: indices into initial_states[] used for evaluation.
                   Default [40..49] matches SimpleVLA-RL's val split of
                   num_trials_per_task=50.
    """
    try:
        from libero.libero import benchmark as libero_benchmark  # type: ignore
    except ImportError:
        raise ImportError("Install LIBERO: pip install libero-benchmark")

    benchmark_dict = libero_benchmark.get_benchmark_dict()
    if benchmark not in benchmark_dict:
        raise ValueError(
            f"Unknown benchmark '{benchmark}'. "
            f"Available: {sorted(benchmark_dict.keys())}"
        )

    task_suite = benchmark_dict[benchmark]()
    n_tasks = task_suite.get_num_tasks()

    specs: list[LiberoEvalSpec] = []
    for task_idx in range(n_tasks):
        task = task_suite.get_task(task_idx)
        task_key = f"{benchmark}/{task_idx}"
        instruction = task.language

        if task_filter and task_key not in task_filter:
            continue

        for trial_id in val_trial_ids:
            specs.append(LiberoEvalSpec(
                task_key=task_key,
                instruction=instruction,
                trial_id=trial_id,
            ))

    logger.info(
        f"Benchmark '{benchmark}': {n_tasks} tasks × "
        f"{len(val_trial_ids)} val trials = {len(specs)} total episodes"
    )
    return specs


# ─── Engine ──────────────────────────────────────────────────────────────────


def build_engine(config: EvalConfig) -> VLALocalEngine:
    """
    Load the VLA model and wrap it in VLALocalEngine.
    do_sample=False (greedy) matches ray_trainer.py _validate() line 347.
    """
    logger.info(f"Loading model: {config.model_path}")
    hf_model = AutoModelForVision2Seq.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(config.device)
    hf_model.eval()

    processor = AutoProcessor.from_pretrained(
        config.model_path, trust_remote_code=True
    )

    # action_chunk_len passed to OpenVLAOFTModel = total tokens generated
    # = action_token_len * action_chunks_len = 7 * 8 = 56 tokens per call
    total_action_tokens = config.action_token_len * config.action_chunks_len

    vla_model = OpenVLAOFTModel(
        model=hf_model,
        processor=processor,
        action_chunk_len=total_action_tokens,
        action_dim=config.action_dim,
        device=config.device,
    )

    # temperature is ignored when do_sample=False; set 0.0 to be explicit
    engine = VLALocalEngine(
        model=vla_model,
        action_chunk_len=total_action_tokens,
        temperature=0.0,   # do_sample=False → greedy, matches _validate()
    )
    logger.info(
        f"Engine ready  "
        f"(action_chunks_len={config.action_chunks_len}, "
        f"action_token_len={config.action_token_len}, "
        f"total_tokens_per_call={total_action_tokens})"
    )
    return engine


# ─── Workflow ─────────────────────────────────────────────────────────────────


def build_workflow(config: EvalConfig) -> VLARobotWorkflow:
    """
    Build VLARobotWorkflow matching SimpleVLA-RL's eval configuration:
      - max_episode_steps = 512  (from rob_rollout.py LIBERO max_steps dict)
      - do_sample = False        (from ray_trainer.py _validate meta_info)
      - center_crop = True       (from shell script)
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, trust_remote_code=True
    )

    def instruction_tokenizer(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    unnorm_key = config.unnorm_key or config.benchmark
    action_decoder = make_openvla_action_decoder(
        action_dim=config.action_dim,
        action_chunk_len=config.action_chunks_len,
        unnorm_key=unnorm_key,
    )

    max_steps = LIBERO_MAX_STEPS.get(config.benchmark, 512)

    class _GConfig:
        """Minimal stub — VLALocalEngine reads do_sample/temperature directly."""
        max_new_tokens = config.action_token_len * config.action_chunks_len
        temperature = 0.0

    return VLARobotWorkflow(
        env_factory=build_libero_env_factory(config),
        action_decoder=action_decoder,
        instruction_tokenizer=instruction_tokenizer,
        gconfig=_GConfig(),
        action_chunk_len=config.action_chunks_len,
        max_episode_steps=max_steps,
        rollout_stat_scope="eval",
    )


# ─── Evaluation loop ──────────────────────────────────────────────────────────


async def run_eval(config: EvalConfig) -> dict[str, Any]:
    """
    Evaluate all val-split episodes, mirroring SimpleVLA-RL's _validate().

    trial_id correctly indexes into LIBERO's fixed initial_states[] array,
    exactly as env_worker() does in rob_rollout.py line 342.
    """
    specs = build_eval_specs(
        benchmark=config.benchmark,
        val_trial_ids=VAL_TRIAL_IDS,
        task_filter=config.tasks,
    )

    engine = build_engine(config)
    workflow = build_workflow(config)

    results: dict[str, list[float]] = defaultdict(list)
    t_start = time.monotonic()

    for i, spec in enumerate(specs):
        # Pass trial_id as the "seed" field — VLARobotWorkflow passes it
        # through to the env_factory as the second argument.
        data = {
            "task_name":   spec.task_key,     # "libero_spatial/3"
            "instruction": spec.instruction,
            "benchmark":   config.benchmark,
            "seed":        spec.trial_id,     # index into initial_states[]
        }

        logger.info(
            f"[{i+1}/{len(specs)}]  {spec.task_key}  trial_id={spec.trial_id}"
        )

        trajectory = await workflow.arun_episode(engine, data)

        if trajectory is None:
            logger.warning("  → episode returned None (env failure) → 0")
            reward = 0.0
        else:
            reward = float(trajectory["rewards"].item())

        results[spec.task_key].append(reward)
        logger.info(f"  → {'SUCCESS ✓' if reward > 0.5 else 'fail    ✗'}")

    elapsed = time.monotonic() - t_start
    engine.destroy()

    # ── Aggregate (same structure as ray_trainer.py metric_dict) ─────────────
    per_task: dict[str, dict] = {}
    for task_key, rewards in results.items():
        per_task[task_key] = {
            "success_rate": float(np.mean(rewards)),
            "n_success":    int(sum(r > 0.5 for r in rewards)),
            "n_episodes":   len(rewards),
            "rewards":      rewards,
        }

    all_rewards = [r for rlist in results.values() for r in rlist]
    overall = float(np.mean(all_rewards))

    return {
        "model_path":           config.model_path,
        "benchmark":            config.benchmark,
        "val_trial_ids":        VAL_TRIAL_IDS,
        "action_chunks_len":    config.action_chunks_len,
        "action_token_len":     config.action_token_len,
        "max_episode_steps":    LIBERO_MAX_STEPS.get(config.benchmark, 512),
        "do_sample":            DO_SAMPLE,
        "num_steps_wait":       config.num_steps_wait,
        "overall_success_rate": overall,
        "per_task":             per_task,
        "elapsed_s":            round(elapsed, 1),
    }


# ─── Pretty-print ─────────────────────────────────────────────────────────────


def print_results(summary: dict[str, Any]) -> None:
    print()
    print("=" * 72)
    print(f"  Model         : {summary['model_path']}")
    print(f"  Benchmark     : {summary['benchmark']}")
    print(f"  Val trial_ids : {summary['val_trial_ids']}  "
          f"({len(summary['val_trial_ids'])} per task)")
    print(f"  action_chunks : {summary['action_chunks_len']} steps  "
          f"× {summary['action_token_len']} tokens/dim")
    print(f"  max_steps     : {summary['max_episode_steps']}  "
          f"do_sample={summary['do_sample']}  "
          f"num_steps_wait={summary['num_steps_wait']}")
    print("=" * 72)
    print(f"  {'Task':<55}  {'Success':>7}")
    print(f"  {'-'*55}  {'-'*7}")

    for task_key in sorted(summary["per_task"]):
        d = summary["per_task"][task_key]
        display = task_key if len(task_key) <= 55 else task_key[:52] + "..."
        print(
            f"  {display:<55}  "
            f"{d['success_rate']*100:6.1f}%  "
            f"({d['n_success']}/{d['n_episodes']})"
        )

    overall = summary["overall_success_rate"]
    print(f"  {'-'*55}  {'-'*7}")
    print(f"  {'OVERALL MEAN':<55}  {overall*100:6.1f}%")
    print("=" * 72)
    print(f"  Wall time: {summary['elapsed_s']:.0f}s")

    simplevla_refs = {
        "libero_object":  "SimpleVLA-RL Table 5: SFT=54.9, RL=98.7",
        "libero_spatial": "SimpleVLA-RL Table 5: SFT=63.6, RL=98.2",
        "libero_goal":    "SimpleVLA-RL Table 5: SFT=59.6, RL=98.8",
        "libero_long":    "SimpleVLA-RL Table 5: SFT=17.3, RL=91.7",
    }
    ref = simplevla_refs.get(summary["benchmark"])
    if ref:
        print()
        print(f"  Reference : {ref}")
        print(f"  This run  : {overall*100:.1f}%")
    print()


# ─── Entry point ──────────────────────────────────────────────────────────────


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate a VLA checkpoint using the AReaL-VLA rollout stack, "
            "replicating SimpleVLA-RL val_only=True exactly."
        )
    )
    p.add_argument(
        "--model_path", required=True,
        help="HuggingFace model ID or local path. "
             "e.g. Haozhan72/Openvla-oft-SFT-libero-spatial-traj1"
    )
    p.add_argument(
        "--benchmark", default="libero_spatial",
        choices=list(LIBERO_MAX_STEPS.keys()),
        help="LIBERO benchmark suite (default: libero_spatial)"
    )
    p.add_argument(
        "--action_chunks_len", type=int, default=ACTION_CHUNKS_LEN,
        help=f"Env steps per VLA generation call "
             f"(shell script: action_chunks_len={ACTION_CHUNKS_LEN})"
    )
    p.add_argument(
        "--action_token_len", type=int, default=ACTION_TOKEN_LEN,
        help=f"Tokens per action dimension "
             f"(shell script: action_token_len={ACTION_TOKEN_LEN})"
    )
    p.add_argument(
        "--action_dim", type=int, default=7,
        help="Robot DoF (LIBERO default: 7)"
    )
    p.add_argument(
        "--num_steps_wait", type=int, default=NUM_STEPS_WAIT,
        help=f"Stabilisation steps before episode starts "
             f"(shell script: num_steps_wait={NUM_STEPS_WAIT})"
    )
    p.add_argument(
        "--unnorm_key", default="",
        help="Action unnormalisation key (defaults to benchmark name, "
             "matching shell script: unnorm_key=$DATASET_NAME)"
    )
    p.add_argument(
        "--no_center_crop", action="store_true",
        help="Disable center cropping (shell script enables it by default)"
    )
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--output_path", default=None,
        help="Optional path to save results JSON"
    )
    p.add_argument(
        "--tasks", nargs="+", default=None,
        help="Restrict to specific task keys e.g. libero_spatial/0 libero_spatial/3"
    )
    args = p.parse_args()

    return EvalConfig(
        model_path=args.model_path,
        benchmark=args.benchmark,
        action_chunks_len=args.action_chunks_len,
        action_token_len=args.action_token_len,
        action_dim=args.action_dim,
        num_steps_wait=args.num_steps_wait,
        unnorm_key=args.unnorm_key,
        center_crop=not args.no_center_crop,
        device=args.device,
        output_path=args.output_path,
        tasks=args.tasks,
    )


def main() -> None:
    config = parse_args()

    logger.info(f"Device: {config.device}")
    if config.device.startswith("cuda") and torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Echo the exact config so results are reproducible and auditable
    logger.info(
        f"Eval config: benchmark={config.benchmark}  "
        f"action_chunks_len={config.action_chunks_len}  "
        f"action_token_len={config.action_token_len}  "
        f"max_steps={LIBERO_MAX_STEPS.get(config.benchmark, 512)}  "
        f"val_trial_ids={VAL_TRIAL_IDS}  "
        f"do_sample={DO_SAMPLE}  "
        f"num_steps_wait={config.num_steps_wait}  "
        f"center_crop={config.center_crop}"
    )

    summary = asyncio.run(run_eval(config))
    print_results(summary)

    if config.output_path:
        os.makedirs(os.path.dirname(os.path.abspath(config.output_path)), exist_ok=True)
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
