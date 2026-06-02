"""
LIBERO evaluation — completely standalone, zero AReaL imports.

Runs in the SimpleVLA conda environment using OpenVLA-OFT directly.
This replicates SimpleVLA-RL's val_only=True mode with all parameters
verified from the source files you provided:

  ray_trainer.py line 347 : do_sample=False  (greedy decoding at eval)
  rob_rollout.py lines 429-435 : max_steps=512 for all LIBERO suites
  shell script : action_chunks_len=8, action_token_len=7, num_steps_wait=10,
                 center_crop=True, unnorm_key=$DATASET_NAME
  LIBERO_Dataset(train_val="valid") : trial_ids [40..49] of 50 total

Usage:
    conda activate simplevla
    python examples/robot/eval_libero.py \
        --model_path Haozhan72/Openvla-oft-SFT-libero-spatial-traj1 \
        --benchmark libero_spatial

    # Save results for comparison
    python examples/robot/eval_libero.py \
        --model_path /path/to/rl_checkpoint \
        --benchmark libero_10 \
        --output_path results/libero_long_rl.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_libero")

# ---------------------------------------------------------------------------
# Constants verified from SimpleVLA-RL source
# ---------------------------------------------------------------------------

# rob_rollout.py __init__ max_steps dict (lines 429-435)
LIBERO_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 512,
    "libero_object":  512,
    "libero_goal":    512,
    "libero_10":      512,  # libero_long
    "libero_90":      512,
}

# data.num_trials_per_task=50, train/val split: last 10 are val
# Matches LIBERO_Dataset(train_val="valid") behaviour
VAL_TRIAL_IDS = list(range(40, 50))

NUM_STEPS_WAIT   = 10   # actor_rollout_ref.rollout.num_steps_wait=10
ACTION_CHUNKS_LEN = 8   # actor_rollout_ref.model.action_chunks_len=8
ACTION_TOKEN_LEN  = 7   # actor_rollout_ref.model.action_token_len=7
DO_SAMPLE        = False  # ray_trainer.py line 347: 'do_sample': False

# SimpleVLA-RL reference numbers for cross-checking
SIMPLEVLA_REFERENCE = {
    "libero_object":  "SFT=88.5%  RL=97.3%",
    "libero_spatial": "SFT=76.0%  RL=92.2%",
    "libero_goal":    "SFT=81.5%  RL=93.0%",
    "libero_10":      "SFT=17.3%  RL=97.6%",
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    model_path: str
    benchmark: str
    action_chunks_len: int = ACTION_CHUNKS_LEN
    action_token_len: int = ACTION_TOKEN_LEN
    num_steps_wait: int = NUM_STEPS_WAIT
    center_crop: bool = True
    device: str = "cuda:0"
    output_path: str | None = None
    tasks: list[str] | None = None  # e.g. ["libero_spatial/0", "libero_spatial/3"]


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def center_crop_image(image: np.ndarray, scale: float = 0.95) -> np.ndarray:
    """Centre-crop + resize. Matches rob_rollout.py center_crop_image()."""
    from PIL import Image as PILImage  # type: ignore

    h, w = image.shape[:2]
    ch, cw = int(h * scale), int(w * scale)
    top, left = (h - ch) // 2, (w - cw) // 2
    cropped = image[top : top + ch, left : left + cw]
    return np.array(PILImage.fromarray(cropped).resize((w, h), PILImage.BILINEAR))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(config: EvalConfig):
    """Load OpenVLA-OFT via transformers-openvla-oft (SimpleVLA env)."""
    from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore

    logger.info(f"Loading model: {config.model_path}  device={config.device}")
    model = AutoModelForVision2Seq.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(config.device)
    model.eval()
    processor = AutoProcessor.from_pretrained(config.model_path, trust_remote_code=True)
    logger.info("Model ready")
    return model, processor


# ---------------------------------------------------------------------------
# Single forward pass
# ---------------------------------------------------------------------------


def generate_actions(
    model,
    processor,
    image: np.ndarray,
    instruction: str,
    config: EvalConfig,
) -> np.ndarray:
    """
    One VLA forward pass → continuous actions.
    Mirrors _generate_one_step_oft() in rob_rollout.py exactly.
    """
    from PIL import Image as PILImage  # type: ignore

    if config.center_crop:
        image = center_crop_image(image)

    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(
        text=prompt,
        images=PILImage.fromarray(image),
        return_tensors="pt",
    )
    input_ids    = inputs["input_ids"].to(config.device)
    attn_mask    = inputs["attention_mask"].to(config.device)
    pixel_values = inputs["pixel_values"].to(config.device, dtype=torch.bfloat16)

    with torch.no_grad():
        actions, _ = model.generate_action_verl(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attn_mask,
            padding_idx=processor.tokenizer.pad_token_id,
            do_sample=DO_SAMPLE,
            unnorm_key=config.benchmark,
            temperature=1.0,
        )

    if isinstance(actions, torch.Tensor):
        actions = actions.squeeze(0).float().cpu().numpy()
    else:
        actions = np.array(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[np.newaxis, :]
    return actions  # (action_chunks_len, action_dim)


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------


def run_episode(
    model,
    processor,
    task_suite,
    task_idx: int,
    trial_id: int,
    config: EvalConfig,
) -> bool:
    """
    Run one LIBERO episode and return True if the task succeeded.
    Mirrors env_worker() + _generate_minibatch_libero() in rob_rollout.py.

    trial_id indexes into task_suite.get_task_init_states(task_idx) —
    a fixed array of pre-generated initial states, not a random seed.
    """
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore

    task = task_suite.get_task(task_idx)
    instruction = task.language
    initial_states = task_suite.get_task_init_states(task_idx)
    initial_state = initial_states[trial_id]

    env = OffScreenRenderEnv(
        bddl_file_name=task.bddl_file,
        camera_heights=256,
        camera_widths=256,
    )
    env.reset()
    obs = env.set_init_state(initial_state)

    # num_steps_wait: let physics settle (rob_rollout.py env_worker lines 365-375)
    dummy = np.zeros(7, dtype=np.float32)
    for _ in range(config.num_steps_wait):
        obs, _, _, _ = env.step(dummy)

    max_steps = LIBERO_MAX_STEPS.get(config.benchmark, 512)
    step = 0
    success = False

    while step < max_steps and not success:
        # Get current image observation
        image = obs.get("agentview_image")
        if image is None:
            try:
                image = env.render(mode="rgb_array")
            except Exception:
                break

        # Generate action chunk
        actions = generate_actions(model, processor, image, instruction, config)

        # Execute each step of the chunk
        for chunk_step in range(config.action_chunks_len):
            if step >= max_steps:
                break
            action = actions[min(chunk_step, len(actions) - 1)]
            obs, _, done, info = env.step(action)
            step += 1
            if info.get("success", False):
                success = True
                break
            if done:
                break
        if done and not success:
            break

    env.close()
    return success


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def run_eval(config: EvalConfig) -> dict:
    try:
        from libero.libero import benchmark as libero_benchmark  # type: ignore
    except ImportError:
        raise ImportError("Install LIBERO: pip install libero-benchmark")

    benchmark_dict = libero_benchmark.get_benchmark_dict()
    if config.benchmark not in benchmark_dict:
        raise ValueError(
            f"Unknown benchmark '{config.benchmark}'. "
            f"Available: {sorted(benchmark_dict.keys())}"
        )

    task_suite = benchmark_dict[config.benchmark]()
    n_tasks = task_suite.get_num_tasks()

    model, processor = load_model(config)

    results: dict[str, list[bool]] = defaultdict(list)
    t_start = time.monotonic()
    episodes_total = n_tasks * len(VAL_TRIAL_IDS)
    ep_idx = 0

    for task_idx in range(n_tasks):
        task = task_suite.get_task(task_idx)
        task_key = f"{config.benchmark}/{task_idx}"

        if config.tasks and task_key not in config.tasks:
            continue

        for trial_id in VAL_TRIAL_IDS:
            ep_idx += 1
            logger.info(
                f"[{ep_idx}/{episodes_total}]  {task.language[:55]}  "
                f"task={task_idx}  trial_id={trial_id}"
            )
            t0 = time.monotonic()
            success = run_episode(
                model, processor, task_suite, task_idx, trial_id, config
            )
            results[task_key].append(success)
            logger.info(
                f"  → {'SUCCESS ✓' if success else 'fail    ✗'}  "
                f"({time.monotonic()-t0:.0f}s)"
            )

    elapsed = time.monotonic() - t_start

    per_task = {}
    for task_key, successes in results.items():
        per_task[task_key] = {
            "success_rate": float(np.mean(successes)),
            "n_success":    int(sum(successes)),
            "n_episodes":   len(successes),
        }

    all_s = [s for sl in results.values() for s in sl]
    overall = float(np.mean(all_s)) if all_s else 0.0

    return {
        "model_path":           config.model_path,
        "benchmark":            config.benchmark,
        "val_trial_ids":        VAL_TRIAL_IDS,
        "action_chunks_len":    config.action_chunks_len,
        "action_token_len":     config.action_token_len,
        "max_episode_steps":    LIBERO_MAX_STEPS.get(config.benchmark, 512),
        "num_steps_wait":       config.num_steps_wait,
        "do_sample":            DO_SAMPLE,
        "center_crop":          config.center_crop,
        "overall_success_rate": overall,
        "per_task":             per_task,
        "elapsed_s":            round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------


def print_results(summary: dict) -> None:
    print()
    print("=" * 72)
    print(f"  Model          : {summary['model_path']}")
    print(f"  Benchmark      : {summary['benchmark']}")
    print(f"  Val trial_ids  : {summary['val_trial_ids']}")
    print(
        f"  action_chunks  : {summary['action_chunks_len']} steps  "
        f"× {summary['action_token_len']} tokens/dim"
    )
    print(
        f"  max_steps={summary['max_episode_steps']}  "
        f"num_steps_wait={summary['num_steps_wait']}  "
        f"do_sample={summary['do_sample']}  "
        f"center_crop={summary['center_crop']}"
    )
    print("=" * 72)
    print(f"  {'Task':<55}  Success")
    print(f"  {'-'*55}  -------")
    for task_key in sorted(summary["per_task"]):
        d = summary["per_task"][task_key]
        display = task_key if len(task_key) <= 55 else task_key[:52] + "..."
        print(
            f"  {display:<55}  "
            f"{d['success_rate']*100:5.1f}%  "
            f"({d['n_success']}/{d['n_episodes']})"
        )
    overall = summary["overall_success_rate"]
    print(f"  {'-'*55}  -------")
    print(f"  {'OVERALL MEAN':<55}  {overall*100:5.1f}%")
    print("=" * 72)
    print(f"  Wall time : {summary['elapsed_s']:.0f}s")
    ref = SIMPLEVLA_REFERENCE.get(summary["benchmark"])
    if ref:
        print(f"\n  SimpleVLA-RL reference : {ref}")
        print(f"  This run               : {overall*100:.1f}%")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate a VLA checkpoint (run in the simplevla conda env)"
    )
    p.add_argument("--model_path", required=True,
                   help="HuggingFace model ID or local path")
    p.add_argument("--benchmark", default="libero_spatial",
                   choices=list(LIBERO_MAX_STEPS.keys()))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no_center_crop", action="store_true")
    p.add_argument("--output_path", default=None,
                   help="Save JSON results to this path")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Restrict to task keys, e.g. libero_spatial/0 libero_spatial/3")
    args = p.parse_args()

    config = EvalConfig(
        model_path=args.model_path,
        benchmark=args.benchmark,
        device=args.device,
        center_crop=not args.no_center_crop,
        output_path=args.output_path,
        tasks=args.tasks,
    )

    logger.info(
        f"Config: benchmark={config.benchmark}  "
        f"val_trial_ids={VAL_TRIAL_IDS}  "
        f"max_steps={LIBERO_MAX_STEPS.get(config.benchmark, 512)}  "
        f"do_sample={DO_SAMPLE}  "
        f"action_chunks_len={config.action_chunks_len}  "
        f"action_token_len={config.action_token_len}  "
        f"num_steps_wait={config.num_steps_wait}  "
        f"center_crop={config.center_crop}"
    )

    summary = run_eval(config)
    print_results(summary)

    if config.output_path:
        os.makedirs(os.path.dirname(os.path.abspath(config.output_path)), exist_ok=True)
        with open(config.output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results saved → {config.output_path}")


if __name__ == "__main__":
    main()
