# AReaL-VLA: SimpleVLA-RL in AReaL

**WORK IN PROGRESS**

This repository ports **SimpleVLA-RL** (ICLR 2026) from its original veRL backend into
**AReaL** — Ant Group / Tsinghua's fully-asynchronous RL infrastructure.

SimpleVLA-RL demonstrated that simple 0/1 outcome rewards can drive online RL for
Vision-Language-Action (VLA) models, raising OpenVLA-OFT performance on LIBERO-Long
from 17.3 → 97.6 points.  This port preserves every algorithmic idea while replacing
the veRL internals with AReaL's cleaner, faster, fully-asynchronous engine.

---

## Why AReaL instead of veRL?

| Property | veRL (SimpleVLA-RL original) | AReaL (this repo) |
|---|---|---|
| Training/inference overlap | Synchronous (blocking) | Fully async (boba²) |
| Distributed backend | FSDP + Ray workers | FSDP2 / Megatron / Archon |
| Inference backend | HF generate / vLLM | SGLang / vLLM (remote) |
| Rollout abstraction | `rob_rollout.py` (custom) | `RolloutWorkflow.arun_episode` |
| Reward plumbing | manual | `AsyncRewardWrapper` |
| Algorithm support | PPO | GRPO, PPO, DAPO, REINFORCE++, … |
| Multi-node | Yes | Yes (+ Slurm / K8s launchers) |

---

## Design Principles and Pre-Implementation Plan

Before a single line of code was written we mapped every SimpleVLA-RL change onto the
corresponding AReaL abstraction. The table below is the architectural blueprint.

### Mapping: SimpleVLA-RL veRL files → AReaL modules

| SimpleVLA-RL file | Purpose | AReaL counterpart |
|---|---|---|
| `verl/utils/dataset/rob_dataset.py` | Robot task rows (task name, seed, instruction) | `areal/dataset/robot_dataset.py` |
| `verl/workers/rollout/rob_rollout.py` | LIBERO env loop, action-token generation, reward | `areal/workflow/vla_robot.py` (`VLARobotWorkflow`) |
| `verl/workers/actor/dp_rob.py` | Custom actor (FSDP + VLA heads) | Thin adapter in `areal/engine/vla_local_engine.py` |
| `verl/trainer/ppo/ray_trainer.py` | Dynamic sampling, mixed-success groups, trainer loop | `examples/robot/libero_rl.py` + config |
| `verl/trainer/main_ppo.py` | Entry point, config hydra | `examples/robot/libero_rl.py` |

### Five algorithmic ideas and their AReaL mapping

#### 1. Robot task dataset (static rows → dynamic environments)

SimpleVLA-RL replaces text prompt rows with *environment-initialization specs*.  In
AReaL, the dataset loader (`areal/dataset/robot_dataset.py`) returns lightweight dicts:

```python
{
    "task_name":   "libero_object/place_soup_in_drawer",
    "instruction": "place the alphabet soup in the top drawer",
    "benchmark":   "libero_object",
    "seed":        42,
}
```

The environment is **not** created here.  `VLARobotWorkflow.arun_episode` instantiates
and resets the environment per episode, exactly as `rob_rollout.py` does.

#### 2. Embodied rollout (env.reset → render → generate → step)

AReaL's `RolloutWorkflow.arun_episode(engine, data)` is the perfect hook.  The entire
LIBERO episode loop fits inside one `arun_episode` call:

```
arun_episode(engine, data):
    env = create_env(data["task_name"], data["seed"])
    obs = env.reset()
    for step in range(max_steps):
        image = env.render()
        req = VLARequest(image, data["instruction_ids"])
        resp = await engine.agenerate(req)        # action tokens
        action = decode_action_tokens(resp.output_tokens)
        obs, _, done, info = env.step(action)
        collect(resp)
        if done: break
    return build_tensor_batch(...)
```

AReaL's `AsyncTaskRunner` runs many `arun_episode` calls concurrently, naturally
achieving *multi-environment parallel rendering* without an explicit env pool.

#### 3. Sparse reward → token-level alignment

SimpleVLA-RL converts binary trajectory reward to per-token supervision.  We replicate
this in `_build_trajectory_tensors`:

```
finish_token = finish_step * action_chunk_len
reward[t] = binary_reward   if t is an action token AND global_action_idx <= finish_token
reward[t] = 0               otherwise
```

In AReaL's GRPO, the `rewards` field is a scalar per trajectory; advantage
normalization happens inside the trainer.  For PPO the per-token alignment feeds
directly into GAE.

#### 4. Post-success loss masking

Tokens generated after task completion carry no useful gradient signal.  We zero
`loss_mask` for all action tokens whose *global action index* exceeds the success
boundary:

```python
loss_mask[i] = 0   if action is post-success
```

This is assembled in `_build_trajectory_tensors` using `finish_step` and
`action_chunk_len`.

#### 5. Dynamic sampling + mixed-success groups

**Mixed-success groups**: GRPO's group-advantage normalization is *identical* to
SimpleVLA-RL's mixed-success strategy.  With `group_size = G` rollouts per task, the
GRPO advantage `A = (r - mean(r_group)) / std(r_group)` is only informative when the
group contains both successes (r=1) and failures (r=0).  We expose `group_size` in
the YAML config and rely on AReaL's standard GRPO trainer.

**Dynamic sampling**: `RobotCurriculumSampler` (in `robot_dataset.py`) tracks
per-task success rates and samples tasks with probability proportional to their
learning signal: `p(task) ∝ clamp(success_rate, ε, 1-ε)`.  The sampler is slotted
into AReaL's `StatefulDataLoader`.

---

## Repository Structure

```
AReaL-VLA/
├── README.md                          ← this file (also the plan doc)
│
├── areal/                             ← add these files into a cloned AReaL
│   ├── workflow/
│   │   └── vla_robot.py              ← VLARobotWorkflow (main contribution)
│   ├── dataset/
│   │   └── robot_dataset.py          ← RobotTaskDataset + CurriculumSampler
│   ├── reward/
│   │   └── robot_reward.py           ← libero_reward_fn, robottwin_reward_fn
│   └── engine/
│       └── vla_local_engine.py       ← VLALocalEngine (wraps VLA model locally)
│
├── examples/
│   └── robot/
│       ├── libero_rl.py              ← training entry point
│       ├── conf/
│       │   ├── libero_grpo.yaml      ← GRPO config
│       │   └── libero_ppo.yaml       ← PPO config (optional)
│       └── README.md                 ← setup + quickstart
│
├── docs/
│   └── vla_integration.md            ← deep-dive architecture notes
│
└── tests/
    ├── test_vla_robot_workflow.py     ← unit tests for the workflow
    └── test_robot_dataset.py         ← dataset tests
```

---

## Quickstart

```bash
# 1. Clone AReaL-VLA
git clone https://github.com/Intelligent-Cloud-Infrastructure-Lab/AReaL-VLA

# 2. Install AReaL + robot sim dependencies
cd AReaL
pip install uv
uv sync --extra cuda
pip install libero-benchmark  # LIBERO env
pip install robottwin          # RoboTwin env (optional)

# 3. Download an SFT VLA model checkpoint
# OpenVLA-OFT: https://huggingface.co/collections/openvla/simplevla-rl
# Or train your own SFT model first

# 4. Run GRPO RL training on LIBERO
python examples/robot/libero_rl.py \
    --config examples/robot/conf/libero_grpo.yaml \
    model.path=/path/to/openvla_oft_sft \
    scheduler.type=local \
    cluster.n_gpus_per_node=8
```
