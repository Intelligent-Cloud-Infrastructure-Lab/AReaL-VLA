# AReaL-VLA: Architecture Deep Dive

This document explains every architectural decision in the port of SimpleVLA-RL into
AReaL.  It is intended for contributors and researchers who want to understand the
implementation deeply, modify it, or debug a training run.

---

## 1. System overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AReaL-VLA Training Loop                          │
│                                                                           │
│  DataLoader          WorkflowExecutor            FSDPPPOActor            │
│ ┌──────────┐        ┌──────────────┐            ┌──────────────┐        │
│ │RobotTask │──────► │AsyncTask     │──────────► │compute_      │        │
│ │Dataset   │        │Runner        │            │advantages()  │        │
│ └──────────┘        │  ┌─────────┐│            │ppo_update()  │        │
│                      │  │arun_    ││            └──────┬───────┘        │
│                      │  │episode()││                   │ update_weights  │
│                      │  │         ││            ┌──────▼───────┐        │
│                      │  │  ┌────┐ ││            │VLALocalEngine│        │
│                      │  │  │env │ ││            │(or SGLang)   │        │
│                      │  │  │    │ ││            └──────────────┘        │
│                      │  │  │step│ ││                                     │
│                      │  │  └────┘ ││                                     │
│                      │  │    ↕    ││                                     │
│                      │  │  engine ││                                     │
│                      │  │.agen-   ││                                     │
│                      │  │erate()  ││                                     │
│                      │  └─────────┘│                                     │
│                      └──────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. RolloutWorkflow: the central abstraction

AReaL's `RolloutWorkflow` requires exactly one method:

```python
async def arun_episode(
    self,
    engine: InferenceEngine,
    data: dict[str, Any],
) -> dict[str, torch.Tensor] | None
```

`VLARobotWorkflow` implements this by running a **complete robot episode** per call.
AReaL's `WorkflowExecutor` calls `arun_episode` concurrently for multiple data items,
so the degree of environment parallelism equals the runner's max concurrency (set via
`rollout.max_concurrent_episodes` in the YAML config).

This is a clean match for SimpleVLA-RL's "multi-environment parallel rendering":
rather than an explicit env pool, parallelism is achieved naturally through asyncio.

---

## 3. Token-level reward alignment: correctness proof

The key identity is:

```
For step s (1-indexed) and action dimension d:
  global_action_token_idx = (s - 1) * action_chunk_len + d

Success boundary (finish_step, 1-indexed):
  cutoff = finish_step * action_chunk_len

Token is valid iff global_action_token_idx <= cutoff
```

**Example** (action_chunk_len = 7, finish_step = 3, total_steps = 5):

```
Step 1: action tokens 1..7    → ALL valid (< cutoff=21)
Step 2: action tokens 8..14   → ALL valid (< cutoff=21)
Step 3: action tokens 15..21  → ALL valid (= cutoff=21)   ← success step
Step 4: action tokens 22..28  → ALL masked (> cutoff=21)
Step 5: action tokens 29..35  → ALL masked (> cutoff=21)
```

The `_EpisodeBuffer.append_step` method sets `step_is_post_success=True` for all
steps AFTER the success step, which zero-masks their action tokens.

**One subtlety**: the step at which `info["success"] = True` is first seen (finish_step)
is INCLUDED in the valid region.  The robot executed the action that caused success, so
that action's tokens receive the positive reward signal.

---

## 4. GRPO vs PPO for mixed-success groups

SimpleVLA-RL uses PPO with a custom "mixed-success group" advantage computation.
The advantage within a group of G episodes of the same task is:

```
A_i = (r_i - mean(r_group)) / std(r_group)
```

This is algebraically identical to **GRPO** (Group Relative Policy Optimisation).
AReaL's standard GRPO trainer handles this automatically with `group_size = G` and
`grpo_normalise_within_group = true`.

Why is this equivalent?
- GRPO samples G completions per prompt (here: G episodes per task).
- It normalises advantages within the group.
- With binary rewards, if all G episodes succeed, all advantages are zero (no signal).
  If all fail, all advantages are zero.  Only mixed groups produce non-zero gradients.
- This matches SimpleVLA-RL's design exactly.

To use PPO instead (with an explicit value function), set `actor.algorithm: ppo` in
the YAML and provide a separate critic network.

---

## 5. Dynamic sampling: curriculum sampler design

`RobotCurriculumSampler` maintains a per-task exponential moving average of
success rates and samples tasks with probability proportional to their "informativeness":

```python
# Informativeness = how often the task transitions between success and failure
p(task) ∝ clamp(success_rate(task), ε, 1 − ε)
```

**Why this works:**
- Tasks with success_rate ≈ 0.0 → always fail → all advantages = 0 → no gradient
- Tasks with success_rate ≈ 1.0 → always succeed → all advantages = 0 → no gradient
- Tasks with success_rate ≈ 0.5 → maximum gradient variance → maximum learning signal

The `ε = 0.05` floor prevents any task from being completely dropped from training.

This closely mirrors the "curriculum" discussion in the SimpleVLA-RL paper.

---

## 6. VLALocalEngine vs RemoteSGLangEngine

The default engine is `VLALocalEngine`, which runs the VLA model inline on the same
process that executes the rollout.  This means:

| Property | VLALocalEngine | RemoteSGLangEngine |
|---|---|---|
| Training/inference overlap | None (blocking) | Full (async) |
| Weight sync | Explicit `set_weights()` call | AReaL's `WeightUpdateMeta` protocol |
| Multi-GPU inference | Manual | SGLang tensor parallel |
| Setup complexity | Minimal | Requires SGLang server |

**Migration path to SGLang:**
1. Implement OpenVLA-OFT as a SGLang-compatible model (see SGLang's custom model docs)
2. Change `rollout.backend: local` to `rollout.backend: sglang:dNtM` in the YAML
3. Replace `VLALocalEngine` with `RemoteSGLangEngine(config.rollout)` in the training script
4. Add `WeightUpdateMeta.from_fsdp_xccl(allocation_mode)` and `actor.connect_engine(...)`

The `VLARobotWorkflow` code is **unchanged** — it only uses the `VLAEngine` interface.

---

## 7. Post-success masking: why it matters for fixed-horizon environments

In LIBERO, most environments terminate immediately when the task succeeds (`done=True`).
In this case, there are no post-success tokens to mask, and the masking logic is a no-op.

For **RoboTwin** and other fixed-horizon environments where the robot keeps running
after success, the masking is critical:
- Without masking: the model receives positive reward for arbitrary post-success actions
- With masking: only the actions that *caused* success are reinforced

The `_EpisodeBuffer.append_step(step_is_post_success=True)` call zeros the `loss_mask`
for all post-success action tokens, preventing their gradients from entering the PPO update.

---

## 8. Trajectory tensor format

Each call to `arun_episode` returns exactly one trajectory dict in AReaL's standard format:

```
{
    "input_ids":       [1, seq_len]  int32   — full token sequence (prompt + actions × T)
    "loss_mask":       [1, seq_len]  int32   — 0=prompt or post-success, 1=action
    "logprobs":        [1, seq_len]  float32 — log p(token) from the rollout model
    "versions":        [1, seq_len]  int32   — weight version when each token was sampled
    "attention_mask":  [1, seq_len]  bool    — all ones (full attention)
    "rewards":         [1, 1]        float32 — scalar 0.0 or 1.0
}
```

The `seq_len` for an episode with T steps and action_chunk_len N:
```
seq_len = T × (prompt_len_per_step + N)
```

Where `prompt_len_per_step` = len(image_tokens) + len(instruction_tokens).
This varies by model: OpenVLA-OFT with SigLIP uses ~256 image tokens + ~30 text tokens.

`concat_padded_tensors(trajectories)` pads to the longest sequence in the group and
stacks into `[group_size, max_seq_len]` tensors, which is what AReaL's actor receives.

---

## 9. Known limitations and future work

1. **SGLang support for VLA models**: VLALocalEngine loses AReaL's async overlap
   benefit.  Adding OpenVLA-OFT / π0 to SGLang is the highest-impact next step.

2. **Multi-node environments**: For very large-scale training, the robot environments
   should run on separate CPU-only workers, connected via gRPC or Ray actors.  The
   current design runs envs on the same nodes as the GPUs.

3. **Action chunking in the sequence**: The current implementation concatenates all
   steps into one long sequence.  An alternative (used in some RLHF systems) would be
   to treat each step as a separate "generation" and aggregate rewards.  Both are valid;
   the single-sequence approach is simpler and matches SimpleVLA-RL.

4. **Off-policy staleness**: `max_head_offpolicyness = 0` (fully on-policy) is the
   default.  Enabling AReaL's async mode (`max_head_offpolicyness > 0`) would improve
   throughput but requires care with importance-sampling corrections.

5. **Vision encoder weight update**: With VLALocalEngine, `set_weights()` updates all
   model parameters including the vision encoder.  For efficiency, it may be desirable
   to freeze the vision encoder and only update the LLM backbone.

6. **RoboTwin 2.0 support**: The `robottwin_reward_fn` is implemented but not fully
   tested; RoboTwin 2.0's env API may differ slightly from 1.0.
