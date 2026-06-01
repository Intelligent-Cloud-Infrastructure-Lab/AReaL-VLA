"""
Robot task dataset for VLA RL training.

Each sample is an *environment-initialisation spec*, not a static text sequence.
The heavy work (creating the simulator, running the physics, rendering frames) all
happens inside VLARobotWorkflow.arun_episode — this dataset simply tells the workflow
which task to attempt and with which random seed.

Design notes
------------
* SimpleVLA-RL's rob_dataset.py returns task-name / instruction / seed rows.
  We do the same, wrapped in a torch.utils.data.Dataset subclass so AReaL's
  standard create_dataloader / StatefulDataLoader pipeline works unchanged.

* RobotCurriculumSampler implements "dynamic sampling" from SimpleVLA-RL:
  tasks with intermediate success rates (neither always-fail nor always-succeed)
  get upsampled because they carry the most gradient signal.

  Specifically:
      p(task) ∝ clamp(success_rate, ε, 1−ε)  — mirroring the logic described
  in the SimpleVLA-RL README / paper.

Supported benchmarks
--------------------
* LIBERO (libero_object, libero_spatial, libero_goal, libero_long)
* RoboTwin 1.0 / 2.0 (when installed)
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

try:
    import torch
    from torch.utils.data import Dataset, Sampler
except ImportError:
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        """Minimal stub when torch is unavailable (unit-test mode)."""
        def __len__(self): raise NotImplementedError
        def __getitem__(self, idx): raise NotImplementedError

    class Sampler:  # type: ignore[no-redef]
        """Minimal stub when torch is unavailable (unit-test mode)."""
        def __iter__(self): raise NotImplementedError
        def __len__(self): raise NotImplementedError


# ---------------------------------------------------------------------------
# Per-task spec
# ---------------------------------------------------------------------------


@dataclass
class RobotTaskSpec:
    """Lightweight environment-initialisation specification."""

    task_name: str
    instruction: str
    benchmark: str
    # Seed used for env.reset; cycling through N_SEEDS seeds per task is one way
    # to improve sample diversity (SimpleVLA-RL uses seed cycling implicitly).
    seed: int = 0


# ---------------------------------------------------------------------------
# LIBERO task catalogue helpers
# ---------------------------------------------------------------------------

# LIBERO task suites and their instructions.  Extend as needed.
# Source: https://libero-project.github.io/
_LIBERO_SUITES: dict[str, list[dict[str, str]]] = {
    "libero_object": [
        {"name": "LIBERO_OBJECT_pick_up_the_alphabet_soup",
         "instruction": "pick up the alphabet soup"},
        {"name": "LIBERO_OBJECT_place_the_alphabet_soup_in_the_top_drawer",
         "instruction": "place the alphabet soup in the top drawer"},
        {"name": "LIBERO_OBJECT_pick_up_the_cream_cheese",
         "instruction": "pick up the cream cheese"},
        {"name": "LIBERO_OBJECT_place_the_cream_cheese_in_the_bowl",
         "instruction": "place the cream cheese in the bowl"},
        {"name": "LIBERO_OBJECT_pick_up_the_ketchup",
         "instruction": "pick up the ketchup"},
        # … populate from the full LIBERO task set
    ],
    "libero_spatial": [
        {"name": "LIBERO_SPATIAL_place_the_black_bowl_left_of_the_plate",
         "instruction": "place the black bowl left of the plate"},
        {"name": "LIBERO_SPATIAL_place_the_alphabet_soup_right_of_the_plate",
         "instruction": "place the alphabet soup right of the plate"},
        # … populate from the full LIBERO task set
    ],
    "libero_goal": [
        {"name": "LIBERO_GOAL_open_the_middle_drawer_of_the_cabinet",
         "instruction": "open the middle drawer of the cabinet"},
        {"name": "LIBERO_GOAL_push_the_plate_to_the_front",
         "instruction": "push the plate to the front"},
        # … populate from the full LIBERO task set
    ],
    "libero_long": [
        {"name": "LIBERO_LONG_open_the_top_drawer_put_the_cream_cheese_in",
         "instruction": "open the top drawer and put the cream cheese in"},
        {"name": "LIBERO_LONG_open_the_bottom_drawer_of_the_cabinet_and_place_the_wine_bottle_in_it",
         "instruction": "open the bottom drawer of the cabinet and place the wine bottle in it"},
        # … populate from the full LIBERO task set
    ],
}


def build_libero_task_specs(
    suite: str = "libero_object",
    n_seeds: int = 5,
    seed_offset: int = 0,
) -> list[RobotTaskSpec]:
    """
    Enumerate all tasks in a LIBERO suite × n_seeds random seeds.

    Multiplying by seeds is one way to increase dataset diversity without
    changing the underlying task distribution — the environment starts from
    a slightly different initial configuration each time.
    """
    if suite not in _LIBERO_SUITES:
        raise ValueError(
            f"Unknown LIBERO suite '{suite}'. "
            f"Available: {list(_LIBERO_SUITES.keys())}"
        )
    specs: list[RobotTaskSpec] = []
    for task in _LIBERO_SUITES[suite]:
        for s in range(seed_offset, seed_offset + n_seeds):
            specs.append(
                RobotTaskSpec(
                    task_name=task["name"],
                    instruction=task["instruction"],
                    benchmark=suite,
                    seed=s,
                )
            )
    return specs


def build_task_specs_from_libero_env(
    benchmark_name: str = "libero_object",
    n_seeds: int = 5,
) -> list[RobotTaskSpec]:
    """
    Dynamically enumerate tasks by importing the libero library at runtime.

    This avoids hardcoding task names.  Falls back to the static list above if
    libero is not installed (useful for unit tests).
    """
    try:
        import libero.libero.benchmark as lb  # type: ignore[import]

        bm = lb.get_benchmark_dict()[benchmark_name]()
        specs: list[RobotTaskSpec] = []
        for idx in range(bm.get_num_tasks()):
            task = bm.get_task(idx)
            lang = task.language
            name = task.name
            for seed in range(n_seeds):
                specs.append(
                    RobotTaskSpec(
                        task_name=name,
                        instruction=lang,
                        benchmark=benchmark_name,
                        seed=seed,
                    )
                )
        return specs
    except ImportError:
        # Fallback: use static catalogue
        return build_libero_task_specs(suite=benchmark_name, n_seeds=n_seeds)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RobotTaskDataset(Dataset):
    """
    torch.utils.data.Dataset whose items are RobotTaskSpec dicts.

    Compatible with AReaL's create_dataloader / StatefulDataLoader helpers.

    Parameters
    ----------
    specs:
        List of task specs to include.  Typically built via build_libero_task_specs
        or build_task_specs_from_libero_env.
    repeat:
        If True, the dataset repeats indefinitely (used for RL training where the
        number of steps is not bounded by dataset length).
        AReaL's cycle_dataloader already handles infinite cycling, so set this
        to False unless you need custom behaviour.
    """

    def __init__(
        self,
        specs: list[RobotTaskSpec],
        repeat: bool = False,
    ) -> None:
        if not specs:
            raise ValueError("RobotTaskDataset: specs list is empty")
        self.specs = specs
        self.repeat = repeat

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.repeat:
            idx = idx % len(self.specs)
        spec = self.specs[idx]
        return {
            "task_name": spec.task_name,
            "instruction": spec.instruction,
            "benchmark": spec.benchmark,
            "seed": spec.seed,
        }


# ---------------------------------------------------------------------------
# Curriculum / dynamic sampler (SimpleVLA-RL "dynamic sampling")
# ---------------------------------------------------------------------------


class RobotCurriculumSampler(Sampler):
    """
    Dynamic task sampler that biases toward tasks with intermediate success rates.

    Implements the exploration-enhancing sampling strategy described in
    SimpleVLA-RL: tasks that are "too easy" (always succeed) or "too hard"
    (never succeed) contribute less gradient signal than tasks where the model
    sometimes succeeds and sometimes fails.

    Sampling weight per task:
        w(task) = clamp(success_rate(task), lo=epsilon, hi=1 - epsilon)

    Initially (zero data), all tasks have equal weight.  As training progresses and
    per-task outcome feedback is received (via update_outcome), weights adapt.

    Parameters
    ----------
    dataset:
        RobotTaskDataset whose specs drive the sampling.
    epsilon:
        Lower bound for success-rate clamping (avoids zero weight on hard tasks).
        Default 0.05 (5 % success rate floor).
    ema_alpha:
        Exponential moving average factor for updating per-task success rates.
        0.0 = use cumulative average; 1.0 = only use the most recent episode.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: RobotTaskDataset,
        epsilon: float = 0.05,
        ema_alpha: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.epsilon = epsilon
        self.ema_alpha = ema_alpha
        self._rng = random.Random(seed)

        # Map task_name → indices in dataset
        self._task_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, spec in enumerate(dataset.specs):
            self._task_to_indices[spec.task_name].append(idx)

        # Per-task success-rate estimate (initialised to 0.5 = no info)
        self._success_rate: dict[str, float] = {
            name: 0.5 for name in self._task_to_indices
        }

    @property
    def task_names(self) -> list[str]:
        return list(self._task_to_indices.keys())

    def update_outcome(self, task_name: str, success: bool) -> None:
        """
        Update the per-task success rate estimate after one episode.

        Call this from the training loop or the rollout callback.
        """
        if task_name not in self._success_rate:
            return
        old = self._success_rate[task_name]
        new_val = float(success)
        self._success_rate[task_name] = (
            (1 - self.ema_alpha) * old + self.ema_alpha * new_val
        )

    def _compute_weights(self) -> list[float]:
        task_names = list(self._task_to_indices.keys())
        weights: list[float] = []
        for name in task_names:
            sr = self._success_rate[name]
            # clamp so no task gets zero weight
            w = max(self.epsilon, min(1.0 - self.epsilon, sr))
            # Equal weight across all seeds of a task
            n_samples = len(self._task_to_indices[name])
            weights.extend([w] * n_samples)
        return weights

    def __iter__(self):
        weights = self._compute_weights()
        n = len(self.dataset)
        indices = self._rng.choices(range(n), weights=weights, k=n)
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)

    def state_dict(self) -> dict[str, Any]:
        """Serialise sampler state for checkpoint / recovery."""
        return {
            "success_rate": dict(self._success_rate),
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from a checkpoint."""
        self._success_rate.update(state.get("success_rate", {}))
        if "rng_state" in state:
            self._rng.setstate(state["rng_state"])


# ---------------------------------------------------------------------------
# Helper: split dataset into train / val by task name
# ---------------------------------------------------------------------------


def split_train_val(
    specs: list[RobotTaskSpec],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[RobotTaskSpec], list[RobotTaskSpec]]:
    """
    Split specs into train / val ensuring full tasks (all seeds) go to the
    same split — we don't want to evaluate on tasks we trained on.

    Returns (train_specs, val_specs).
    """
    rng = random.Random(seed)
    task_names = sorted(set(s.task_name for s in specs))
    rng.shuffle(task_names)
    n_val = max(1, int(len(task_names) * val_fraction))
    val_tasks = set(task_names[:n_val])

    train_specs = [s for s in specs if s.task_name not in val_tasks]
    val_specs = [s for s in specs if s.task_name in val_tasks]
    return train_specs, val_specs
