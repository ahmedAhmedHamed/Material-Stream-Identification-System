from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Dict, List


@dataclass(frozen=True)
class ClassBalancePlan:
    selected_paths: Dict[str, List[str]]
    augmented_needed: Dict[str, int]
    target_per_class: int


def build_class_balance_plan(
    class_to_paths: Dict[str, List[str]],
    *,
    random_state: int,
    target_multiplier: float = 1.0,
    target: str = "median",
) -> ClassBalancePlan:
    counts = [len(v) for v in class_to_paths.values() if v]
    if not counts:
        return ClassBalancePlan({}, {}, 0)

    base = _pick_target_count(counts, target)
    target_n = max(1, int(round(base * float(target_multiplier))))
    rng = Random(int(random_state))

    selected: Dict[str, List[str]] = {}
    aug_needed: Dict[str, int] = {}
    for label, paths in class_to_paths.items():
        keep = min(len(paths), target_n)
        selected[label] = _sample_without_replacement(paths, keep, rng)
        aug_needed[label] = max(0, target_n - keep)
    return ClassBalancePlan(selected, aug_needed, target_n)


def _pick_target_count(counts: List[int], target: str) -> int:
    sorted_counts = sorted(counts)
    if target == "min":
        return int(sorted_counts[0])
    if target == "max":
        return int(sorted_counts[-1])
    if target == "mean":
        return int(round(sum(sorted_counts) / len(sorted_counts)))
    if target == "median":
        mid = len(sorted_counts) // 2
        if len(sorted_counts) % 2 == 1:
            return int(sorted_counts[mid])
        return int(round((sorted_counts[mid - 1] + sorted_counts[mid]) / 2))
    raise ValueError(f"Unknown target strategy: {target!r}")


def _sample_without_replacement(paths: List[str], k: int, rng: Random) -> List[str]:
    if k <= 0:
        return []
    if k >= len(paths):
        return list(paths)
    # Random.sample preserves uniqueness without replacement.
    return rng.sample(list(paths), k)


