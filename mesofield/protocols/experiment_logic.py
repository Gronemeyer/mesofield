"""Very small helpers for expanding experiment definitions."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

TrialSpec = Dict[str, Any]
RoutineSpec = Dict[str, Any]

__all__ = [
    "TrialSpec",
    "RoutineSpec",
    "StructuredTrial",
    "compile_trials",
    "compute_routine_duration",
    "build_structured_trials",
    "iter_structured_trials",
]


@dataclass
class StructuredTrial:
    label: str
    mode: Optional[str]
    duration: Optional[float]
    block_name: Optional[str]
    routines: List[RoutineSpec]
    sequence_index: int
    definition: TrialSpec


def compile_trials(definition: Dict[str, Any]) -> List[TrialSpec]:
    """Return trial dictionaries in the order they should run."""

    blocks = definition.get("blocks") or []
    try:
        seed = int(definition.get("rng_seed", 0) or 0)
    except (TypeError, ValueError):
        seed = 0
    rng = random.Random(seed)

    plan: List[TrialSpec] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        raw_trials = block.get("trials") or []
        trials = [t for t in raw_trials if isinstance(t, dict)]
        if not trials:
            continue

        policy = str(block.get("policy", "sequential") or "sequential").lower()
        repeats = block.get("repeats", 1)
        try:
            repeat_count = max(1, int(repeats))
        except (TypeError, ValueError):
            repeat_count = 1

        order = list(range(len(trials)))
        for _ in range(repeat_count):
            run_order = list(order)
            if policy == "random_no_replacement":
                rng.shuffle(run_order)
            for idx in run_order:
                trial_copy = copy.deepcopy(trials[idx])
                trial_copy.setdefault("block_name", block.get("name"))
                plan.append(trial_copy)
    return plan


def compute_routine_duration(routines: Iterable[RoutineSpec]) -> Optional[float]:
    """Sum routine durations unless an open-ended action is present."""

    total = 0.0
    for routine in routines:
        if not isinstance(routine, dict):
            return None
        if routine.get("action") == "wait_for_key":
            return None
        duration = routine.get("duration_seconds")
        if duration is None:
            return None
        try:
            total += float(duration)
        except (TypeError, ValueError):
            return None
    return total


def build_structured_trials(definition: Dict[str, Any]) -> Tuple[List[StructuredTrial], Dict[str, Any]]:
    """Create StructuredTrial objects and a tiny metadata dictionary."""

    plan = compile_trials(definition)
    trials: List[StructuredTrial] = []
    required_keys: set[str] = set()

    for index, trial in enumerate(plan, start=1):
        routines = _clone_routines(trial.get("routines", []))
        required_keys.update(_find_required_keys(routines))
        trials.append(
            StructuredTrial(
                label=_resolve_trial_label(trial, index),
                mode=_resolve_trial_mode(trial),
                duration=compute_routine_duration(routines),
                block_name=trial.get("block_name"),
                routines=routines,
                sequence_index=index,
                definition=copy.deepcopy(trial),
            )
        )

    metadata = {"required_keys": sorted(required_keys)}
    return trials, metadata


def iter_structured_trials(definition: Dict[str, Any]) -> Iterator[StructuredTrial]:
    trials, _ = build_structured_trials(definition)
    return iter(trials)


def _clone_routines(routines: Iterable[RoutineSpec]) -> List[RoutineSpec]:
    copies: List[RoutineSpec] = []
    for entry in routines:
        if isinstance(entry, dict):
            copies.append(copy.deepcopy(entry))
    return copies


def _find_required_keys(routines: Iterable[RoutineSpec]) -> List[str]:
    keys: List[str] = []
    for routine in routines:
        if not isinstance(routine, dict):
            continue
        if routine.get("action") == "wait_for_key":
            name = routine.get("key_name")
            if isinstance(name, str) and name:
                keys.append(name)
    return keys


def _resolve_trial_label(trial: TrialSpec, fallback_index: int) -> str:
    for key in ("label", "name"):
        value = trial.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return f"trial_{fallback_index}"


def _resolve_trial_mode(trial: TrialSpec) -> Optional[str]:
    value = trial.get("mode")
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return None
