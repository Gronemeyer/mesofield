from __future__ import annotations

import json
from pathlib import Path

import pytest

from mesofield.protocols.experiment_logic import (
    build_structured_trials,
    compute_routine_duration,
    compile_trials,
)


def _load_sample_definition() -> dict:
    config_path = Path(__file__).with_name("sample_experiment") / "devcfg.json"
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data["Plugins"]["mouseportal"]["config"]["experiment"]


def test_compile_trials_applies_repeats_and_policies() -> None:
    definition = _load_sample_definition()
    plan = compile_trials(definition)

    assert len(plan) == 2

    durations = [compute_routine_duration(trial.get("routines", [])) for trial in plan]
    assert durations[0] == pytest.approx(5.0)
    assert durations[1] is None


def test_build_structured_trials_enriches_metadata() -> None:
    definition = _load_sample_definition()
    trials, metadata = build_structured_trials(definition)

    assert len(trials) == 2
    first, second = trials

    assert first.label == "show_twice"
    assert first.block_name == "show_texture_block"
    assert first.duration == pytest.approx(5.0)
    assert second.duration is None
    assert metadata["required_keys"] == ["space"]
