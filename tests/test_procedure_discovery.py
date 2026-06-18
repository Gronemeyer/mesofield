"""Tests for custom Procedure discovery via experiment.json."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import pytest

from mesofield.base import Procedure, load_procedure_from_config


def test_sample_experiment_loads_subclass(sample_experiment_dir, no_hardware_init) -> None:
    """A discoverable experiment.json resolves to its declared Procedure subclass.

    Uses a fixture-synthesized experiment dir (procedure.py + experiment.json +
    mock hardware.yaml) instead of a checked-in sample, so the test is portable.
    """
    exp_dir = sample_experiment_dir("SampleProcedure")

    proc = load_procedure_from_config(str(exp_dir / "experiment.json"))

    assert proc.__class__.__name__ == "SampleProcedure"
    assert isinstance(proc, Procedure)


def test_missing_fields_falls_back_to_base(tmp_path: Path) -> None:
    cfg = tmp_path / "experiment.json"
    cfg.write_text(json.dumps({"Configuration": {"protocol": "X"}}))

    proc = load_procedure_from_config(str(cfg))
    assert type(proc) is Procedure


def test_non_procedure_class_rejected(tmp_path: Path) -> None:
    proc_file = tmp_path / "bad.py"
    proc_file.write_text("class NotAProcedure:\n    pass\n")

    cfg = tmp_path / "experiment.json"
    cfg.write_text(
        json.dumps(
            {
                "Configuration": {},
                "procedure_file": "bad.py",
                "procedure_class": "NotAProcedure",
            }
        )
    )

    with pytest.raises((TypeError, ValueError, ImportError)):
        load_procedure_from_config(str(cfg))


def test_bad_path_raises(tmp_path: Path) -> None:
    cfg = tmp_path / "experiment.json"
    cfg.write_text(
        json.dumps(
            {
                "Configuration": {},
                "procedure_file": "does_not_exist.py",
                "procedure_class": "Whatever",
            }
        )
    )

    with pytest.raises((FileNotFoundError, ImportError, OSError)):
        load_procedure_from_config(str(cfg))


def test_relative_path_resolved_against_json_dir(tmp_path: Path) -> None:
    proc_file = tmp_path / "myproc.py"
    proc_file.write_text(
        textwrap.dedent(
            """
            from mesofield.base import Procedure

            class MyProc(Procedure):
                pass
            """
        )
    )

    cfg = tmp_path / "experiment.json"
    cfg.write_text(
        json.dumps(
            {
                "Configuration": {},
                "procedure_file": "myproc.py",
                "procedure_class": "MyProc",
            }
        )
    )

    proc = load_procedure_from_config(str(cfg))
    assert proc.__class__.__name__ == "MyProc"
    assert isinstance(proc, Procedure)
