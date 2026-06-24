"""Tests for the procedure -> experiment.json link.

A custom procedure declares its self-contained ``experiment.json`` via the
class-level ``experiment`` path; launching the procedure (or its directory)
loads those parameters and the embedded rig.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from mesofield.base import Procedure, _load_procedure_from_py, load_procedure


def test_directory_loads_subclass(sample_experiment_dir, no_hardware_init) -> None:
    """A directory with a procedure.py resolves to its Procedure subclass."""
    exp_dir = sample_experiment_dir("SampleProcedure")

    proc = load_procedure(str(exp_dir))

    assert proc.__class__.__name__ == "SampleProcedure"
    assert isinstance(proc, Procedure)


def test_procedure_loads_declared_experiment(tmp_path: Path, no_hardware_init) -> None:
    """``experiment = "..."`` loads the sibling JSON's parameters."""
    (tmp_path / "experiment.json").write_text(
        json.dumps({
            "Configuration": {"protocol": "DECLARED", "duration": 1},
            "Subjects": {"S": {"session": "01", "task": "demo"}},
        })
    )
    proc_file = tmp_path / "procedure.py"
    proc_file.write_text(
        textwrap.dedent(
            """
            from mesofield.base import Procedure

            class MyProc(Procedure):
                experiment = "experiment.json"
            """
        )
    )

    proc = _load_procedure_from_py(str(proc_file))
    assert proc.__class__.__name__ == "MyProc"
    assert proc.config.get("protocol") == "DECLARED"


def test_no_declared_experiment_is_base_defaults(tmp_path: Path, no_hardware_init) -> None:
    proc_file = tmp_path / "procedure.py"
    proc_file.write_text(
        textwrap.dedent(
            """
            from mesofield.base import Procedure

            class Bare(Procedure):
                pass
            """
        )
    )

    proc = _load_procedure_from_py(str(proc_file))
    assert proc.__class__.__name__ == "Bare"
    assert not proc.config.hardware.is_configured


def test_plain_json_is_base_procedure(tmp_path: Path, no_hardware_init) -> None:
    cfg = tmp_path / "experiment.json"
    cfg.write_text(json.dumps({"Configuration": {"protocol": "X"}}))

    proc = load_procedure(str(cfg))
    assert type(proc) is Procedure
