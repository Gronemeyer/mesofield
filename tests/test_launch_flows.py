"""Launch-flow resolution -- ``_resolve_target_path`` / ``load_procedure``.

Covers the single resolver that maps a launch target (rig name / ``dev`` / yaml
/ json / py / directory / None) to a :class:`Procedure`, plus the scripted
``procedure.py`` loader.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from mesofield.base import (
    Procedure,
    _load_procedure_from_py,
    _resolve_target_path,
    load_procedure,
)


# --------------------------------------------------------------------------- #
# _resolve_target_path: argument -> filesystem path
# --------------------------------------------------------------------------- #
def test_resolve_none_or_empty():
    assert _resolve_target_path(None) is None
    assert _resolve_target_path("") is None


def test_resolve_existing_path_is_verbatim(tmp_path):
    p = tmp_path / "rig.yaml"
    p.write_text("memory_buffer_size: 1\n")
    assert _resolve_target_path(str(p)) == str(p)


def test_resolve_unknown_name_is_none():
    assert _resolve_target_path("definitely-not-a-rig-name") is None


def test_resolve_dev_writes_mock_yaml():
    path = _resolve_target_path("dev")
    assert path is not None and path.endswith(".yaml")


# --------------------------------------------------------------------------- #
# load_procedure: target -> Procedure
# --------------------------------------------------------------------------- #
def test_load_none_is_unconfigured_base(no_hardware_init):
    proc = load_procedure(None)
    assert type(proc) is Procedure
    assert not proc.config.hardware.is_configured


def test_load_yaml_only_builds_base_procedure(hardware_yaml, no_hardware_init):
    proc = load_procedure(str(hardware_yaml()))
    assert type(proc) is Procedure
    assert proc.config.hardware.is_configured


def test_load_self_contained_json(tmp_path, no_hardware_init):
    doc = {
        "Configuration": {"protocol": "T", "duration": 1},
        "Subjects": {"S": {"session": "01", "task": "demo"}},
        "hardware": {
            "memory_buffer_size": 10,
            "wheel": {
                "type": "mock_wheel", "primary": True, "sample_interval_ms": 20,
                "cpr": 2400, "diameter_mm": 80,
                "output": {"suffix": "wheel", "file_type": "csv", "bids_type": "beh"},
            },
        },
    }
    j = tmp_path / "experiment.json"
    j.write_text(json.dumps(doc))
    proc = load_procedure(str(j))
    assert type(proc) is Procedure
    assert proc.config.hardware.is_configured
    assert proc.config.get("protocol") == "T"


def test_load_json_falls_back_to_sibling_yaml(tmp_path, hardware_yaml, no_hardware_init):
    hardware_yaml(name="hardware.yaml")  # writes into tmp_path
    j = tmp_path / "experiment.json"
    j.write_text(json.dumps({"Configuration": {"protocol": "T", "duration": 1}}))
    proc = load_procedure(str(j))
    assert proc.config.hardware.is_configured


def test_load_directory_prefers_procedure_py(sample_experiment_dir, no_hardware_init):
    exp_dir = sample_experiment_dir("DirProc")
    proc = load_procedure(str(exp_dir))
    assert proc.__class__.__name__ == "DirProc"
    assert isinstance(proc, Procedure)


# --------------------------------------------------------------------------- #
# _load_procedure_from_py: scripted procedure selection
# --------------------------------------------------------------------------- #
def _write_proc_py(tmp_path, body: str):
    p = tmp_path / "procedure.py"
    p.write_text(textwrap.dedent(body))
    return p


def test_scripted_single_subclass_is_selected(tmp_path):
    p = _write_proc_py(
        tmp_path,
        """
        from mesofield.base import Procedure
        class OnlyProc(Procedure):
            pass
        """,
    )
    proc = _load_procedure_from_py(str(p))
    assert proc.__class__.__name__ == "OnlyProc"
    assert isinstance(proc, Procedure)


def test_scripted_PROCEDURE_attr_disambiguates(tmp_path):
    p = _write_proc_py(
        tmp_path,
        """
        from mesofield.base import Procedure
        class A(Procedure):
            pass
        class B(Procedure):
            pass
        PROCEDURE = B
        """,
    )
    assert _load_procedure_from_py(str(p)).__class__.__name__ == "B"


def test_scripted_multiple_without_marker_raises(tmp_path):
    p = _write_proc_py(
        tmp_path,
        """
        from mesofield.base import Procedure
        class A(Procedure):
            pass
        class B(Procedure):
            pass
        """,
    )
    with pytest.raises(AttributeError):
        _load_procedure_from_py(str(p))


def test_scripted_no_subclass_raises(tmp_path):
    p = _write_proc_py(tmp_path, "x = 1\n")
    with pytest.raises(AttributeError):
        _load_procedure_from_py(str(p))
