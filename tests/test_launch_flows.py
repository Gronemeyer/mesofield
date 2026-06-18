"""Launch-flow resolution -- load_procedure_from_config / _resolve_target.

Covers the file-discovery convention that maps a launch target (yaml / json /
py / directory / None) to ``(hardware, experiment, procedure)`` and the scripted
``procedure.py`` loader. Complements ``test_procedure_discovery`` (the
experiment.json -> declared-subclass path).
"""

from __future__ import annotations

import textwrap

import pytest

from mesofield.base import (
    Procedure,
    _load_procedure_from_py,
    _resolve_target,
    load_procedure_from_config,
)


# --------------------------------------------------------------------------- #
# _resolve_target: pure resolution matrix
# --------------------------------------------------------------------------- #
def test_resolve_none_or_empty():
    assert _resolve_target(None) == (None, None, None)
    assert _resolve_target("") == (None, None, None)


@pytest.mark.parametrize("ext", [".yaml", ".yml"])
def test_resolve_yaml_is_hardware_only(tmp_path, ext):
    p = tmp_path / f"rig{ext}"
    p.write_text("memory_buffer_size: 1\n")
    assert _resolve_target(str(p)) == (str(p), None, None)


def test_resolve_py_is_procedure_only(tmp_path):
    p = tmp_path / "procedure.py"
    p.write_text("x = 1\n")
    assert _resolve_target(str(p)) == (None, None, str(p))


def test_resolve_json_with_sibling_hardware(tmp_path):
    hw = tmp_path / "hardware.yaml"
    hw.write_text("memory_buffer_size: 1\n")
    j = tmp_path / "experiment.json"
    j.write_text("{}")
    assert _resolve_target(str(j)) == (str(hw), str(j), None)


def test_resolve_json_without_sibling(tmp_path):
    j = tmp_path / "experiment.json"
    j.write_text("{}")
    assert _resolve_target(str(j)) == (None, str(j), None)


def test_resolve_directory_discovers_pair(tmp_path):
    hw = tmp_path / "hardware.yaml"
    hw.write_text("memory_buffer_size: 1\n")
    exp = tmp_path / "experiment.json"
    exp.write_text("{}")
    assert _resolve_target(str(tmp_path)) == (str(hw), str(exp), None)


def test_resolve_directory_missing_files(tmp_path):
    assert _resolve_target(str(tmp_path)) == (None, None, None)


def test_resolve_unknown_path(tmp_path):
    assert _resolve_target(str(tmp_path / "nope.txt")) == (None, None, None)


# --------------------------------------------------------------------------- #
# load_procedure_from_config: end-to-end target -> Procedure
# --------------------------------------------------------------------------- #
def test_load_yaml_only_builds_base_procedure(hardware_yaml, no_hardware_init):
    proc = load_procedure_from_config(str(hardware_yaml()))
    assert type(proc) is Procedure
    assert proc.config.hardware.is_configured


def test_load_directory_uses_declared_subclass(sample_experiment_dir, no_hardware_init):
    exp_dir = sample_experiment_dir("DirProc")
    proc = load_procedure_from_config(str(exp_dir))
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
