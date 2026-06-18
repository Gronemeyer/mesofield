"""Shared fixtures and configuration for the mesofield test suite.

Responsibilities:

* Force a **headless Qt platform before any Qt import** (``offscreen``), so GUI
  tests run without a display.
* Rely on pytest-qt's single **session ``QApplication``** (``qapp``/``qtbot``).
  This fixes the cross-test QObject-lifetime leak where a per-module
  ``QCoreApplication`` torn down mid-session orphaned later ``ProcedureSignals``
  objects (``RuntimeError: wrapped C/C++ object ... has been deleted``).
* Register ``--run-hardware`` and auto-skip ``@pytest.mark.hardware`` tests by
  default.
* Snapshot/restore the global :class:`~mesofield.DeviceRegistry` around every
  test so a test that registers ad-hoc device classes can't leak into others.
* Provide factories for the mock rig (``hardware.yaml`` / ``experiment.json`` /
  ``ExperimentConfig`` / a discoverable ``sample_experiment`` dir) and a
  deterministic ``wait_until`` helper that replaces ``time.sleep`` polling.

Mocks used here come from ``mesofield/devices/mocks.py`` -- a *shipped* surface
that backs the ``mesofield init --rig dev`` template, not throwaway test fakes.
"""

from __future__ import annotations

import os

# Must be set before PyQt6 is imported anywhere (pytest-qt imports it lazily via
# the ``qapp`` fixture, and mesofield GUI modules import it at module load).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import time
from pathlib import Path
from typing import Callable

import pytest


# --------------------------------------------------------------------------- #
# Hardware marker gating
# --------------------------------------------------------------------------- #
def pytest_addoption(parser):
    parser.addoption(
        "--run-hardware",
        action="store_true",
        default=False,
        help="Run @pytest.mark.hardware tests (real camera/serial/MicroManager).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-hardware"):
        return
    skip_hw = pytest.mark.skip(reason="needs real hardware; pass --run-hardware to run")
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip_hw)


# --------------------------------------------------------------------------- #
# Global DeviceRegistry isolation (autouse -- guards every test)
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _isolate_device_registry():
    """Snapshot and restore ``DeviceRegistry._registry`` around each test.

    Some tests register throwaway device classes; this keeps that mutation from
    leaking and making the suite order-dependent.
    """
    from mesofield import DeviceRegistry

    saved = dict(DeviceRegistry._registry)
    try:
        yield
    finally:
        DeviceRegistry._registry.clear()
        DeviceRegistry._registry.update(saved)


# --------------------------------------------------------------------------- #
# Modal-dialog guard: keep a stray .exec() from hanging the suite.
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _no_blocking_dialogs(monkeypatch):
    """Make modal dialogs non-blocking so an unexpected ``.exec()`` can't hang.

    Once a GUI test creates the session ``QApplication``, a ``QMessageBox`` /
    ``QDialog`` ``.exec()`` opens a real modal and blocks on user input (e.g.
    ``PsychoPyDevice.confirm_ready_to_record``). Default them to an *accepted*
    result; tests that assert a specific dialog outcome re-patch as needed.
    """
    try:
        from PyQt6.QtWidgets import QDialog, QMessageBox
    except Exception:
        return
    monkeypatch.setattr(
        QMessageBox, "exec",
        lambda self, *a, **k: QMessageBox.StandardButton.Ok, raising=False,
    )
    monkeypatch.setattr(
        QDialog, "exec",
        lambda self, *a, **k: QDialog.DialogCode.Accepted, raising=False,
    )


# --------------------------------------------------------------------------- #
# Deterministic polling helper (replaces time.sleep-based waits)
# --------------------------------------------------------------------------- #
@pytest.fixture
def wait_until() -> Callable[..., bool]:
    """Return ``wait(predicate, timeout=2.0, interval=0.01, message=None)``.

    Polls ``predicate`` until it is truthy or ``timeout`` elapses; raises
    ``AssertionError`` on timeout. Prefer ``qtbot.waitSignal``/``waitUntil`` for
    Qt-driven paths; use this for thread/daemon-driven device data.
    """

    def _wait(predicate, timeout: float = 2.0, interval: float = 0.01, message=None):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(interval)
        if predicate():
            return True
        raise AssertionError(message or f"condition not met within {timeout}s")

    return _wait


# --------------------------------------------------------------------------- #
# Mock-rig YAML helpers + fixtures
# --------------------------------------------------------------------------- #
_MOCK_WHEEL_STANZA = """\
wheel:
  type: mock_wheel
  primary: true
  sample_interval_ms: 20
  cpr: 2400
  diameter_mm: 80
  output:
    suffix: wheel
    file_type: csv
    bids_type: beh
"""

_MOCK_CAMERA_STANZA = """\
camera:
  type: mock_camera
  width: 32
  height: 32
  frame_interval_ms: 20
  output:
    suffix: meso
    file_type: ome.tiff
    bids_type: func
"""


def mock_rig_yaml(camera: bool = False, extra: str = "") -> str:
    """The text of a minimal mock ``hardware.yaml`` (primary mock wheel).

    ``camera=True`` appends a non-primary mock camera. ``extra`` appends raw
    YAML for additional device stanzas.
    """
    body = "memory_buffer_size: 1000\n\n" + _MOCK_WHEEL_STANZA
    if camera:
        body += "\n" + _MOCK_CAMERA_STANZA
    if extra:
        body += "\n" + extra + "\n"
    return body


@pytest.fixture
def hardware_yaml(tmp_path) -> Callable[..., Path]:
    """Factory: write a mock ``hardware.yaml`` into ``tmp_path`` and return it."""

    def _make(camera: bool = False, name: str = "hardware.yaml", extra: str = "") -> Path:
        path = tmp_path / name
        path.write_text(mock_rig_yaml(camera=camera, extra=extra))
        return path

    return _make


def _default_experiment_doc() -> dict:
    return {
        "Configuration": {
            "experimenter": "tester",
            "protocol": "exp1",
            # ``duration`` is registered as an int (seconds); keep it a whole
            # number so the wall-clock duration cap actually arms (a sub-second
            # value coerces to 0 and disables the timer).
            "duration": 1,
            "start_on_trigger": False,
        },
        "Subjects": {
            "SUBJ1": {
                "sex": "F",
                "genotype": "test",
                "DOB": "2024-01-01",
                "DOS": "2024-01-02",
                "session": "01",
                "task": "wf",
            },
        },
        "DisplayKeys": ["duration", "start_on_trigger", "task", "session"],
    }


@pytest.fixture
def experiment_json(tmp_path) -> Callable[..., Path]:
    """Factory: write a minimal ``experiment.json`` and return its path.

    Keyword args override fields in the ``Configuration`` block.
    """

    def _make(name: str = "experiment.json", **config_overrides) -> Path:
        doc = _default_experiment_doc()
        doc["Configuration"].update(config_overrides)
        path = tmp_path / name
        path.write_text(json.dumps(doc))
        return path

    return _make


@pytest.fixture
def experiment_config(hardware_yaml, experiment_json):
    """Factory: build an :class:`ExperimentConfig` from the mock rig + JSON."""
    from mesofield.config import ExperimentConfig

    def _make(camera: bool = False, with_json: bool = True):
        cfg = ExperimentConfig(str(hardware_yaml(camera=camera)))
        if with_json:
            cfg.load_json(str(experiment_json()))
        return cfg

    return _make


@pytest.fixture
def sample_experiment_dir(tmp_path) -> Callable[..., Path]:
    """Synthesize a discoverable experiment dir with a custom ``SampleProcedure``.

    Replaces the missing shipped ``experiments/sample_experiment/`` fixture: a
    ``procedure.py`` (declaring ``SampleProcedure``), an ``experiment.json`` that
    points at it, and a mock ``hardware.yaml`` -- all in ``tmp_path``.
    """

    def _make(class_name: str = "SampleProcedure") -> Path:
        d = tmp_path / "sample_experiment"
        d.mkdir(exist_ok=True)
        (d / "procedure.py").write_text(
            "from mesofield.base import Procedure\n\n\n"
            f"class {class_name}(Procedure):\n"
            "    pass\n"
        )
        (d / "hardware.yaml").write_text(mock_rig_yaml())
        doc = _default_experiment_doc()
        doc["procedure_file"] = "procedure.py"
        doc["procedure_class"] = class_name
        (d / "experiment.json").write_text(json.dumps(doc))
        return d

    return _make


@pytest.fixture
def no_hardware_init(monkeypatch):
    """Stub ``HardwareManager.initialize`` so procedure-loading tests stay
    hermetic (no device instantiation / daemon threads)."""
    from mesofield.hardware import HardwareManager

    monkeypatch.setattr(HardwareManager, "initialize", lambda self, cfg: None)
    return monkeypatch
