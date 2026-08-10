"""GUI controller -- run-state transitions and ExperimentConfig mutation.

Builds the real ``ConfigController`` headlessly (offscreen Qt via conftest +
pytest-qt's session QApplication). Covers:

* run-state: record() -> recordStarted, the run-lifecycle button toggles, and
  the start gate (stimulus vs manual);
* config mutation: adding subjects / parameters / notes goes through the
  controller into the live ExperimentConfig.

Dialogs (QInputDialog / QMessageBox / force_foreground) are monkeypatched so no
modal ever blocks.
"""

from __future__ import annotations

import threading

import pytest

# Register mock device types for the hardware_yaml fixture.
import mesofield.devices.mocks  # noqa: F401
from mesofield.base import Procedure

pytestmark = pytest.mark.gui


@pytest.fixture
def controller(qtbot, hardware_yaml, experiment_json, tmp_path):
    from mesofield.gui.controller import ConfigController

    proc = Procedure(
        hardware=str(hardware_yaml()),
        config=str(experiment_json()),
        experiment_directory=str(tmp_path / "out"),
    )
    ctrl = ConfigController(proc)
    qtbot.addWidget(ctrl)
    return ctrl


class _FakeStim:
    device_type = "stimulus"
    launch_phase = "start"
    enabled = True
    device_id = "stim"

    def __init__(self, ok: bool):
        self._ok = ok
        self.started = False

    def start(self) -> bool:
        self.started = True
        return self._ok


# --------------------------------------------------------------------------- #
# Run-state / threading
# --------------------------------------------------------------------------- #
def test_run_lifecycle_toggles_buttons(controller):
    controller._on_run_started()
    assert controller.record_button.isEnabled() is False
    assert controller.abort_button.isEnabled() is True

    controller._on_run_finished()
    assert controller.record_button.isEnabled() is True
    assert controller.abort_button.isEnabled() is False


def test_record_emits_record_started(controller, qtbot, monkeypatch):
    # Don't drive a real acquisition; just prove record() runs + signals.
    monkeypatch.setattr(controller.procedure, "run", lambda: None)
    with qtbot.waitSignal(controller.recordStarted, timeout=1000):
        controller.record()


def test_start_gate_no_stimulus_uses_manual(controller, monkeypatch):
    monkeypatch.setattr(controller, "_manual_start_gate", lambda: True)
    assert controller._start_gate(controller.procedure) is True


def test_start_gate_with_stimulus_proceeds(controller):
    controller.procedure.hardware.devices["stim"] = _FakeStim(ok=True)
    assert controller._start_gate(controller.procedure) is True
    assert controller.procedure.hardware.devices["stim"].started is True


def test_start_gate_with_failed_stimulus_cancels(controller):
    controller.procedure.hardware.devices["stim"] = _FakeStim(ok=False)
    assert controller._start_gate(controller.procedure) is False


def test_manual_start_gate_ok_proceeds(controller, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    # _manual_start_gate imports force_foreground lazily from its source module.
    monkeypatch.setattr(
        "mesofield.devices.subprocesses.psychopy.force_foreground", lambda w: None
    )
    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.StandardButton.Ok)
    assert controller._manual_start_gate() is True


def test_manual_start_gate_from_record_thread_builds_dialog_on_gui_thread(
    controller, qtbot, monkeypatch
):
    """Regression: the gate used to hang when reached off the GUI thread.

    ``record()`` runs the procedure on a worker thread, so the gate builds its
    dialog there; a QWidget created off the GUI thread can't be parented to one
    and its ``exec()`` never returns. The other gate tests call it on the main
    thread, which is why none of them caught it.
    """
    from PyQt6.QtCore import QThread
    from PyQt6.QtWidgets import QApplication, QMessageBox

    monkeypatch.setattr(
        "mesofield.devices.subprocesses.psychopy.force_foreground", lambda w: None
    )
    seen: dict = {}

    def _exec(box):
        seen["thread"] = box.thread()
        seen["exec_thread"] = QThread.currentThread()
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "exec", _exec)

    # No start-phase stimulus on the mock rig -> the manual gate.
    controller.config.set("start_on_trigger", True)
    result: dict = {}
    thread = threading.Thread(
        target=lambda: result.update(gate=controller._start_gate(controller.procedure)),
        name="mesofield-record",
        daemon=True,
    )
    thread.start()
    qtbot.waitUntil(lambda: "gate" in result, timeout=5000)
    thread.join(timeout=1)

    assert result["gate"] is True
    gui_thread = QApplication.instance().thread()
    assert seen["thread"] is gui_thread, "dialog was created off the GUI thread"
    assert seen["exec_thread"] is gui_thread, "dialog was executed off the GUI thread"


# --------------------------------------------------------------------------- #
# Config-state mutation via the controller
# --------------------------------------------------------------------------- #
def test_add_note_appends_to_config(controller, monkeypatch):
    monkeypatch.setattr(
        "mesofield.gui.controller.QInputDialog.getText",
        lambda *a, **k: ("hello world", True),
    )
    before = len(controller.config.notes)
    controller._add_note()
    assert len(controller.config.notes) == before + 1
    assert controller.config.notes[-1].endswith("hello world")


def test_add_subject_updates_config_and_dropdown(controller, monkeypatch):
    monkeypatch.setattr(
        "mesofield.gui.controller.QInputDialog.getText",
        lambda *a, **k: ("M2", True),
    )
    controller._add_subject()
    assert "M2" in controller.config.subjects
    assert controller.subject_dropdown.findText("M2") >= 0


def test_add_parameter_applies_to_config(controller, monkeypatch):
    import mesofield.gui.controller as ctrl_mod

    # name -> "trials"; getItem -> type "int"; getInt -> default 5.
    monkeypatch.setattr(ctrl_mod.QInputDialog, "getText", lambda *a, **k: ("trials", True))
    monkeypatch.setattr(ctrl_mod.QInputDialog, "getItem", lambda *a, **k: ("int", True))
    monkeypatch.setattr(ctrl_mod.QInputDialog, "getInt", lambda *a, **k: (5, True))

    controller._add_parameter()
    assert controller.config.get("trials") == 5
