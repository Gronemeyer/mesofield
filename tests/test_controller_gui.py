"""GUI controller -- run-state transitions and ExperimentConfig mutation.

Builds the real ``ConfigController`` headlessly (offscreen Qt via conftest +
pytest-qt's session QApplication). Covers run-state (record() -> recordStarted,
the run-lifecycle button toggles) and config mutation (adding subjects /
parameters / notes lands in the live ExperimentConfig). Dialogs are
monkeypatched so no modal ever blocks.
"""

from __future__ import annotations

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


# --------------------------------------------------------------------------- #
# BIDS picker row <-> config state
# --------------------------------------------------------------------------- #
@pytest.fixture
def multi_subject_controller(qtbot, hardware_yaml, tmp_path):
    """Controller over two subjects whose stored ``task`` values differ.

    One task is served by a PsychoPy script, the other is stimulus-free -- the
    shape that used to leave the picker and the config disagreeing.
    """
    import json

    from mesofield.gui.controller import ConfigController

    doc = {
        "Configuration": {"duration": 1},
        "Subjects": {
            "JG01": {"session": "01", "task": "VECr"},
            "JG04": {"session": "01", "task": "405nm"},
        },
        "PsychoPy": {"VECr": "vis_stim_task-VECr.py"},
        "DisplayKeys": ["duration", "task", "session"],
    }
    path = tmp_path / "multi_subject.json"
    path.write_text(json.dumps(doc))

    proc = Procedure(
        hardware=str(hardware_yaml()),
        config=str(path),
        experiment_directory=str(tmp_path / "out"),
    )
    ctrl = ConfigController(proc)
    qtbot.addWidget(ctrl)
    return ctrl


def test_task_picker_matches_config_on_build(multi_subject_controller):
    ctrl = multi_subject_controller
    assert ctrl.task_dropdown.currentText() == ctrl.config.get("task")
    assert f"task-{ctrl.config.get('task')}_" in ctrl.filename_preview_label.text()


def test_task_picker_follows_subject_switch(multi_subject_controller):
    """Selecting a subject applies its stored task -- picker and preview follow."""
    ctrl = multi_subject_controller
    idx = ctrl.subject_dropdown.findText("JG04")
    assert idx >= 0
    ctrl.subject_dropdown.setCurrentIndex(idx)

    assert ctrl.config.get("task") == "405nm"
    assert ctrl.task_dropdown.currentText() == "405nm"
    assert "task-405nm_" in ctrl.filename_preview_label.text()


def test_task_picker_follows_external_config_write(multi_subject_controller):
    """A config-side ``task`` write re-syncs the dropdown, not just the preview."""
    ctrl = multi_subject_controller
    ctrl.config.set("task", "VECr")
    assert ctrl.task_dropdown.currentText() == "VECr"
    assert "task-VECr_" in ctrl.filename_preview_label.text()
