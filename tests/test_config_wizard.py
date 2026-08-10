import json
import sys

import pytest
import mesofield.devices.mocks  # noqa: F401
from mesofield.base import Procedure

pytestmark = pytest.mark.gui


@pytest.fixture
def wizard(qtbot, hardware_yaml, experiment_json, tmp_path):
    from mesofield.gui.config_wizard import ConfigWizard

    proc = Procedure(
        hardware=str(hardware_yaml()),
        config=str(experiment_json()),
        experiment_directory=str(tmp_path / "out"),
    )
    w = ConfigWizard(proc)
    qtbot.addWidget(w)
    return w


def test_wizard_builds_without_mm_section(wizard, tmp_path):
    assert wizard._mm_section.isHidden()
    assert wizard._mm_section.parent() is None
    wizard._on_outdir_changed(str(tmp_path))
    wizard.refresh_mm_section()


def test_wizard_window_hosts_wizard_and_console(qtbot, wizard):
    from mesofield.gui.wizard_window import WizardWindow

    win = WizardWindow(wizard)
    qtbot.addWidget(win)
    assert win.windowTitle() == "Mesofield Wizard"
    assert wizard.parent() is not None

    # Console captures both loguru records and bare prints.
    from loguru import logger

    logger.info("hello from loguru")
    print("hello from stdout")
    text = win.console.toPlainText()
    assert "hello from loguru" in text
    assert "hello from stdout" in text

    # Closing releases the borrowed wizard and restores the streams.
    win.close()
    assert wizard.parent() is None
    assert not isinstance(sys.stdout, type(win.console)) and sys.stdout is not None
    print("back to the real stdout")
    assert "back to the real stdout" not in win.console.toPlainText()


def test_console_replays_records_logged_before_it_existed(qtbot, wizard):
    """The launch preamble used to scroll past in the terminal only."""
    from mesofield.gui.wizard_window import WizardWindow
    from mesofield.utils._logger import get_logger

    get_logger("test").info("logged well before any window opened")

    win = WizardWindow(wizard)
    qtbot.addWidget(win)
    assert "logged well before any window opened" in win.console.toPlainText()
    win.close()


def test_tiff_viewer_reachable_from_wizard_and_main_toolbar(qtbot, wizard):
    """Both entry points exist: the wizard button and the main window toolbar."""
    from mesofield.gui.wizard_window import WizardWindow
    from mesofield.gui.maingui import MainWindow

    from PyQt6.QtWidgets import QPushButton

    win = WizardWindow(wizard)
    qtbot.addWidget(win)
    labels = [b.text() for b in win.findChildren(QPushButton)]
    assert any("TIFF Viewer" in t for t in labels)
    assert hasattr(MainWindow, "_open_tiff_viewer")
    win.close()


def test_startup_banner_reports_versions_and_machine(qtbot, wizard):
    from mesofield.gui import wizard_window as ww

    win = ww.WizardWindow(wizard)
    qtbot.addWidget(win)
    text = win.console.toPlainText()
    for label in ("mesofield", "pymmcore-plus", "os", "cpu", "gpu", "ram"):
        assert f"\n{label}" in f"\n{text}" or text.startswith(label)
    # Values are probed, never left as the raw label.
    assert "GB" in text
    win.close()


def test_custom_startup_lines_are_appended_and_failures_contained(qtbot, wizard, monkeypatch):
    from mesofield.gui import wizard_window as ww

    monkeypatch.setattr(ww, "CUSTOM_STARTUP_LINES", [
        "plain line",
        lambda: "computed line",
        lambda: ["a", "b"],
        lambda: 1 / 0,
    ])
    win = ww.WizardWindow(wizard)
    qtbot.addWidget(win)
    text = win.console.toPlainText()
    assert "plain line" in text
    assert "computed line" in text
    assert "\na\nb\n" in text
    assert "ZeroDivisionError" in text  # reported inline, window still built
    win.close()


def test_console_header_links_to_the_log_file(qtbot, wizard):
    from PyQt6.QtWidgets import QLabel
    from mesofield.gui.wizard_window import WizardWindow
    from mesofield.utils._logger import log_file, setup_logging

    setup_logging()  # no-op if already configured
    win = WizardWindow(wizard)
    qtbot.addWidget(win)
    texts = [lbl.text() for lbl in win.findChildren(QLabel)]
    assert any(t == "Console" for t in texts)
    assert any("mesofield.log" in t and "<a href=" in t for t in texts)
    assert log_file() is not None and log_file().name == "mesofield.log"
    win.close()


def test_experiment_builder_round_trips_an_existing_json(qtbot, tmp_path):
    from mesofield.gui.config_builder import ExperimentBuilderDialog

    path = tmp_path / "experiment.json"
    path.write_text(json.dumps({
        "Configuration": {
            "experimenter": "jgronemeyer",
            "protocol": "strehab",
            "duration": 60,
            "task": ["baseline", "stim"],
        },
        "Subjects": {
            "SUBJ01": {"session": "01", "task": "baseline", "sex": "F", "age": 12},
            "SUBJ02": {"session": "02", "task": "stim", "sex": "M", "age": 14},
        },
        "DisplayKeys": ["subject", "session", "task", "sex"],
    }), encoding="utf-8")

    dlg = ExperimentBuilderDialog(json_path=str(path))
    qtbot.addWidget(dlg)

    assert dlg.windowTitle() == "Edit experiment.json"
    assert dlg._tasks == ["baseline", "stim"]
    assert [s["subject"] for s in dlg._subjects] == ["SUBJ01", "SUBJ02"]
    assert dlg._session_form.values()["experimenter"] == "jgronemeyer"
    assert dlg._session_form.values()["duration"] == 60
    # `age` is int-valued everywhere -> number; `sex` is in DisplayKeys -> shown.
    assert ("age", int, False) in dlg._variables
    assert ("sex", str, True) in dlg._variables
