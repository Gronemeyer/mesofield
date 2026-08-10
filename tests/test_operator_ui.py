"""Operator port: Qt dialogs raised from a run's worker thread.

The run never owns the GUI thread, so ``QtOperatorUI`` must build and execute
its dialogs there. A widget created on the calling thread cannot be parented
and its ``exec()`` never returns -- the hang this port exists to prevent.
"""

from __future__ import annotations

import threading

import pytest

from mesofield.gui.operator_ui import on_main_thread, run_on_main_thread

pytestmark = pytest.mark.gui


def _in_worker(fn, qtbot, timeout: int = 5000):
    """Call ``fn`` on a worker thread, pumping the GUI thread until it returns."""
    box: dict = {}

    def _body():
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised by the caller
            box["error"] = exc
        finally:
            box["done"] = True

    thread = threading.Thread(target=_body, name="test-worker", daemon=True)
    thread.start()
    qtbot.waitUntil(lambda: box.get("done", False), timeout=timeout)
    thread.join(timeout=1)
    if "error" in box:
        raise box["error"]
    return box["result"]


# --------------------------------------------------------------------------- #
# run_on_main_thread
# --------------------------------------------------------------------------- #
def test_returns_result_from_worker_thread(qtbot):
    from PyQt6.QtCore import QThread
    from PyQt6.QtWidgets import QApplication

    seen: dict = {}

    def _on_gui():
        seen["thread"] = QThread.currentThread()
        return 42

    assert _in_worker(lambda: run_on_main_thread(_on_gui), qtbot) == 42
    assert seen["thread"] is QApplication.instance().thread()


def test_arguments_are_forwarded(qtbot):
    assert _in_worker(lambda: run_on_main_thread(lambda a, b=0: a + b, 1, b=2), qtbot) == 3


def test_exception_propagates_to_the_calling_thread(qtbot):
    def _boom():
        raise ValueError("gate exploded")

    with pytest.raises(ValueError, match="gate exploded"):
        _in_worker(lambda: run_on_main_thread(_boom), qtbot)


def test_call_on_the_main_thread_does_not_deadlock(qapp):
    # A BlockingQueuedConnection to one's own thread deadlocks; the same-thread
    # path must run inline instead.
    assert on_main_thread() is True
    assert run_on_main_thread(lambda: "inline") == "inline"


# --------------------------------------------------------------------------- #
# QtOperatorUI
# --------------------------------------------------------------------------- #
@pytest.fixture
def qt_ui(qtbot, monkeypatch):
    from mesofield.gui.operator_ui import QtOperatorUI

    monkeypatch.setattr("mesofield.gui.operator_ui.force_foreground", lambda w: None)
    return QtOperatorUI()


def test_confirm_from_worker_thread_builds_the_dialog_on_the_gui_thread(
    qt_ui, qtbot, monkeypatch
):
    from PyQt6.QtWidgets import QApplication, QMessageBox

    seen: dict = {}

    def _exec(box):
        seen["box"] = box.thread()
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "exec", _exec)

    assert _in_worker(lambda: qt_ui.confirm("Start", "go?"), qtbot) is True
    assert seen["box"] is QApplication.instance().thread()


def test_confirm_cancel_returns_false(qt_ui, qtbot, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "exec", lambda self: QMessageBox.StandardButton.Cancel
    )
    assert _in_worker(lambda: qt_ui.confirm("Start", "go?"), qtbot) is False


def test_busy_handle_closes_from_a_worker_thread(qt_ui, qtbot):
    from PyQt6 import sip

    busy = _in_worker(lambda: qt_ui.busy("Launching", "waiting..."), qtbot)
    box = busy._box
    assert box.isVisible()

    _in_worker(busy.close, qtbot)
    assert busy._box is None
    # A button-less QMessageBox ignores close(); it must be hidden outright.
    assert sip.isdeleted(box) or not box.isVisible()
    busy.close()  # idempotent
