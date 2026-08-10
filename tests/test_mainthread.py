"""Marshaling worker-thread work onto the Qt GUI thread.

``run_on_main_thread`` is what keeps a run's operator dialogs on the GUI thread.
Covers worker -> GUI with a result, exception propagation back to the worker,
and the same-thread call that must not deadlock.
"""

from __future__ import annotations

import threading

import pytest

from mesofield.gui._mainthread import on_main_thread, run_on_main_thread

pytestmark = pytest.mark.gui


def _run_in_worker(fn, qtbot, timeout: int = 5000):
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
    # waitUntil spins the main event loop, which is what services the call.
    qtbot.waitUntil(lambda: box.get("done", False), timeout=timeout)
    thread.join(timeout=1)
    if "error" in box:
        raise box["error"]
    return box["result"]


def test_returns_result_from_worker_thread(qtbot):
    from PyQt6.QtCore import QThread
    from PyQt6.QtWidgets import QApplication

    seen: dict = {}

    def _on_gui():
        seen["thread"] = QThread.currentThread()
        return 42

    assert _run_in_worker(lambda: run_on_main_thread(_on_gui), qtbot) == 42
    assert seen["thread"] is QApplication.instance().thread()


def test_arguments_are_forwarded(qtbot):
    result = _run_in_worker(
        lambda: run_on_main_thread(lambda a, b=0: a + b, 1, b=2), qtbot
    )
    assert result == 3


def test_exception_propagates_to_the_calling_thread(qtbot):
    def _boom():
        raise ValueError("gate exploded")

    with pytest.raises(ValueError, match="gate exploded"):
        _run_in_worker(lambda: run_on_main_thread(_boom), qtbot)


def test_call_on_the_main_thread_does_not_deadlock(qapp):
    # A BlockingQueuedConnection to one's own thread deadlocks; the same-thread
    # path must run inline instead.
    assert on_main_thread() is True
    assert run_on_main_thread(lambda: "inline") == "inline"


def test_widget_built_through_helper_lives_on_the_gui_thread(qtbot):
    from PyQt6.QtWidgets import QApplication, QWidget

    def _build():
        return run_on_main_thread(QWidget)

    widget = _run_in_worker(_build, qtbot)
    qtbot.addWidget(widget)
    assert widget.thread() is QApplication.instance().thread()
