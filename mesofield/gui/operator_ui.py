"""Qt implementation of the :mod:`mesofield.ui` operator port.

A run is driven from a worker thread, but Qt widgets may only be created and
driven from the GUI thread -- a modal opened on a worker thread never returns.
So each dialog here is built and executed through :func:`run_on_main_thread`,
which blocks the caller until the GUI thread has run it and hands back the
result (the usual QObject + signal seam is fire-and-forget, and a gate needs an
answer). Callers already on the GUI thread, or running headless, run inline.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional, Tuple

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QMessageBox


# --------------------------------------------------------------------------- #
# GUI-thread marshaling
# --------------------------------------------------------------------------- #
class _Call:
    """Mutable carrier for one cross-thread call and its outcome."""

    __slots__ = ("fn", "args", "kwargs", "result", "error")

    def __init__(self, fn: Callable[..., Any], args: Tuple, kwargs: dict) -> None:
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.result: Any = None
        self.error: Optional[BaseException] = None


class _Invoker(QObject):
    """Lives on the GUI thread; runs whatever a worker thread hands it."""

    invoke = pyqtSignal(object)

    def _run(self, call: _Call) -> None:
        try:
            call.result = call.fn(*call.args, **call.kwargs)
        except BaseException as exc:  # re-raised on the calling thread
            call.error = exc


# Keyed by app so a new QApplication never reuses an invoker bound to a dead one.
_INVOKER: Optional[Tuple[Any, _Invoker]] = None
_INVOKER_LOCK = threading.Lock()


def _qapp():
    """Return the live QApplication, or None when running headless."""
    try:
        from PyQt6.QtWidgets import QApplication

        return QApplication.instance()
    except Exception:
        return None


def on_main_thread() -> bool:
    """True when the caller is running on the Qt GUI thread.

    False headless, where there is no GUI thread to be on.
    """
    app = _qapp()
    if app is None:
        return False
    try:
        return QThread.currentThread() is app.thread()
    except RuntimeError:
        return False


def _invoker_for(app) -> _Invoker:
    """The GUI-thread invoker for *app*, built on first use from any thread."""
    global _INVOKER
    with _INVOKER_LOCK:
        if _INVOKER is not None and _INVOKER[0] is app:
            return _INVOKER[1]
        invoker = _Invoker()
        # Affinity first, connection second: connecting while the invoker still
        # belongs to the calling thread records a same-thread connection that
        # survives the move, and the later emit then blocks forever.
        invoker.moveToThread(app.thread())
        invoker.invoke.connect(
            invoker._run, Qt.ConnectionType.BlockingQueuedConnection
        )
        _INVOKER = (app, invoker)
        return invoker


def run_on_main_thread(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run *fn* on the Qt GUI thread and return its result, blocking the caller.

    Exceptions raised by *fn* propagate to the calling thread. Runs *fn* inline
    when headless or when the caller is already on the GUI thread.
    """
    app = _qapp()
    if app is None or on_main_thread():
        return fn(*args, **kwargs)

    call = _Call(fn, args, kwargs)
    _invoker_for(app).invoke.emit(call)
    if call.error is not None:
        raise call.error
    return call.result


# --------------------------------------------------------------------------- #
# Dialogs
# --------------------------------------------------------------------------- #
def force_foreground(widget) -> None:
    """Raise *widget* over a full-screen stimulus window with keyboard focus.

    Windows blocks a background process from calling ``SetForegroundWindow``,
    so nudge it via Win32; failures are cosmetic (the dialog still works, it
    just may not auto-focus) and leave the portable path below.
    """
    widget.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    widget.show()
    widget.raise_()
    widget.activateWindow()
    try:
        import ctypes

        hwnd = int(widget.winId())
        ctypes.windll.user32.BringWindowToTop(hwnd)
        ctypes.windll.user32.SetForegroundWindow(hwnd)
    except Exception:
        pass


class _Busy:
    def __init__(self, box) -> None:
        self._box = box

    def close(self) -> None:
        box, self._box = self._box, None
        if box is None:
            return

        def _dismiss() -> None:
            # hide(), not close(): a button-less QMessageBox has no escape
            # action, so it ignores a close event and stays on screen.
            box.hide()
            box.deleteLater()

        run_on_main_thread(_dismiss)


class QtOperatorUI:
    """Modal dialogs, forced to the foreground so a keypress lands on them."""

    def __init__(self, parent=None) -> None:
        self._parent = parent

    def confirm(self, title: str, text: str) -> bool:
        def _ask() -> bool:
            box = QMessageBox(self._parent)
            box.setWindowTitle(title)
            box.setText(text)
            box.setStandardButtons(
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
            )
            box.setDefaultButton(QMessageBox.StandardButton.Ok)
            box.setWindowModality(Qt.WindowModality.ApplicationModal)
            force_foreground(box)
            return box.exec() == QMessageBox.StandardButton.Ok

        return run_on_main_thread(_ask)

    def alert(self, title: str, text: str, detail: str = "") -> None:
        def _show() -> None:
            box = QMessageBox(self._parent)
            box.setIcon(QMessageBox.Icon.Critical)
            box.setWindowTitle(title)
            box.setText(text)
            if detail.strip():
                box.setDetailedText(detail.strip())
            box.exec()

        run_on_main_thread(_show)

    def busy(self, title: str, text: str) -> _Busy:
        def _show():
            box = QMessageBox(self._parent)
            box.setWindowTitle(title)
            box.setText(text)
            box.setStandardButtons(QMessageBox.StandardButton.NoButton)
            box.setWindowModality(Qt.WindowModality.ApplicationModal)
            box.show()
            box.raise_()
            return box

        return _Busy(run_on_main_thread(_show))
