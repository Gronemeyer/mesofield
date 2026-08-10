"""Run a callable on the Qt GUI thread and return its result.

Qt widgets may only be created and driven from the GUI thread, so work running
off it cannot build a dialog or read the operator's answer directly -- a modal
opened on a worker thread never returns. The usual worker->GUI seam (a QObject
plus a ``pyqtSignal``) is fire-and-forget; this one blocks the caller and hands
back a value, which is what a gate needs.

Callers already on the GUI thread, or running headless, invoke the callable
directly.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional, Tuple

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal

__all__ = ["run_on_main_thread", "on_main_thread"]


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
