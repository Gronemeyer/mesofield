"""Surface initialization failures in the GUI instead of only in the log.

Hardware bring-up is deliberately forgiving: :class:`~mesofield.hardware.HardwareManager`
skips a device it cannot construct so the rest of the rig still comes up. That
is the right runtime behaviour and the wrong reporting behaviour -- a camera
that fails to open disappears from the acquisition UI with nothing but a log
line to explain it. These helpers put the same information in front of the
operator.

Import lazily from non-GUI code: this module needs a ``QApplication``.
"""

from __future__ import annotations

import traceback
from typing import Iterable, Optional

from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget


def _detail(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def show_error(parent: Optional[QWidget], title: str, exc: BaseException) -> None:
    """Report *exc* as a critical dialog, with its traceback behind Details."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle(title)
    box.setText(f"{type(exc).__name__}: {exc}")
    box.setDetailedText(_detail(exc))
    box.exec()


def show_startup_error(title: str, exc: BaseException) -> None:
    """Report a failure that happened before any window exists.

    Creates a throwaway ``QApplication`` when needed so ``mesofield launch``
    fails with a dialog rather than a bare traceback in a terminal the user may
    not even be looking at.
    """
    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication([])
    try:
        show_error(None, title, exc)
    finally:
        if owns_app:
            app.quit()


def show_init_warnings(parent: Optional[QWidget], messages: Iterable[str]) -> None:
    """Report devices that were skipped during hardware initialization."""
    items = [m for m in messages if m]
    if not items:
        return
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Hardware initialization")
    box.setText(
        f"{len(items)} device(s) were skipped while bringing up the rig.\n"
        "The rest of the rig started normally."
    )
    box.setInformativeText("\n".join(f"• {m}" for m in items))
    box.exec()
