"""Operator port: how a run asks the person at the rig something.

Core and device code calls :func:`ui`; the front-end installs an
implementation with :func:`set_ui` at startup. The default answers headlessly,
so scripts and tests run without a UI toolkit.
"""

from __future__ import annotations

from typing import Protocol

from mesofield.utils._logger import get_logger

__all__ = ["OperatorUI", "Busy", "HeadlessUI", "ui", "set_ui"]

logger = get_logger("mesofield.ui")


class Busy(Protocol):
    """Handle to a dismissable progress indicator."""

    def close(self) -> None: ...


class OperatorUI(Protocol):
    def confirm(self, title: str, text: str) -> bool:
        """Ask the operator to proceed; False cancels."""

    def alert(self, title: str, text: str, detail: str = "") -> None:
        """Report a failure."""

    def busy(self, title: str, text: str) -> Busy:
        """Show a non-blocking indicator until the handle is closed."""


class _NullBusy:
    def close(self) -> None:
        pass


class HeadlessUI:
    """No operator present: gates pass, messages go to the log."""

    def confirm(self, title: str, text: str) -> bool:
        return True

    def alert(self, title: str, text: str, detail: str = "") -> None:
        logger.error(f"{title}: {text}" + (f"\n{detail}" if detail else ""))

    def busy(self, title: str, text: str) -> Busy:
        logger.info(text)
        return _NullBusy()


_ui: OperatorUI = HeadlessUI()


def ui() -> OperatorUI:
    return _ui


def set_ui(impl: OperatorUI | None) -> None:
    """Install the front-end's implementation; None restores headless."""
    global _ui
    _ui = impl or HeadlessUI()
