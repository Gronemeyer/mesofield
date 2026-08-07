"""GUI status panel for the MousePortal stimulus device.

Surfaces in the ConfigController (via :class:`DynamicController`) so the user
can see at a glance that MousePortal is loaded, what gain experiment is queued,
and whether the subprocess launched and reached readiness -- instead of the
launch failing silently in the logs.

The panel reflects :class:`~mesofield.devices.stimulus_base.SubprocessStimulusDevice`
lifecycle status (``loaded → launching → ready / running / failed → stopped``)
via the device's ``status_changed`` psygnal, bridged onto a ``pyqtSignal`` so
updates always land on the GUI thread.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QGroupBox, QGridLayout, QLabel

from mesofield.signals import qt_bridge


# status key -> (dot color, human label)
_STATUS = {
    "loaded":    ("#6e7681", "Loaded — ready to record"),
    "launching": ("#d29922", "Launching…"),
    "running":   ("#d29922", "Running (no readiness handshake)"),
    "ready":     ("#3fb950", "Ready"),
    "failed":    ("#f85149", "Failed — check logs"),
    "stopped":   ("#6e7681", "Stopped"),
}


class MousePortalPanel(QGroupBox):
    """Compact load/readiness indicator + experiment summary for MousePortal."""

    # Bridge target so cross-thread status emits marshal onto the GUI thread.
    _status_relay = pyqtSignal(str)

    def __init__(self, config: Any, device: Any, parent=None) -> None:
        super().__init__("MousePortal", parent)
        self._config = config
        self._device = device

        layout = QGridLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setHorizontalSpacing(8)

        self._dot = QLabel()
        self._dot.setFixedSize(14, 14)
        layout.addWidget(self._dot, 0, 0)

        self._status_label = QLabel()
        layout.addWidget(self._status_label, 0, 1)

        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)
        self._summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        try:
            from mesofield.gui import theme
            self._summary_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        except Exception:
            pass
        layout.addWidget(self._summary_label, 1, 0, 1, 2)

        self.refresh_summary()
        self._apply_status(getattr(device, "gui_status", "loaded"))

        # Device status -> GUI. psygnal (may fire from the subprocess reader
        # thread) -> pyqtSignal -> slot (queued onto the GUI thread).
        self._status_relay.connect(self._apply_status)
        status_sig = getattr(device, "status_changed", None)
        if status_sig is not None:
            qt_bridge(status_sig, self._status_relay)

    # -- rendering ------------------------------------------------------
    def _apply_status(self, status: str) -> None:
        color, label = _STATUS.get(status, ("#6e7681", status))
        self._dot.setStyleSheet(
            f"background-color: {color}; border-radius: 7px;"
        )
        self._status_label.setText(f"<b>{label}</b>")
        self.setToolTip(f"MousePortal status: {status}")

    def refresh_summary(self) -> None:
        """Re-render the experiment summary from the current config block."""
        from mesofield.gui.mouseportal_config import summarize_experiment
        getter = getattr(self._config, "get", None)
        block = getter("mouseportal") if callable(getter) else None
        self._summary_label.setText(summarize_experiment(block if isinstance(block, dict) else {}))
