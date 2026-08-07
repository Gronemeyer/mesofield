"""Compact status panel for an NI-DAQ trigger + TTL counter device.

- an indicator **light** that goes green once the start-trigger pulse has been
  sent (the device's ``started`` signal), and
- a running **count** of the TTL rising edges received back from the triggered
  system (one per ``signals.data`` emission), so the operator can confirm the
  external system is actually pulsing.

The device stays pure-Python (psygnal); ``signals.data`` fires from the
device's polling thread, so a small :class:`QObject` bridge re-emits through
``pyqtSignal``s, which Qt auto-queues onto the GUI thread.
"""
from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import Qt, QObject, QTimer, pyqtSignal
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel

from mesofield.gui import theme


class _NidaqBridge(QObject):
    """Marshal the device's psygnal emissions onto the GUI thread."""

    triggerSent = pyqtSignal()
    pulse = pyqtSignal()
    stopped = pyqtSignal()

    _SLOTS = ("started", "data", "finished")

    def __init__(self, device: Any) -> None:
        super().__init__()
        self._device = device
        self._handlers = {
            "started": self._on_started,
            "data": self._on_data,
            "finished": self._on_finished,
        }
        signals = getattr(device, "signals", None)
        for name in self._SLOTS:
            sig = getattr(signals, name, None) if signals is not None else None
            if sig is not None and hasattr(sig, "connect"):
                try:
                    sig.connect(self._handlers[name])
                except Exception:
                    pass

    def _on_started(self, *_a) -> None:
        self.triggerSent.emit()

    def _on_data(self, *_a) -> None:
        self.pulse.emit()

    def _on_finished(self, *_a) -> None:
        self.stopped.emit()

    def cleanup(self) -> None:
        """Sever the psygnal connections so a torn-down indicator stops firing."""
        signals = getattr(self._device, "signals", None)
        for name in self._SLOTS:
            sig = getattr(signals, name, None) if signals is not None else None
            if sig is not None and hasattr(sig, "disconnect"):
                try:
                    sig.disconnect(self._handlers[name])
                except Exception:
                    pass


class NidaqIndicator(QWidget):
    """Trigger-sent light + TTL edge counter for an NI-DAQ device."""

    _LIGHT_SIZE = 16
    _FLASH_MS = 120

    def __init__(self, device: Any, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._device = device
        self._count = 0
        self._armed = False

        self._bridge = _NidaqBridge(device)
        self._bridge.triggerSent.connect(self._on_trigger_sent)
        self._bridge.pulse.connect(self._on_pulse)
        self._bridge.stopped.connect(self._on_stopped)

        # One-shot timer flashes the light brighter on each incoming edge.
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._end_flash)

        self._build_ui()
        self._set_light(theme.TEXT_DIM)

    # -- UI -----------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 2, 4, 2)
        outer.setSpacing(2)

        row = QHBoxLayout()
        row.setSpacing(8)

        self._light = QLabel()
        self._light.setFixedSize(self._LIGHT_SIZE, self._LIGHT_SIZE)
        row.addWidget(self._light)

        self._name_label = QLabel(self._device_title())
        self._name_label.setToolTip(
            "NI-DAQ start-trigger output + TTL rising-edge counter"
        )
        row.addWidget(self._name_label)
        row.addStretch(1)

        self._count_label = QLabel("0")
        self._count_label.setStyleSheet(
            f"color: {theme.ACCENT}; font-family: {theme.MONO_FONT}; font-weight: bold;"
        )
        self._count_label.setToolTip("TTL rising edges counted since start")
        row.addWidget(self._count_label)
        outer.addLayout(row)

        self._status_label = QLabel("Idle — waiting for start trigger")
        self._status_label.setStyleSheet(
            f"color: {theme.TEXT_DIM}; font-size: 11px;"
        )
        outer.addWidget(self._status_label)

    def _device_title(self) -> str:
        dev_id = getattr(self._device, "device_id", "nidaq")
        device_name = getattr(self._device, "device_name", None)
        ctr = getattr(self._device, "ctr", None)
        if device_name and ctr:
            return f"{dev_id}  ({device_name}/{ctr})"
        return str(dev_id)

    def _set_light(self, color: str) -> None:
        self._light.setStyleSheet(
            f"background-color: {color}; border-radius: {self._LIGHT_SIZE // 2}px;"
            f" border: 1px solid {theme.BORDER};"
        )

    # -- slots (GUI thread) -------------------------------------------------
    def _on_trigger_sent(self) -> None:
        self._armed = True
        self._count = 0
        self._count_label.setText("0")
        self._set_light(theme.ACCENT)
        self._status_label.setText("Trigger sent — counting TTL edges")

    def _on_pulse(self) -> None:
        self._count += 1
        self._count_label.setText(str(self._count))
        # Flash bright phosphor briefly to register the edge visually.
        self._set_light(theme.ACCENT_HI)
        self._flash_timer.start(self._FLASH_MS)

    def _end_flash(self) -> None:
        self._set_light(theme.ACCENT if self._armed else theme.TEXT_DIM)

    def _on_stopped(self) -> None:
        self._armed = False
        self._set_light(theme.TEXT_DIM)
        self._status_label.setText(
            f"Stopped — {self._count} TTL edge(s) received"
        )

    # -- teardown -----------------------------------------------------------
    def cleanup(self) -> None:
        """Disconnect the device bridge before this widget is destroyed."""
        try:
            self._flash_timer.stop()
        except Exception:
            pass
        try:
            self._bridge.cleanup()
        except Exception:
            pass
