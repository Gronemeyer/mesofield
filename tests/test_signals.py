"""Unit tests for mesofield.signals.

Validates that DeviceSignals provides the documented contract
(started / finished / data), that Bindings severs exactly what it connected,
and that qt_relay forwards psygnal -> Qt pyqtSignal emissions.
"""

from __future__ import annotations

import pytest

from mesofield.signals import Bindings, DeviceSignals, qt_relay


def test_device_signals_lifecycle_payload() -> None:
    sigs = DeviceSignals()

    started_calls: list = []
    finished_calls: list = []
    data_calls: list = []

    sigs.started.connect(lambda: started_calls.append(True))
    sigs.finished.connect(lambda: finished_calls.append(True))
    sigs.data.connect(lambda payload, ts: data_calls.append((payload, ts)))

    sigs.started.emit()
    sigs.data.emit({"idx": 0}, 123.456)
    sigs.finished.emit()

    assert started_calls == [True]
    assert finished_calls == [True]
    assert data_calls == [({"idx": 0}, 123.456)]


def test_device_signals_disconnect() -> None:
    sigs = DeviceSignals()
    seen: list = []

    def cb(payload, ts):
        seen.append(payload)

    sigs.data.connect(cb)
    sigs.data.emit("a", 0.0)
    sigs.data.disconnect(cb)
    sigs.data.emit("b", 0.0)

    assert seen == ["a"]


def test_qt_relay_forwards_to_a_pyqt_signal() -> None:
    from PyQt6.QtCore import QObject, pyqtSignal

    class Holder(QObject):
        forwarded = pyqtSignal(object, object)

    sigs = DeviceSignals()
    holder = Holder()
    received: list = []
    holder.forwarded.connect(lambda p, t: received.append((p, t)))

    binds = Bindings()
    binds.connect(sigs.data, qt_relay(holder.forwarded))
    sigs.data.emit({"frame": 1}, 9.99)
    assert received == [({"frame": 1}, 9.99)]

    binds.close()
    sigs.data.emit({"frame": 2}, 0.0)
    assert len(received) == 1


def test_bindings_close_is_idempotent_and_ordered() -> None:
    sigs = DeviceSignals()
    seen: list = []
    binds = Bindings()
    binds.connect(sigs.started, lambda: seen.append("a"))
    binds.connect(sigs.started, lambda: seen.append("b"))

    sigs.started.emit()
    assert seen == ["a", "b"]

    binds.close()
    binds.close()  # nothing left to disconnect
    sigs.started.emit()
    assert seen == ["a", "b"]


def test_mouseportal_panel_cleanup_severs_device_bridge(qtbot) -> None:
    """A cleaned-up panel stops reacting to the (longer-lived) device."""
    pytest.importorskip("PyQt6")
    from psygnal import SignalInstance

    from mesofield.gui.mouseportal_panel import MousePortalPanel

    class _Device:
        gui_status = "loaded"

        def __init__(self):
            self.status_changed = SignalInstance((str,))

    device = _Device()
    panel = MousePortalPanel({}, device)
    qtbot.addWidget(panel)

    device.status_changed.emit("ready")
    assert panel.toolTip() == "MousePortal status: ready"

    panel.cleanup()
    panel.cleanup()  # idempotent
    device.status_changed.emit("failed")
    assert panel.toolTip() == "MousePortal status: ready"
