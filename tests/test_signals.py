"""Unit tests for mesofield.signals.

Validates that DeviceSignals provides the documented contract
(started / finished / data) and that qt_bridge correctly forwards
psygnal -> Qt pyqtSignal emissions.
"""

from __future__ import annotations

import time

import pytest

from mesofield.signals import DeviceSignals, qt_bridge


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


def test_qt_bridge_propagates_to_pyqt_signal() -> None:
    pytest.importorskip("PyQt6")
    from PyQt6.QtCore import QObject, pyqtSignal

    class Holder(QObject):
        forwarded = pyqtSignal(object, object)

    sigs = DeviceSignals()
    holder = Holder()
    received: list = []
    holder.forwarded.connect(lambda p, t: received.append((p, t)))

    qt_bridge(sigs.data, holder.forwarded)

    sigs.data.emit({"frame": 1}, 9.99)

    # qt_bridge uses a direct connection; emit synchronously propagates.
    assert received == [({"frame": 1}, 9.99)]
