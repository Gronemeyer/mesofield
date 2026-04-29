"""Lightweight signaling hub used across mesofield.

Every :class:`~mesofield.protocols.HardwareDevice` exposes a
:class:`DeviceSignals` instance on ``self.signals`` so the rest of the
system (e.g. :class:`~mesofield.data.manager.DataManager`,
:class:`~mesofield.base.Procedure`) can connect uniformly without caring
which backend the device uses.

Three signals form the standard contract:

``started()``
    Emitted once the device is actively acquiring / running.
``finished()``
    Emitted when the device has stopped on its own (e.g. an MDA sequence
    completed) *or* in response to ``stop()``.  The ``primary`` device's
    ``finished`` is what triggers :meth:`Procedure._cleanup_procedure`.
``data(payload, device_ts)``
    Emitted for every datum that should land on
    :class:`~mesofield.data.manager.DataQueue`.  ``payload`` is the raw
    sample (frame index, encoder click count, NIDAQ count, ...) and
    ``device_ts`` is the device-side timestamp (float seconds, optional).

The implementation wraps :mod:`psygnal` so emission is Qt-free, weakly
referenced and thread-safe.  GUI code that needs a Qt slot can use
:func:`qt_bridge` to forward a :class:`psygnal.Signal` into a
``pyqtSignal``.
"""

from __future__ import annotations

from typing import Any, Optional

from psygnal import Signal

__all__ = ["Signal", "DeviceSignals", "qt_bridge"]


class DeviceSignals:
    """Standard bundle of signals carried by every hardware device.

    Defined as instance attributes (not class attributes) so each device
    owns its own emitters.  ``psygnal.Signal`` instances are descriptors
    when declared on a class; we instantiate them directly here so they
    behave as plain emitters on the bundle.
    """

    __slots__ = ("started", "finished", "data")

    def __init__(self) -> None:
        from psygnal import SignalInstance

        # Construct lightweight SignalInstance objects directly so the
        # bundle is independent of any owning class.
        self.started: SignalInstance = SignalInstance(())
        self.finished: SignalInstance = SignalInstance(())
        self.data: SignalInstance = SignalInstance((object, object))

    def disconnect_all(self) -> None:
        for sig in (self.started, self.finished, self.data):
            try:
                sig.disconnect()
            except Exception:
                pass


def qt_bridge(signal: Any, qt_signal: Any) -> None:
    """Forward emissions of a ``psygnal`` signal to a ``pyqtSignal``.

    Use from GUI code only.  Both signals must accept the same argument
    arity.  The connection is one-way: psygnal -> Qt.
    """

    def _relay(*args: Any) -> None:
        try:
            qt_signal.emit(*args)
        except Exception:
            pass

    signal.connect(_relay)
