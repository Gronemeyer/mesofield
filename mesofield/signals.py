"""Lightweight signaling hub used across mesofield.

Every :class:`~mesofield.protocols.HardwareDevice` exposes a
:class:`DeviceSignals` instance on ``self.signals`` so the rest of the
system (e.g. :class:`~mesofield.data.manager.DataManager`,
:class:`~mesofield.base.Procedure`) can connect uniformly without caring
which backend the device uses.

Five signals form the standard contract:

``started()``
    Emitted once the device is actively acquiring / running.
``finished()``
    Emitted exactly once when the device reaches a terminal state — on
    its own (e.g. an MDA sequence completed), in response to ``stop()``,
    or after an unrecoverable error.  The ``primary`` device's
    ``finished`` is what triggers :meth:`Procedure._cleanup_procedure`.
    Devices built on :class:`mesofield.devices.base.BaseDevice` route
    every exit path through ``BaseDevice._finalize`` which guarantees
    the exactly-once emission.
``error(exc)``
    Emitted (before ``finished``) when the device reaches its terminal
    state because of an unrecoverable failure.  ``exc`` is the exception
    instance.  The :class:`~mesofield.base.Procedure` subscribes on every
    device so mid-run failures of *non-primary* producers are surfaced
    instead of silently yielding empty data files.
``data(payload, device_ts)``
    Emitted for every datum that should land on
    :class:`~mesofield.data.manager.DataQueue`.  ``payload`` is the raw
    sample (frame index, encoder click count, NIDAQ count, ...) and
    ``device_ts`` is the device-side timestamp (float seconds, optional).
``frame(img, idx, device_ts)``
    Optional.  Emitted by camera-like producers carrying the raw frame
    array in addition to the lightweight ``data`` emission.  Subscribers
    use this for real-time processing (see ``mesofield.processors``).
    Producers without per-sample raw payloads never emit on this signal.

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

    __slots__ = ("started", "finished", "error", "data", "frame")

    def __init__(self) -> None:
        from psygnal import SignalInstance

        # Construct lightweight SignalInstance objects directly so the
        # bundle is independent of any owning class.
        self.started: SignalInstance = SignalInstance(())
        self.finished: SignalInstance = SignalInstance(())
        self.error: SignalInstance = SignalInstance((object,))
        self.data: SignalInstance = SignalInstance((object, object))
        self.frame: SignalInstance = SignalInstance((object, object, object))

    def disconnect_all(self) -> None:
        for sig in (self.started, self.finished, self.error, self.data, self.frame):
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
