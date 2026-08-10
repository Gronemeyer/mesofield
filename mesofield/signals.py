"""Lightweight signaling hub used across mesofield.

Every :class:`~mesofield.protocols.HardwareDevice` exposes a
:class:`DeviceSignals` instance on ``self.signals`` so the rest of the
system (e.g. :class:`~mesofield.data.manager.DataManager`,
:class:`~mesofield.base.Procedure`) can connect uniformly without caring
which backend the device uses.

Four signals form the standard contract:

``started()``
    Emitted once the device is actively acquiring / running.
``finished()``
    Emitted when the device has stopped on its own (e.g. an MDA sequence
    completed) *or* in response to ``stop()``.  The ``primary`` device's
    ``finished`` is what triggers :meth:`Procedure.cleanup`.
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
:func:`qt_relay` to forward a :class:`psygnal.Signal` into a ``pyqtSignal``,
and should connect through :class:`Bindings` so the subscription is severed
before the widget is destroyed.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from psygnal import Signal

__all__ = ["Signal", "DeviceSignals", "Bindings", "qt_relay"]


class Bindings:
    """Records signal->slot connections so an owner can sever them all at once.

    Devices and the procedure outlive the widgets that subscribe to them, so
    every subscriber has to disconnect before it is destroyed or it goes on
    firing into a deleted object.  Connect through here and call :meth:`close`
    in the owner's teardown; a missed connection is then a missing line rather
    than a forgotten disconnect.  Works with psygnal and pyqtSignal alike.
    """

    def __init__(self) -> None:
        self._bound: list[tuple[Any, Callable]] = []

    def connect(self, signal: Any, slot: Callable, **kwargs: Any) -> Callable:
        signal.connect(slot, **kwargs)
        self._bound.append((signal, slot))
        return slot

    def close(self) -> None:
        """Disconnect everything, once. Must run before the owner is destroyed."""
        bound, self._bound = self._bound, []
        for signal, slot in reversed(bound):
            signal.disconnect(slot)


class DeviceSignals:
    """Standard bundle of signals carried by every hardware device.

    Defined as instance attributes (not class attributes) so each device
    owns its own emitters.  ``psygnal.Signal`` instances are descriptors
    when declared on a class; we instantiate them directly here so they
    behave as plain emitters on the bundle.
    """

    __slots__ = ("started", "finished", "data", "frame")

    def __init__(self) -> None:
        from psygnal import SignalInstance

        # Construct lightweight SignalInstance objects directly so the
        # bundle is independent of any owning class.
        self.started: SignalInstance = SignalInstance(())
        self.finished: SignalInstance = SignalInstance(())
        self.data: SignalInstance = SignalInstance((object, object))
        self.frame: SignalInstance = SignalInstance((object, object, object))

    def disconnect_all(self) -> None:
        """Drop every subscriber, e.g. when the owning device shuts down."""
        for sig in (self.started, self.finished, self.data, self.frame):
            sig.disconnect()


def qt_relay(qt_signal: Any) -> Callable[..., None]:
    """Return a slot that forwards a ``psygnal`` emission to a ``pyqtSignal``.

    Use from GUI code only, and connect it through :class:`Bindings` so the
    device stops relaying into the widget before that widget is destroyed.
    Both signals must accept the same argument arity; the relay is one-way.
    """

    def _relay(*args: Any) -> None:
        qt_signal.emit(*args)

    return _relay
