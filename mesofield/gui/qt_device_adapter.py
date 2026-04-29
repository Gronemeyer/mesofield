"""Qt adapter that bridges pure-Python device signals into ``pyqtSignal``s.

GUI code (e.g. live plotting) needs Qt signals so that emissions are
delivered on the main thread.  Devices built on
:class:`mesofield.devices.base.BaseDataProducer` use ``psygnal`` and
remain Qt-free.  This module is the seam:  attach a
:class:`QtDeviceAdapter` to a device and read its ``pyqtSignal``s
(``serialDataReceived``, ``serialSpeedUpdated``) instead of the device
attributes directly.

Used internally by ``EncoderDevice`` / ``TreadmillDevice`` so that legacy
GUI widgets (``SerialWidget``) keep working unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import QObject, pyqtSignal


class QtDeviceAdapter(QObject):
    """Bridges ``device.signals.data`` into Qt-friendly emissions.

    Subscribes to the device's ``signals.data`` and re-emits:

    - ``serialDataReceived(object)``  — the raw payload.
    - ``serialSpeedUpdated(float, float)`` — ``(time_s, speed)`` whenever
      the payload is a ``dict`` carrying a ``"speed"`` key.  ``time_s``
      defaults to the device timestamp; if absent, the queue timestamp.
    """

    serialDataReceived = pyqtSignal(object)
    serialSpeedUpdated = pyqtSignal(float, float)

    def __init__(self, device: Any) -> None:
        super().__init__()
        self._device = device
        signals = getattr(device, "signals", None)
        data_sig = getattr(signals, "data", None) if signals is not None else None
        if data_sig is not None and hasattr(data_sig, "connect"):
            try:
                data_sig.connect(self._on_data)
            except Exception:
                pass

    def _on_data(self, payload: Any, ts: Any = None) -> None:
        try:
            self.serialDataReceived.emit(payload)
        except Exception:
            pass

        # Pull a scalar "value" out of the payload for the live trace.
        value: Optional[float] = None
        t: Any = None
        if isinstance(payload, dict):
            if "speed" in payload:
                value = payload["speed"]
                t = payload.get("device_us")
        elif isinstance(payload, (int, float)):
            value = payload

        if value is None:
            return
        if t is None:
            t = ts if ts is not None else 0.0
        try:
            self.serialSpeedUpdated.emit(float(t), float(value))
        except (TypeError, ValueError):
            pass
