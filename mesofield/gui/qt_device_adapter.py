"""Qt adapter that bridges pure-Python device signals into ``pyqtSignal``s.

GUI code (e.g. live plotting, image previews) needs Qt signals so that
emissions are delivered on the main thread with QueuedConnection
semantics.  Devices built on
:class:`mesofield.devices.base.BaseDataProducer` use ``psygnal`` and
remain Qt-free.  This module is the seam: attach an adapter to a
device, expose the adapter's ``pyqtSignal`` as an attribute on the
device, and the GUI reads that attribute.

Two adapters live here:

- :class:`QtDeviceAdapter` — for serial-style devices. Bridges
  ``signals.data`` into ``serialDataReceived`` / ``serialSpeedUpdated``.
- :class:`QtImageAdapter` — for camera-shaped devices. Provides an
  ``image_ready(np.ndarray)`` pyqtSignal that the MDA viewer subscribes
  to. The device pushes frames into the adapter via
  ``adapter.emit_frame(frame)``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
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
        # First live-trace timestamp seen; used to rebase x to ~0 (see _on_data).
        self._t0: Optional[float] = None
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
        # Rebase the live-trace x to the first sample so it advances from ~0
        # regardless of the device's absolute time basis (wall-clock seconds for
        # serial / mock wheels, device_us for treadmills).
        try:
            t = float(t)
        except (TypeError, ValueError):
            return
        if self._t0 is None:
            self._t0 = t
        try:
            self.serialSpeedUpdated.emit(t - self._t0, float(value))
        except (TypeError, ValueError):
            pass


class QtImageAdapter(QObject):
    """Bridges per-frame ndarray emissions into a Qt ``image_ready`` signal.

    The MDA gui's static-viewer branch subscribes via
    ``preview = ImagePreview(image_payload=cam.image_ready, ...)``, which
    calls ``image_payload.connect(cb, type=QueuedConnection)`` -- so
    ``image_ready`` MUST be a real ``pyqtSignal``. Devices built on
    :class:`BaseDataProducer` are non-Qt; they hold an instance of this
    adapter and expose its ``image_ready`` attribute as their own.

    Usage::

        class MyCam(BaseDataProducer):
            def __init__(self, cfg=None, **kwargs):
                super().__init__(cfg, **kwargs)
                self._qt_image_adapter = QtImageAdapter()
                self.image_ready = self._qt_image_adapter.image_ready

            def _run_loop(self):
                ...
                self._qt_image_adapter.emit_frame(frame)
    """

    image_ready = pyqtSignal(np.ndarray)

    def emit_frame(self, frame: np.ndarray) -> None:
        try:
            self.image_ready.emit(frame)
        except Exception:
            pass
