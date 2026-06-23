"""Qt adapter that bridges pure-Python device signals into ``pyqtSignal``s.

GUI code (e.g. live plotting, image previews) needs Qt signals so that
emissions are delivered on the main thread with QueuedConnection
semantics.  Devices built on
:class:`mesofield.devices.base.BaseDataProducer` use ``psygnal`` and
remain Qt-free.  This module is the seam: attach an adapter to a
device, expose the adapter's ``pyqtSignal`` as an attribute on the
device, and the GUI reads that attribute.

Three adapters live here:

- :class:`QtDeviceAdapter` — for single-channel serial-style devices.
  Bridges ``signals.data`` into ``serialDataReceived`` / ``serialSpeedUpdated``.
- :func:`build_channel_adapter` — for multi-channel devices (e.g. a lick
  detector plotting both lick events and capacitance). Bridges
  ``signals.data`` into one ``{channel}Updated`` pyqtSignal per channel,
  each fed by a payload extractor.
- :class:`QtImageAdapter` — for camera-shaped devices. Provides an
  ``image_ready(np.ndarray)`` pyqtSignal that the MDA viewer subscribes
  to. The device pushes frames into the adapter via
  ``adapter.emit_frame(frame)``.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Callable, Dict, Mapping, Optional, Union

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


# Per-channel source: a payload-dict key (str) or a callable(payload)->value.
ChannelSource = Union[str, Callable[[Any], Any]]


class DeviceChannelSampler:
    """Pull-based bridge from ``device.signals.data`` to live plots.

    A fast serial device (e.g. ~1 kHz licks) emitting one Qt ``pyqtSignal``
    per sample floods the GUI thread's event queue: the queue grows faster
    than it drains regardless of how cheap the slot is, and the window stalls.

    This sampler avoids Qt entirely on the hot path. It subscribes to the
    device's ``signals.data`` (a psygnal, invoked on the *device* thread) and
    only appends each channel's scalar to a bounded per-channel ring buffer --
    no Qt signal, no cross-thread event. The GUI side *pulls* a snapshot on its
    own redraw timer via :meth:`provider`, so the GUI touches the data at a
    fixed ~30 Hz no matter how fast the device streams.

    ``channel_sources`` maps each channel name to the payload-dict key (or a
    ``callable(payload) -> value``) yielding that channel's scalar. ``t`` is
    rebased to the first sample so traces start at ~0 regardless of the
    device's absolute clock. ``max_points`` caps each buffer (and thus memory).
    """

    def __init__(
        self,
        device: Any,
        channel_sources: Mapping[str, ChannelSource],
        max_points: int = 2000,
    ) -> None:
        self._sources: Dict[str, ChannelSource] = dict(channel_sources)
        self._lock = threading.Lock()
        self._t0: Optional[float] = None
        self._buffers: Dict[str, tuple] = {
            ch: (deque(maxlen=max_points), deque(maxlen=max_points))
            for ch in self._sources
        }
        # Monotonic count of samples appended per channel. A saturated ring
        # buffer has constant length, so length can't tell the GUI whether new
        # data arrived -- this counter can.
        self._counts: Dict[str, int] = {ch: 0 for ch in self._sources}
        signals = getattr(device, "signals", None)
        self._data_sig = getattr(signals, "data", None) if signals is not None else None
        if self._data_sig is not None and hasattr(self._data_sig, "connect"):
            try:
                self._data_sig.connect(self._on_data)
            except Exception:
                pass

    def _on_data(self, payload: Any, ts: Any = None) -> None:
        """Runs on the device thread -- append only, never touch Qt."""
        if not isinstance(payload, dict):
            return
        try:
            t = float(ts) if ts is not None else 0.0
        except (TypeError, ValueError):
            t = 0.0
        with self._lock:
            if self._t0 is None:
                self._t0 = t
            t -= self._t0
            for ch, source in self._sources.items():
                try:
                    value = source(payload) if callable(source) else payload.get(source)
                except Exception:
                    value = None
                if value is None:
                    continue
                try:
                    fval = float(value)
                except (TypeError, ValueError):
                    continue
                xs, ys = self._buffers[ch]
                xs.append(t)
                ys.append(fval)
                self._counts[ch] += 1

    def snapshot(self, channel: str) -> tuple[list, list, int]:
        """Thread-safe copy of one channel's ``(times, values, count)``.

        ``count`` is the total number of samples ever appended to this channel
        (monotonic), so the GUI can detect new data even once the ring buffer
        is saturated and its length stops changing.
        """
        with self._lock:
            xs, ys = self._buffers[channel]
            return list(xs), list(ys), self._counts[channel]

    def provider(self, channel: str) -> Callable[[], tuple]:
        """Return a zero-arg callable the GUI timer pulls for ``channel``."""
        return lambda: self.snapshot(channel)

    def disconnect(self) -> None:
        if self._data_sig is not None:
            try:
                self._data_sig.disconnect(self._on_data)
            except Exception:
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
