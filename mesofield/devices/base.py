"""Concrete base class for non-Qt hardware devices.

Devices that are not bound to ``QThread``/``QObject`` should subclass
:class:`BaseDevice` to inherit a uniform shape:
``self.signals`` (:class:`~mesofield.signals.DeviceSignals`),
``_started``/``_stopped`` timestamps, and no-op defaults for ``arm``,
``status``, ``metadata``.

Qt-based devices (e.g. ``OpenCVCamera(QThread)`` / ``SerialWorker(QThread)``)
cannot inherit from this class due to metaclass conflicts.  They satisfy
the same contract via duck typing -- they instantiate
``self.signals = DeviceSignals()`` in ``__init__`` and implement the
lifecycle methods directly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from mesofield.signals import DeviceSignals


class BaseDevice:
    """Default lifecycle skeleton for non-Qt hardware devices."""

    device_type: str = "device"
    device_id: str = "device"

    def __init__(self) -> None:
        self.signals = DeviceSignals()
        self._started: datetime | None = None
        self._stopped: datetime | None = None

    # -- lifecycle (subclasses override as needed) -----------------------
    def initialize(self) -> bool:
        return True

    def arm(self, config: Any) -> None:  # noqa: D401 - default no-op
        """Default no-op per-run preparation."""
        return None

    def start(self) -> bool:
        self._started = datetime.now()
        self.signals.started.emit()
        return True

    def stop(self) -> bool:
        self._stopped = datetime.now()
        self.signals.finished.emit()
        return True

    def shutdown(self) -> None:
        return None

    # -- introspection ---------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "started": self._started,
            "stopped": self._stopped,
        }

    @property
    def metadata(self) -> Dict[str, Any]:
        return {"device_id": self.device_id, "device_type": self.device_type}
