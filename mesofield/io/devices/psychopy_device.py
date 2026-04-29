"""PsychoPy stimulus device.

Wraps :class:`mesofield.subprocesses.psychopy.PsychoPyProcess` behind the
standard :class:`~mesofield.protocols.StimulusDevice` interface so the
:class:`~mesofield.base.Procedure` can drive it through the same
``arm/start/stop/shutdown`` lifecycle as any other hardware device.

PsychoPy is *not* a :class:`~mesofield.protocols.DataProducer`: it never
emits on ``signals.data`` and does not implement ``save_data`` /
``get_data``.  Its ``signals.started`` fires when the subprocess prints
``PSYCHOPY_READY``; ``signals.finished`` fires when ``QProcess`` exits.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, Dict, Optional

from mesofield import DeviceRegistry
from mesofield.signals import DeviceSignals
from mesofield.utils._logger import get_logger


@DeviceRegistry.register("psychopy")
class PsychoPyDevice:
    """Stimulus device that launches a PsychoPy script as a subprocess."""

    device_type: ClassVar[str] = "stimulus"
    file_type: str = ""
    bids_type: Optional[str] = None

    def __init__(self, cfg: Dict[str, Any]):
        self.signals = DeviceSignals()
        self.cfg = dict(cfg)
        self.device_id: str = cfg.get("id", "psychopy")
        self.is_primary: bool = bool(cfg.get("primary", False))
        self.logger = get_logger(f"{__name__}.PsychoPyDevice[{self.device_id}]")
        self._process = None  # mesofield.subprocesses.psychopy.PsychoPyProcess
        self._config = None
        self._started: Optional[datetime] = None
        self._stopped: Optional[datetime] = None

    # -- lifecycle ------------------------------------------------------
    def initialize(self) -> bool:
        return True

    def arm(self, config) -> None:
        """Per-run prep: stash the experiment config so ``start`` can launch."""
        self._config = config

    def start(self) -> bool:
        from mesofield.subprocesses.psychopy import PsychoPyProcess

        if self._config is None:
            self.logger.error("PsychoPyDevice.start called before arm()")
            return False

        self._process = PsychoPyProcess(self._config)
        self._process.ready.connect(self._on_ready)
        self._process.finished.connect(self._on_finished)
        self._started = datetime.now()
        self._process.start()
        return True

    def stop(self) -> bool:
        self._stopped = datetime.now()
        if self._process is not None and self._process.process is not None:
            try:
                self._process.process.terminate()
                if not self._process.process.waitForFinished(2000):
                    self._process.process.kill()
            except Exception as exc:
                self.logger.warning(f"PsychoPy stop failed: {exc}")
        self.signals.finished.emit()
        return True

    def shutdown(self) -> None:
        self.stop()
        self._process = None

    # -- callbacks ------------------------------------------------------
    def _on_ready(self) -> None:
        self.signals.started.emit()

    def _on_finished(self, *_args) -> None:
        self.signals.finished.emit()

    # -- introspection --------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "running": self._process is not None,
            "started": self._started,
            "stopped": self._stopped,
        }

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
        }
