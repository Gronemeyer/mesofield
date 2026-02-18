from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from mesofield.io.events import DataEvent
from mesofield.utils._logger import get_logger


class BaseDataProducer:
    """Minimal base class for mesofield data-producing devices."""

    device_id: str
    device_type: str
    file_type: str = "dat"
    bids_type: Optional[str] = None
    output_path: Optional[str] = None
    metadata_path: Optional[str] = None
    path_args: dict[str, Any]
    data_event: DataEvent
    is_active: bool
    _started: Optional[datetime]
    _stopped: Optional[datetime]
    _recording: bool

    def __init__(
        self,
        *,
        device_id: Optional[str] = None,
        device_type: Optional[str] = None,
        file_type: Optional[str] = None,
        bids_type: Optional[str] = None,
        logger_name: Optional[str] = None,
    ) -> None:
        self.output_path = None
        self.metadata_path = None
        self.path_args = {}
        self.is_active = False
        self._started = None
        self._stopped = None
        self._recording = False
        if device_id is not None:
            self.device_id = device_id
        if device_type is not None:
            self.device_type = device_type
        if file_type is not None:
            self.file_type = file_type
        if bids_type is not None:
            self.bids_type = bids_type
        self.logger = get_logger(logger_name or f"{__name__}.{self.__class__.__name__}")
        self.data_event = DataEvent(on_error=lambda exc: self._log_exception("data_event", exc))

    def _log_exception(self, action: str, exc: Exception) -> None:
        self.logger.exception(
            "%s failed for %s (%s): %s",
            action,
            getattr(self, "device_id", "unknown"),
            getattr(self, "device_type", "unknown"),
            exc,
        )

    def _guard(self, action: str, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            self._log_exception(action, exc)
            return None

    def emit_data(self, payload: Any, device_ts: Any = None) -> None:
        self._guard("emit_data", self.data_event.emit, payload, device_ts)

    def mark_started(self) -> None:
        self._started = datetime.now()
        self.is_active = True

    def mark_stopped(self) -> None:
        self._stopped = datetime.now()
        self.is_active = False

    def set_output_path(self, path: Optional[str]) -> None:
        self.output_path = path

    def set_metadata_path(self, path: Optional[str]) -> None:
        self.metadata_path = path

    def set_path_args(
        self,
        *,
        suffix: Optional[str] = None,
        extension: Optional[str] = None,
        bids_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """Standardize device output path arguments."""
        if suffix is not None:
            self.path_args["suffix"] = suffix
        if extension is not None:
            self.path_args["extension"] = extension
            self.file_type = extension
        if bids_type is not None:
            self.path_args["bids_type"] = bids_type
            self.bids_type = bids_type
        return dict(self.path_args)

    def initialize(self) -> None:
        """Initialize hardware resources."""
        self._guard("initialize", self._initialize)

    def start(self) -> Any:
        """Start device acquisition or operation."""
        return self._guard("start", self._start)

    def stop(self) -> Any:
        """Stop device acquisition or operation."""
        return self._guard("stop", self._stop)

    def shutdown(self) -> None:
        """Close hardware resources."""
        self._guard("shutdown", self._shutdown)

    def save_data(self, path: Optional[str] = None) -> None:
        """Save recorded data (override in subclasses)."""
        if path is not None:
            self.output_path = path
        self._guard("save_data", self._save_data)

    def get_data(self) -> Any:
        """Return buffered data (override in subclasses)."""
        return self._guard("get_data", self._get_data)

    def status(self) -> dict[str, Any]:
        """Return device status."""
        return {
            "device_id": getattr(self, "device_id", ""),
            "device_type": getattr(self, "device_type", ""),
            "active": self.is_active,
            "recording": self._recording,
            "started": self._started,
            "stopped": self._stopped,
            "file_type": self.file_type,
            "bids_type": self.bids_type,
            "output_path": self.output_path,
            "metadata_path": self.metadata_path,
        }

    def get_status(self) -> dict[str, Any]:
        """Legacy alias for status()."""
        return self.status()

    def _initialize(self) -> None:
        return None

    def start_recording(self, path: Optional[str] = None) -> None:
        """Standard recording entry point for data-producing devices."""
        if path is not None:
            self.output_path = path
        self._recording = True
        self.mark_started()
        self.start()
        self._guard("start_recording", self._start_recording)

    def stop_recording(self) -> None:
        """Standard recording exit point for data-producing devices."""
        self._recording = False
        self.mark_stopped()
        self._guard("stop_recording", self._stop_recording)

    def _start(self) -> Any:
        raise NotImplementedError

    def _stop(self) -> Any:
        raise NotImplementedError

    def _shutdown(self) -> None:
        return None

    def _save_data(self) -> None:
        return None

    def _get_data(self) -> Any:
        return None

    def _start_recording(self) -> None:
        return None

    def _stop_recording(self) -> None:
        return None
