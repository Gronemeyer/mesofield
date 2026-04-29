"""Authorable base classes for non-Qt hardware devices.

Layered hierarchy that lets a user write a new device with as little
boilerplate as possible:

``BaseDevice``
    Lifecycle skeleton, ``self.signals`` (DeviceSignals), logger, cfg
    storage, ``is_primary`` / ``is_running`` tracking, default
    ``status``/``metadata``.

``BaseDataProducer(BaseDevice)``
    Adds the DataProducer surface: ``output_path``/``metadata_path``,
    a thread-safe in-memory buffer, a ``record(payload, ts=None)``
    helper that timestamps + buffers + emits ``signals.data`` in one
    call, and a default CSV ``save_data`` / ``get_data``.

``BaseSerialDevice(BaseDataProducer)``
    Specialization for line-oriented serial protocols (Teensy /
    Arduino).  Owns its own daemon read thread; subclasses only need
    to override ``parse_line(line)``.  A ``development_mode`` flag
    skips opening the port so authors can iterate without hardware
    connected.

Qt devices (``QThread`` subclasses) cannot inherit from these due to
metaclass conflicts; they continue to duck-type the contract by
instantiating ``self.signals = DeviceSignals()`` directly.
"""

from __future__ import annotations

import csv
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

from mesofield.signals import DeviceSignals
from mesofield.utils._logger import get_logger


__all__ = [
    "BaseDevice",
    "BaseDataProducer",
    "BaseSerialDevice",
]


# ---------------------------------------------------------------------------
# BaseDevice
# ---------------------------------------------------------------------------


class BaseDevice:
    """Default lifecycle skeleton for non-Qt hardware devices.

    Constructor accepts an optional ``cfg`` mapping (the YAML stanza
    for this device).  Common keys are auto-extracted:

    - ``id`` / ``device_id`` -> ``self.device_id``
    - ``primary: true`` -> ``self.is_primary``

    A logger is created automatically as
    ``f"{module}.{class}[{device_id}]"``.
    """

    device_type: ClassVar[str] = "device"
    device_id: str = "device"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self.cfg: Dict[str, Any] = dict(cfg or {})
        self.cfg.update(kwargs)

        cfg_id = self.cfg.get("id") or self.cfg.get("device_id")
        if cfg_id:
            self.device_id = str(cfg_id)
        self.is_primary: bool = bool(self.cfg.get("primary", False))

        self.signals = DeviceSignals()

        self._started: Optional[datetime] = None
        self._stopped: Optional[datetime] = None
        self.is_running: bool = False

        self.logger = get_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}[{self.device_id}]"
        )

    # -- lifecycle (subclasses override as needed) -----------------------
    def initialize(self) -> bool:
        return True

    def arm(self, config: Any) -> None:  # noqa: D401 - default no-op
        """Per-run preparation.  No-op by default."""
        return None

    def start(self) -> bool:
        self._started = datetime.now()
        self.is_running = True
        self.signals.started.emit()
        return True

    def stop(self) -> bool:
        self._stopped = datetime.now()
        self.is_running = False
        self.signals.finished.emit()
        return True

    def shutdown(self) -> None:
        return None

    # -- introspection ---------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "is_primary": self.is_primary,
            "is_running": self.is_running,
            "started": self._started.isoformat() if self._started else None,
            "stopped": self._stopped.isoformat() if self._stopped else None,
        }

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "is_primary": self.is_primary,
        }


# ---------------------------------------------------------------------------
# BaseDataProducer
# ---------------------------------------------------------------------------


class BaseDataProducer(BaseDevice):
    """Base class for devices that stream samples to the DataQueue.

    Subclasses produce data by calling :meth:`record`, which
    timestamps, appends to an in-memory buffer, and emits
    ``signals.data(payload, ts)`` in one step.

    The default :meth:`save_data` writes the buffer as a two-column
    CSV (``timestamp,payload``).  Override for binary or
    domain-specific formats.
    """

    file_type: ClassVar[str] = "csv"
    bids_type: ClassVar[Optional[str]] = "beh"
    sampling_rate: float = 0.0
    data_type: ClassVar[str] = "samples"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)

        self.output_path: Optional[str] = None
        self.metadata_path: Optional[str] = None
        self.is_active: bool = False

        self._buffer: list[Tuple[float, Any]] = []
        self._buffer_lock = threading.Lock()

    # -- helpers ---------------------------------------------------------
    def record(self, payload: Any, ts: Optional[float] = None) -> float:
        """Buffer ``payload`` and emit ``signals.data``.

        Returns the timestamp used.
        """
        if ts is None:
            ts = time.time()
        with self._buffer_lock:
            self._buffer.append((ts, payload))
        try:
            self.signals.data.emit(payload, ts)
        except Exception as exc:
            self.logger.warning("signals.data emit failed: %s", exc)
        return ts

    def clear_buffer(self) -> None:
        with self._buffer_lock:
            self._buffer.clear()

    # -- lifecycle overrides --------------------------------------------
    def arm(self, config: Any) -> None:
        """Default ``arm``: clear buffer and resolve ``output_path``.

        ``config`` is expected to expose ``make_path(name, ext, bids)``
        (see :class:`mesofield.config.ExperimentConfig`).
        """
        self.clear_buffer()
        make_path = getattr(config, "make_path", None)
        if make_path is not None:
            try:
                self.output_path = make_path(
                    self.device_id, self.file_type, self.bids_type
                )
            except Exception as exc:
                self.logger.debug("make_path failed for %s: %s", self.device_id, exc)

    def start(self) -> bool:
        self.is_active = True
        return super().start()

    def stop(self) -> bool:
        self.is_active = False
        return super().stop()

    # -- data access -----------------------------------------------------
    def get_data(self) -> list[Tuple[float, Any]]:
        with self._buffer_lock:
            return list(self._buffer)

    def save_data(self, path: Optional[str] = None) -> Optional[str]:
        target = path or self.output_path
        if not target:
            self.logger.debug("save_data: no path provided for %s", self.device_id)
            return None

        snapshot = self.get_data()
        Path(target).parent.mkdir(parents=True, exist_ok=True)

        # If any payload is a dict, fan keys out to columns.
        keys: list[str] = []
        seen: set[str] = set()
        for _ts, p in snapshot:
            if isinstance(p, dict):
                for k in p:
                    if k not in seen:
                        seen.add(k)
                        keys.append(k)

        with open(target, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if keys:
                writer.writerow(["timestamp", *keys])
                for ts, payload in snapshot:
                    if isinstance(payload, dict):
                        writer.writerow([ts, *(payload.get(k, "") for k in keys)])
                    else:
                        writer.writerow([ts, payload, *([""] * (len(keys) - 1))])
            else:
                writer.writerow(["timestamp", "payload"])
                for ts, payload in snapshot:
                    writer.writerow([ts, payload])
        self.logger.info("Saved %d samples to %s", len(snapshot), target)
        return target


# ---------------------------------------------------------------------------
# BaseSerialDevice
# ---------------------------------------------------------------------------


class BaseSerialDevice(BaseDataProducer):
    """Polling device for line-based serial protocols (Arduino/Teensy/etc.).

    Subclasses override :meth:`parse_line`.  Optionally override
    :meth:`setup_serial` for post-open initialisation (handshakes,
    buffer drain, configuring device-side parameters).

    Configuration keys read from ``cfg``:

    - ``port`` (str, required when ``development_mode=False``)
    - ``baudrate`` (int, default 115200)
    - ``timeout`` (float, default 0.1) — pyserial readline timeout.
    - ``dtr`` (bool | None, default None) — set to ``False`` to
      suppress Arduino auto-reset on connect.  ``None`` keeps the
      OS default.
    - ``connect_delay`` (float, default 0.0) — seconds to wait after
      opening the port before reads begin; common Arduinos need ~2.0.
      The input buffer is flushed after the delay.
    - ``development_mode`` (bool, default False) — skip opening the
      port so the GUI / Procedure can launch without hardware.
      ``send_line`` becomes a no-op; the polling thread idles.
    """

    default_baudrate: ClassVar[int] = 115_200
    default_timeout: ClassVar[float] = 0.1
    poll_interval: float = 0.0
    join_timeout: float = 2.0

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self.port: Optional[str] = self.cfg.get("port")
        self.baudrate: int = int(self.cfg.get("baudrate", self.default_baudrate))
        self.timeout: float = float(self.cfg.get("timeout", self.default_timeout))
        self.dtr: Optional[bool] = self.cfg.get("dtr")
        self.connect_delay: float = float(self.cfg.get("connect_delay", 0.0))
        self.development_mode: bool = bool(self.cfg.get("development_mode", False))

        self._serial: Any = None
        self._serial_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- subclass hooks -------------------------------------------------
    def parse_line(self, line: bytes) -> Optional[Tuple[Any, Optional[float]]]:
        """Decode one raw serial line into ``(payload, ts)`` or ``None``."""
        raise NotImplementedError("Subclasses must override parse_line()")

    def setup_serial(self) -> None:
        """Hook called once after the port is opened.  Default no-op.

        Override to send a handshake, query firmware version, configure
        device-side parameters, etc.
        """
        return None

    # -- write path -----------------------------------------------------
    def send_line(self, payload: str | bytes, *, newline: bytes = b"\n") -> bytes:
        """Write a command to the device.  Thread-safe with the reader.

        Accepts ``str`` (UTF-8 encoded) or ``bytes``.  ``newline`` is
        appended unless ``payload`` already ends with it.  In
        ``development_mode`` the bytes are logged and discarded.
        Returns the bytes that were written (or would have been).
        """
        data = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
        if not data.endswith(newline):
            data += newline

        if self.development_mode:
            self.logger.debug("send_line (dev_mode, no-op): %r", data)
            return data
        if self._serial is None:
            raise RuntimeError(
                f"{self.device_id}: cannot send_line before initialize()/start()"
            )
        with self._serial_lock:
            self._serial.write(data)
        return data

    # -- lifecycle ------------------------------------------------------
    def initialize(self) -> bool:
        if self.development_mode or self._serial is not None:
            return True
        if not self.port:
            raise ValueError(
                f"{self.device_id}: 'port' is required when development_mode is False"
            )
        try:
            import serial
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pyserial is required for BaseSerialDevice; install with `pip install pyserial`"
            ) from exc

        if self.dtr is None:
            ser = serial.Serial(self.port, baudrate=self.baudrate, timeout=self.timeout)
        else:
            # DTR must be set before the port physically opens
            # (Arduino-no-reset idiom).
            ser = serial.Serial(baudrate=self.baudrate, timeout=self.timeout)
            ser.port = self.port
            ser.dtr = self.dtr
            ser.open()
        self._serial = ser

        if self.connect_delay > 0:
            time.sleep(self.connect_delay)
            try:
                ser.reset_input_buffer()
            except Exception:
                pass

        self.logger.info("Opened serial %s @ %d baud", self.port, self.baudrate)
        try:
            self.setup_serial()
        except Exception:
            self.logger.exception("setup_serial failed; continuing")
        return True

    def shutdown(self) -> None:
        if self.is_running:
            self.stop()
        super().shutdown()
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception as exc:
                self.logger.debug("serial close failed: %s", exc)
            self._serial = None

    def start(self) -> bool:
        if self._serial is None and not self.development_mode:
            self.initialize()
        if self._thread is not None and self._thread.is_alive():
            self.logger.debug("start() called but thread already running")
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"{self.__class__.__name__}-{self.device_id}",
            daemon=True,
        )
        self._thread.start()
        return super().start()

    def stop(self) -> bool:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self.join_timeout)
            if thread.is_alive():
                self.logger.warning(
                    "read thread for %s did not exit within %.1fs",
                    self.device_id,
                    self.join_timeout,
                )
        self._thread = None
        return super().stop()

    # -- read loop ------------------------------------------------------
    def _run_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    sample = self._read_once()
                except Exception as exc:
                    self.logger.exception("read raised: %s", exc)
                    sample = None

                if sample is not None:
                    payload, ts = sample
                    self.record(payload, ts)
                elif self.poll_interval > 0:
                    self._stop_event.wait(self.poll_interval)
        finally:
            self.logger.debug("read loop exited")

    def _read_once(self) -> Optional[Tuple[Any, Optional[float]]]:
        if self._serial is None:
            return None
        try:
            with self._serial_lock:
                line = self._serial.readline()
        except Exception as exc:
            self.logger.exception("serial read failed: %s", exc)
            return None
        if not line:
            return None
        try:
            return self.parse_line(line)
        except Exception as exc:
            self.logger.exception("parse_line raised on %r: %s", line, exc)
            return None
