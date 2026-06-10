"""Authorable base classes for non-Qt hardware devices.

Layered hierarchy that lets a user write a new device with as little
boilerplate as possible:

``BaseDevice``
    Lifecycle skeleton, ``self.signals`` (DeviceSignals), logger, cfg
    storage, ``is_primary`` / ``is_active`` tracking, default
    ``status``/``metadata``.

    Every exit path — natural completion, ``stop()``, or an unrecoverable
    error — funnels through :meth:`BaseDevice._finalize`, the only code
    that flips the device inactive, stamps ``_stopped``, runs subclass
    teardown (:meth:`BaseDevice._on_finalize`) and emits ``signals.error``
    / ``signals.finished``.  ``_finalize`` is idempotent, so ``finished``
    fires exactly once per run no matter how many paths race into it.

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

Qt devices (``QThread`` subclasses) inherit :class:`BaseDevice` through
:class:`mesofield.devices.base_camera.BaseCamera` (BaseDevice never calls
``super().__init__``, so it composes cleanly with QThread's sip metaclass);
see ``OpenCVCamera`` for the pattern.
"""

from __future__ import annotations

import csv
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Tuple

from mesofield.signals import DeviceSignals
from mesofield.utils._logger import get_logger

if TYPE_CHECKING:
    from mesofield.config import ExperimentConfig


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

        # Injected once by HardwareManager.initialize() so producers can reach
        # `make_path` and experiment state outside the per-run `arm(config)`.
        self.config: Optional["ExperimentConfig"] = None

        self.signals = DeviceSignals()

        self._started: Optional[datetime] = None
        self._stopped: Optional[datetime] = None
        self.is_active: bool = False
        # Set by `_finalize` when the run ended in failure; None otherwise.
        self.error: Optional[BaseException] = None
        # Recording length in seconds for this run (None = unbounded).
        # Captured by `arm(config)`; primary loop-based devices use it to
        # self-terminate (which is what ends the whole Procedure).
        self._planned_duration: Optional[float] = None
        # `_finalize` exactly-once guard.
        self._finalize_lock = threading.Lock()
        self._finalized = False

        self.logger = get_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}[{self.device_id}]"
        )

    # -- lifecycle (subclasses override the _do_* / _on_* hooks) ----------
    def initialize(self) -> bool:
        return True

    def arm(self, config: "ExperimentConfig") -> None:
        """Per-run preparation: capture the planned recording duration and
        reset the terminal state.  Subclasses extend (writer setup, buffer
        clearing) and call ``super().arm(config)`` — or
        ``BaseDevice.arm(self, config)`` when multiple inheritance makes
        ``super()`` ambiguous.
        """
        try:
            duration = getattr(config, "sequence_duration", None)
            self._planned_duration = float(duration) if duration else None
        except (TypeError, ValueError):
            self._planned_duration = None
        with self._finalize_lock:
            self._finalized = False
        self.error = None

    def start(self) -> bool:
        """Mark the device running and emit ``signals.started``.

        Subclasses that spawn worker threads should do so via the
        acquisition-loop scaffold (see :class:`BaseDataProducer`) or call
        ``super().start()`` after their backend-specific launch succeeds.
        """
        with self._finalize_lock:
            self._finalized = False
        self.error = None
        self._started = datetime.now()
        self.is_active = True
        self.signals.started.emit()
        return True

    def stop(self) -> bool:
        """Stop the device.  Idempotent; safe to call from any thread."""
        self._finalize()
        return True

    def _finalize(self, error: Optional[BaseException] = None) -> None:
        """Single terminal transition for the run.

        Exactly-once: the first caller wins (natural loop exit, ``stop()``,
        or a crashed acquisition thread); later calls are no-ops.  Runs the
        subclass :meth:`_on_finalize` teardown, then emits ``signals.error``
        (when failing) and ``signals.finished``.
        """
        with self._finalize_lock:
            if self._finalized:
                return
            self._finalized = True
        self._stopped = datetime.now()
        self.is_active = False
        self.error = error
        try:
            self._on_finalize(error)
        except Exception as exc:
            self.logger.error(f"_on_finalize failed for {self.device_id}: {exc}")
        if error is not None:
            self.logger.error(f"{self.device_id} terminated with error: {error}")
            self.signals.error.emit(error)
        self.signals.finished.emit()

    def _on_finalize(self, error: Optional[BaseException]) -> None:
        """Subclass teardown hook (close writers, flush buffers, release
        handles).  Runs exactly once per run, on every exit path, before
        ``finished`` is emitted.  Default: no-op.
        """
        return None

    def shutdown(self) -> None:
        return None

    # -- introspection ---------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "is_primary": self.is_primary,
            "is_active": self.is_active,
            "error": str(self.error) if self.error else None,
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

    # Keys that describe orchestration / GUI / routing rather than calibration.
    # Stripped from `cfg` when building the `calibration` payload that ends up
    # in the AcquisitionManifest. Subclasses can extend this tuple.
    _MANIFEST_RESERVED_KEYS: ClassVar[tuple[str, ...]] = (
        "id", "device_id", "type", "primary", "widgets", "output",
    )

    @property
    def calibration(self) -> Dict[str, Any]:
        """Device-specific constants worth recording with the data.

        Default: everything in `cfg` that isn't an orchestration key. Override
        on a subclass to curate the list explicitly.
        """
        reserved = set(self._MANIFEST_RESERVED_KEYS)
        return {k: v for k, v in self.cfg.items() if k not in reserved}

    def sidecars(self) -> list:
        """Auxiliary files this device writes alongside its primary output.

        Default: none. Override to declare extra sidecars (masks, regions,
        derived parameter files) so they ride in the manifest with a `role`
        and `schema_version` instead of being discovered by glob.

        The camera classes' per-frame metadata JSON is the primary sidecar
        and lives on `self.metadata_path` -- not here. Use this method for
        the *extra* files only.

        Returns a list of `mesokit_schema.SidecarEntry`-shaped mappings or
        instances. The Procedure relativises any absolute paths.
        """
        return []


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
    # Declared clock source for the AcquisitionManifest's TimeBasis. Matches
    # what `record()` stamps (`time.time()`). Camera subclasses override.
    clock_source: ClassVar[str] = "wall_unix_s"
    # Typed contract for what this producer pushes onto the session-wide
    # dataqueue (see mesokit_schema.DataqueuePayloadSchema). Set on a
    # subclass when the parser needs to locate this producer's rows in
    # dataqueue.csv. Leave `None` for producers that don't write to the
    # queue (e.g. cameras whose alignment is per-frame metadata, not queue
    # rows). The Procedure copies the value into ProducerEntry.dataqueue_schema.
    dataqueue_payload_schema: ClassVar[Optional[dict]] = None

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)

        self.output_path: Optional[str] = None
        self.metadata_path: Optional[str] = None

        self._buffer: list[Tuple[float, Any]] = []
        self._buffer_lock = threading.Lock()

        # Acquisition-loop state (thread spawned by `start`).
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop_t0: float = 0.0
        self.join_timeout: float = getattr(self, "join_timeout", 2.0)

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
            self.logger.warning(f"signals.data emit failed: {exc}")
        return ts

    def clear_buffer(self) -> None:
        with self._buffer_lock:
            self._buffer.clear()

    # -- lifecycle overrides --------------------------------------------
    def arm(self, config: "ExperimentConfig") -> None:
        """Default ``arm``: clear buffer and resolve ``output_path``.

        ``config`` is expected to expose ``make_path(name, ext, bids)``
        (see :class:`mesofield.config.ExperimentConfig`).
        """
        super().arm(config)
        self.clear_buffer()
        make_path = getattr(config, "make_path", None)
        if make_path is not None:
            try:
                self.output_path = make_path(
                    self.device_id, self.file_type, self.bids_type
                )
            except Exception as exc:
                self.logger.debug(f"make_path failed for {self.device_id}: {exc}")

    # -- acquisition loop --------------------------------------------------
    # Threaded producers implement `_acquire_loop`; `start`/`stop` own the
    # thread, and `_run_acquisition` guarantees the terminal transition
    # (exactly-once `finished`, `error` on a crash) on every exit path.

    def start(self) -> bool:
        """Mark the device running and spawn the acquisition thread."""
        if self._thread is not None and self._thread.is_alive():
            self.logger.debug("start() called but acquisition thread already running")
            return False
        started = super().start()
        self._stop_event.clear()
        self._loop_t0 = time.monotonic()
        self._thread = threading.Thread(
            target=self._run_acquisition,
            name=f"{self.__class__.__name__}-{self.device_id}",
            daemon=True,
        )
        self._thread.start()
        return started

    def stop(self) -> bool:
        """Signal the loop to stop, join it (unless called *from* it), and
        finalize."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.join_timeout)
            if thread.is_alive():
                self.logger.warning(
                    f"acquisition thread for {self.device_id} did not exit "
                    f"within {self.join_timeout:.1f}s"
                )
        self._thread = None
        return super().stop()

    def _acquire_loop(self) -> None:
        """Acquisition loop body (subclass hook).

        Loop on ``while not self._loop_should_stop(): ...`` and call
        :meth:`record` per sample.  Raising ends the run in failure
        (``signals.error`` then ``finished``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _acquire_loop() "
            f"(or override start/stop for non-threaded producers)"
        )

    def _run_acquisition(self) -> None:
        error: Optional[BaseException] = None
        try:
            self._acquire_loop()
        except Exception as exc:
            error = exc
            self.logger.exception(f"acquisition loop crashed: {exc}")
        finally:
            # A natural end (planned duration reached, source exhausted) or a
            # crash must finalize from here because nothing external will call
            # stop(). When stop() initiated the exit it finalizes after the
            # join; `_finalize` is a no-op for whichever caller arrives second.
            if error is not None or not self._stop_event.is_set():
                self._finalize(error)

    def _loop_should_stop(self) -> bool:
        """Loop-exit test: external stop, or — for the primary device of a
        run with a known duration — planned duration reached."""
        if self._stop_event.is_set():
            return True
        if (
            self.is_primary
            and self._planned_duration
            and (time.monotonic() - self._loop_t0) >= self._planned_duration
        ):
            self.logger.info(
                f"planned duration {self._planned_duration:.1f}s reached; stopping"
            )
            return True
        return False

    # -- data access -----------------------------------------------------
    def get_data(self) -> list[Tuple[float, Any]]:
        with self._buffer_lock:
            return list(self._buffer)

    def save_data(self, path: Optional[str] = None) -> Optional[str]:
        target = path or self.output_path
        if not target:
            self.logger.debug(f"save_data: no path provided for {self.device_id}")
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
            self.logger.debug(f"Saved {len(snapshot)} samples to {target}")
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
    # Consecutive read failures tolerated before the run ends in failure.
    max_consecutive_failures: ClassVar[int] = 25
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
            self.logger.debug(f"send_line (dev_mode, no-op): {data!r}")
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
            
        try:
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
        except Exception as exc:
            raise RuntimeError(f"Failed to open serial port {self.port}: {exc}") from exc

        self.logger.info(f"Opened serial {self.port} @ {self.baudrate} baud")
        try:
            self.setup_serial()
        except Exception:
            self.logger.exception("setup_serial failed; continuing")
        return True

    def shutdown(self) -> None:
        if self.is_active:
            self.stop()
        super().shutdown()
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception as exc:
                self.logger.debug(f"serial close failed: {exc}")
            self._serial = None

    def start(self) -> bool:
        if self._serial is None and not self.development_mode:
            self.initialize()
        return super().start()

    # -- read loop --------------------------------------------------------
    def _acquire_loop(self) -> None:
        failures = 0
        while not self._loop_should_stop():
            try:
                sample = self._read_once()
                failures = 0
            except Exception as exc:
                self.logger.exception(f"read raised: {exc}")
                failures += 1
                # A genuinely dead port should end the run in failure, not
                # spin logging exceptions forever.
                if failures >= self.max_consecutive_failures:
                    raise RuntimeError(
                        f"{self.device_id}: {failures} consecutive read "
                        f"failures (last: {exc})"
                    ) from exc
                sample = None

            if sample is not None:
                payload, ts = sample
                self.record(payload, ts)
            elif self.poll_interval > 0:
                self._stop_event.wait(self.poll_interval)

    def _read_once(self) -> Optional[Tuple[Any, Optional[float]]]:
        if self._serial is None:
            return None
        with self._serial_lock:
            line = self._serial.readline()
        if not line:
            return None
        try:
            return self.parse_line(line)
        except Exception as exc:
            self.logger.exception(f"parse_line raised on {line!r}: {exc}")
            return None
