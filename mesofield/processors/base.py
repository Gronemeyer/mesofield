"""Threaded base class for real-time camera frame processors."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, ClassVar, Dict, Optional, TYPE_CHECKING

from PyQt6.QtCore import QObject, pyqtSignal

from mesofield.signals import DeviceSignals
from mesofield.utils._logger import get_logger

if TYPE_CHECKING:
    from mesofield.protocols import DataProducer


class _QtAdapter(QObject):
    """QObject carrier so the processor can expose a pyqtSignal that
    cross-thread emits are auto-queued to the GUI thread."""

    valueUpdated = pyqtSignal(float, float)


class FrameProcessor:
    """Base class for per-frame, real-time processors.

    Acquisition stays on the camera thread; ``compute`` runs on a daemon
    worker thread fed by a bounded queue (size 1, drop-oldest replace)
    so processing latency cannot stall the camera. Subclasses implement
    one method::

        def compute(self, img, idx, ts) -> float | None: ...

    Returning ``None`` skips emission for that frame. Otherwise the
    scalar is emitted on:

    * ``self.signals.data(value, ts)`` -- psygnal; pushed onto the
      DataQueue / CSV logger when the processor is registered with
      :meth:`DataManager.register_hardware_device`.
    * ``self.valueUpdated(t, value)`` -- ``pyqtSignal(float, float)``;
      Qt's cross-thread queued connection makes
      :class:`~mesofield.gui.speedplotter.SerialWidget` safe to drive
      from the worker thread.
    """

    device_type: ClassVar[str] = "processor"
    data_type: ClassVar[str] = "scalar"
    file_type: ClassVar[str] = "csv"
    bids_type: ClassVar[Optional[str]] = None

    # Recognized SerialWidget styling kwargs collected into ``plot_config``.
    _PLOT_KWARGS = (
        "label",
        "value_label",
        "value_units",
        "y_range",
        "value_scale",
        "max_points",
    )

    def __init__(
        self,
        name: Optional[str] = None,
        camera: Optional["DataProducer"] = None,
        sampling_rate: float = 0.0,
        plot: bool = False,
        **kwargs: Any,
    ) -> None:
        # Default name = lowercased class name so ``FrameMean(camera=cam)`` works.
        if name is None:
            name = self.__class__.__name__.lower()
        self.device_id: str = name
        self.id: str = name
        self.name: str = name
        self.signals = DeviceSignals()
        self.sampling_rate: float = (
            sampling_rate
            or float(getattr(camera, "sampling_rate", 0.0) or 0.0)
        )
        self.is_active: bool = False
        self.output_path: Optional[str] = None
        self.metadata_path: Optional[str] = None
        self.logger = get_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}[{self.device_id}]"
        )

        self._qt = _QtAdapter()
        self.valueUpdated = self._qt.valueUpdated

        self._queue: "queue.Queue[tuple]" = queue.Queue(maxsize=1)
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._camera: Optional["DataProducer"] = None
        # Remembered camera reference for no-arg attach() / status reporting.
        self.camera: Optional["DataProducer"] = camera

        # ---- compute-load counters (see ``status()``) ------------------
        self.n_enqueued: int = 0
        self.n_dropped: int = 0
        self.n_processed: int = 0
        self.n_errors: int = 0
        self.compute_ms_ewma: float = 0.0
        self.compute_ms_max: float = 0.0
        self._ewma_alpha: float = 0.1

        # ---- GUI plot wiring (opt-in) ----------------------------------
        self.plot_enabled: bool = bool(plot)
        self.plot_config: Dict[str, Any] = {
            k: kwargs.pop(k) for k in list(kwargs) if k in self._PLOT_KWARGS
        }
        if kwargs:
            # Unknown kwargs are usually a typo (e.g. ``y_lim`` for ``y_range``).
            raise TypeError(
                f"{type(self).__name__}: unexpected keyword argument(s) "
                f"{sorted(kwargs)}. Recognized plot kwargs: {self._PLOT_KWARGS}"
            )

        # ---- auto-attach if a camera was supplied ----------------------
        if camera is not None:
            self.attach(camera)

    # ---- subclass contract --------------------------------------------
    def compute(self, img: Any, idx: Any, ts: Any) -> Optional[float]:
        """Subclass hook: return one scalar (or ``None``) per frame.

        Called on every frame the attached camera emits. Implementations
        should be fast — anything heavy will starve the camera buffer.
        Return ``None`` to skip emitting a sample for this frame.

        Args:
            img: Frame from the camera (typically a 2-D ``ndarray``).
            idx: Frame index assigned by the camera (monotonic).
            ts: Device timestamp for the frame.

        Returns:
            A single ``float`` to be pushed to the data queue / plot, or
            ``None`` to skip this frame.
        """
        raise NotImplementedError

    # ---- compute-load reporting ----------------------------------------
    def status(self) -> Dict[str, Any]:
        """Snapshot of the per-run compute-load counters."""
        enq = self.n_enqueued or 1
        return {
            "device_id": self.device_id,
            "n_enqueued": self.n_enqueued,
            "n_processed": self.n_processed,
            "n_dropped": self.n_dropped,
            "n_errors": self.n_errors,
            "drop_ratio": self.n_dropped / enq,
            "compute_ms_ewma": round(self.compute_ms_ewma, 3),
            "compute_ms_max": round(self.compute_ms_max, 3),
        }

    def _log_status(self) -> None:
        s = self.status()
        self.logger.info(
            f"compute-load: processed={s['n_processed']} "
            f"dropped={s['n_dropped']} ({s['drop_ratio']*100:.1f}%) "
            f"compute_ms ewma={s['compute_ms_ewma']} max={s['compute_ms_max']} "
            f"errors={s['n_errors']}"
        )

    # ---- public API ----------------------------------------------------
    def attach(self, camera: Optional["DataProducer"] = None) -> None:
        """Attach to a camera's ``signals.frame`` and start the worker.

        ``camera`` defaults to whatever was passed at construction time
        (``self.camera``); callers building processors first and attaching
        later may pass it explicitly.
        """
        if self._camera is not None:
            return
        if camera is None:
            camera = self.camera
        if camera is None:
            raise ValueError(
                f"{type(self).__name__}: no camera to attach to "
                "(pass camera= to attach() or to the constructor)"
            )
        self._camera = camera
        self.camera = camera
        # Reset counters at the start of each attach session.
        self.n_enqueued = 0
        self.n_dropped = 0
        self.n_processed = 0
        self.n_errors = 0
        self.compute_ms_ewma = 0.0
        self.compute_ms_max = 0.0
        self._stop.clear()
        self._worker = threading.Thread(
            target=self._loop,
            name=f"FrameProcessor-{self.device_id}",
            daemon=True,
        )
        self._worker.start()
        self.is_active = True
        try:
            camera.signals.frame.connect(self._enqueue)
        except Exception as exc:
            self.logger.warning(f"attach: signals.frame.connect failed: {exc}")
        try:
            camera.signals.finished.connect(self._log_status)
        except Exception:
            pass

    def detach(self) -> None:
        cam = self._camera
        if cam is not None:
            try:
                cam.signals.frame.disconnect(self._enqueue)
            except Exception:
                pass
            try:
                cam.signals.finished.disconnect(self._log_status)
            except Exception:
                pass
        self._camera = None
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=1.0)
            self._worker = None
        self.is_active = False

    # SerialWidget toggles call device.start()/device.stop().
    def start(self) -> bool:
        return True

    def stop(self) -> bool:
        return True

    # ---- internals -----------------------------------------------------
    def _enqueue(self, img: Any, idx: Any, ts: Any) -> None:
        self.n_enqueued += 1
        try:
            self._queue.put_nowait((img, idx, ts))
            return
        except queue.Full:
            pass
        # Drop the stale item so we always work on the freshest frame.
        self.n_dropped += 1
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait((img, idx, ts))
        except queue.Full:
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                img, idx, ts = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            try:
                value = self.compute(img, idx, ts)
            except Exception as exc:
                self.n_errors += 1
                self.logger.warning(f"compute failed: {exc}")
                continue
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.n_processed += 1
            if dt_ms > self.compute_ms_max:
                self.compute_ms_max = dt_ms
            self.compute_ms_ewma = (
                self._ewma_alpha * dt_ms
                + (1.0 - self._ewma_alpha) * self.compute_ms_ewma
            )
            if value is None:
                continue
            try:
                self.signals.data.emit(value, ts)
            except Exception:
                pass
            try:
                self._qt.valueUpdated.emit(
                    float(ts) if ts is not None else 0.0,
                    float(value),
                )
            except Exception:
                pass
