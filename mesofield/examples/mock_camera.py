"""Synthetic camera device for headless / GUI-only iteration.

Subclasses :class:`BaseCamera` for the cosmetic + manifest surface every
camera shares, and :class:`BaseDataProducer` for the buffer + record()
mechanism every queue-pushing producer needs.

Contract with the real cameras:

  - Writes an actual OME-TIFF stack on :meth:`save_data` (real file, real
    tifffile output).
  - Writes a ``<output_path>_frame_metadata.json`` sidecar mirroring what
    ``CustomWriter.finalize_metadata`` would emit on a real MMCamera.
  - Sets ``self.metadata_path`` so the AcquisitionManifest writer picks it
    up via :attr:`ProducerEntry.metadata_path`.
  - Exposes ``self.image_ready`` as a Qt :class:`pyqtSignal` (via a lazy
    :class:`QtImageAdapter`) so the MDA viewer's static-camera branch can
    subscribe to live frames.

The producer-side data path still uses ``self.record({"frame_index": N}, ts)``
to push lightweight per-frame markers onto the dataqueue. Raw image
ndarrays travel out-of-band via ``image_ready`` (Qt) and the in-memory
``_frames`` buffer (saved by ``save_data``); they intentionally do NOT
travel on ``self.signals.data``, because the dataqueue is a row-per-emit
CSV and streaming raw frames through it would balloon the file.

YAML registration via ``type: mock_camera``::

    camera:
      type: mock_camera
      primary: true
      width: 64
      height: 64
      frame_interval_ms: 50
      output:
        suffix: meso
        file_type: ome.tiff
        bids_type: func
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

import numpy as np

from mesofield.devices.base import BaseDataProducer
from mesofield.io.devices.base_camera import BaseCamera


class MockFrameProducer(BaseCamera, BaseDataProducer):
    """Synthetic camera producing real OME-TIFF + frame metadata JSON."""

    # BaseCamera defaults are already camera-friendly; lock the data_type
    # to `frames` for downstream parsers.
    device_type: ClassVar[str] = "camera"
    file_type: ClassVar[str] = "ome.tiff"
    bids_type: ClassVar[Optional[str]] = "func"
    data_type: ClassVar[str] = "frames"
    clock_source: ClassVar[str] = "wall_unix_s"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        # BaseDataProducer.__init__ sets self.signals + the record() buffer.
        BaseDataProducer.__init__(self, cfg, **kwargs)
        # BaseCamera surface (identity, viewer cosmetics, output slots).
        # Idempotent on `self.signals` -- preserves the DeviceSignals already
        # created by BaseDataProducer.__init__.
        self._init_camera_surface(self.cfg, backend="mock")
        # Frame-generation knobs from cfg.
        self.width: int = int(self.cfg.get("width", 32))
        self.height: int = int(self.cfg.get("height", 32))
        self.frame_interval_s: float = float(self.cfg.get("frame_interval_ms", 50)) / 1000.0
        if self.frame_interval_s:
            self.sampling_rate = 1.0 / self.frame_interval_s
        # Capture thread state.
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frames: list[np.ndarray] = []
        self._frames_lock = threading.Lock()
        self._frame_records: list[dict[str, Any]] = []
        # Lazy Qt bridge for live preview. None in headless mode (no PyQt6
        # installed); the MDA viewer short-circuits when image_ready is None.
        self._qt_image_adapter = None
        self.image_ready = None
        try:
            from mesofield.gui.qt_device_adapter import QtImageAdapter

            self._qt_image_adapter = QtImageAdapter()
            self.image_ready = self._qt_image_adapter.image_ready
        except Exception:
            self.logger.debug("QtImageAdapter unavailable; running headless.")

    # ---- lifecycle ------------------------------------------------------
    def arm(self, config: Any) -> None:
        # BaseDataProducer.arm() resolves self.output_path via make_path.
        # We also need BaseCamera's set_sequence call to run (no-op for mock).
        BaseDataProducer.arm(self, config)
        self.set_sequence(config.build_sequence)
        # Reset capture state.
        self._frames.clear()
        self._frame_records.clear()
        # NOTE: don't compute metadata_path here. DataSaver rewrites the
        # final save path using path_args['suffix'] at save time, which can
        # differ from self.output_path computed during arm(). The sidecar is
        # derived from the final target inside save_data().

    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"MockCamera-{self.device_id}",
            daemon=True,
        )
        self._thread.start()
        return BaseDataProducer.start(self)

    def stop(self) -> bool:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None
        return BaseDataProducer.stop(self)

    def _run_loop(self) -> None:
        rng = np.random.default_rng(seed=0)
        while not self._stop_event.is_set():
            ts = time.time()
            frame = self._synthesise_frame(rng, len(self._frames))
            with self._frames_lock:
                self._frames.append(frame)
                self._frame_records.append({
                    "frame_index": len(self._frames) - 1,
                    "TimeReceivedByCore": datetime.fromtimestamp(ts, tz=timezone.utc)
                                                    .isoformat().replace("+00:00", "Z"),
                })
            # Push a lightweight marker onto the DataQueue (signals.data) so
            # session-wide timing is visible without bloating the queue CSV
            # with full frame arrays.
            self.record({"frame_index": len(self._frames) - 1}, ts=ts)
            # Fan the raw frame out to the GUI preview, if Qt is available.
            if self._qt_image_adapter is not None:
                self._qt_image_adapter.emit_frame(frame)
            self._stop_event.wait(self.frame_interval_s)

    # --- live preview contract (BaseCamera abstract methods) ------------
    # The mock's run loop already drives "live" frames; start_live aliases
    # start(), stop_live aliases stop(). snap synthesises a one-off frame
    # using the same generator so calibration/colour behaviour is identical.

    def snap(self) -> np.ndarray:
        rng = np.random.default_rng(seed=0)
        frame = self._synthesise_frame(rng, 0)
        if self._qt_image_adapter is not None:
            self._qt_image_adapter.emit_frame(frame)
        return frame

    def start_live(self) -> None:
        self.start()

    def stop_live(self) -> None:
        self.stop()

    def _synthesise_frame(self, rng: np.random.Generator, frame_index: int) -> np.ndarray:
        # Four-quadrant baseline so a "regional means" processor downstream
        # produces 4 distinguishable traces, not just noise.
        h, w = self.height, self.width
        frame = np.zeros((h, w), dtype=np.uint16)
        half_h, half_w = h // 2, w // 2
        baselines = (1000, 2000, 3000, 4000)
        slow_drift = int(50 * np.sin(frame_index / 5.0))
        frame[:half_h, :half_w] = baselines[0] + slow_drift
        frame[:half_h, half_w:] = baselines[1] + slow_drift
        frame[half_h:, :half_w] = baselines[2] + slow_drift
        frame[half_h:, half_w:] = baselines[3] + slow_drift
        frame += rng.integers(0, 50, size=(h, w), dtype=np.uint16)
        return frame

    # ---- save -----------------------------------------------------------
    def save_data(self, path: Optional[str] = None) -> Optional[str]:
        """Replay buffered frames through the standard mesofield writer.

        MockFrameProducer captures frames in RAM during the run and pushes
        them through ``BaseCamera._make_writer`` (defaulting to
        :class:`CustomWriter`) at save time. The writer's
        ``sequenceStarted`` / ``frameReady`` / ``sequenceFinished``
        lifecycle is the same one pymmcore-plus drives for real MMCameras,
        so the on-disk OME-TIFF + ``_frame_metadata.json`` sidecar are
        produced by the same code path -- the mock is now a real-world
        testing fixture for the writer, not a parallel implementation.
        """
        target = path or self.output_path
        if not target:
            self.logger.debug("save_data: no path for %s", self.device_id)
            return None
        Path(target).parent.mkdir(parents=True, exist_ok=True)

        with self._frames_lock:
            frames = list(self._frames)
            records = list(self._frame_records)

        if not frames:
            self.logger.warning("MockFrameProducer captured 0 frames")
            return None

        # Anchor output_path to the final save target before constructing the
        # writer; BaseCamera._make_writer derives the sidecar filename from it.
        self.output_path = str(target)
        self.writer = self._make_writer(target)
        # CustomWriter sets _frame_metadata_filename = filename + _frame_metadata.json
        self.metadata_path = getattr(
            self.writer, "_frame_metadata_filename", str(target) + "_frame_metadata.json",
        )

        # Build a synthetic MDASequence + MDAEvent stream so the writer's
        # ``_5DWriterBase`` lifecycle has the metadata it expects. The
        # sequence size is `t=len(frames)` (with implicit y/x derived from
        # the first frame shape); each event carries `index={"t": i}`.
        import useq

        seq = useq.MDASequence(
            time_plan={"loops": len(frames), "interval": self.frame_interval_s},
        )
        # Empty dict satisfies pymmcore-plus's "meta required" deprecation
        # warning; the actual per-frame metadata still rides on the
        # `frameReady` calls below.
        self.writer.sequenceStarted(seq, {})
        try:
            for i, (frame, record) in enumerate(zip(frames, records)):
                event = useq.MDAEvent(index={"t": i})
                # The mock's per-frame record (frame_index + TimeReceivedByCore)
                # becomes the `meta` that finalize_metadata serializes to JSON.
                self.writer.frameReady(frame, event, record)
        finally:
            self.writer.sequenceFinished(seq)

        self.logger.info(
            "Wrote %d frames to %s via %s",
            len(frames), target, type(self.writer).__name__,
        )
        return target

    # ---- manifest hint --------------------------------------------------
    @property
    def calibration(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "frame_interval_ms": int(self.frame_interval_s * 1000),
        }
