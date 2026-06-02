"""Synthetic devices for headless / GUI-only iteration.

These mock devices produce realistic data on daemon threads without any
serial or camera hardware. They are used by the scaffold ``--rig dev``
template and by the test suite.

.. rubric:: Mock encoder

Produces random click counts via :class:`BaseDataProducer`.

Usage in ``hardware.yaml``::

    encoder:
      type: mock
      sample_interval_ms: 100

Or programmatically::

    from mesofield.devices.mocks import MockEncoderDevice
    dev = MockEncoderDevice({"id": "encoder", "sample_interval_ms": 50})
    dev.start()
    ...
    dev.stop()

.. rubric:: Mock camera

Subclasses :class:`BaseCamera` for the cosmetic + manifest surface every
camera shares, and :class:`BaseDataProducer` for the buffer + record()
mechanism every queue-pushing producer needs.

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

import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

import numpy as np

from mesofield.devices.base import BaseDataProducer
from mesofield.devices.base_camera import BaseCamera


# ---------------------------------------------------------------------------
# Mock encoder
# ---------------------------------------------------------------------------


class MockEncoderDevice(BaseDataProducer):
    """Synthetic encoder that records random click counts."""

    device_type: ClassVar[str] = "encoder"
    file_type: ClassVar[str] = "csv"
    bids_type: ClassVar[Optional[str]] = "beh"

    # Declare the dataqueue payload contract so the test_pipeline e2e can
    # round-trip it and prove the schema reaches the manifest.
    dataqueue_payload_schema: ClassVar[Optional[dict]] = {
        "device_id": "wheel",
        "payload_format": "scalar",
        "payload_fields": {},
        "description": "Mock click count pushed by _run_loop().",
    }

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self.sample_interval_s: float = float(
            self.cfg.get("sample_interval_ms", 100)
        ) / 1000.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self.record(random.randint(1, 10))
            self._stop_event.wait(self.sample_interval_s)

    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"MockEncoder-{self.device_id}",
            daemon=True,
        )
        self._thread.start()
        return super().start()

    def stop(self) -> bool:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None
        return super().stop()


# ---------------------------------------------------------------------------
# Mock treadmill
# ---------------------------------------------------------------------------


class MockTreadmillDevice(BaseDataProducer):
    """Synthetic treadmill that mimics the real :class:`EncoderSerialInterface`.

    Emits the same ``{"distance", "speed", "device_us"}`` dict payload as the
    Teensy treadmill, with a monotonic microsecond ``device_us`` clock and a
    smoothly varying speed.  This lets the MousePortal integration run end to
    end without hardware *and* exercises the ``device_us`` master-clock
    alignment path used by ``TreadmillSource`` / ``MousePortalSource``.

    YAML registration via ``type: mock_treadmill``::

        treadmill:
          type: mock_treadmill
          sample_interval_ms: 20
          speed_mm_s: 8.0
    """

    device_type: ClassVar[str] = "encoder"
    file_type: ClassVar[str] = "csv"
    bids_type: ClassVar[Optional[str]] = "beh"
    data_type: ClassVar[str] = "treadmill"

    # Mirror the real treadmill's dataqueue contract so the manifest + parser
    # round-trip identically (see EncoderSerialInterface.dataqueue_payload_schema).
    dataqueue_payload_schema: ClassVar[Optional[dict]] = {
        "device_id": "treadmill",
        "payload_format": "dict",
        "payload_fields": {
            "distance": "float",
            "speed": "float",
            "device_us": "int",
        },
        "description": "Mock dict payload; device_us is the master clock anchor.",
    }

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        if not self.cfg.get("id") and not self.cfg.get("device_id"):
            self.device_id = "treadmill"
        self.sample_interval_s: float = float(
            self.cfg.get("sample_interval_ms", 20)
        ) / 1000.0
        self._peak_speed: float = float(self.cfg.get("speed_mm_s", 8.0))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run_loop(self) -> None:
        import math

        t0 = time.perf_counter()
        distance = 0.0
        while not self._stop_event.is_set():
            elapsed = time.perf_counter() - t0
            # Smooth, always-forward locomotion (half-rectified sine).
            speed = self._peak_speed * (0.5 + 0.5 * math.sin(elapsed))
            distance += speed * self.sample_interval_s
            device_us = int(elapsed * 1_000_000)
            self.record({
                "distance": round(distance, 4),
                "speed": round(speed, 4),
                "device_us": device_us,
            })
            self._stop_event.wait(self.sample_interval_s)

    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"MockTreadmill-{self.device_id}",
            daemon=True,
        )
        self._thread.start()
        return super().start()

    def stop(self) -> bool:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None
        return super().stop()


# ---------------------------------------------------------------------------
# Mock camera
# ---------------------------------------------------------------------------


class MockFrameProducer(BaseCamera, BaseDataProducer):
    """Synthetic camera producing real OME-TIFF + frame metadata JSON."""

    device_type: ClassVar[str] = "camera"
    file_type: ClassVar[str] = "ome.tiff"
    bids_type: ClassVar[Optional[str]] = "func"
    data_type: ClassVar[str] = "frames"
    clock_source: ClassVar[str] = "wall_unix_s"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        BaseDataProducer.__init__(self, cfg, **kwargs)
        self._init_camera_surface(self.cfg, backend="mock")
        self.width: int = int(self.cfg.get("width", 32))
        self.height: int = int(self.cfg.get("height", 32))
        self.frame_interval_s: float = float(self.cfg.get("frame_interval_ms", 50)) / 1000.0
        if self.frame_interval_s:
            self.sampling_rate = 1.0 / self.frame_interval_s
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frames: list[np.ndarray] = []
        self._frames_lock = threading.Lock()
        self._frame_records: list[dict[str, Any]] = []
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
        BaseDataProducer.arm(self, config)
        self.set_sequence(config.build_sequence)
        self._frames.clear()
        self._frame_records.clear()

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
            self.record({"frame_index": len(self._frames) - 1}, ts=ts)
            if self._qt_image_adapter is not None:
                self._qt_image_adapter.emit_frame(frame)
            self._stop_event.wait(self.frame_interval_s)

    # --- live preview contract (BaseCamera abstract methods) ------------
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
        """Replay buffered frames through the standard mesofield writer."""
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

        self.output_path = str(target)
        self.writer = self._make_writer(target)
        self.metadata_path = getattr(
            self.writer, "_frame_metadata_filename", str(target) + "_frame_metadata.json",
        )

        import useq

        seq = useq.MDASequence(
            time_plan={"loops": len(frames), "interval": self.frame_interval_s},
        )
        self.writer.sequenceStarted(seq, {})
        try:
            for i, (frame, record) in enumerate(zip(frames, records)):
                event = useq.MDAEvent(index={"t": i})
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
