"""Synthetic camera device for headless / GUI-only iteration.

Mirrors the real cameras' contract:

  - Writes an actual OME-TIFF stack on `save_data`, so downstream tools
    (tifffile, mesomap, your own processors) get a real file to load.
  - Writes a `<output_path>_frame_metadata.json` sidecar with one record
    per frame (frame index, wall-clock receive time).
  - Sets `self.metadata_path` so the AcquisitionManifest writer picks it
    up via the standard `ProducerEntry.metadata_path` path.

Where the real MMCamera relies on micromanager to drive frame timing,
this class fires a synthetic frame every `frame_interval_ms` on a
daemon thread.

YAML registration via `type: mock_camera`:

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

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

import numpy as np

from mesofield.devices.base import BaseDataProducer


class MockFrameProducer(BaseDataProducer):
    """Synthetic camera producing real OME-TIFF + frame metadata JSON."""

    device_type: ClassVar[str] = "camera"
    file_type: ClassVar[str] = "ome.tiff"
    bids_type: ClassVar[Optional[str]] = "func"
    data_type: ClassVar[str] = "frames"
    clock_source: ClassVar[str] = "wall_unix_s"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self.width: int = int(self.cfg.get("width", 32))
        self.height: int = int(self.cfg.get("height", 32))
        self.frame_interval_s: float = float(self.cfg.get("frame_interval_ms", 50)) / 1000.0
        self.sampling_rate = 1.0 / self.frame_interval_s if self.frame_interval_s else 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frames: list[np.ndarray] = []
        self._frames_lock = threading.Lock()
        self._frame_records: list[dict[str, Any]] = []

    # ---- lifecycle ------------------------------------------------------
    def arm(self, config: Any) -> None:
        super().arm(config)
        self._frames.clear()
        self._frame_records.clear()
        # NOTE: don't compute metadata_path here. DataSaver rewrites the
        # final save path using path_args['suffix'] at save time, which
        # can differ from self.output_path computed during arm(). The
        # sidecar is derived from the final target inside save_data().

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
        return super().start()

    def stop(self) -> bool:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None
        return super().stop()

    def _run_loop(self) -> None:
        rng = np.random.default_rng(seed=0)
        while not self._stop_event.is_set():
            ts = time.time()
            # Quadrant intensity baseline so a "regional means" processor
            # downstream sees a structured signal, not just noise.
            frame = self._synthesise_frame(rng, len(self._frames))
            with self._frames_lock:
                self._frames.append(frame)
                self._frame_records.append({
                    "frame_index": len(self._frames) - 1,
                    "TimeReceivedByCore": datetime.fromtimestamp(ts, tz=timezone.utc)
                                                    .isoformat().replace("+00:00", "Z"),
                })
            self.record({"frame_index": len(self._frames) - 1}, ts=ts)
            self._stop_event.wait(self.frame_interval_s)

    def _synthesise_frame(self, rng: np.random.Generator, frame_index: int) -> np.ndarray:
        # Four-quadrant baseline; each quadrant has a different mean so a
        # 4-region processor downstream produces 4 distinguishable traces.
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
        target = path or self.output_path
        if not target:
            self.logger.debug("save_data: no path for %s", self.device_id)
            return None
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        # Anchor the sidecar to the final target so the manifest's
        # metadata_path matches the on-disk filename exactly.
        self.output_path = str(target)
        self.metadata_path = str(target) + "_frame_metadata.json"

        with self._frames_lock:
            frames = list(self._frames)
            records = list(self._frame_records)

        if not frames:
            self.logger.warning("MockFrameProducer captured 0 frames")
            return None

        stack = np.stack(frames)
        try:
            import tifffile
        except ImportError:  # pragma: no cover
            raise ImportError(
                "tifffile is required to write MockFrameProducer output. "
                "Install with `pip install tifffile` (it's already in mesofield[rig])."
            )
        tifffile.imwrite(target, stack)
        self.logger.info("Wrote %d frames to %s (shape=%s)", len(frames), target, stack.shape)

        if self.metadata_path:
            with open(self.metadata_path, "w", encoding="utf-8") as fh:
                json.dump({"p0": records}, fh, indent=2)
            self.logger.info("Wrote %d frame metadata records to %s",
                             len(records), self.metadata_path)

        return target

    # ---- manifest hint --------------------------------------------------
    @property
    def calibration(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "frame_interval_ms": int(self.frame_interval_s * 1000),
        }
