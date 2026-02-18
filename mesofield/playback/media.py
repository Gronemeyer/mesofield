from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

import numpy as np
import tifffile

if TYPE_CHECKING:  # pragma: no cover - only for typing
    from PyQt6.QtGui import QPixmap

try:  # Optional dependency
    import cv2
except Exception:  # pragma: no cover - handled gracefully at runtime
    cv2 = None


def _normalize_to_uint8(img: np.ndarray, clims: tuple[float, float] | str = "auto") -> np.ndarray:
    """Scale arbitrary image data to uint8 for display.

    Mirrors the contrast handling used in :mod:`mesofield.gui.viewer` to avoid
    washed-out or speckled rendering when displaying higher-bit-depth frames
    (e.g., 16-bit OME-TIFF stacks). When ``clims`` is ``"auto"``, the limits are
    derived per-frame from the min/max of the data.
    """

    if img is None:
        return img

    arr = img.astype(np.float32, copy=False)
    if clims == "auto":
        min_val, max_val = float(np.min(arr)), float(np.max(arr))
    else:
        min_val, max_val = clims

    scale = 255.0 / (max_val - min_val) if max_val != min_val else 255.0
    arr = np.clip((arr - min_val) * scale, 0, 255).astype(np.uint8, copy=False)
    return arr


@dataclass
class FrameSource:
    path: Path
    fps: float | None = None

    def frame_at_fraction(self, fraction: float) -> np.ndarray | None:  # pragma: no cover - interface
        raise NotImplementedError

    def to_pixmap(self, frame: np.ndarray, *, target_size) -> QPixmap | None:
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QImage, QPixmap

        if frame is None:
            return None
        if frame.ndim == 2:
            data = np.ascontiguousarray(_normalize_to_uint8(frame))
            h, w = data.shape
            qimage = QImage(data.data, w, h, w, QImage.Format.Format_Grayscale8)
        elif frame.ndim == 3:
            data = np.ascontiguousarray(_normalize_to_uint8(frame))
            h, w, _ = data.shape
            qimage = QImage(data.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        else:
            return None

        pixmap = QPixmap.fromImage(qimage)
        if target_size:
            pixmap = pixmap.scaled(
                target_size, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio
            )
        return pixmap


class TiffFrameSource(FrameSource):
    def __init__(self, path: str | Path, *, duration_hint: float | None = None) -> None:
        super().__init__(Path(path))
        self._data = tifffile.memmap(self.path)
        self._frame_count = self._data.shape[0]
        self._duration = duration_hint
        self._timestamps = self._load_frame_times()
        self.fps = self._estimate_fps()

    def _load_frame_times(self) -> np.ndarray | None:
        metadata_path = self.path.with_name(self.path.name.replace(".ome.tiff", ".ome.tiff_frame_metadata.json"))
        if not metadata_path.exists():
            return None
        try:
            meta = json.loads(metadata_path.read_text())
        except Exception:
            return None

        times: list[float] = []
        if isinstance(meta, Iterable) and not isinstance(meta, (str, bytes, dict)):
            for item in meta:
                if isinstance(item, dict):
                    if "ElapsedTime-ms" in item:
                        times.append(float(item["ElapsedTime-ms"]) / 1000.0)
                    elif "elapsed_ms" in item:
                        times.append(float(item["elapsed_ms"]) / 1000.0)
        return np.array(times) if times else None

    def _estimate_fps(self) -> float | None:
        if self._timestamps is not None and self._timestamps.size > 1:
            diffs = np.diff(self._timestamps)
            diffs = diffs[diffs > 0]
            if diffs.size:
                median = float(np.median(diffs))
                return 1.0 / median if median > 0 else None
        if self._duration and self._duration > 0:
            return float(self._frame_count) / float(self._duration)
        return None

    def frame_at_fraction(self, fraction: float) -> np.ndarray | None:
        fraction = max(0.0, min(1.0, fraction))
        if self._timestamps is not None and self._timestamps.size:
            target = fraction * (self._timestamps[-1] if self._duration is None else self._duration)
            idx = int(np.searchsorted(self._timestamps, target, side="right"))
        else:
            idx = int(fraction * self._frame_count)
        idx = min(self._frame_count - 1, max(0, idx))
        return np.asarray(self._data[idx])


class Mp4FrameSource(FrameSource):
    def __init__(self, path: str | Path, *, duration_hint: float | None = None) -> None:
        super().__init__(Path(path))
        self._frames: list[np.ndarray] = []
        self._duration = duration_hint
        self.fps = None
        self._load_frames()

    def _load_frames(self) -> None:
        if cv2 is None:
            return
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and fps > 0 else None
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._frames.append(frame)
        cap.release()
        if self.fps is None and self._duration and self._duration > 0 and self._frames:
            self.fps = float(len(self._frames)) / float(self._duration)

    def frame_at_fraction(self, fraction: float) -> np.ndarray | None:
        if not self._frames:
            return None
        fraction = max(0.0, min(1.0, fraction))
        idx = int(fraction * len(self._frames))
        idx = min(len(self._frames) - 1, max(0, idx))
        return self._frames[idx]


def discover_media_paths(root: Path) -> tuple[Path | None, Path | None, Path | None]:
    meso = next(root.rglob("*meso*.tif*"), None)
    pupil = next(root.rglob("*pupil*.mp4"), None)
    treadmill = next(root.rglob("*treadmill.csv"), None)
    return meso, pupil, treadmill


def load_treadmill_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load treadmill/encoder CSV data into time/value arrays.

    The loader is intentionally permissive: it looks for timestamp-ish columns
    (``time``, ``timestamp``, ``elapsed``, ``ms``) and value-ish columns
    (``speed``, ``value``, ``encoder``) and converts any ISO or numeric timestamp
    into seconds relative to the first sample.
    """

    import csv
    from datetime import datetime

    times: list[float] = []
    values: list[float] = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return np.array([], dtype=float), np.array([], dtype=float)

        time_candidates = [
            name
            for name in reader.fieldnames
            if name and any(token in name.lower() for token in ("time", "ts", "ms", "elapsed"))
        ]
        value_candidates = [
            name
            for name in reader.fieldnames
            if name and any(token in name.lower() for token in ("speed", "value", "encoder"))
        ]

        if not time_candidates or not value_candidates:
            return np.array([], dtype=float), np.array([], dtype=float)

        time_key = time_candidates[0]
        value_key = value_candidates[0]

        for row in reader:
            ts_raw = row.get(time_key)
            val_raw = row.get(value_key)
            if ts_raw is None or val_raw is None:
                continue
            try:
                if any(ch in str(ts_raw) for ch in ("-", ":", "T")):
                    ts = datetime.fromisoformat(str(ts_raw)).timestamp()
                else:
                    ts = float(ts_raw)
                val = float(val_raw)
            except Exception:
                continue
            times.append(ts)
            values.append(val)

    if not times:
        return np.array([], dtype=float), np.array([], dtype=float)

    start = times[0]
    norm_times = np.array([t - start for t in times], dtype=float)
    norm_vals = np.array(values, dtype=float)
    return norm_times, norm_vals

