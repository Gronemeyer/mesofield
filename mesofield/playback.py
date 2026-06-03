"""Seek-based playback model for recorded mesofield sessions.

This module is **headless** (no Qt). It discovers recordings under an
experiment directory and loads a chosen ``(subject, session, task)`` cell into a
:class:`PlaybackSession` of lazily-read :class:`CameraStream` objects plus an
optional :class:`TreadmillTrack`, all placed on one **elapsed-seconds**
timeline. The GUI (:mod:`mesofield.gui.playback_window`) renders by calling
:meth:`PlaybackSession.seek` with a timeline position; nothing is pushed through
the live-acquisition pipeline, so playback is fully decoupled and read-only.

File detection is **manifest-driven**:

- ``data/sub-*/ses-*/manifest.json`` for single-task sessions, or
- ``manifest_task-<T>.json`` files for multi-task sessions, or
- when neither exists, manifests are synthesized *in memory* via
  :func:`mesofield.utils.retrofit.synthesize_manifests` so un-retrofitted
  folders play too.

Treadmill data is reconstructed from the master ``*_dataqueue.csv`` via
:class:`mesofield.datakit.sources.behavior.treadmill.TreadmillSource`; the
per-session ``treadmill.csv`` is **never** used.

Every stream loads independently: a missing/corrupt camera file becomes a
placeholder, a missing dataqueue simply hides the treadmill track.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from mesofield.utils._logger import get_logger
from mesofield.utils.retrofit import (
    discover_sessions as _discover_session_dirs,
    synthesize_manifests,
)

logger = get_logger(__name__)


__all__ = [
    "CameraStream",
    "TreadmillTrack",
    "FrameResult",
    "SessionRef",
    "PlaybackSession",
    "discover_recordings",
    "iter_refs",
    "default_ref",
    "load_session",
]


_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
_VIDEO_FILE_TYPES = {"mp4", "avi", "mov", "mkv"}
_SUPPORTED_CAMERA_TYPES = {
    "ome.tiff", "ome.tif", "tiff", "tif", *_VIDEO_FILE_TYPES,
}


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------


def _parse_iso_to_unix(value: str) -> float:
    """Parse an ISO-8601 timestamp (``...Z`` or offset) to UNIX seconds."""
    from datetime import datetime

    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).timestamp()


def _norm_rel(value: Optional[str]) -> Optional[str]:
    """Normalize a manifest-relative path (manifests store Windows ``\\``)."""
    if not value:
        return None
    return str(value).replace("\\", "/")


def _load_frame_timestamps(meta_path: Path) -> List[float]:
    """Return per-frame timestamps (seconds) from a ``*_frame_metadata.json``.

    Handles every on-disk shape mesofield writers produce. The returned values
    may be wall-clock UNIX seconds (OME-TIFF) or ``perf_counter`` seconds
    (OpenCV ``mp4``); callers only use *relative* spacing, so the clock origin
    does not matter.

    - ``OpenCVCamera`` direct capture: ``{"frames": [{"index", "device_ts"}, ...]}``
    - ``CustomWriter`` (OME-TIFF) / ``CV2Writer`` MDA mode: ``{"p0": [...]}``
      keyed by position, timestamps under ``TimeReceivedByCore``.
    - ``CV2Writer.finish`` wraps the latter under ``{"frame_metadatas": {...}}``.
    - A bare ``[{...}, ...]`` list is tolerated.
    """
    with meta_path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)

    # OpenCV direct-capture sidecar: top-level "frames" list with perf_counter.
    if isinstance(records, dict) and isinstance(records.get("frames"), list):
        out: List[float] = []
        for entry in records["frames"]:
            if not isinstance(entry, dict):
                continue
            ts = entry.get("device_ts", entry.get("timestamp"))
            if ts is None:
                continue
            try:
                out.append(float(ts))
            except (TypeError, ValueError):
                continue
        if out:
            return out

    # Unwrap CV2Writer.finish's outer envelope.
    if isinstance(records, dict) and "frame_metadatas" in records:
        records = records["frame_metadatas"]

    timestamps: List[float] = []
    if isinstance(records, dict):
        for _pos_key, entries in records.items():
            for entry in entries or []:
                ts_raw = entry.get("TimeReceivedByCore") if isinstance(entry, dict) else None
                if ts_raw is None:
                    continue
                timestamps.append(_parse_iso_to_unix(ts_raw))
    elif isinstance(records, list):
        for entry in records:
            ts_raw = entry.get("TimeReceivedByCore") if isinstance(entry, dict) else None
            if ts_raw is None:
                continue
            timestamps.append(_parse_iso_to_unix(ts_raw))
    return timestamps


def _frame_timeline(
    ts_list: List[float], n_frames: int, sampling_rate: float
) -> np.ndarray:
    """Map per-frame timestamps to a monotonic elapsed-seconds axis of length ``n_frames``.

    Falls back to uniform spacing from ``sampling_rate`` (default 30 fps) when
    timestamps are missing. When fewer timestamps than frames are present the
    tail is extrapolated at the median frame interval.
    """
    n_frames = max(int(n_frames), 0)
    if n_frames == 0:
        return np.zeros(0, dtype=np.float64)

    ts = np.asarray([t for t in ts_list if t is not None], dtype=np.float64)
    ts = ts[np.isfinite(ts)]
    if ts.size >= 2:
        elapsed = np.maximum.accumulate(ts - ts[0])  # enforce non-decreasing
        if elapsed.size >= n_frames:
            return elapsed[:n_frames]
        # Extrapolate the tail at the median interval.
        diffs = np.diff(elapsed)
        dt = float(np.median(diffs)) if diffs.size else 0.0
        if not math.isfinite(dt) or dt <= 0:
            dt = 1.0 / sampling_rate if sampling_rate > 0 else 1.0 / 30.0
        pad = elapsed[-1] + dt * np.arange(1, n_frames - elapsed.size + 1)
        return np.concatenate([elapsed, pad])

    fps = sampling_rate if sampling_rate and sampling_rate > 0 else 30.0
    return np.arange(n_frames, dtype=np.float64) / fps


# ---------------------------------------------------------------------------
# Frame reader (OME-TIFF memmap + cv2 video with random-access seek)
# ---------------------------------------------------------------------------


class _FrameReader:
    """Uniform random-access reader over a recorded camera output.

    Hides the difference between ``tifffile``-readable stacks (O(1) page
    indexing) and ``cv2.VideoCapture``-readable videos (seek to the requested
    frame). Open with :meth:`open` and release with :meth:`close`; also usable
    as a context manager.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.kind = "video" if self.path.suffix.lower() in _VIDEO_SUFFIXES else "tiff"
        self.n_frames: int = 0
        self._tf: Any = None
        self._pages: Optional[List[Any]] = None
        self._cap: Any = None
        self._next_video_idx: int = 0
        self._linear_seek_window: int = 8

    def open(self) -> "_FrameReader":
        if self.kind == "tiff":
            import tifffile

            self._tf = tifffile.TiffFile(str(self.path))
            # Keep the lazy TiffPages sequence (don't materialize every page
            # object) so opening a multi-thousand-frame stack stays cheap.
            self._pages = self._tf.pages
            self.n_frames = len(self._tf.pages)
        else:
            import cv2

            self._cap = cv2.VideoCapture(str(self.path))
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open video {self.path}")
            count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.n_frames = max(count, 0)
            self._next_video_idx = 0
        return self

    def __enter__(self) -> "_FrameReader":
        return self.open()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._tf is not None:
            try:
                self._tf.close()
            except Exception:
                pass
            self._tf = None
        self._pages = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def read(self, idx: int) -> Optional[np.ndarray]:
        """Return frame ``idx`` (0-based) as an ndarray, or ``None`` if absent."""
        if idx < 0:
            return None
        if self.kind == "tiff":
            if self._pages is None or idx >= len(self._pages):
                return None
            return np.asarray(self._pages[idx].asarray())
        if self._cap is None:
            return None
        import cv2

        # Small forward jumps are cheaper as linear decode than repeated random
        # CAP_PROP seeks, which can force heavy keyframe re-decode.
        if idx > self._next_video_idx:
            gap = idx - self._next_video_idx
            if gap <= self._linear_seek_window:
                frame = None
                for _ in range(gap + 1):
                    ret, frame = self._cap.read()
                    if not ret:
                        return None
                self._next_video_idx = idx + 1
                return frame

        # Backward jumps or large forward jumps still use explicit random seek.
        if idx != self._next_video_idx:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if not ret:
            return None
        self._next_video_idx = idx + 1
        return frame  # BGR for color video; the GUI converts to RGB


# ---------------------------------------------------------------------------
# Camera path resolution (handles stale manifest output_path)
# ---------------------------------------------------------------------------


def _resolve_camera_path(entry: Dict[str, Any], session_dir: Path) -> Optional[Path]:
    """Resolve a camera producer's frames file on disk.

    Some writers name the file by the device's display name while the manifest
    ``output_path`` is populated from the YAML ``suffix:`` — the two diverge and
    ``output_path`` ends up pointing at a file that was never written (e.g. the
    mesoscope's ``_meso.ome.tiff`` vs the real ``_mesoscope.ome.tiff``). Fall
    back to ``metadata_path`` minus the ``_frame_metadata.json`` tail, which
    matches the writer's actual filename.
    """
    output_path = _norm_rel(entry.get("output_path"))
    if output_path:
        abs_output = (session_dir / output_path).resolve()
        if abs_output.is_file():
            return abs_output

    meta_path = _norm_rel(entry.get("metadata_path"))
    if meta_path and meta_path.endswith("_frame_metadata.json"):
        derived = (session_dir / meta_path[: -len("_frame_metadata.json")]).resolve()
        if derived.is_file():
            logger.warning(
                "%s: manifest output_path %r not on disk; using %s derived "
                "from metadata_path",
                entry.get("device_id"), output_path, derived.name,
            )
            return derived
    return None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CameraStream:
    """One recorded camera, lazily read and placed on the elapsed-seconds axis."""

    device_id: str
    file_type: str
    is_video: bool
    frames_path: Optional[Path]
    reader: Optional[_FrameReader]
    frame_times_s: np.ndarray  # elapsed seconds per frame, monotonic, len == n_frames
    n_frames: int
    bids_type: Optional[str] = None
    error: Optional[str] = None
    _last_idx: int = field(default=-1, repr=False)
    _last_frame: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def duration_s(self) -> float:
        if self.frame_times_s.size:
            return float(self.frame_times_s[-1])
        return 0.0


@dataclass
class TreadmillTrack:
    """Sparse treadmill samples reconstructed from the master dataqueue."""

    time_s: np.ndarray      # elapsed seconds (relative to the camera window)
    speed: np.ndarray       # mm/s
    distance: np.ndarray    # mm
    source_file: Optional[str] = None

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1]) if self.time_s.size else 0.0

    def sample_at(self, t_s: float) -> Optional[Dict[str, float]]:
        """Return the most recent sample at or before ``t_s`` (hold-last)."""
        if not self.time_s.size:
            return None
        idx = int(np.searchsorted(self.time_s, t_s, side="right") - 1)
        if idx < 0:
            return None
        return {
            "time_s": float(self.time_s[idx]),
            "speed": float(self.speed[idx]),
            "distance": float(self.distance[idx]),
        }


@dataclass
class FrameResult:
    """A single camera's frame at a seek position."""

    device_id: str
    index: int
    frame: Optional[np.ndarray]
    time_s: float


@dataclass
class SessionRef:
    """A selectable ``(subject, session, task)`` cell + its manifest."""

    subject: str
    session: str
    task: str
    session_dir: Path
    manifest: Dict[str, Any]

    @property
    def label(self) -> str:
        return f"sub-{self.subject} / ses-{self.session} / task-{self.task}"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


_MANIFEST_TASK_RE = re.compile(r"manifest_task-(?P<task>.+)\.json$")


def _manifest_to_dict(manifest: Any) -> Dict[str, Any]:
    """Coerce a ``mesokit_schema.AcquisitionManifest`` (pydantic) to a plain dict."""
    for attempt in (
        lambda: manifest.model_dump(mode="json"),       # pydantic v2
        lambda: json.loads(manifest.to_json()),          # schema helper
        lambda: manifest.model_dump(),
    ):
        try:
            result = attempt()
            if isinstance(result, dict):
                return result
        except Exception:
            continue
    raise TypeError(f"Cannot serialize manifest of type {type(manifest).__name__}")


def _task_of(manifest: Dict[str, Any], fallback: str = "session") -> str:
    session = manifest.get("session") or {}
    task = session.get("task")
    return str(task) if task else fallback


def _refs_for_session_dir(session_dir: Path) -> List[SessionRef]:
    """Build one :class:`SessionRef` per task available in ``session_dir``.

    Prefers on-disk manifests; if none exist, synthesizes them in memory via
    the retrofit helper so un-retrofitted sessions are still playable.
    """
    subject = session_dir.parent.name.removeprefix("sub-")
    session = session_dir.name.removeprefix("ses-")
    refs: List[SessionRef] = []

    single = session_dir / "manifest.json"
    per_task = sorted(session_dir.glob("manifest_task-*.json"))

    manifests: List[Dict[str, Any]] = []
    if single.is_file():
        try:
            with single.open("r", encoding="utf-8") as fh:
                manifests.append(json.load(fh))
        except Exception as exc:
            logger.warning("Failed to read %s: %s", single, exc)
    elif per_task:
        for path in per_task:
            try:
                with path.open("r", encoding="utf-8") as fh:
                    manifests.append(json.load(fh))
            except Exception as exc:
                logger.warning("Failed to read %s: %s", path, exc)
    else:
        # No manifest on disk — synthesize in memory (no files written).
        try:
            synthesized = synthesize_manifests(session_dir)
        except Exception as exc:
            logger.warning("Manifest synthesis failed for %s: %s", session_dir, exc)
            synthesized = {}
        for man in synthesized.values():
            try:
                manifests.append(_manifest_to_dict(man))
            except Exception as exc:
                logger.warning("Could not serialize synthesized manifest: %s", exc)

    for manifest in manifests:
        refs.append(
            SessionRef(
                subject=subject,
                session=session,
                task=_task_of(manifest),
                session_dir=session_dir,
                manifest=manifest,
            )
        )
    return refs


def discover_recordings(
    experiment_dir: Path | str,
) -> Dict[str, Dict[str, Dict[str, SessionRef]]]:
    """Index an experiment directory as ``{subject: {session: {task: SessionRef}}}``.

    Accepts an experiment root (containing ``data/sub-*/ses-*``), a subject or
    session directory — anything :func:`retrofit.discover_sessions` understands.
    """
    experiment_dir = Path(experiment_dir).expanduser().resolve()
    index: Dict[str, Dict[str, Dict[str, SessionRef]]] = {}
    for session_dir in _discover_session_dirs(experiment_dir):
        for ref in _refs_for_session_dir(session_dir):
            index.setdefault(ref.subject, {}).setdefault(ref.session, {})[ref.task] = ref
    return index


def iter_refs(
    index: Dict[str, Dict[str, Dict[str, SessionRef]]]
) -> List[SessionRef]:
    """Flatten a discovery index to a sorted list of refs."""
    refs: List[SessionRef] = []
    for subject in sorted(index):
        for session in sorted(index[subject]):
            for task in sorted(index[subject][session]):
                refs.append(index[subject][session][task])
    return refs


def default_ref(
    index: Dict[str, Dict[str, Dict[str, SessionRef]]]
) -> Optional[SessionRef]:
    """Return the first ref in the index, or ``None`` when empty."""
    refs = iter_refs(index)
    return refs[0] if refs else None


# ---------------------------------------------------------------------------
# Stream loading
# ---------------------------------------------------------------------------


def _build_camera_streams(
    manifest: Dict[str, Any],
    session_dir: Path,
    dataqueue: Optional["object"] = None,
    t0: float = 0.0,
) -> List[CameraStream]:
    """Build camera streams, placing frames on the dataqueue master clock.

    Each frame's time comes from the camera's ``queue_elapsed`` rows in the
    dataqueue (minus the shared ``t0``), so every camera and the treadmill share
    one timeline. The per-frame metadata sidecar is only a fallback when the
    camera has no rows in the dataqueue.
    """
    streams: List[CameraStream] = []
    dq_cam_counts = _dataqueue_camera_devices(dataqueue)
    used_dq_devices: set = set()
    for entry in manifest.get("producers", []) or []:
        if entry.get("device_type") != "camera":
            continue
        file_type = (entry.get("file_type") or "").lower()
        if file_type not in _SUPPORTED_CAMERA_TYPES:
            logger.debug("Skipping %s: unsupported file_type %r",
                         entry.get("device_id"), file_type)
            continue
        device_id = str(entry.get("device_id") or "camera")
        is_video = file_type in _VIDEO_FILE_TYPES

        frames_path = _resolve_camera_path(entry, session_dir)
        if frames_path is None:
            streams.append(CameraStream(
                device_id=device_id, file_type=file_type, is_video=is_video,
                frames_path=None, reader=None,
                frame_times_s=np.zeros(0), n_frames=0,
                bids_type=entry.get("bids_type"),
                error="frames file not found",
            ))
            continue

        try:
            reader = _FrameReader(frames_path).open()
        except Exception as exc:
            logger.error("Failed to open %s: %s", frames_path, exc)
            streams.append(CameraStream(
                device_id=device_id, file_type=file_type, is_video=is_video,
                frames_path=frames_path, reader=None,
                frame_times_s=np.zeros(0), n_frames=0,
                bids_type=entry.get("bids_type"),
                error=f"could not open: {exc}",
            ))
            continue

        meta_rel = _norm_rel(entry.get("metadata_path"))
        meta_path = (session_dir / meta_rel).resolve() if meta_rel else None
        ts: List[float] = []
        if meta_path and meta_path.is_file():
            try:
                ts = _load_frame_timestamps(meta_path)
            except Exception as exc:
                logger.warning("Failed to read sidecar %s: %s", meta_path, exc)

        sampling_rate = float(entry.get("sampling_rate_hz") or 0.0)
        n_frames = reader.n_frames

        # Match this camera to its dataqueue device (ids may differ) and take
        # per-frame times from the master clock; fall back to the sidecar.
        dq_dev = _resolve_dq_device(device_id, n_frames, dq_cam_counts, used_dq_devices)
        dq_times = _camera_times_from_dataqueue(dataqueue, dq_dev, t0) if dq_dev else None
        if dq_dev and dq_times is not None and dq_times.size:
            used_dq_devices.add(dq_dev)

        if n_frames <= 0:
            n_frames = int(dq_times.size) if dq_times is not None else len(ts)

        if dq_times is not None and dq_times.size:
            times = _reconcile_times(dq_times, n_frames, sampling_rate)
            source = f"dataqueue[{dq_dev}]"
        else:
            # Fallback: the camera's own frame metadata, relative to its first
            # frame. Note this no longer shares the master t=0 exactly.
            times = _frame_timeline(ts, n_frames, sampling_rate)
            source = "sidecar" if ts else "uniform"
        logger.debug("camera %s: %d frames, timeline from %s", device_id, n_frames, source)

        streams.append(CameraStream(
            device_id=device_id, file_type=file_type, is_video=is_video,
            frames_path=frames_path, reader=reader,
            frame_times_s=times, n_frames=n_frames,
            bids_type=entry.get("bids_type"),
        ))
    return streams


def _find_dataqueue(session_dir: Path) -> Optional[Path]:
    """Locate the master ``*_dataqueue.csv`` for a session."""
    beh = sorted((session_dir / "beh").glob("*_dataqueue.csv"))
    if beh:
        return beh[0]
    deep = sorted(session_dir.rglob("*_dataqueue.csv"))
    return deep[0] if deep else None


_NUM = r"(-?\d+(?:\.\d+)?)"
# Tolerate ``distance=1.5`` (EncoderData repr) and ``'distance': 1.5`` (dict repr).
_DIST_RE = re.compile(r"distance['\"]?\s*[=:]\s*" + _NUM, re.IGNORECASE)
_SPEED_RE = re.compile(r"speed['\"]?\s*[=:]\s*" + _NUM, re.IGNORECASE)
_MASTER_CAMERA_RE = r"dhyana|mesoscope"


def _read_dataqueue(session_dir: Path):
    """Read the session's master ``*_dataqueue.csv`` once (or return ``None``)."""
    dq_path = _find_dataqueue(session_dir)
    if dq_path is None:
        return None
    try:
        import pandas as pd

        df = pd.read_csv(dq_path, low_memory=False)
    except Exception as exc:
        logger.warning("Failed to read dataqueue %s (%s)", dq_path, exc)
        return None
    if "queue_elapsed" not in df.columns:
        logger.warning("Dataqueue %s has no 'queue_elapsed' column", dq_path.name)
        return None
    df.attrs["source_file"] = str(dq_path)
    return df


def _master_t0(dataqueue) -> float:
    """Master-clock zero: the first master-camera frame, else the first row."""
    if dataqueue is None:
        return 0.0
    import pandas as pd

    q = pd.to_numeric(dataqueue["queue_elapsed"], errors="coerce")
    if "device_id" in dataqueue.columns:
        cam = dataqueue["device_id"].astype(str).str.contains(
            _MASTER_CAMERA_RE, case=False, na=False, regex=True
        )
        cam_q = q[cam].dropna()
        if len(cam_q):
            return float(cam_q.min())
    q = q.dropna()
    return float(q.min()) if len(q) else 0.0


def _camera_times_from_dataqueue(
    dataqueue, device_id: str, t0: float
) -> Optional[np.ndarray]:
    """Per-frame elapsed seconds for ``device_id`` from the master dataqueue."""
    if dataqueue is None or "device_id" not in dataqueue.columns:
        return None
    import pandas as pd

    mask = dataqueue["device_id"].astype(str) == str(device_id)
    if not mask.any():
        return None
    q = pd.to_numeric(dataqueue.loc[mask, "queue_elapsed"], errors="coerce").dropna()
    if q.empty:
        return None
    return np.sort(q.to_numpy(dtype=np.float64)) - t0


# Dataqueue device_ids that are not cameras (so they're never count-matched to
# a camera producer).
_NONCAMERA_DEV_RE = re.compile(r"tread|encoder|wheel|nidaq|daq|psychopy|arduino|teensy", re.IGNORECASE)


def _dataqueue_camera_devices(dataqueue) -> Dict[str, int]:
    """Map camera-like dataqueue ``device_id`` -> row (frame) count."""
    if dataqueue is None or "device_id" not in dataqueue.columns:
        return {}
    counts = dataqueue["device_id"].astype(str).value_counts()
    return {
        str(dev): int(n)
        for dev, n in counts.items()
        if not _NONCAMERA_DEV_RE.search(str(dev))
    }


def _resolve_dq_device(
    device_id: str,
    n_frames: int,
    dq_cam_counts: Dict[str, int],
    used: set,
) -> Optional[str]:
    """Match a manifest camera to its dataqueue device_id.

    The manifest device_id often differs from the dataqueue's (e.g. synthesized
    manifests use the filename suffix ``mesoscope`` while the dataqueue logged
    the hardware id ``Dhyana``). Prefer an exact id match, then the unused
    camera device whose frame count is closest to ``n_frames``.
    """
    if not dq_cam_counts:
        return None
    if device_id in dq_cam_counts and device_id not in used:
        return device_id
    if n_frames and n_frames > 0:
        best, best_diff = None, None
        for dev, count in dq_cam_counts.items():
            if dev in used:
                continue
            diff = abs(count - n_frames)
            if best_diff is None or diff < best_diff:
                best, best_diff = dev, diff
        if best is not None and best_diff is not None and best_diff <= max(50, int(0.02 * n_frames)):
            return best
    return None


def _reconcile_times(times: np.ndarray, n_frames: int, sampling_rate: float) -> np.ndarray:
    """Force a monotonic timeline of exactly ``n_frames`` entries.

    Truncates when there are more timestamps than frames; extrapolates the tail
    at the median interval (or ``sampling_rate``) when there are fewer.
    """
    times = np.maximum.accumulate(np.asarray(times, dtype=np.float64))
    n_frames = max(int(n_frames), 0)
    if n_frames == 0:
        return np.zeros(0, dtype=np.float64)
    if times.size >= n_frames:
        return times[:n_frames]
    diffs = np.diff(times)
    dt = float(np.median(diffs)) if diffs.size else 0.0
    if not math.isfinite(dt) or dt <= 0:
        dt = 1.0 / sampling_rate if sampling_rate > 0 else 1.0 / 30.0
    start = times[-1] if times.size else 0.0
    pad = start + dt * np.arange(1, n_frames - times.size + 1)
    return np.concatenate([times, pad]) if times.size else pad


def _treadmill_from_dataqueue(dataqueue, t0: float) -> Optional[TreadmillTrack]:
    """Extract sparse treadmill samples from the already-read dataqueue.

    Anchored to the shared master ``t0`` so the track lines up with the cameras.
    Handles fanned ``distance``/``speed`` columns and dict-/``EncoderData``-repr
    payload strings. ``device_ts`` is intentionally ignored (it is only good for
    relative spacing, not absolute time). Returns ``None`` when the dataqueue
    carries no encoder samples.
    """
    if dataqueue is None:
        return None
    import pandas as pd

    df = dataqueue
    q = pd.to_numeric(df["queue_elapsed"], errors="coerce").to_numpy(dtype=np.float64)
    cols = {c.lower(): c for c in df.columns}

    dist_col = next((cols[c] for c in ("distance_mm", "distance") if c in cols), None)
    speed_col = next((cols[c] for c in ("speed_mm", "speed_mm_s", "speed") if c in cols), None)
    if dist_col is not None or speed_col is not None:
        dist = (pd.to_numeric(df[dist_col], errors="coerce").to_numpy(np.float64)
                if dist_col else np.full(len(df), np.nan))
        speed = (pd.to_numeric(df[speed_col], errors="coerce").to_numpy(np.float64)
                 if speed_col else np.full(len(df), np.nan))
    elif "payload" in df.columns:
        text = df["payload"].astype(str)
        dist = pd.to_numeric(text.str.extract(_DIST_RE, expand=False), errors="coerce").to_numpy(np.float64)
        speed = pd.to_numeric(text.str.extract(_SPEED_RE, expand=False), errors="coerce").to_numpy(np.float64)
    else:
        return None

    keep = np.isfinite(q) & (np.isfinite(dist) | np.isfinite(speed))
    if not keep.any():
        return None

    time_s = q[keep] - t0
    order = np.argsort(time_s, kind="stable")
    return TreadmillTrack(
        time_s=time_s[order],
        speed=np.nan_to_num(speed[keep])[order],
        distance=np.nan_to_num(dist[keep])[order],
        source_file=str(df.attrs.get("source_file", "")),
    )


def _manifest_duration(manifest: Dict[str, Any]) -> Optional[float]:
    started, ended = manifest.get("started_at"), manifest.get("ended_at")
    if started and ended:
        try:
            return _parse_iso_to_unix(ended) - _parse_iso_to_unix(started)
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# PlaybackSession
# ---------------------------------------------------------------------------


@dataclass
class PlaybackSession:
    """A loaded recording ready for seeking. Call :meth:`close` when done."""

    ref: SessionRef
    cameras: List[CameraStream]
    treadmill: Optional[TreadmillTrack]
    duration_s: float

    def frame_index_at(self, stream: CameraStream, t_s: float) -> int:
        """Index of the last frame whose elapsed time is <= ``t_s`` (clamped)."""
        if stream.n_frames <= 0:
            return -1
        idx = int(np.searchsorted(stream.frame_times_s, t_s, side="right") - 1)
        return max(0, min(idx, stream.n_frames - 1))

    def seek(self, t_s: float) -> List[FrameResult]:
        """Return the current frame of every camera at timeline position ``t_s``."""
        results: List[FrameResult] = []
        for stream in self.cameras:
            if stream.error or stream.reader is None or stream.n_frames <= 0:
                results.append(FrameResult(stream.device_id, -1, None, t_s))
                continue
            idx = self.frame_index_at(stream, t_s)
            if idx == stream._last_idx and stream._last_frame is not None:
                frame = stream._last_frame
            else:
                frame = stream.reader.read(idx)
                if frame is not None:
                    stream._last_idx = idx
                    stream._last_frame = frame
            results.append(FrameResult(stream.device_id, idx, frame, t_s))
        return results

    def close(self) -> None:
        for stream in self.cameras:
            if stream.reader is not None:
                stream.reader.close()
                stream.reader = None


def load_session(ref: SessionRef) -> PlaybackSession:
    """Materialize a :class:`SessionRef` into a seekable :class:`PlaybackSession`.

    The session's ``*_dataqueue.csv`` is read once and used as the master clock:
    camera frames and treadmill samples are both placed relative to the same
    ``t0`` (the first master-camera frame), so they stay synchronized.
    """
    dataqueue = _read_dataqueue(ref.session_dir)
    t0 = _master_t0(dataqueue)

    cameras = _build_camera_streams(ref.manifest, ref.session_dir, dataqueue, t0)
    treadmill = _treadmill_from_dataqueue(dataqueue, t0)

    duration = _manifest_duration(ref.manifest)
    if duration is None or duration <= 0:
        candidates = [c.duration_s for c in cameras if c.n_frames > 0]
        if treadmill is not None:
            candidates.append(treadmill.duration_s)
        duration = max(candidates) if candidates else 0.0

    playable = sum(1 for c in cameras if c.error is None)
    logger.info(
        "Loaded %s: %d/%d camera stream(s), treadmill=%s, duration=%.1fs",
        ref.label, playable, len(cameras),
        "yes" if treadmill is not None else "no", duration,
    )
    return PlaybackSession(
        ref=ref, cameras=cameras, treadmill=treadmill, duration_s=float(duration)
    )
