"""Replay a saved mesofield session as if it were a live acquisition.

A playback run instantiates :class:`PlaybackCamera` and :class:`PlaybackEncoder`
devices that read recorded OME-TIFF frames and encoder CSVs off disk and re-emit
them through the standard :class:`~mesofield.signals.DeviceSignals` bundle. The
:class:`~mesofield.data.manager.DataManager` registers them like any other
producer, so the in-memory :class:`~mesofield.data.manager.DataQueue`, every
subscribed processor and the GUI viewer see the data move through the same
pipeline they would during a live capture.

Public surface used by ``mesofield playback <experiment_dir>``:

- :func:`discover_playback_context` -- locate ``manifest.json``, build a
  :class:`PlaybackProcedure` with one playback device per producer.
- :func:`launch_playback_app` -- run that procedure inside the standard
  :class:`~mesofield.gui.maingui.MainWindow`.

Playback is read-only: devices do not allocate writers, and the procedure
skips the dataqueue logger, the per-device ``save_data`` step, and the
``manifest.json`` re-write so the original session folder is left untouched.
"""

from __future__ import annotations

import csv
import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np

import importlib.util
import sys
import uuid

from mesofield import DeviceRegistry
from mesofield.base import Procedure
from mesofield.devices.base import BaseDataProducer
from mesofield.devices.base_camera import BaseCamera
from mesofield.utils._logger import get_logger

if TYPE_CHECKING:
    from mesofield.config import ExperimentConfig


__all__ = [
    "PlaybackClock",
    "PlaybackCamera",
    "PlaybackEncoder",
    "PlaybackContext",
    "PlaybackSession",
    "discover_playback_context",
    "discover_playback_sessions",
    "PlaybackBrowser",
    "launch_playback_app",
]


# ---------------------------------------------------------------------------
# Shared clock
# ---------------------------------------------------------------------------


@dataclass
class PlaybackClock:
    """Wall-clock anchor shared by every playback device in a session.

    ``t0_original`` is the earliest device timestamp across all producers in
    the manifest; ``t0_wall`` is captured at ``arm()`` time. Each replay thread
    sleeps until ``t0_wall + (original_ts - t0_original) / speed``, which keeps
    the two cameras and the encoder in their original relative order.
    """

    t0_original: float = 0.0
    speed: float = 1.0
    t0_wall: float = 0.0

    def reset(self, wall_now: Optional[float] = None) -> None:
        self.t0_wall = time.time() if wall_now is None else wall_now

    def sleep_until(self, original_ts: float, stop_event: threading.Event) -> bool:
        """Block until the wall-clock moment matching ``original_ts``.

        Returns ``True`` when interrupted via ``stop_event``.
        """
        if self.speed <= 0:
            return stop_event.wait(0)
        delay = (original_ts - self.t0_original) / self.speed
        target = self.t0_wall + delay
        wait = target - time.time()
        if wait <= 0:
            return stop_event.is_set()
        return stop_event.wait(wait)


# ---------------------------------------------------------------------------
# Frame metadata / CSV readers
# ---------------------------------------------------------------------------


def _parse_iso_to_unix(value: str) -> float:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).timestamp()


def _load_frame_timestamps(meta_path: Path) -> List[float]:
    """Return UNIX-second timestamps from a ``*_frame_metadata.json`` sidecar.

    Handles three on-disk shapes mesofield writers produce:

    - ``CustomWriter`` (OME-TIFF) and ``CV2Writer`` MDA-mode emit
      ``{"p0": [...]}`` keyed by position.
    - ``CV2Writer.finish`` (direct capture-loop mode used by ``OpenCVCamera``)
      wraps the same payload under ``{"frame_metadatas": {"p0": [...]}}``.
    - A bare ``[{...}, ...]`` list is also tolerated.
    """
    with meta_path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)

    # Unwrap CV2Writer.finish's outer envelope.
    if isinstance(records, dict) and "frame_metadatas" in records:
        records = records["frame_metadatas"]

    timestamps: List[float] = []
    if isinstance(records, dict):
        for _pos_key, entries in records.items():
            for entry in entries or []:
                ts_raw = entry.get("TimeReceivedByCore")
                if ts_raw is None:
                    continue
                timestamps.append(_parse_iso_to_unix(ts_raw))
    elif isinstance(records, list):
        for entry in records:
            ts_raw = entry.get("TimeReceivedByCore")
            if ts_raw is None:
                continue
            timestamps.append(_parse_iso_to_unix(ts_raw))
    return timestamps


# ---------------------------------------------------------------------------
# Frame readers (OME-TIFF and video)
# ---------------------------------------------------------------------------


_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


class _FrameReader:
    """Uniform iterator over a recorded camera output.

    Hides the difference between ``tifffile``-readable stacks and
    ``cv2.VideoCapture``-readable videos so :meth:`PlaybackCamera._run_loop`
    only writes the timing logic once. Used as a context manager; supports
    ``rewind()`` for the ``loop=True`` mode.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.kind = "video" if path.suffix.lower() in _VIDEO_SUFFIXES else "tiff"
        self.n_frames: int = 0
        self._tf: Any = None
        self._pages: Optional[List[Any]] = None
        self._cap: Any = None

    def __enter__(self) -> "_FrameReader":
        if self.kind == "tiff":
            import tifffile

            self._tf = tifffile.TiffFile(str(self.path))
            self._pages = list(self._tf.pages)
            self.n_frames = len(self._pages)
        else:
            import cv2

            self._cap = cv2.VideoCapture(str(self.path))
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open video {self.path}")
            count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.n_frames = max(count, 0)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._tf is not None:
            try:
                self._tf.close()
            except Exception:
                pass
            self._tf = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def read(self, idx: int) -> Optional[np.ndarray]:
        """Return frame ``idx`` (0-based) as an ndarray, or ``None`` if absent."""
        if self.kind == "tiff":
            if self._pages is None or idx >= len(self._pages):
                return None
            return np.asarray(self._pages[idx].asarray())
        # Video: cv2 advances sequentially; idx is informational.
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def rewind(self) -> None:
        if self.kind == "video" and self._cap is not None:
            import cv2

            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # tiff is indexed, no state to reset.


def _load_encoder_rows(csv_path: Path) -> Tuple[List[str], List[Tuple[float, Any]]]:
    """Read a saved producer CSV.

    Returns ``(payload_columns, rows)`` where each row is ``(ts, payload)``.
    ``payload`` is the scalar from the single ``payload`` column, or a dict
    keyed by ``payload_columns`` when the CSV fans multiple columns out (the
    treadmill case).
    """
    rows: List[Tuple[float, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return [], []
        if not header or header[0] != "timestamp":
            raise ValueError(
                f"{csv_path}: expected first column 'timestamp', got {header!r}"
            )
        data_cols = header[1:]
        scalar = len(data_cols) == 1 and data_cols[0] == "payload"

        for raw in reader:
            if not raw:
                continue
            try:
                ts = float(raw[0])
            except ValueError:
                continue
            if scalar:
                payload: Any = _coerce_scalar(raw[1] if len(raw) > 1 else "")
            else:
                payload = {
                    col: _coerce_scalar(raw[i + 1]) if i + 1 < len(raw) else None
                    for i, col in enumerate(data_cols)
                }
            rows.append((ts, payload))
    return data_cols, rows


def _coerce_scalar(text: str) -> Any:
    if text == "" or text is None:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


# ---------------------------------------------------------------------------
# Playback camera
# ---------------------------------------------------------------------------


@DeviceRegistry.register("playback_camera")
class PlaybackCamera(BaseCamera, BaseDataProducer):
    """Replays an OME-TIFF + sidecar JSON pair through the standard camera surface."""

    device_type: ClassVar[str] = "camera"
    file_type: ClassVar[str] = "ome.tiff"
    bids_type: ClassVar[Optional[str]] = "func"
    data_type: ClassVar[str] = "frames"
    clock_source: ClassVar[str] = "wall_unix_s"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        BaseDataProducer.__init__(self, cfg, **kwargs)
        self._init_camera_surface(self.cfg, backend="playback")

        frames_path = self.cfg.get("frames_path")
        meta_path = self.cfg.get("metadata_path")
        if not frames_path:
            raise ValueError(
                f"PlaybackCamera({self.device_id}): 'frames_path' is required"
            )
        self._frames_path = Path(frames_path)
        self._meta_path = Path(meta_path) if meta_path else None
        self._timestamps: List[float] = []
        # Reflect the on-disk format in status()/metadata even though playback
        # never allocates a writer.
        ft = self.cfg.get("file_type")
        if ft:
            self.file_type = str(ft).lower()

        self._clock: PlaybackClock = self.cfg.get("clock") or PlaybackClock()
        self._loop: bool = bool(self.cfg.get("loop", False))

        self.sampling_rate = float(
            self.cfg.get("sampling_rate") or self.cfg.get("fps") or 0.0
        )

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_count: Optional[int] = None

        self._qt_image_adapter = None
        self.image_ready = None
        try:
            from mesofield.gui.qt_device_adapter import QtImageAdapter

            self._qt_image_adapter = QtImageAdapter()
            self.image_ready = self._qt_image_adapter.image_ready
        except Exception:
            self.logger.debug("QtImageAdapter unavailable; running headless.")

    # ---- lifecycle -----------------------------------------------------
    def initialize(self) -> bool:
        if not self._frames_path.is_file():
            raise FileNotFoundError(
                f"PlaybackCamera({self.device_id}): frames file missing: "
                f"{self._frames_path}"
            )
        if self._meta_path and self._meta_path.is_file():
            self._timestamps = _load_frame_timestamps(self._meta_path)
        else:
            self.logger.warning(
                f"No frame metadata sidecar for {self.device_id}; "
                f"synthesising timestamps from sampling_rate={self.sampling_rate}"
            )
            self._timestamps = []
        return True

    def arm(self, config: "ExperimentConfig") -> None:
        # Read-only: clear the buffer but skip writer / output_path setup.
        self.clear_buffer()

    def set_writer(self, make_path: Any) -> None:  # noqa: D401
        """No-op: playback never allocates a writer."""
        return None

    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"PlaybackCamera-{self.device_id}",
            daemon=True,
        )
        self._thread.start()
        return BaseDataProducer.start(self)

    def stop(self) -> bool:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None
        return BaseDataProducer.stop(self)

    # ---- live preview contract (abstract on BaseCamera) ----------------
    def snap(self) -> Optional[np.ndarray]:
        try:
            with _FrameReader(self._frames_path) as reader:
                if reader.n_frames == 0:
                    return None
                frame = reader.read(0)
        except Exception as exc:
            self.logger.error(f"snap failed for {self._frames_path}: {exc}")
            return None
        if frame is not None and self._qt_image_adapter is not None:
            self._qt_image_adapter.emit_frame(frame)
        return frame

    def start_live(self) -> None:
        self.start()

    def stop_live(self) -> None:
        self.stop()

    # ---- run loop ------------------------------------------------------
    def _run_loop(self) -> None:
        try:
            with _FrameReader(self._frames_path) as reader:
                self._frame_count = reader.n_frames
                if reader.n_frames == 0:
                    self.logger.warning(f"No frames in {self._frames_path}")
                    return

                timestamps = self._effective_timestamps(reader.n_frames)

                while not self._stop_event.is_set():
                    for idx in range(reader.n_frames):
                        if self._stop_event.is_set():
                            return
                        ts = timestamps[idx]
                        if self._clock.sleep_until(ts, self._stop_event):
                            return
                        try:
                            frame = reader.read(idx)
                        except Exception as exc:
                            self.logger.exception(f"frame read failed: {exc}")
                            continue
                        if frame is None:
                            continue
                        if self._qt_image_adapter is not None:
                            try:
                                self._qt_image_adapter.emit_frame(frame)
                            except Exception:
                                pass
                        try:
                            self.signals.frame.emit(frame, idx, ts)
                        except Exception:
                            pass
                        self.record({"frame_index": idx}, ts=ts)

                    if not self._loop:
                        return
                    # Re-anchor the clock so the next pass replays at speed.
                    reader.rewind()
                    self._clock.reset()
        except Exception as exc:
            self.logger.exception(f"playback loop crashed: {exc}")
        finally:
            self._emit_finished_if_natural()
            self.logger.debug(f"playback camera {self.device_id} exited")

    def _emit_finished_if_natural(self) -> None:
        """Emit ``signals.finished`` when the loop exits without a stop call.

        :meth:`stop` already emits via :class:`BaseDataProducer`; we only
        emit here when the thread ran out of frames on its own, so the
        :class:`~mesofield.base.Procedure`'s primary-finished hook fires
        even though nothing explicitly called ``stop()``.
        """
        if self._stop_event.is_set():
            return
        self.is_active = False
        self.is_running = False
        try:
            self.signals.finished.emit()
        except Exception:
            pass

    def _effective_timestamps(self, n_pages: int) -> List[float]:
        if self._timestamps and len(self._timestamps) >= n_pages:
            return self._timestamps[:n_pages]
        if self._timestamps:
            # Pad with cadence of the last delta to match page count.
            ts = list(self._timestamps)
            if len(ts) >= 2:
                step = ts[-1] - ts[-2]
            elif self.sampling_rate:
                step = 1.0 / self.sampling_rate
            else:
                step = 0.05
            while len(ts) < n_pages:
                ts.append(ts[-1] + step)
            return ts
        # No metadata at all: synthesise from sampling_rate, anchored at t0.
        step = 1.0 / self.sampling_rate if self.sampling_rate else 0.05
        t0 = self._clock.t0_original
        return [t0 + i * step for i in range(n_pages)]

    @property
    def first_timestamp(self) -> Optional[float]:
        if self._timestamps:
            return self._timestamps[0]
        return None

    @property
    def calibration(self) -> Dict[str, Any]:
        return {
            "source": str(self._frames_path),
            "frame_count": self._frame_count,
            "sampling_rate_hz": self.sampling_rate,
        }


# ---------------------------------------------------------------------------
# Playback encoder
# ---------------------------------------------------------------------------


@DeviceRegistry.register("playback_encoder")
class PlaybackEncoder(BaseDataProducer):
    """Replays a saved encoder/treadmill CSV through ``signals.data``."""

    device_type: ClassVar[str] = "encoder"
    file_type: ClassVar[str] = "csv"
    bids_type: ClassVar[Optional[str]] = "beh"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        csv_path = self.cfg.get("csv_path")
        if not csv_path:
            raise ValueError(
                f"PlaybackEncoder({self.device_id}): 'csv_path' is required"
            )
        self._csv_path = Path(csv_path)
        self._rows: List[Tuple[float, Any]] = []
        self._payload_columns: List[str] = []
        self._clock: PlaybackClock = self.cfg.get("clock") or PlaybackClock()
        self._loop: bool = bool(self.cfg.get("loop", False))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---- lifecycle -----------------------------------------------------
    def initialize(self) -> bool:
        if not self._csv_path.is_file():
            raise FileNotFoundError(
                f"PlaybackEncoder({self.device_id}): csv missing: {self._csv_path}"
            )
        self._payload_columns, self._rows = _load_encoder_rows(self._csv_path)
        return True

    def arm(self, config: "ExperimentConfig") -> None:
        # Read-only: clear buffer, skip make_path.
        self.clear_buffer()

    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"PlaybackEncoder-{self.device_id}",
            daemon=True,
        )
        self._thread.start()
        return super().start()

    def stop(self) -> bool:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None
        return super().stop()

    def save_data(self, path: Optional[str] = None) -> Optional[str]:
        # Read-only.
        return None

    # ---- run loop ------------------------------------------------------
    def _run_loop(self) -> None:
        try:
            if not self._rows:
                self.logger.warning(f"No rows in {self._csv_path}")
                return
            while not self._stop_event.is_set():
                for ts, payload in self._rows:
                    if self._stop_event.is_set():
                        return
                    if self._clock.sleep_until(ts, self._stop_event):
                        return
                    self.record(payload, ts=ts)
                if not self._loop:
                    return
                self._clock.reset()
        finally:
            if not self._stop_event.is_set():
                self.is_active = False
                self.is_running = False
                try:
                    self.signals.finished.emit()
                except Exception:
                    pass

    @property
    def first_timestamp(self) -> Optional[float]:
        return self._rows[0][0] if self._rows else None

    @property
    def calibration(self) -> Dict[str, Any]:
        return {
            "source": str(self._csv_path),
            "row_count": len(self._rows),
            "payload_columns": list(self._payload_columns),
        }


# ---------------------------------------------------------------------------
# Context + discovery
# ---------------------------------------------------------------------------


@dataclass
class PlaybackContext:
    """Bundle returned by :func:`discover_playback_context`."""

    procedure: Procedure
    session_dir: Path
    speed: float
    loop: bool
    clock: PlaybackClock
    producers: List[str] = field(default_factory=list)


def _find_manifest(experiment_dir: Path) -> Path:
    """Locate a ``manifest.json`` for *experiment_dir*.

    Accepts either:

    - a session directory containing ``manifest.json`` directly, or
    - an experiment root with ``data/sub-*/ses-*/manifest.json`` below it
      (the first match in lexicographic order is used).
    """
    direct = experiment_dir / "manifest.json"
    if direct.is_file():
        return direct
    candidates = sorted(experiment_dir.glob("data/sub-*/ses-*/manifest.json"))
    if candidates:
        return candidates[-1]  # most recent session lexicographically
    # As a last resort, recursive search (capped by glob).
    deep = sorted(experiment_dir.rglob("manifest.json"))
    if deep:
        return deep[-1]
    raise FileNotFoundError(
        f"No manifest.json found under {experiment_dir}. Pass either a session "
        "directory or an experiment root containing data/sub-*/ses-*/manifest.json."
    )


def _find_experiment_root(session_dir: Path) -> Optional[Path]:
    """Walk up from *session_dir* looking for an experiment.json/hardware.yaml."""
    for parent in [session_dir, *session_dir.parents]:
        if (parent / "experiment.json").is_file() or (parent / "hardware.yaml").is_file():
            return parent
    return None


def _resolve_user_procedure_class(experiment_json: Path) -> Optional[type]:
    """Return the :class:`Procedure` subclass declared in *experiment_json*.

    Mirrors :func:`mesofield.base.load_procedure_from_config` but stops before
    instantiation -- we only want the class so we can subclass it and inject
    playback devices via :meth:`define_hardware`. Avoiding the instantiation
    side-steps the live ``initialize_hardware`` step that would otherwise
    try to open real cameras and serial ports during ``mesofield playback``.
    """
    try:
        with experiment_json.open("r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception:
        return None
    proc_file = cfg.get("procedure_file")
    proc_class = cfg.get("procedure_class")
    if not proc_file or not proc_class:
        return None

    if not Path(proc_file).is_absolute():
        proc_file = str(experiment_json.parent / proc_file)
    if not Path(proc_file).is_file():
        return None

    mod_name = f"mesofield_playback_procedure_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, proc_file)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    cls = getattr(module, proc_class, None)
    if isinstance(cls, type) and issubclass(cls, Procedure):
        return cls
    return None


def _read_primary_device_id(hardware_yaml: Path) -> Optional[str]:
    """Return the device_id flagged ``primary: true`` in a hardware.yaml.

    Matches the convention :class:`HardwareManager` enforces: each top-level
    stanza is a device, and the one with ``primary: true`` is the timing
    anchor. Returns ``None`` if the file is unreadable or has no primary.
    """
    try:
        import yaml

        with hardware_yaml.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception:
        return None
    for key, stanza in data.items():
        if isinstance(stanza, dict) and stanza.get("primary") is True:
            return str(stanza.get("id", key))
    return None


def _resolve_playback_camera_path(
    entry: Dict[str, Any],
    session_dir: Path,
    logger: Any,
) -> Optional[Path]:
    """Resolve a camera producer's frames path with a metadata-sidecar fallback.

    Some writers (e.g. ``OpenCVCamera`` -> ``CV2Writer``) name the on-disk file
    by the device's display name while :class:`DataSaver` populates the
    manifest's ``output_path`` from the YAML ``suffix:`` -- the two diverge
    and the recorded ``output_path`` ends up pointing at a file that was
    never written. Fall back to ``metadata_path`` minus the
    ``_frame_metadata.json`` tail, which matches the writer's actual filename.
    """
    output_path = entry.get("output_path")
    if output_path:
        abs_output = (session_dir / output_path).resolve()
        if abs_output.is_file():
            return abs_output

    meta_path = entry.get("metadata_path")
    if meta_path and meta_path.endswith("_frame_metadata.json"):
        derived = (session_dir / meta_path[: -len("_frame_metadata.json")]).resolve()
        if derived.is_file():
            logger.warning(
                f"{entry.get('device_id')}: manifest output_path "
                f"{output_path!r} not on disk; using {derived.name} derived "
                "from metadata_path"
            )
            return derived

    return None


def _build_devices(
    manifest: Dict[str, Any],
    session_dir: Path,
    clock: PlaybackClock,
    loop: bool,
    logger: Any,
    primary_device_id: Optional[str] = None,
) -> Tuple[List[Any], List[str], List[float]]:
    """Construct one playback device per replayable producer in *manifest*."""
    devices: List[Any] = []
    producer_ids: List[str] = []
    first_timestamps: List[float] = []
    producers = manifest.get("producers", []) or []

    for entry in producers:
        device_id = entry.get("device_id") or "device"
        device_type = entry.get("device_type")

        if device_type == "camera":
            file_type = (entry.get("file_type") or "").lower()
            if file_type not in {
                "ome.tiff", "ome.tif", "tiff", "tif",
                "mp4", "avi", "mov", "mkv",
            }:
                logger.warning(
                    f"Skipping camera {device_id}: file_type "
                    f"{file_type!r} not supported for playback"
                )
                continue
            frames_path = _resolve_playback_camera_path(entry, session_dir, logger)
            if frames_path is None:
                logger.error(
                    f"Skipping camera {device_id}: cannot find frames file on "
                    f"disk (manifest output_path={entry.get('output_path')!r})"
                )
                continue
            meta_path = entry.get("metadata_path")
            abs_meta = (session_dir / meta_path).resolve() if meta_path else None
            cfg = {
                "id": device_id,
                "frames_path": str(frames_path),
                "metadata_path": str(abs_meta) if abs_meta else None,
                "file_type": file_type,
                "sampling_rate": entry.get("sampling_rate_hz") or 0.0,
                "clock": clock,
                "loop": loop,
            }
            device = PlaybackCamera(cfg)
        elif device_type == "encoder":
            output_path = entry.get("output_path")
            if not output_path:
                logger.warning(f"Skipping {device_id}: no output_path in manifest")
                continue
            abs_output = (session_dir / output_path).resolve()
            cfg = {
                "id": device_id,
                "csv_path": str(abs_output),
                "clock": clock,
                "loop": loop,
            }
            device = PlaybackEncoder(cfg)
        else:
            logger.warning(
                f"Skipping {device_id}: device_type "
                f"{device_type!r} not supported in playback v1"
            )
            continue

        try:
            device.initialize()
        except Exception as exc:
            logger.error(f"initialize() failed for {device_id}: {exc}")
            continue

        ts0 = getattr(device, "first_timestamp", None)
        if ts0 is not None:
            first_timestamps.append(float(ts0))

        devices.append(device)
        producer_ids.append(device_id)

    # Promote the device matching the recorded primary (preferred) or the
    # first camera (fallback) to is_primary=True. HardwareManager enforces
    # exactly-one-primary; Procedure.run wires cleanup off its `finished`.
    primary_set = False
    if primary_device_id:
        for dev in devices:
            if dev.device_id == primary_device_id:
                dev.is_primary = True
                primary_set = True
                break
        if not primary_set:
            logger.warning(
                f"hardware.yaml's primary device {primary_device_id!r} is not "
                "in the replayable set; falling back to first camera"
            )
    if not primary_set:
        for dev in devices:
            if isinstance(dev, PlaybackCamera):
                dev.is_primary = True
                primary_set = True
                break
    if not primary_set and devices:
        devices[0].is_primary = True

    return devices, producer_ids, first_timestamps


def _make_playback_procedure(
    base_cls: type,
    *,
    devices: List[Any],
    session_dir: Path,
    session_info: Dict[str, Any],
    speed: float,
    loop: bool,
) -> Procedure:
    """Dynamically subclass *base_cls* to inject playback devices.

    The wrapper sets ``playback = True`` (so the base ``Procedure`` skips its
    disk-writing side-effects), overrides ``define_hardware`` to return the
    pre-built playback devices, and -- only when the base class has no real
    ``experiment.json`` to drive it -- overrides ``define_config`` from the
    manifest's session block. When an experiment.json is available we let the
    user's class load it normally so duration/notes/processors line up with
    the original run.
    """

    class _PlaybackWrapper(base_cls):  # type: ignore[valid-type, misc]
        playback = True

        def define_hardware(self):  # noqa: D401
            return list(devices)

    _PlaybackWrapper.__name__ = f"Playback[{base_cls.__name__}]"
    _PlaybackWrapper.__qualname__ = _PlaybackWrapper.__name__

    if base_cls is Procedure:
        # No user procedure -> manufacture a config from the manifest so
        # subject/session/task/protocol are populated for logging/paths.
        def _define_config(self):  # type: ignore[no-redef]
            info = session_info
            return {
                "subject": str(info.get("subject", "playback")),
                "session": str(info.get("session", "00")),
                "task": str(info.get("task", "playback")),
                "protocol": str(info.get("protocol", "playback")),
                "experimenter": str(info.get("experimenter", "playback")),
                "duration": int(info.get("duration", 0) or 0),
            }

        _PlaybackWrapper.define_config = _define_config  # type: ignore[assignment]
        proc = _PlaybackWrapper(config_path=None)
        try:
            proc.config.experiment_dir = str(session_dir.parent.parent.parent)
        except Exception:
            pass
        return proc

    # The user's procedure class drives __init__ from its own experiment.json.
    # _PlaybackWrapper.define_hardware short-circuits the YAML path so devices
    # come from our pre-built list.
    experiment_root = _find_experiment_root(session_dir)
    config_path = None
    if experiment_root is not None:
        candidate = experiment_root / "experiment.json"
        if candidate.is_file():
            config_path = str(candidate)
    proc = _PlaybackWrapper(config_path)
    return proc


def discover_playback_context(
    experiment_dir: Path,
    *,
    speed: float = 1.0,
    loop: bool = False,
) -> PlaybackContext:
    """Build a :class:`PlaybackContext` from a recorded session directory."""
    logger = get_logger(__name__)
    experiment_dir = Path(experiment_dir).expanduser().resolve()
    manifest_path = _find_manifest(experiment_dir)
    session_dir = manifest_path.parent

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    experiment_root = _find_experiment_root(session_dir)
    primary_id: Optional[str] = None
    base_cls: type = Procedure
    if experiment_root is not None:
        hw_yaml = experiment_root / "hardware.yaml"
        if hw_yaml.is_file():
            primary_id = _read_primary_device_id(hw_yaml)
        exp_json = experiment_root / "experiment.json"
        if exp_json.is_file():
            resolved = _resolve_user_procedure_class(exp_json)
            if resolved is not None:
                base_cls = resolved
                logger.info(
                    f"Using user procedure class {base_cls.__name__} from "
                    f"{exp_json}"
                )
            else:
                logger.debug(
                    f"No procedure_file/procedure_class in {exp_json}; "
                    "using base Procedure."
                )

    clock = PlaybackClock(speed=float(speed))
    devices, producer_ids, first_timestamps = _build_devices(
        manifest, session_dir, clock, loop, logger,
        primary_device_id=primary_id,
    )
    if not devices:
        raise RuntimeError(
            f"No replayable producers found in {manifest_path}. "
            "Playback supports camera (OME-TIFF / mp4) and encoder (CSV) producers."
        )
    clock.t0_original = min(first_timestamps) if first_timestamps else 0.0
    clock.reset()  # wall-clock anchor; reset again at procedure start.

    session_info = dict(manifest.get("session") or {})
    procedure = _make_playback_procedure(
        base_cls,
        devices=devices,
        session_dir=session_dir,
        session_info=session_info,
        speed=float(speed),
        loop=bool(loop),
    )
    logger.info(
        f"Playback context ready: {len(producer_ids)} producers from "
        f"{manifest_path} (speed={speed:.2f}x, loop={loop})"
    )
    return PlaybackContext(
        procedure=procedure,
        session_dir=session_dir,
        speed=float(speed),
        loop=bool(loop),
        clock=clock,
        producers=producer_ids,
    )


# ---------------------------------------------------------------------------
# Session-list discovery (drives the ConfigController dropdowns)
# ---------------------------------------------------------------------------


@dataclass
class PlaybackSession:
    """One ``(subject, session)`` directory discovered on disk.

    Holds just enough metadata for the ConfigController dropdowns to present
    a choice: the parsed subject/session ids, the absolute session directory,
    the path to its manifest, and the task/protocol read from the manifest's
    session block (used as display hints and to seed ``ExperimentConfig``).
    """

    subject: str
    session: str
    session_dir: Path
    manifest_path: Path
    task: Optional[str] = None
    protocol: Optional[str] = None


_SUB_RE = re.compile(r"^sub-(.+)$")
_SES_RE = re.compile(r"^ses-(.+)$")


def discover_playback_sessions(experiment_dir: Path) -> List[PlaybackSession]:
    """Enumerate every ``data/sub-*/ses-*/manifest.json`` under *experiment_dir*.

    Falls back to scanning *experiment_dir* itself when it points directly at
    a session directory. Sessions are returned sorted by (subject, session) so
    the dropdowns have a predictable order.
    """
    experiment_dir = Path(experiment_dir).expanduser().resolve()
    candidates = sorted(experiment_dir.glob("data/sub-*/ses-*/manifest.json"))
    if not candidates and (experiment_dir / "manifest.json").is_file():
        candidates = [experiment_dir / "manifest.json"]
    sessions: List[PlaybackSession] = []
    for manifest_path in candidates:
        session_dir = manifest_path.parent
        sub_match = _SUB_RE.match(session_dir.parent.name)
        ses_match = _SES_RE.match(session_dir.name)
        if not (sub_match and ses_match):
            continue
        info: Dict[str, Any] = {}
        try:
            with manifest_path.open("r", encoding="utf-8") as fh:
                info = (json.load(fh) or {}).get("session") or {}
        except Exception:
            pass
        sessions.append(
            PlaybackSession(
                subject=sub_match.group(1),
                session=ses_match.group(1),
                session_dir=session_dir,
                manifest_path=manifest_path,
                task=info.get("task"),
                protocol=info.get("protocol"),
            )
        )
    sessions.sort(key=lambda s: (s.subject, s.session))
    return sessions


# ---------------------------------------------------------------------------
# Browser: ties subject/session dropdowns to procedure swaps
# ---------------------------------------------------------------------------


class PlaybackBrowser:
    """Drives :class:`~mesofield.gui.controller.ConfigController` for playback.

    Two responsibilities:

    1. Re-seed ``ExperimentConfig.subjects`` from the set of recorded sessions
       on disk so the subject dropdown lists what is actually playable, and
       register the per-subject session ids as ``register_choices`` on the
       ``session`` key so it renders as a dropdown.
    2. When the user picks a different subject or session, tear down the
       current playback procedure and swap a freshly-built one into
       :class:`~mesofield.gui.maingui.MainWindow`, re-using the existing
       widget-rebuild path (``_build_acquisition_ui`` + ``_on_config_applied``).

    Implementation note: the actual swap is deferred via ``QTimer.singleShot``
    because the change callback fires inside a QComboBox signal handler --
    rebuilding the parent ConfigController during that callback would delete
    the QObject that emitted the signal.
    """

    def __init__(
        self,
        main_window: Any,
        sessions: List[PlaybackSession],
        *,
        speed: float = 1.0,
        loop: bool = False,
    ) -> None:
        self.main_window = main_window
        self.sessions = list(sessions)
        self.speed = float(speed)
        self.loop = bool(loop)
        self.logger = get_logger(f"{__name__}.PlaybackBrowser")

        self._by_subject: Dict[str, List[PlaybackSession]] = {}
        for ses in self.sessions:
            self._by_subject.setdefault(ses.subject, []).append(ses)

        self._swapping: bool = False
        self._current: Optional[PlaybackSession] = None
        self._install()

    # -- public API -----------------------------------------------------
    @property
    def current(self) -> Optional[PlaybackSession]:
        return self._current

    # -- install ---------------------------------------------------------
    def _install(self) -> None:
        """Apply browser-aware behaviour to the live ConfigController.

        Called once at startup and again after every procedure swap (because
        the new procedure has a fresh ``ExperimentConfig`` object and the
        ConfigController is rebuilt by ``MainWindow._on_config_applied``).
        """
        proc = self.main_window.procedure
        cfg = proc.config

        # Mirror the disk inventory onto cfg.subjects so the existing subject
        # dropdown wiring (built in ConfigController._populate_subjects from
        # cfg.subjects.keys()) reflects what's available for replay.
        cfg.subjects = {
            sub: {
                "session": ses_list[0].session,
                "task": ses_list[0].task or cfg.get("task", "playback") or "playback",
            }
            for sub, ses_list in self._by_subject.items()
        }

        current_sub = cfg.get("subject")
        if current_sub not in cfg.subjects and cfg.subjects:
            current_sub = next(iter(cfg.subjects))

        # Build (and lock in) session choices for the current subject so the
        # ConfigController's form renders a session dropdown rather than a
        # free-text editor.
        self._set_session_choices(current_sub)

        # When _change_subject was last called by the ConfigController, it ran
        # select_subject(initial) which set "session" to whatever cfg.subjects
        # said. If we have a more precise current_session (from the active
        # PlaybackContext), enforce it here so the form picks the right entry.
        if self._current is not None and current_sub == self._current.subject:
            cfg.set("session", self._current.session)

        # Listen for subject/session changes; subject changes also update
        # session choices (the dropdown gets repopulated for the new subject).
        cfg.register_callback("subject", self._on_subject_changed)
        cfg.register_callback("session", self._on_session_changed)

        # Push a refreshed ConfigController tab (the form widget reads
        # choices at construction time, so a rebuild is required for the new
        # session list to render).
        try:
            self.main_window._on_config_applied()
        except Exception as exc:
            self.logger.debug(f"_on_config_applied() reinstall failed: {exc}")

        # Relabel the Record button now that the controller has been rebuilt.
        cc = getattr(self.main_window, "_config_controller", None)
        if cc is not None and hasattr(cc, "record_button"):
            cc.record_button.setText("Play")
            cc.record_button.setToolTip("Play back the selected session")

    def _set_session_choices(self, subject: str) -> None:
        cfg = self.main_window.procedure.config
        sess_list = [s.session for s in self._by_subject.get(subject, [])]
        if sess_list:
            cfg.register_choices("session", sess_list)

    # -- callbacks -------------------------------------------------------
    def _on_subject_changed(self, _key: str, value: Any) -> None:
        if self._swapping:
            return
        # Update choices for the new subject so the form, when ConfigController
        # rebuilds it after select_subject returns, renders the right options.
        self._set_session_choices(str(value))
        # The session change that select_subject() fires next will dispatch
        # the actual procedure swap.

    def _on_session_changed(self, _key: str, value: Any) -> None:
        if self._swapping:
            return
        cfg = self.main_window.procedure.config
        subject = cfg.get("subject")
        # Defer to the event loop so the QComboBox that emitted this change
        # is not destroyed in the middle of its own signal dispatch.
        try:
            from PyQt6.QtCore import QTimer

            QTimer.singleShot(0, lambda: self._swap_to(str(subject), str(value)))
        except Exception:
            # Headless / no Qt loop -- swap synchronously (e.g. tests).
            self._swap_to(str(subject), str(value))

    # -- swap ------------------------------------------------------------
    def _swap_to(self, subject: str, session: str) -> None:
        target = next(
            (s for s in self.sessions if s.subject == subject and s.session == session),
            None,
        )
        if target is None:
            self.logger.warning(
                f"No session found on disk for sub-{subject}/ses-{session}; "
                "ignoring change"
            )
            return
        if self._current is not None and target.session_dir == self._current.session_dir:
            return  # no-op
        self.logger.info(
            f"Swapping playback target -> sub-{subject}/ses-{session} "
            f"({target.session_dir})"
        )
        self._swapping = True
        try:
            self._teardown_current()
            new_ctx = discover_playback_context(
                target.session_dir, speed=self.speed, loop=self.loop
            )
            self.main_window.procedure = new_ctx.procedure
            try:
                self.main_window._build_acquisition_ui()
            except Exception as exc:
                self.logger.exception(f"_build_acquisition_ui failed: {exc}")
            self._current = target
            # Re-install on the new procedure's fresh config object.
            self._install()
        finally:
            self._swapping = False

    def _teardown_current(self) -> None:
        proc = self.main_window.procedure
        try:
            proc.cleanup()
        except Exception:
            pass
        try:
            proc.config.hardware.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# GUI launcher
# ---------------------------------------------------------------------------


def launch_playback_app(
    context: PlaybackContext,
    *,
    browser_sessions: Optional[List[PlaybackSession]] = None,
) -> int:
    """Open the standard mesofield GUI against a :class:`PlaybackContext`.

    When ``browser_sessions`` is supplied, a :class:`PlaybackBrowser` is wired
    into the MainWindow's ConfigController so the subject/session dropdowns
    swap playback targets on change.
    """
    from PyQt6.QtWidgets import QApplication

    from mesofield.gui.maingui import MainWindow
    from mesofield.gui import theme

    app = QApplication.instance() or QApplication([])
    theme.apply_theme(app)

    window = MainWindow(context.procedure)
    title = (
        f"Mesofield Playback (speed={context.speed:g}x"
        f"{', loop' if context.loop else ''})"
    )
    window.setWindowTitle(title)
    window.show()

    browser: Optional[PlaybackBrowser] = None
    if browser_sessions:
        # Find the session entry that matches the loaded context so the
        # browser's _current pointer is correct before any change events.
        match = next(
            (s for s in browser_sessions if s.session_dir == context.session_dir),
            None,
        )
        browser = PlaybackBrowser(
            window,
            browser_sessions,
            speed=context.speed,
            loop=context.loop,
        )
        browser._current = match
        # Hold a reference on the window so the browser isn't GC'd while the
        # GUI is alive.
        window._playback_browser = browser  # type: ignore[attr-defined]

    # Re-anchor the wall clock at the moment we kick off, so any GUI startup
    # delay does not eat the head of the recording, then start the procedure.
    context.clock.reset()
    try:
        context.procedure.run()
    except Exception:
        get_logger(__name__).exception("playback procedure.run failed")

    return app.exec()
