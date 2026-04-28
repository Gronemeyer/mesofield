"""OpenCV-backed camera device.

A formal :class:`~mesofield.protocols.HardwareDevice` /
:class:`~mesofield.protocols.DataProducer` implementation that captures frames
from any OpenCV-accessible camera (webcam, UVC, etc.), writes them to MP4 via
``cv2.VideoWriter`` (configured with the project's bundled OpenH264 codec),
and pushes per-frame events onto the :class:`mesofield.data.manager.DataQueue`.

YAML usage (within ``hardware.yaml``)::

    cameras:
      - id: webcam              # unique device id (used by DataManager)
        name: webcam            # path suffix; used to build BIDS filename
        backend: opencv         # selects this class via the DeviceRegistry
        device_index: 0         # OpenCV VideoCapture index
        cv_backend: MSMF        # MSMF | DSHOW | ANY  (Windows: MSMF preferred)
        fps: 30                 # capture + writer fps (matches camera default)
        fourcc: H264            # uses bundled OpenH264 codec
        # width/height are optional; omit to use the camera's native resolution
        # width: 1080
        # height: 1920
        viewer_type: static     # required to render in the MDA Live Viewer
        properties:
          viewer_type: static
        output:
          suffix: webcam
          file_type: mp4
          bids_type: beh

The class implements the protocol via duck typing so the Qt
:class:`QThread` metaclass does not conflict with :class:`Protocol`.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from mesofield import DeviceRegistry
from mesofield.data.writer import configure_opencv_codec
from mesofield.utils._logger import get_logger


_DEFAULT_CV_BACKEND = "MSMF"


def _resolve_cv_backend(name: str | int | None) -> int:
    """Resolve a backend name (e.g. ``"MSMF"``) or numeric id to an OpenCV
    ``CAP_*`` constant. Falls back to ``CAP_ANY`` when unknown.
    """
    import cv2

    if isinstance(name, int):
        return name
    if not name:
        return cv2.CAP_ANY
    upper = name.upper()
    if not upper.startswith("CAP_"):
        upper = f"CAP_{upper}"
    return int(getattr(cv2, upper, cv2.CAP_ANY))


@DeviceRegistry.register("opencv_camera")
class OpenCVCamera(QThread):
    """Background-thread OpenCV camera capturing to MP4.

    Emits:
      - ``frame_ready(np.ndarray)``: raw BGR frame for live preview / GUI.
      - ``data_event(payload, device_ts)``: lightweight per-frame metadata
        consumed by :meth:`DataManager.register_hardware_device` and pushed
        onto the global :class:`DataQueue`.
    """

    # ----- Qt signals ------------------------------------------------------
    frame_ready = pyqtSignal(np.ndarray)
    # Alias used by the mesofield GUI's `InteractivePreview` (matches the
    # signal name expected by `arducam.VideoThread`).
    image_ready = pyqtSignal(np.ndarray)
    data_event = pyqtSignal(object, object)  # payload, device_ts
    captureStarted = pyqtSignal()
    captureStopped = pyqtSignal()

    # ----- Protocol attributes --------------------------------------------
    device_type: ClassVar[str] = "camera"
    file_type: str = "mp4"
    bids_type: Optional[str] = "beh"
    sampling_rate: float = 30.0
    data_type: str = "video"
    is_active: bool = False

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg: Dict[str, Any] = dict(cfg)
        self.id: str = cfg.get("id", "opencv_camera")
        self.name: str = cfg.get("name", self.id)
        self.device_id: str = self.id
        self.backend: str = "opencv"
        self.viewer: str = cfg.get("viewer_type", "live")
        self.auto_contrast: Any = cfg.get("auto_contrast")

        self.device_index: int = int(cfg.get("device_index", 0))
        self.cv_backend_name: str = str(cfg.get("cv_backend", _DEFAULT_CV_BACKEND))
        self.sampling_rate = float(cfg.get("fps", self.sampling_rate))
        self.fourcc: str = str(cfg.get("fourcc", "H264"))
        self.is_color: bool = bool(cfg.get("color", True))

        # Optional explicit width/height overrides; otherwise driver default
        self._req_width: Optional[int] = cfg.get("width")
        self._req_height: Optional[int] = cfg.get("height")

        # Apply YAML "properties" block (mirrors MMCamera semantics)
        self.properties: Dict[str, Any] = cfg.get("properties", {}) or {}
        for key, val in self.properties.items():
            if key == "fps":
                self.sampling_rate = float(val)
            elif key == "viewer_type":
                self.viewer = val
            elif key == "auto_contrast":
                self.auto_contrast = val

        # Filled in later
        self.output_path: Optional[str] = None
        self.metadata_path: Optional[str] = None
        self.writer: Optional[Any] = None  # cv2.VideoWriter
        self._capture: Optional[Any] = None  # cv2.VideoCapture
        self._frame_index: int = 0
        self._frame_timestamps: list[tuple[int, float]] = []  # (idx, perf_counter)
        self._stop = False

        # Match the MMCamera surface: a few attributes that other parts of
        # mesofield poke at directly. ``core`` / ``camera_device`` are kept
        # as ``None`` so isinstance() checks against pymmcore types fail
        # cleanly without raising.
        self.core: Optional[Any] = None
        self.camera_device: Optional[Any] = None
        self._engine = None

        self._started: Optional[datetime] = None
        self._stopped: Optional[datetime] = None

        self.logger = get_logger(f"{__name__}.OpenCVCamera[{self.id}]")
        self.initialize()

    # ----- HardwareDevice protocol ----------------------------------------
    def initialize(self) -> bool:
        """Verify the camera can be opened. Returns immediately afterwards."""
        configure_opencv_codec()
        import cv2

        backend_id = _resolve_cv_backend(self.cv_backend_name)
        cap = cv2.VideoCapture(self.device_index, backend_id)
        if not cap.isOpened():
            cap.release()
            self.logger.error(
                f"Could not open OpenCV camera index={self.device_index} "
                f"backend={self.cv_backend_name}"
            )
            return False
        # Detect actual resolution; honour overrides if given
        if self._req_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self._req_width))
        if self._req_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self._req_height))
        self._frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if cap_fps > 0 and not self.cfg.get("fps") and "fps" not in self.properties:
            self.sampling_rate = cap_fps
        cap.release()
        self.logger.info(
            f"OpenCV camera ready: index={self.device_index} "
            f"backend={self.cv_backend_name} "
            f"{self._frame_width}x{self._frame_height} @ {self.sampling_rate} fps"
        )
        return True

    def status(self) -> Dict[str, Any]:
        return {
            "active": self.is_active,
            "device_index": self.device_index,
            "cv_backend": self.cv_backend_name,
            "frames_written": self._frame_index,
            "output_path": self.output_path,
        }

    # Backwards-compat alias used elsewhere in the codebase
    def get_status(self) -> Dict[str, Any]:
        return self.status()

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "backend": self.backend,
            "device_index": self.device_index,
            "cv_backend": self.cv_backend_name,
            "fps": self.sampling_rate,
            "width": getattr(self, "_frame_width", None),
            "height": getattr(self, "_frame_height", None),
            "fourcc": self.fourcc,
            "file_type": self.file_type,
            "bids_type": self.bids_type,
        }

    # ----- DataProducer protocol -------------------------------------------
    def set_writer(self, make_path: Callable[[str, str, str, bool], str]) -> None:
        """Generate the output path and create the MP4 writer.

        Mirrors :meth:`MMCamera.set_writer` so :class:`Procedure.prerun` can
        treat both camera classes interchangeably. The capture thread may
        already be running for live-preview; once the writer is attached,
        subsequent frames are written to disk.
        """
        self.output_path = make_path(self.name, self.file_type, self.bids_type, True)
        self.metadata_path = self.output_path + "_frame_metadata.json"
        self._open_writer(self.output_path)
        # Reset timing so saved timestamps reflect the recording start, not
        # whenever the live-preview thread first launched.
        self._started = datetime.now()
        self._frame_timestamps = []
        self._frame_index = 0
        self.logger.info(f"Writer set to {self.output_path}")

    def set_sequence(self, build_mda: Callable[[Any], Any]) -> None:
        """No-op for OpenCV backend (no MDA sequence)."""
        self.logger.debug("set_sequence: OpenCV backend has no MDA sequence")

    def start(self) -> bool:
        if self.isRunning():
            self.logger.warning("OpenCVCamera.start: already running")
            return False
        self._stop = False
        self._frame_index = 0
        self._frame_timestamps = []
        self._started = datetime.now()
        self.is_active = True
        self.captureStarted.emit()
        super().start()  # spawns QThread.run
        return True

    # alias for HardwareManager compatibility
    def start_recording(self) -> bool:
        return self.start()

    def stop(self) -> bool:
        if not self.isRunning() and not self._stop:
            return True
        self._stop = True
        self.requestInterruption()
        self.wait(5000)
        self.is_active = False
        self._stopped = datetime.now()
        self.captureStopped.emit()
        return True

    def shutdown(self) -> None:
        self.stop()
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
            self.writer = None

    def get_data(self) -> Optional[Dict[str, Any]]:
        """Return the most recent frame index/timestamp pair, or ``None``."""
        if not self._frame_timestamps:
            return None
        idx, ts = self._frame_timestamps[-1]
        return {"frame_index": idx, "device_ts": ts}

    def save_data(self, path: Optional[str] = None) -> None:
        """The video file is already written on the fly; this method writes a
        sidecar JSON of frame timestamps next to the MP4.
        """
        import json

        target = path or self.metadata_path
        if not target:
            return
        try:
            with open(target, "w") as fh:
                json.dump(
                    {
                        "device": self.metadata,
                        "started": self._started.isoformat() if self._started else None,
                        "stopped": self._stopped.isoformat() if self._stopped else None,
                        "frames": [
                            {"index": i, "device_ts": ts}
                            for i, ts in self._frame_timestamps
                        ],
                    },
                    fh,
                    indent=2,
                )
            self.logger.info(f"Frame metadata saved to {target}")
        except Exception as exc:
            self.logger.error(f"Failed to save frame metadata: {exc}")

    # ----- Internal --------------------------------------------------------
    def _open_writer(self, path: str) -> None:
        import cv2

        if not path.endswith((".mp4", ".avi")):
            raise ValueError("OpenCVCamera output_path must end with .mp4 or .avi")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter.fourcc(*self.fourcc)
        writer = cv2.VideoWriter(
            path,
            fourcc,
            float(self.sampling_rate),
            (self._frame_width, self._frame_height),
            isColor=self.is_color,
        )
        if not writer.isOpened():
            raise RuntimeError(
                f"cv2.VideoWriter failed to open '{path}' "
                f"(fourcc={self.fourcc}, fps={self.sampling_rate})"
            )
        self.writer = writer

    def run(self) -> None:  # QThread entry point
        import cv2

        backend_id = _resolve_cv_backend(self.cv_backend_name)
        cap = cv2.VideoCapture(self.device_index, backend_id)
        if not cap.isOpened():
            cap.release()
            self.is_active = False
            self.logger.error("Failed to open camera in capture thread")
            return

        if self._req_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self._req_width))
        if self._req_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self._req_height))

        self._capture = cap
        period = 1.0 / self.sampling_rate if self.sampling_rate > 0 else 0.0
        next_t = time.perf_counter()
        try:
            while not self.isInterruptionRequested() and not self._stop:
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.msleep(1)
                    continue

                ts = time.perf_counter()
                idx = self._frame_index
                self._frame_index += 1
                self._frame_timestamps.append((idx, ts))

                # Write to disk
                if self.writer is not None:
                    if not self.is_color and frame.ndim == 3:
                        frame_to_write = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        frame_to_write = frame
                    try:
                        self.writer.write(frame_to_write)
                    except Exception as exc:  # pragma: no cover - codec failure
                        self.logger.error(f"VideoWriter.write failed: {exc}")

                # Emit signals (queue + GUI). `image_ready` mirrors the
                # arducam VideoThread signal name expected by the GUI viewer.
                self.frame_ready.emit(frame)
                self.image_ready.emit(frame)
                self.data_event.emit(idx, ts)

                # Frame pacing — only sleep if we have headroom
                if period:
                    next_t += period
                    delay = next_t - time.perf_counter()
                    if delay > 0:
                        self.msleep(int(delay * 1000))
                    else:
                        next_t = time.perf_counter()
        finally:
            try:
                cap.release()
            except Exception:
                pass
            self._capture = None
            if self.writer is not None:
                try:
                    self.writer.release()
                except Exception:
                    pass
            self.logger.info(
                f"OpenCV capture stopped: wrote {self._frame_index} frames"
            )
