"""Shared surface for camera-shaped devices.

Three concrete cameras converge on this base:

  - :class:`mesofield.devices.cameras.MMCamera`  (Micro-Manager backend)
  - :class:`mesofield.devices.cameras.OpenCVCamera`  (OpenCV/UVC)
  - :class:`mesofield.devices.mocks.MockFrameProducer`  (synthetic)

They have wildly different acquisition loops (mmcore-driven MDA vs Qt thread
vs synthetic frame generator) so we don't try to unify the run-loop here.
What we DO unify is the *surface* every camera must expose:

  - Identity & cosmetic attrs the MDA GUI reads
    (``name``, ``viewer``, ``auto_contrast``, ``core``, ``backend``).
  - Output paths and metadata sidecar plumbing.
  - The standard lifecycle hooks (``arm``, ``set_writer``, ``set_sequence``,
    ``shutdown``, ``status``, ``calibration``).
  - The :class:`DeviceSignals` bundle that :class:`DataManager` subscribes to
    (``signals.started``, ``signals.finished``, ``signals.data``).

Subclasses keep their backend-specific ``initialize``, ``start``, ``stop``,
``save_data``, and writer setup.

Design notes:

- :class:`BaseCamera` is a *regular* class -- not a Protocol -- so subclasses
  can multiply-inherit alongside Qt's :class:`QThread` (``OpenCVCamera``),
  pymmcore-plus's ``DataProducer``/``HardwareDevice`` Protocols
  (``MMCamera``), and our pure-Python :class:`BaseDataProducer`
  (``MockFrameProducer``).
- We avoid ``BaseCamera.__init__`` to dodge multiple-inheritance ``super()``
  chains with QThread. Subclasses call :meth:`_init_camera_surface` after
  whatever parent ``__init__`` they need.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, ClassVar, Dict, Optional, Type

from mesofield.signals import DeviceSignals
from mesofield.utils._logger import get_logger


class BaseCamera:
    """Common camera surface (no Qt, no acquisition loop).

    Subclasses call :meth:`_init_camera_surface` from their ``__init__``
    after any superclass ``super().__init__()`` calls. Override
    :meth:`set_writer`, :meth:`set_sequence`, :meth:`start`, :meth:`stop`,
    :meth:`save_data`, and :meth:`get_data` as needed.
    """

    # --- class-level defaults; subclasses override as appropriate -------
    device_type: ClassVar[str] = "camera"
    viewer: ClassVar[str] = "static"
    file_type: ClassVar[str] = "ome.tiff"
    bids_type: ClassVar[Optional[str]] = "func"
    data_type: ClassVar[str] = "frames"

    # --- initialisation helper ------------------------------------------
    def _init_camera_surface(self, cfg: Dict[str, Any], *, backend: str) -> None:
        """Populate the shared camera attributes from a YAML stanza.

        Sets identity (id/device_id/name), primary flag, viewer cosmetics,
        sampling rate, the :class:`DeviceSignals` bundle, output paths,
        and timing slots. Idempotent on ``signals`` if the subclass has
        already created it (e.g. :class:`BaseDataProducer` does).
        """
        self.cfg: Dict[str, Any] = dict(cfg or {})
        self.id: str = str(self.cfg.get("id", "camera"))
        self.device_id: str = self.id
        self.name: str = str(self.cfg.get("name", self.id))
        self.backend: str = backend
        self.is_primary: bool = bool(self.cfg.get("primary", False))
        self.is_active: bool = False
        self.auto_contrast: Any = self.cfg.get("auto_contrast", True)
        self.viewer = self.cfg.get("viewer_type", self.viewer)
        # `sampling_rate` is the canonical fps for the camera. Subclasses
        # may overwrite from cfg["fps"] or driver introspection.
        self.sampling_rate: float = float(
            self.cfg.get("sampling_rate", self.cfg.get("fps", 0.0)) or 0.0
        )
        # Output state (filled in by set_writer/save_data).
        self.output_path: Optional[str] = None
        self.metadata_path: Optional[str] = None
        self.writer: Any = None
        # Injected once by HardwareManager.initialize() so the camera can
        # resolve paths (`config.make_path`) outside the per-run `arm()`.
        self.config: Any = None
        # MDA gui reads `cam.core` (None for non-mmcore cameras). MMCamera
        # overrides during backend setup; others leave it None.
        self.core: Any = None
        self.camera_device: Any = None
        self._engine = None
        # Lifecycle timestamps.
        self._started: Optional[datetime] = None
        self._stopped: Optional[datetime] = None
        # `image_ready` is a pyqtSignal (or psygnal wrapper) the MDA gui's
        # non-mmcore viewer subscribes to. Subclasses set this when they
        # have a Qt-friendly emitter (OpenCVCamera does directly via
        # `pyqtSignal`; MockFrameProducer composes a `QtImageAdapter`).
        if not hasattr(self, "image_ready"):
            self.image_ready = None
        # `signals` may already be created by a parent class (e.g.
        # BaseDataProducer.__init__). Don't clobber if so.
        if not hasattr(self, "signals"):
            self.signals = DeviceSignals()
        # Per-instance logger keyed on the actual subclass name + id.
        self.logger = get_logger(
            f"{type(self).__module__}.{type(self).__name__}[{self.id}]"
        )

    # --- per-run prep ---------------------------------------------------
    def arm(self, config: Any) -> None:
        """Per-run prep: set up the writer + (optionally) an MDA sequence.

        The default body fits both MMCamera (which needs a sequence) and
        OpenCV/Mock (where ``set_sequence`` is a no-op). Subclasses
        override only when they need additional prep.
        """
        self.set_writer(config.make_path)
        self.set_sequence(config.build_sequence)

    # --- writer selection ------------------------------------------------
    # The default mapping is `file_type` (the YAML stanza's `output.file_type`
    # key) -> writer class. Subclasses can override the mapping to add new
    # output formats (e.g. a future Zarr writer) without touching `set_writer`.
    # GUI code wanting to switch writers at runtime only has to update
    # `self.file_type` and call `set_writer` again.
    _WRITER_FOR_FILE_TYPE: ClassVar[Dict[str, str]] = {
        "ome.tiff": "CustomWriter",
        "tiff": "CustomWriter",
        "mp4": "CV2Writer",
        "avi": "CV2Writer",
    }

    def _make_writer(self, filename: str) -> Any:
        """Construct the writer matching ``self.file_type``.

        Default selection: OME-TIFF (``CustomWriter``) for ``ome.tiff`` /
        ``tiff``, MP4 (``CV2Writer``) for ``mp4`` / ``avi``. Override
        :attr:`_WRITER_FOR_FILE_TYPE` on a subclass to register additional
        writers, or override this method entirely to control construction
        per-format (e.g. passing ``fps`` to a video writer).
        """
        from mesofield.data import CustomWriter, CV2Writer

        name = self._WRITER_FOR_FILE_TYPE.get(self.file_type)
        if name is None:
            raise ValueError(
                f"No writer registered for file_type {self.file_type!r}. "
                f"Known: {sorted(self._WRITER_FOR_FILE_TYPE)}"
            )
        if name == "CustomWriter":
            return CustomWriter(filename=filename)
        if name == "CV2Writer":
            fps = int(self.sampling_rate) if self.sampling_rate else 30
            return CV2Writer(filename=filename, fps=fps)
        raise ValueError(f"Unknown writer class name {name!r}")

    def set_writer(self, make_path: Callable[[str, str, str, bool], str]) -> None:
        """Generate the camera's output path and construct the writer.

        Resolves ``self.output_path`` via ``make_path``, then instantiates
        the writer that :meth:`_make_writer` picks for ``self.file_type``
        and copies the writer's sidecar filename onto ``self.metadata_path``
        for the AcquisitionManifest. Subclasses override only when they
        need additional plumbing on top (e.g. OpenCVCamera resetting its
        capture-loop timing).
        """
        self.output_path = make_path(self.name, self.file_type, self.bids_type, True)
        self.writer = self._make_writer(self.output_path)
        if hasattr(self.writer, "_frame_metadata_filename"):
            self.metadata_path = self.writer._frame_metadata_filename

    def set_sequence(self, build_mda: Callable[[Any], Any]) -> None:
        """Default no-op (only MMCamera with an MDA backend overrides this)."""
        return None

    # --- live-view + snap contract --------------------------------------
    # The MDA GUI's snap/live buttons used to call mmcore directly; with
    # BaseCamera in place they can call these methods on the camera and
    # work uniformly across MMCamera (mmcore-driven), OpenCVCamera
    # (cv2.VideoCapture-driven), and MockFrameProducer (synthetic).
    # Subclasses MUST implement; the base raises NotImplementedError so a
    # mis-wired GUI surfaces it immediately instead of silently no-op'ing.

    def snap(self):
        """Capture a single frame outside any recording, return it as an ndarray.

        Used by the GUI's snap button. Implementations should NOT alter
        recording state -- snap is preview-only -- but they SHOULD call
        :meth:`_save_snap_png` so each snap also lands a ``*_snap.png``.
        """
        raise NotImplementedError(f"{type(self).__name__}.snap() not implemented")

    def _save_snap_png(self, frame: Any) -> Optional[str]:
        """Write a snapped frame to ``<name>_snap.png`` at the BIDS path.

        No-op when no :class:`ExperimentConfig` has been injected. Uses
        ``config.make_path`` so the snapshot follows the same BIDS layout
        (and ``bids_type``) as the camera's recordings.
        """
        if frame is None or self.config is None:
            return None
        make_path = getattr(self.config, "make_path", None)
        if make_path is None:
            return None
        try:
            import numpy as np
            from PIL import Image

            path = make_path(f"{self.name}_snap", "png", self.bids_type, True)
            arr = np.asarray(frame)
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = arr[..., ::-1]  # OpenCV BGR -> RGB
            if arr.dtype != np.uint8:
                # Scale to 8-bit so the snapshot PNG is viewable.
                lo, hi = float(arr.min()), float(arr.max())
                scale = 255.0 / (hi - lo) if hi > lo else 1.0
                arr = np.clip((arr.astype(np.float32) - lo) * scale, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(path)
            self.logger.info(f"Snapshot saved to {path}")
            return path
        except Exception as exc:
            self.logger.error(f"Failed to save snapshot PNG: {exc}")
            return None

    def start_live(self) -> None:
        """Begin continuous live preview WITHOUT writing to disk.

        Subscribers receive frames via ``image_ready`` (Qt) / ``signals.data``
        (psygnal). No recording side-effects; pair with :meth:`stop_live`.
        """
        raise NotImplementedError(f"{type(self).__name__}.start_live() not implemented")

    def stop_live(self) -> None:
        """End the continuous live preview started by :meth:`start_live`."""
        raise NotImplementedError(f"{type(self).__name__}.stop_live() not implemented")

    # --- standard introspection -----------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "backend": self.backend,
            "is_primary": self.is_primary,
            "is_active": self.is_active,
            "output_path": self.output_path,
            "metadata_path": self.metadata_path,
            "started": self._started.isoformat() if self._started else None,
            "stopped": self._stopped.isoformat() if self._stopped else None,
        }

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "backend": self.backend,
            "name": self.name,
            "fps": self.sampling_rate,
            "file_type": self.file_type,
            "bids_type": self.bids_type,
        }

    @property
    def calibration(self) -> Dict[str, Any]:
        """Camera-specific constants worth recording in the AcquisitionManifest."""
        return {
            "name": self.name,
            "sampling_rate_hz": self.sampling_rate,
            "viewer": self.viewer,
        }

    def sidecars(self) -> list:
        """Extra sidecars beyond ``metadata_path``. Cameras typically have none."""
        return []

    def shutdown(self) -> None:
        """Default cleanup: stop the camera. Subclasses extend if needed."""
        try:
            self.stop()
        except Exception:
            pass
