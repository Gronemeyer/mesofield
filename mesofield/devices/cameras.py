from __future__ import annotations

import time
from typing import Optional, Callable, Any, ClassVar, Dict
from datetime import datetime
from pathlib import Path
import inspect

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from pymmcore_plus import CMMCorePlus, DeviceType
from pymmcore_plus.core._device import CameraDevice

from mesofield.protocols import HardwareDevice, DataProducer
from mesofield.engines import DevEngine, MesoEngine, PupilEngine
#from tests.arducam import VideoThread
from mesofield.devices.base_camera import BaseCamera
from mesofield.data import CustomWriter, CV2Writer
from mesofield.data.writer import configure_opencv_codec
from mesofield import DeviceRegistry


@DeviceRegistry.register("camera")
class MMCamera(BaseCamera, DataProducer, HardwareDevice):
    """Micro-Manager-backed camera.

    Inherits the common camera surface (identity, output paths, manifest
    metadata, ``arm`` / ``set_sequence`` defaults, ``status``, ``calibration``)
    from :class:`BaseCamera`, and duck-types the ``DataProducer`` /
    ``HardwareDevice`` Protocols so existing isinstance() checks keep
    working. The actual frame flow is driven by pymmcore-plus's MDA event
    system; this class wires those events into the standard
    :class:`DeviceSignals` bundle and constructs a :class:`CustomWriter`
    (OME-TIFF) or :class:`CV2Writer` (MP4) for the output.
    """

    sampling_rate: float = 30.0  # Default sampling rate in Hz
    file_type: ClassVar[str] = "ome.tiff"
    bids_type: ClassVar[Optional[str]] = "func"
    writer: CustomWriter | CV2Writer

    def __init__(self, cfg: dict):
        # BaseCamera surface (signals, identity, viewer cosmetics, output
        # slots, logger). MMCamera carries an explicit `backend` (set from
        # cfg), so we pass through the cfg-declared value here.
        backend = str(cfg.get("backend", "")).lower()
        if backend not in {"micromanager", "opencv"}:
            raise ValueError(f"Unknown camera backend '{backend}'")
        self._init_camera_surface(cfg, backend=backend)
        # MMCamera-specific state.
        self.camera_device: Optional[CameraDevice] = None
        self.core: Optional[CMMCorePlus] = None
        self.properties = self.cfg.get("properties", {})

        if self.backend == "micromanager":
            self._setup_micromanager(self.cfg)
        elif self.backend == "opencv":
            self._setup_opencv()

        # automatically apply all YAML properties
        self.initialize()
        self._wire_signals()

    def _wire_signals(self) -> None:
        """Bridge MMCore MDA events to the standard DeviceSignals."""
        if self.backend != "micromanager" or self.core is None:
            return
        evt = self.core.mda.events  # type: ignore[union-attr]

        def _on_started(*_args, **_kw):
            self.signals.started.emit()

        def _on_finished(*_args, **_kw):
            self.signals.finished.emit()

        def _on_frame(_img, _event, metadata):
            idx = ts = None
            try:
                cam_meta = metadata.get("camera_metadata", {}) if isinstance(metadata, dict) else {}
                idx = cam_meta.get("ImageNumber")
                ts = cam_meta.get("TimeReceivedByCore")
                self.signals.data.emit(idx, ts)
            except Exception:
                pass
            try:
                self.signals.frame.emit(_img, idx, ts)
            except Exception:
                pass

        try:
            evt.sequenceStarted.connect(_on_started)
        except Exception:
            pass
        try:
            evt.sequenceFinished.connect(_on_finished)
        except Exception:
            pass
        try:
            evt.frameReady.connect(_on_frame)
        except Exception:
            pass

    def _setup_micromanager(self, cfg):
        core = CMMCorePlus(cfg.get("micromanager_path"))
        cfg_path = cfg.get("configuration_path")
        core.loadSystemConfiguration(cfg_path) if cfg_path else core.loadSystemConfiguration()
        self.camera_device = core.getDeviceObject(core.getCameraDevice(),
                                                  DeviceType.Camera)
        Engine = {"ThorCam": PupilEngine,
                  "Dhyana": MesoEngine}.get(self.id, DevEngine)
        self._engine = Engine(core, use_hardware_sequencing=True)
        # Back-reference so engines can call camera helpers (e.g. LED control).
        try:
            self._engine.camera = self
        except Exception:
            pass
        core.mda.set_engine(self._engine)
        self.core = core

    # `set_writer` is inherited from BaseCamera: it resolves the output path
    # and picks the writer (CustomWriter for OME-TIFF, CV2Writer for MP4)
    # from the shared `_WRITER_FOR_FILE_TYPE` mapping. Override only if you
    # need to swap the mapping for a specific MMCamera subclass.

    def set_sequence(self, build_mda: Callable[[DataProducer], Any]):
        """Build the MDA sequence (Micro-Manager backend only)."""
        if self.backend == "micromanager":
            self.sequence = build_mda(self)
            self.logger.info("MDA sequence set for Micromanager backend.")
        else:
            self.logger.warning("Setting sequence is not supported for OpenCV backend.")

    def initialize(self):
        """Apply the YAML ``properties`` block to the underlying camera.

        Each ``{device_id: {property: value}}`` pair is forwarded to the
        backend (``core.setROI`` for ROIs, ``core.setProperty`` otherwise)
        with special handling for the synthetic ``fps``, ``viewer_type``,
        and ``auto_contrast`` keys.
        """
        for dev_id, props in self.properties.items():
            if not isinstance(props, dict):
                continue
            for prop, val in props.items():
                self.logger.info(f"Setting {dev_id}.{prop} → {val}")
                if prop == "ROI":
                    roi_setter = getattr(self.core, "setROI", None) if self.backend == "micromanager" else None
                    if roi_setter:
                        roi_setter(dev_id, *val)
                elif prop == "fps":
                    setattr(self, "sampling_rate", val)
                elif prop == "viewer_type":
                    setattr(self, "viewer", val)
                elif prop == "auto_contrast":
                    setattr(self, "auto_contrast", val)
                else:
                    if self.backend == "micromanager":
                        setter = getattr(self.core, "setProperty", None)
                    else:
                        setter = getattr(self.camera_device, "setProperty", None)
                    if setter:
                        setter(dev_id, prop, val)

    # `arm()` is inherited from BaseCamera: it calls set_writer + set_sequence.

    # ------------------------------------------------------------------
    # LED control
    # ------------------------------------------------------------------
    # The Arduino-Switch device adapter in some Micro-Manager configs becomes
    # unusable when certain Dhyana camera properties are set (its sequenced
    # ``State`` property no longer responds to ``loadSequence``/``startSequence``).
    # The underlying serial port still works, so when ``led_serial`` is present
    # in the camera YAML block we bypass the MM Arduino device and write raw
    # bytes through MM's SerialManager (which already owns the COM port).
    #
    # YAML schema (sibling of ``properties``)::
    #
    #     led_serial:
    #       port: "COM3"                 # MM SerialManager device label
    #       start_command:               # list of byte sequences sent in order
    #         - [5, 0, 4]
    #         - [6, 1]
    #         - [8]
    #       stop_command:
    #         - [9]
    def _led_serial_cfg(self) -> Optional[dict]:
        cfg = self.cfg.get("led_serial") if isinstance(self.cfg, dict) else None
        return cfg if isinstance(cfg, dict) else None

    def _send_serial_commands(self, commands) -> None:
        """Write a list of byte sequences to the configured MM serial port."""
        led = self._led_serial_cfg() or {}
        port = led.get("port")
        if not port or not commands or self.core is None:
            return
        writer = getattr(self.core, "writeToSerialPort", None)
        if writer is None:
            self.logger.warning("core has no writeToSerialPort; cannot send LED bytes")
            return
        for cmd in commands:
            byte_list = [int(b) & 0xFF for b in cmd]
            self.logger.info(f"writeToSerialPort({port!r}, {byte_list})")
            writer(port, byte_list)

    def start_led_sequence(self, pattern) -> None:
        """Start the LED pattern.

        If ``led_serial`` is configured on this camera, sends the configured
        raw byte sequences via MM's SerialManager.  Otherwise falls back to
        the original ``Arduino-Switch.State.loadSequence/startSequence`` path.
        """
        led = self._led_serial_cfg()
        self.logger.info(
            f"start_led_sequence: led_serial={'present' if led else 'absent'} "
            f"pattern={pattern}"
        )
        if led is not None:
            self._send_serial_commands(led.get("start_command", []))
            return
        if self.backend != "micromanager" or self.core is None:
            return
        prop = self.core.getPropertyObject("Arduino-Switch", "State")
        prop.loadSequence(pattern)
        prop.setValue(4)  # seems essential to initiate serial communication
        prop.startSequence()

    def stop_led_sequence(self) -> None:
        """Stop the LED pattern (mirror of :meth:`start_led_sequence`)."""
        led = self._led_serial_cfg()
        self.logger.info(
            f"stop_led_sequence: led_serial={'present' if led else 'absent'}"
        )
        if led is not None:
            self._send_serial_commands(led.get("stop_command", []))
            return
        if self.backend != "micromanager" or self.core is None:
            return
        try:
            self.core.getPropertyObject("Arduino-Switch", "State").stopSequence()
        except Exception as exc:
            self.logger.warning(f"stopSequence failed: {exc}")

    def start(self) -> bool:
        """Launch the MDA sequence non-blocking.

        Returns:
            Always ``True``. The sequence runs asynchronously on the
            camera backend; lifecycle is reported via ``self.signals``.
        """
        self._started = datetime.now()
        self.core.run_mda(events=self.sequence, output=self.writer, block=False)  # type: ignore
        return True

    def stop(self) -> bool:
        """Stop acquisition.

        Non-primary Micro-Manager cameras must be told explicitly to halt
        their sequence acquisition — the primary camera's MDA driver
        does not stop them.
        """
        self._stopped = datetime.now()
        if (
            self.backend == "micromanager"
            and not self.is_primary
            and self.core is not None
        ):
            try:
                self.core.stopSequenceAcquisition()
            except Exception as exc:
                self.logger.warning(f"stopSequenceAcquisition failed: {exc}")
        return True

    def get_data(self):
        """Return the latest captured frame, or ``None`` if not active."""
        return getattr(self.camera_device, "get_frame", lambda: None)() if self.is_active else None

    # --- live preview contract (BaseCamera abstract methods) ------------
    # Implementations delegate to mmcore so the GUI's snap/live buttons
    # work without touching pymmcore directly.

    def snap(self):
        """Capture a single frame via ``mmcore.snap()`` and save a snapshot PNG."""
        if self.backend != "micromanager" or self.core is None:
            raise RuntimeError("snap() requires the micromanager backend")
        frame = self.core.snap()
        self._save_snap_png(frame)
        return frame

    def start_live(self) -> None:
        """Begin continuous (untimed) sequence acquisition for preview."""
        if self.backend != "micromanager" or self.core is None:
            raise RuntimeError("start_live() requires the micromanager backend")
        if not self.core.isSequenceRunning():
            self.core.startContinuousSequenceAcquisition()

    def stop_live(self) -> None:
        """End the continuous sequence acquisition started by start_live."""
        if self.backend != "micromanager" or self.core is None:
            return
        try:
            self.core.stopSequenceAcquisition()
        except Exception as exc:
            self.logger.debug(f"stopSequenceAcquisition: {exc}")

    def shutdown(self):
        """Cancel any in-flight MDA on the Micro-Manager backend."""
        if self.backend == "micromanager" and hasattr(self.core, "reset"):
            self.core.mda.cancel()
    
    def __getattr__(self, name: str):
        """
        Any attribute not found on MMCamera will be looked up
        on the wrapped camera_device automatically.
        """
        if self.camera_device is not None and hasattr(self.camera_device, name):
            return getattr(self.camera_device, name)
        raise AttributeError(f"{self.__class__.__name__!r} has no attribute {name!r}")

    def __dir__(self):
        """
        Include camera_device’s public attributes in dir(self) so
        tab‐complete / introspection still works.
        """
        base = set(super().__dir__())
        if self.camera_device is not None:
            base.update(n for n in dir(self.camera_device) if not n.startswith("_"))
        return sorted(base)
    
    def __repr__(self):
        # Compact one-liner so HardwareManager listings stay readable.
        # Use ``cam.info()`` for the verbose dump.
        cam_props = self.properties.get(self.id, {}) if isinstance(self.properties, dict) else {}
        fps = cam_props.get("fps")
        out = self.cfg.get("output", {}) if isinstance(self.cfg, dict) else {}
        bits = [f"id={self.id!r}", f"backend={self.backend!r}"]
        if fps is not None:
            bits.append(f"fps={fps}")
        if self.is_primary:
            bits.append("primary=True")
        if out.get("suffix") or out.get("file_type"):
            bits.append(
                f"output={out.get('suffix','?')}.{out.get('file_type','?')!r}"
            )
        if self._engine is not None:
            bits.append(f"engine={type(self._engine).__name__!r}")
        return f"<MMCamera {' '.join(bits)}>"

    def info(self) -> str:
        """Return the verbose multi-line description (module path, MRO, properties)."""
        module = inspect.getmodule(self)
        module_name = module.__name__ if module else "<unknown>"
        module_file = getattr(module, "__file__", "<built-in>")
        mro = inspect.getmro(self.__class__)
        inheritance = " -> ".join(cls.__name__ for cls in reversed(mro))
        return (
            f"<MMCamera.{self.id}>\n"
            f"  backend     = {self.backend!r}\n"
            f"  module      = {module_name!r} ({module_file!r})\n"
            f"  properties  = {self.properties!r}\n"
            f"  engine      = {type(self._engine).__name__!r}\n"
            f"  inheritance = {inheritance}\n"
        )

# ============================ OpenCV camera ============================ #
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
class OpenCVCamera(BaseCamera, QThread):
    """Background-thread OpenCV camera capturing to MP4.

    Emits via ``self.signals`` (a :class:`mesofield.signals.DeviceSignals`):
      - ``signals.started`` / ``signals.finished`` for lifecycle.
      - ``signals.data(idx, device_ts)`` per frame, consumed by
        :meth:`DataManager.register_hardware_device`.
    Plus Qt live-preview signals (GUI-only, decoupled from DataQueue):
      - ``frame_ready(np.ndarray)`` / ``image_ready(np.ndarray)``.

    Inherits the common camera surface (identity, output paths, manifest
    metadata, ``arm``/``set_sequence`` defaults) from :class:`BaseCamera`,
    and runs its own capture loop on top of :class:`QThread`.
    """

    # ----- Qt signals (GUI-only live preview) ----------------------------
    # Class-level pyqtSignals are valid because QThread is a QObject.
    frame_ready = pyqtSignal(np.ndarray)
    # Alias used by the mesofield GUI's `InteractivePreview` (matches the
    # signal name expected by `arducam.VideoThread`).
    image_ready = pyqtSignal(np.ndarray)
    # Recording progress: (frames_written, expected_total). expected_total is
    # 0 when the duration is unknown (drives an indeterminate progress bar).
    progress = pyqtSignal(int, int)

    # ----- Camera surface overrides (BaseCamera defaults are tiff/func) --
    file_type: ClassVar[str] = "mp4"
    bids_type: ClassVar[Optional[str]] = "beh"
    data_type: ClassVar[str] = "video"

    def __init__(self, cfg: Dict[str, Any]):
        QThread.__init__(self)
        # BaseCamera surface (signals, identity, viewer cosmetics, output
        # slots, logger). image_ready is already a class-level pyqtSignal,
        # so _init_camera_surface's `if not hasattr(self, 'image_ready')`
        # guard leaves it alone.
        self._init_camera_surface(cfg, backend="opencv")
        # OpenCV-specific knobs.
        self.device_index: int = int(self.cfg.get("device_index", 0))
        self.cv_backend_name: str = str(self.cfg.get("cv_backend", _DEFAULT_CV_BACKEND))
        # Default fps if cfg didn't carry one (BaseCamera reads `fps` from cfg).
        if not self.sampling_rate:
            self.sampling_rate = 30.0
        self.fourcc: str = str(self.cfg.get("fourcc", "H264"))
        self.is_color: bool = bool(self.cfg.get("color", True))

        # Optional explicit width/height overrides; otherwise driver default
        self._req_width: Optional[int] = self.cfg.get("width")
        self._req_height: Optional[int] = self.cfg.get("height")

        # Apply YAML "properties" block (mirrors MMCamera semantics).
        self.properties: Dict[str, Any] = self.cfg.get("properties", {}) or {}
        for key, val in self.properties.items():
            if key == "fps":
                self.sampling_rate = float(val)
            elif key == "viewer_type":
                self.viewer = val
            elif key == "auto_contrast":
                self.auto_contrast = val

        # Capture loop state.
        self._capture: Optional[Any] = None  # cv2.VideoCapture
        self._frame_index: int = 0
        self._frame_timestamps: list[tuple[int, float]] = []  # (idx, perf_counter)
        self._stop = False
        # Expected recording length in frames (0 = unknown); set by set_writer.
        self._expected_frames: int = 0

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
        st = super().status()
        st.update({
            "device_index": self.device_index,
            "cv_backend": self.cv_backend_name,
            "frames_written": self._frame_index,
        })
        return st

    # Backwards-compat alias used elsewhere in the codebase
    def get_status(self) -> Dict[str, Any]:
        return self.status()

    @property
    def metadata(self) -> Dict[str, Any]:
        md = super().metadata
        md.update({
            "device_index": self.device_index,
            "cv_backend": self.cv_backend_name,
            "width": getattr(self, "_frame_width", None),
            "height": getattr(self, "_frame_height", None),
            "fourcc": self.fourcc,
        })
        return md

    # ----- DataProducer protocol -------------------------------------------
    def set_writer(self, make_path: Callable[[str, str, str, bool], str]) -> None:
        """Resolve the output path and build the :class:`CV2Writer`.

        ``BaseCamera.set_writer`` resolves ``output_path``, constructs the
        ``CV2Writer`` (the project's shared MP4 writer), and copies its
        sidecar path onto ``metadata_path``. The capture loop drives the
        writer directly via ``begin``/``add_frame``/``finish``.
        """
        super().set_writer(make_path)
        # Reset timing so saved timestamps reflect the recording start, not
        # whenever the live-preview thread first launched.
        self._started = datetime.now()
        self._frame_timestamps = []
        self._frame_index = 0
        # Expected frame count drives the GUI progress bar. Falls back to 0
        # (indeterminate) when the experiment duration is unavailable.
        duration = getattr(self.config, "sequence_duration", None)
        try:
            self._expected_frames = int(self.sampling_rate * float(duration))
        except (TypeError, ValueError):
            self._expected_frames = 0
        self.logger.info(f"Writer set to {self.output_path}")

    def start(self) -> bool:
        """Spawn the capture thread and begin writing frames to MP4.

        Returns:
            ``True`` if the thread started, ``False`` if it was already
            running.
        """
        if self.isRunning():
            self.logger.warning("OpenCVCamera.start: already running")
            return False
        self._stop = False
        self._frame_index = 0
        self._frame_timestamps = []
        self._started = datetime.now()
        self.is_active = True
        self.signals.started.emit()
        super().start()  # spawns QThread.run
        return True

    def stop(self) -> bool:
        """Signal the capture thread to stop and wait for it to join."""
        if not self.isRunning() and not self._stop:
            return True
        self._stop = True
        self.requestInterruption()
        self.wait(5000)
        self.is_active = False
        self._stopped = datetime.now()
        self.signals.finished.emit()
        return True

    def shutdown(self) -> None:
        """Tear down the capture thread.

        ``stop()`` joins the capture thread; its ``run()`` finally-block
        releases the :class:`CV2Writer` and writes the sidecar JSON.
        """
        self.stop()

    # --- live preview contract (BaseCamera abstract methods) ------------
    # `snap()` reads a frame from a one-shot VideoCapture; the live capture
    # thread is the same one driven by start()/stop(), just without a
    # writer attached.

    def snap(self) -> np.ndarray:
        """Open the camera, read one frame, return it without recording."""
        import cv2

        backend_id = _resolve_cv_backend(self.cv_backend_name)
        cap = cv2.VideoCapture(self.device_index, backend_id)
        try:
            if not cap.isOpened():
                raise RuntimeError("OpenCVCamera.snap: could not open camera")
            if self._req_width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self._req_width))
            if self._req_height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self._req_height))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("OpenCVCamera.snap: VideoCapture.read failed")
            # Fan the snapped frame out to GUI subscribers too.
            try:
                self.image_ready.emit(frame)
            except Exception:
                pass
            self._save_snap_png(frame)
            return frame
        finally:
            cap.release()

    def start_live(self) -> None:
        """Start the capture thread WITHOUT a writer (preview-only)."""
        if self.isRunning():
            return
        # Ensure no writer is attached so the capture loop's
        # `if self.writer is not None` guard skips disk writes.
        self.writer = None
        self._stop = False
        self._frame_index = 0
        self._frame_timestamps = []
        self.is_active = True
        super().start()

    def stop_live(self) -> None:
        """End preview capture started by start_live."""
        if not self.isRunning() and not self._stop:
            return
        self._stop = True
        self.requestInterruption()
        self.wait(5000)
        self.is_active = False

    def get_data(self) -> Optional[Dict[str, Any]]:
        """Return the most recent frame index/timestamp pair, or ``None``."""
        if not self._frame_timestamps:
            return None
        idx, ts = self._frame_timestamps[-1]
        return {"frame_index": idx, "device_ts": ts}

    def save_data(self, path: Optional[str] = None) -> None:
        """No-op: the MP4 and its ``_frame_metadata.json`` sidecar are written
        by the :class:`CV2Writer` when the capture loop finishes.
        """
        return None

    def _frame_metadata(self) -> Dict[str, Any]:
        """Per-frame timestamp metadata handed to ``CV2Writer.finish``."""
        return {
            "device": self.metadata,
            "started": self._started.isoformat() if self._started else None,
            "stopped": self._stopped.isoformat() if self._stopped else None,
            "frames": [
                {"index": i, "device_ts": ts} for i, ts in self._frame_timestamps
            ],
        }

    # ----- Internal --------------------------------------------------------
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
        # Open the CV2Writer for this recording (skipped for live preview,
        # where `start_live()` clears `self.writer`).
        if self.writer is not None:
            try:
                self.writer.begin(self._frame_width, self._frame_height, self.is_color)
            except Exception as exc:  # pragma: no cover - codec failure
                self.logger.error(f"CV2Writer.begin failed: {exc}")
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

                # Write to disk via the shared CV2Writer
                if self.writer is not None:
                    if not self.is_color and frame.ndim == 3:
                        frame_to_write = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        frame_to_write = frame
                    try:
                        self.writer.add_frame(frame_to_write)
                    except Exception as exc:  # pragma: no cover - codec failure
                        self.logger.error(f"CV2Writer.add_frame failed: {exc}")

                # GUI live-preview signals (Qt-native, decoupled from queue)
                self.frame_ready.emit(frame)
                self.image_ready.emit(frame)
                # Recording progress (only meaningful while a writer is attached)
                if self.writer is not None:
                    self.progress.emit(idx + 1, self._expected_frames)
                # Standardized data signal -> DataQueue
                self.signals.data.emit(idx, ts)
                # Optional raw-frame signal for real-time processors.
                try:
                    self.signals.frame.emit(frame, idx, ts)
                except Exception:
                    pass

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
                    self.writer.finish(extra_metadata=self._frame_metadata())
                except Exception as exc:
                    self.logger.error(f"CV2Writer.finish failed: {exc}")
                # `-1` is the "recording done -- hide the progress bar" sentinel.
                self.progress.emit(-1, 0)
            self.logger.info(
                f"OpenCV capture stopped: wrote {self._frame_index} frames"
            )
