from typing import Optional, Callable, Any, ClassVar
from datetime import datetime
import inspect

from pymmcore_plus import CMMCorePlus, DeviceType
from pymmcore_plus.core._device import CameraDevice

from mesofield.protocols import HardwareDevice, DataProducer
from mesofield.engines import DevEngine, MesoEngine, PupilEngine
from mesofield.io.devices.arducam import VideoThread
from mesofield.io.devices.base_camera import BaseCamera
from mesofield.io import CustomWriter, CV2Writer
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
        self.camera_device: Optional[CameraDevice | VideoThread] = None
        self.core: Optional[CMMCorePlus | VideoThread] = None
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
            try:
                cam_meta = metadata.get("camera_metadata", {}) if isinstance(metadata, dict) else {}
                idx = cam_meta.get("ImageNumber")
                ts = cam_meta.get("TimeReceivedByCore")
                self.signals.data.emit(idx, ts)
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
        core.mda.set_engine(self._engine)
        self.core = core

    def _setup_opencv(self):
        vid = VideoThread()
        self.camera_device = vid
        self.core = vid

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

    def start(self) -> bool:
        #self.is_active = True
        self._started = datetime.now()
        self.core.run_mda(events=self.sequence, output=self.writer, block=False) #type: ignore
        return True

    def stop(self) -> bool:
        """Stop acquisition.  Non-primary MM cameras must be told explicitly
        to halt their sequence acquisition (the primary camera's MDA driver
        does not stop them)."""
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
        return getattr(self.camera_device, "get_frame", lambda: None)() if self.is_active else None

    # --- live preview contract (BaseCamera abstract methods) ------------
    # Implementations delegate to mmcore so the GUI's snap/live buttons
    # work without touching pymmcore directly.

    def snap(self):
        """Capture and return a single frame via ``mmcore.snap()``."""
        if self.backend != "micromanager" or self.core is None:
            raise RuntimeError("snap() requires the micromanager backend")
        return self.core.snap()

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
        if self.backend == "micromanager" and hasattr(self.core, "reset"):
            self.core.mda.cancel()
            #self.core.reset()
    
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