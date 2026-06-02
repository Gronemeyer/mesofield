VALID_BACKENDS = {"micromanager", "opencv"}

import os
from typing import Dict, Any, List, Optional, ClassVar
import yaml

from mesofield.protocols import HardwareDevice, DataProducer
from mesofield.devices import Nidaq, MMCamera, SerialWorker, EncoderSerialInterface
from mesofield.utils._logger import get_logger, log_this_fr
from mesofield import DeviceRegistry

class HardwareManager():
    """
    High-level class that initializes all hardware (cameras, encoder, etc.)
    using the ParameterManager. Keeps references easily accessible.

    When *config_file* is ``None`` or the file does not exist the manager
    starts in an **unconfigured** default state.  Call
    :meth:`load_config` later to point it at a real YAML file and then
    :meth:`initialize` to bring hardware up.
    """

    def __init__(self, config_file: Optional[str] = None, devices=None):
        self.logger = get_logger(f'{__name__}.{self.__class__.__name__}')
        self.logger.info(f"Initializing HardwareManager with config: {config_file}")

        self.config_file = config_file
        self.devices: Dict[str, DataProducer] = {}
        self._configured: bool = False
        # Devices constructed programmatically (scripted procedures). When set,
        # ``initialize`` registers these directly instead of parsing YAML.
        self._prebuilt_devices: Optional[List[Any]] = (
            list(devices) if devices else None
        )

        if self._prebuilt_devices:
            self.yaml = {}
            self._configured = True
            self.logger.info(
                f"Initialized with {len(self._prebuilt_devices)} pre-built device(s)."
            )
        elif config_file and os.path.isfile(config_file):
            try:
                self.yaml = self._load_yaml(config_file)
                self._configured = True
                self.logger.info("Successfully loaded hardware configuration")
            except Exception as e:
                self.logger.error(f"Failed to load hardware configuration: {e}")
                self.yaml = {}
        else:
            self.yaml = {}
            if config_file:
                self.logger.debug(
                    f"Hardware config not found: {config_file}. "
                    "Starting in unconfigured state."
                )
            else:
                self.logger.info("No hardware config provided. Starting in default state.")

        self.widgets: List[str] = self._aggregate_widgets()
        self.cameras: tuple[MMCamera, ...] = ()
        self.encoder = None
        self.nidaq = None
        self.psychopy = None
        self._viewer = self.yaml.get('viewer_type', 'static')

    @property
    def is_configured(self) -> bool:
        """``True`` when a valid YAML config has been loaded."""
        return self._configured

    def load_config(self, config_file: str) -> None:
        """Load (or reload) a hardware YAML file.

        This does **not** initialise devices – call :meth:`initialize` afterwards.
        """
        self.config_file = config_file
        self.yaml = self._load_yaml(config_file)
        self._configured = True
        self.widgets = self._aggregate_widgets()
        self._viewer = self.yaml.get('viewer_type', 'static')
        self.logger.info(f"Loaded hardware config: {config_file}")

    def __repr__(self):
        # Compact, IPython-friendly summary. Devices are introspected
        # generically: any attribute in ``_REPR_FIELDS`` that exists and
        # has a meaningful value is shown. Top-level YAML config is
        # summarized to its keys (full content is still in ``self.yaml``).
        _REPR_FIELDS = (
            "device_id", "device_type", "backend",
            "port", "baudrate", "sampling_rate",
            "is_primary", "is_active",
        )

        def _short(dev) -> str:
            attrs = []
            for k in _REPR_FIELDS:
                if not hasattr(dev, k):
                    continue
                v = getattr(dev, k)
                if v in (None, "", False):
                    continue
                attrs.append(f"{k}={v!r}")
            # Pull fps from the device's camera-style cfg if available.
            cfg = getattr(dev, "cfg", None)
            if isinstance(cfg, dict):
                cam_id = getattr(dev, "id", "") or getattr(dev, "device_id", "")
                fps = cfg.get("properties", {}).get(cam_id, {}).get("fps") \
                    or cfg.get("fps")
                if fps is not None:
                    attrs.append(f"fps={fps!r}")
                output = cfg.get("output") or {}
                suffix = output.get("suffix")
                ftype = output.get("file_type")
                if suffix or ftype:
                    attrs.append(
                        "output=" + repr(f"{suffix or '?'}.{ftype or '?'}")
                    )
            return f"{type(dev).__name__}(" + ", ".join(attrs) + ")"

        lines = ["<HardwareManager>"]
        if self.cameras:
            lines.append("  Cameras:")
            for cam in self.cameras:
                lines.append(f"    - {_short(cam)}")
        else:
            lines.append("  Cameras: <none>")

        extras = {
            k: v for k, v in self.devices.items() if v not in self.cameras
        }
        if extras:
            lines.append("  Devices:")
            for name, dev in extras.items():
                lines.append(f"    - {name}: {_short(dev)}")

        if self.config_file:
            lines.append(f"  Config file: {self.config_file}")
        if self.yaml:
            lines.append(f"  Config keys: {sorted(self.yaml.keys())}")
        lines.append("</HardwareManager>")
        return "\n".join(lines)

    # ---- Public interface --------------------------------------------------

    def initialize(self, cfg) -> None:
        """Initialize all devices from YAML and configure engines.

        Does nothing if the manager has no loaded YAML configuration.
        Validates that exactly one device is flagged ``primary: true``.
        """
        if not self._configured:
            self.logger.warning("Cannot initialize hardware: no YAML config loaded.")
            return
        if self._prebuilt_devices is not None:
            self.logger.info("Initializing hardware devices from pre-built objects...")
            self._register_prebuilt_devices()
        else:
            self.logger.info("Initializing hardware devices from YAML configuration...")
            self._init_cameras()
            self._init_encoder()
            self._init_daq()
            self._init_psychopy()
            self._init_extras()
        self._configure_engines(cfg)
        # Inject the ExperimentConfig onto every device so producers can reach
        # `make_path` and experiment state outside the per-run `arm(config)`
        # call (e.g. a camera's snap-and-save before a run is armed).
        for device in self.devices.values():
            device.config = cfg
        self._validate_primary()

    # Top-level YAML keys handled by dedicated initializers above.
    # Anything else with a ``type:`` field is dispatched through
    # ``_init_extras`` against the global :class:`DeviceRegistry`.
    _RESERVED_YAML_KEYS = frozenset({
        "cameras", "encoder", "nidaq", "psychopy",
        "memory_buffer_size", "blue_led_power_mw", "violet_led_power_mw",
        "viewer_type", "widgets",
    })

    def _init_extras(self) -> None:
        """Instantiate any extra YAML stanza with a registered ``type:``.

        Lets users add custom devices to ``hardware.yaml`` without
        editing :class:`HardwareManager`. The stanza must be a mapping
        whose ``type`` field matches a key registered via
        ``@DeviceRegistry.register(...)``.
        """
        for key, params in (self.yaml or {}).items():
            if key in self._RESERVED_YAML_KEYS:
                continue
            if not isinstance(params, dict):
                continue
            type_key = params.get("type")
            if not type_key:
                continue
            Cls = DeviceRegistry.get_class(type_key)
            if Cls is None:
                self.logger.warning(
                    f"YAML stanza '{key}' has type='{type_key}' but no class "
                    f"is registered for it; skipping."
                )
                continue
            cfg = dict(params)
            cfg.setdefault("id", key)
            try:
                device = Cls(cfg)
            except Exception as exc:
                self.logger.error(f"Failed to construct '{key}' ({type_key}): {exc}")
                continue
            self._apply_output_args(device, params.get("output", {}), key)
            device.is_primary = bool(params.get("primary", False))
            try:
                if hasattr(device, "initialize"):
                    device.initialize()
            except Exception as exc:
                self.logger.error(f"initialize() failed for '{key}': {exc}")
                continue
            dev_id = getattr(device, "device_id", key)
            self.devices[dev_id] = device
            setattr(self, dev_id, device)
            # Camera-class extras land in self.cameras too so the MDA gui
            # builds a live-view widget for them. This is the path
            # MockFrameProducer takes; real MMCameras come through
            # _init_cameras (which keys off the dedicated `cameras:` YAML
            # stanza) and never touch this branch.
            if getattr(device, "device_type", None) == "camera" and device not in self.cameras:
                self.cameras = self.cameras + (device,)
            self.logger.info(f"Registered extra device '{dev_id}' (type={type_key}).")

    # ---- Pre-built device path (scripted procedures) ----------------------

    def _register_prebuilt_devices(self) -> None:
        """Register devices constructed programmatically by a Procedure.

        Mirrors :meth:`_init_extras` but skips construction -- the objects are
        already instantiated. Camera-type devices are folded into
        ``self.cameras``; everything else lives in ``self.devices`` and as an
        attribute keyed by ``device_id``. Dedicated slots (``self.encoder`` /
        ``self.nidaq`` / ``self.psychopy``) are intentionally not populated, so
        a scripted setup exported via :meth:`to_yaml` and re-loaded as YAML
        extras behaves the same way.
        """
        cams = list(self.cameras)
        for device in self._prebuilt_devices or []:
            dev_id = getattr(device, "device_id", None) or getattr(device, "id", None)
            if not dev_id:
                self.logger.error(
                    f"Pre-built device {device!r} has no device_id/id; skipping."
                )
                continue
            cfg = getattr(device, "cfg", {}) or {}
            self._apply_output_args(device, cfg.get("output", {}), dev_id)
            if not getattr(device, "is_primary", False):
                device.is_primary = bool(cfg.get("primary", False))
            try:
                if hasattr(device, "initialize"):
                    device.initialize()
            except Exception as exc:
                self.logger.error(f"initialize() failed for '{dev_id}': {exc}")
                continue
            self.devices[dev_id] = device
            setattr(self, dev_id, device)
            if getattr(device, "device_type", None) == "camera" and device not in cams:
                cams.append(device)
            self.logger.info(
                f"Registered pre-built device '{dev_id}' "
                f"(device_type={getattr(device, 'device_type', None)})."
            )
        self.cameras = tuple(cams)

    def to_yaml(self, path: Optional[str] = None) -> dict:
        """Serialize the current devices into a ``hardware.yaml`` mapping.

        Each device becomes a top-level ``type:``-tagged stanza keyed by its
        device id, re-importable through :meth:`_init_extras`. This is the
        migration path from a scripted procedure to a reusable rig file.

        When *path* is given the mapping is also written to disk. Raises
        :class:`RuntimeError` if any device's class was never registered via
        ``@DeviceRegistry.register`` (no ``registry_key`` -> not migratable).
        """
        out: dict = {}
        for dev_id, device in self.devices.items():
            registry_key = getattr(type(device), "registry_key", None)
            if not registry_key:
                raise RuntimeError(
                    f"Device '{dev_id}' ({type(device).__name__}) has no "
                    f"registry_key; decorate its class with "
                    f"@DeviceRegistry.register(...) to make it exportable."
                )
            cfg = dict(getattr(device, "cfg", {}) or {})
            cfg.pop("id", None)
            cfg.pop("type", None)
            cfg.pop("output", None)
            stanza: dict = {"type": registry_key}
            stanza.update(cfg)
            if getattr(device, "is_primary", False):
                stanza["primary"] = True
            path_args = getattr(device, "path_args", None)
            if path_args:
                output = {
                    "suffix": path_args.get("suffix"),
                    "file_type": path_args.get("extension"),
                    "bids_type": path_args.get("bids_type"),
                }
                stanza["output"] = {k: v for k, v in output.items() if v is not None}
            out[dev_id] = stanza
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(out, fh, sort_keys=False)
            self.logger.info(f"Exported hardware configuration to: {path}")
        return out

    def _validate_primary(self) -> None:
        """Require exactly one device flagged ``primary: true`` in YAML."""
        primaries = [d for d in self.devices.values() if getattr(d, "is_primary", False)]
        if len(primaries) != 1:
            ids = [getattr(d, "device_id", getattr(d, "id", "?")) for d in primaries]
            raise RuntimeError(
                f"Exactly one device must be flagged 'primary: true' in hardware.yaml; "
                f"found {len(primaries)}: {ids}."
            )

    @property
    def primary(self):
        """Return the device flagged ``primary: true`` in YAML."""
        for d in self.devices.values():
            if getattr(d, "is_primary", False):
                return d
        raise RuntimeError("No device is flagged 'primary: true'.")

    def arm_all(self, cfg) -> None:
        """Call ``arm(cfg)`` on every device for per-run preparation."""
        for name, device in self.devices.items():
            arm = getattr(device, "arm", None)
            if callable(arm):
                try:
                    arm(cfg)
                except Exception as exc:
                    self.logger.error(f"Error arming {name}: {exc}")

    def start_all(self) -> None:
        """Call ``start()`` on every device."""
        for name, device in self.devices.items():
            try:
                device.start()
            except Exception as exc:
                self.logger.error(f"Error starting {name}: {exc}")

    def deinitialize(self):
        """Tear down all devices and reset to unconfigured state.

        After this call the manager can be re-initialised with a fresh
        :meth:`load_config` / :meth:`initialize` cycle.
        """
        self.logger.info("Deinitializing hardware – shutting down all devices...")
        self.shutdown()
        # Release pymmcore-held hardware so cameras can be re-acquired
        for cam in self.cameras:
            if cam.backend == "micromanager" and cam.core is not None:
                try:
                    cam.core.unloadAllDevices()
                except Exception as e:
                    self.logger.error(f"Error unloading devices for {cam.id}: {e}")
        self.devices.clear()
        self.cameras = ()
        self.encoder = None
        self.nidaq = None
        self.psychopy = None
        self._configured = False
        self.logger.info("Hardware deinitialized – ready for reconfiguration.")

    def stop(self):
        """Stop all devices."""
        for name, device in self.devices.items():
            try:
                device.stop()
            except Exception as e:
                self.logger.error(f"Error stopping {name}: {e}")

    # Symmetric alias to ``start_all`` / ``arm_all`` used by ``Procedure``.
    stop_all = stop

    def shutdown(self):
        """Shutdown all devices."""
        for name, device in self.devices.items():
            try:
                device.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down {name}: {e}")

    def get_device(self, device_id: str) -> Optional[HardwareDevice]:
        """Get a device by its ID."""
        return self.devices.get(device_id)

    def cam_backends(self, backend):
        """Generator to iterate through cameras with a specific backend."""
        for cam in self.cameras:
            if cam.backend == backend:
                yield cam

    # ---- YAML loading ------------------------------------------------------

    @staticmethod
    def _load_yaml(path: str) -> dict:
        if not path:
            raise FileNotFoundError(f"Cannot find config file at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _aggregate_widgets(self) -> List[str]:
        """Collect unique widget keys from every device stanza.

        Scans the top-level ``widgets`` list plus a ``widgets:`` key on any
        stanza -- mappings (``encoder``/``nidaq``/``psychopy`` and registered
        extras like ``mouseportal``) and lists of mappings (``cameras``) -- so
        any device can contribute a GUI widget without special-casing here.
        """
        sources: List[list] = [self.yaml.get('widgets', []) or []]
        for key, val in (self.yaml or {}).items():
            if key == 'widgets':
                continue
            if isinstance(val, dict):
                sources.append(val.get('widgets', []) or [])
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        sources.append(item.get('widgets', []) or [])
        # flatten and dedupe while preserving order
        seen: dict[str, None] = {}
        for group in sources:
            for w in group or []:
                seen.setdefault(w, None)
        return list(seen)

    # ---- Device helpers ----------------------------------------------------

    @staticmethod
    def _apply_output_args(device, output: dict, default_suffix: str):
        """Set path_args, file_type, and bids_type on a device from its YAML output block."""
        device.path_args = {
            'suffix':    output.get('suffix', default_suffix),
            'extension': output.get('file_type', getattr(device, 'file_type', 'csv')),
            'bids_type': output.get('bids_type', getattr(device, 'bids_type', None)),
        }
        device.file_type = device.path_args['extension']
        device.bids_type = device.path_args['bids_type']

    # ---- Device init -------------------------------------------------------

    def _init_cameras(self):
        cams = []
        for cfg in self.yaml.get("cameras", []):
            backend = str(cfg.get("backend", "")).lower()
            registry_key = "opencv_camera" if backend == "opencv" else "camera"
            CameraClass = DeviceRegistry.get_class(registry_key)
            if CameraClass is None:
                self.logger.error(
                    f"No camera class registered under '{registry_key}' "
                    f"(backend='{backend}')"
                )
                continue
            cam = CameraClass(cfg)
            cam.is_primary = bool(cfg.get("primary", False))
            self._apply_output_args(cam, cfg.get('output', {}), cam.name)
            setattr(self, cam.id, cam)
            self.devices[cam.id] = cam
            cams.append(cam)
        self.cameras = tuple(cams)

    def _init_encoder(self):
        params = self.yaml.get("encoder")
        if not params:
            return

        enc_type = params.get('type')
        try:
            if enc_type == 'wheel':
                self.encoder = SerialWorker(
                    serial_port=params.get('port'),
                    baud_rate=params.get('baudrate'),
                    sample_interval=params.get('sample_interval_ms'),
                    wheel_diameter=params.get('diameter_mm'),
                    cpr=params.get('cpr'),
                    development_mode=params.get('development_mode'),
                )
            elif enc_type == 'treadmill':
                try:
                    self.encoder = EncoderSerialInterface(
                        port=params.get('port'),
                        baudrate=params.get('baudrate'),
                    )
                    self.encoder.initialize()
                except Exception as e:
                    raise RuntimeError(f"Failed to initialize EncoderSerialInterface: {e}") from e

            else:
                self.logger.warning(f"Unknown encoder type: {enc_type}")
                return
        except Exception as e:
            self.logger.warning(f"Could not open encoder on {params.get('port')}: {e}")
            self.encoder = None
            return

        self._apply_output_args(self.encoder, params.get('output', {}), 'encoder')
        self.encoder.is_primary = bool(params.get("primary", False))
        self.devices["encoder"] = self.encoder

    def _init_daq(self):
        params = self.yaml.get("nidaq")
        if not params:
            return
        self.nidaq = Nidaq(
            device_name=params.get('device_name'),
            lines=params.get('lines'),
            io_type=params.get('io_type'),
            ctr=params.get('crt', 'ctr0'),
        )
        self.nidaq.is_primary = bool(params.get("primary", False))
        self.devices["nidaq"] = self.nidaq

    def _init_psychopy(self):
        params = self.yaml.get("psychopy")
        if not params:
            return
        Cls = DeviceRegistry.get_class("psychopy")
        if Cls is None:
            self.logger.error("No class registered under 'psychopy'")
            return
        cfg = dict(params)
        cfg.setdefault("id", "psychopy")
        device = Cls(cfg)
        device.is_primary = bool(params.get("primary", False))
        self.psychopy = device
        self.devices[device.device_id] = device

    # ---- Engine configuration ----------------------------------------------

    def _configure_engines(self, cfg):
        """If using micromanager cameras, configure the engines."""
        if not self.cameras:
            return
        from pymmcore_plus import CMMCorePlus
        for cam in self.cameras:
            if isinstance(cam.core, CMMCorePlus):
                cam.core.mda.engine.set_config(cfg)