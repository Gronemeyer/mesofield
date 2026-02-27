from __future__ import annotations

VALID_BACKENDS = {"micromanager", "opencv"}

from typing import Any, ClassVar
import yaml

from mesofield.protocols import HardwareDevice, DataProducer
from mesofield.io.devices import Nidaq, MMCamera, SerialWorker, EncoderSerialInterface
from mesofield.utils._logger import get_logger, log_this_fr
from mesofield import DeviceRegistry

class HardwareManager():
    """
    High-level class that initializes all hardware (cameras, encoder, etc.)
    using the ParameterManager. Keeps references easily accessible.
    """

    def __init__(self, config_file: str):
        self.logger = get_logger(f'{__name__}.{self.__class__.__name__}')
        self.logger.info(f"Initializing HardwareManager with config: {config_file}")

        self.config_file = config_file
        self.devices: dict[str, DataProducer] = {}

        try:
            self.yaml = self._load_yaml(config_file)
            self.logger.info("Successfully loaded hardware configuration")
        except Exception as e:
            self.logger.error(f"Failed to load hardware configuration: {e}")
            raise

        self.widgets: list[str] = self._aggregate_widgets()
        self.cameras: tuple[MMCamera, ...] = ()
        self.encoder = None
        self.nidaq = None
        self._viewer = self.yaml.get('viewer_type', 'static')

    def __repr__(self):
        return (
            "<HardwareManager>\n"
            f"  Cameras: {[cam for cam in self.cameras]}\n"
            f"  Devices: {list(self.devices.keys())}\n"
            f"  Config: {self.yaml}\n"
            "</HardwareManager>"
        )

    # ---- Public interface --------------------------------------------------

    def initialize(self, cfg) -> None:
        """Initialize all devices from YAML and configure engines."""
        self.logger.info("Initializing hardware devices from YAML configuration...")
        self._init_cameras()
        self._init_encoder()
        self._init_daq()
        self._configure_engines(cfg)

    def stop(self):
        """Stop all devices."""
        for name, device in self.devices.items():
            try:
                device.stop()
            except Exception as e:
                self.logger.error(f"Error stopping {name}: {e}")

    def shutdown(self):
        """Shutdown all devices."""
        for name, device in self.devices.items():
            try:
                device.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down {name}: {e}")

    def get_device(self, device_id: str) -> HardwareDevice | None:
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

    def _aggregate_widgets(self) -> list[str]:
        """Collect unique widget keys from all device sections."""
        sources = [
            self.yaml.get('widgets', []),
            *[cam.get('widgets', []) for cam in self.yaml.get('cameras', [])],
            self.yaml.get('encoder', {}).get('widgets', []),
            self.yaml.get('nidaq', {}).get('widgets', []),
        ]
        # flatten and dedupe while preserving order
        seen: dict[str, None] = {}
        for group in sources:
            for w in group:
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
        CameraClass = DeviceRegistry.get_class("camera")
        cams = []
        for cfg in self.yaml.get("cameras", []):
            cam = CameraClass(cfg)
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
                self.encoder = EncoderSerialInterface(
                    port=params.get('port'),
                    baudrate=params.get('baudrate'),
                )
            else:
                self.logger.warning(f"Unknown encoder type: {enc_type}")
                return
        except Exception as e:
            self.logger.warning(f"Could not open encoder on {params.get('port')}: {e}")
            self.encoder = None
            return

        self._apply_output_args(self.encoder, params.get('output', {}), 'encoder')
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
        self.devices["nidaq"] = self.nidaq

    # ---- Engine configuration ----------------------------------------------

    def _configure_engines(self, cfg):
        """If using micromanager cameras, configure the engines."""
        from pymmcore_plus import CMMCorePlus
        for cam in self.cameras:
            if isinstance(cam.core, CMMCorePlus):
                cam.core.mda.engine.set_config(cfg)