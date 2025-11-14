VALID_BACKENDS = {"micromanager", "opencv"}

from typing import Dict, Any, List, Optional, Type, TypeVar, Callable

from mesofield.protocols import HardwareDevice, DataProducer
from mesofield.io.devices import Nidaq, MMCamera, SerialWorker, EncoderSerialInterface
from mesofield.utils._logger import get_logger, log_this_fr
from mesofield import DeviceRegistry
from mesofield.utils.config import load_hardware_config

class HardwareManager():
    """
    High-level class that initializes all hardware (cameras, encoder, etc.)
    using the ParameterManager. Keeps references easily accessible.
    """

    def __init__(self, config_file: str):
        self.logger = get_logger(f'{__name__}.{self.__class__.__name__}')
        self.logger.info(f"Initializing HardwareManager with config: {config_file}")

        self.config_file = config_file
        # every entry here is a DataProducer (and thus also a HardwareDevice)
        self.devices: Dict[str, DataProducer] = {}

        try:
            self.yaml = self._load_hardware_config(config_file)
            self.logger.info("Successfully loaded hardware configuration")
        except Exception as e:
            self.logger.error(f"Failed to load hardware configuration: {e}")
            raise

        # Build canonical widget list from YAML
        self.widgets: List[str] = self._aggregate_widgets()
        self.cameras: tuple[MMCamera, ...] = ()
        self._viewer = self.yaml.get('viewer_type', 'static')


    def __repr__(self):
        return (
            "<HardwareManager>\n"
            f"  Cameras: {[cam for cam in self.cameras]}\n"
            f"  Devices: {list(self.devices.keys())}\n"
            f"  Config: {self.yaml}\n"
            "</HardwareManager>"
        )


    def stop(self):
        """Stop all devices."""
        for device in self.devices.values():
            try:
                device.stop()
            except Exception as e:
                self.logger.error(f"Error stopping device: {e}")


    def shutdown(self):
        """Shutdown all devices."""
        for device in self.devices.values():
            try:
                device.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down device: {e}")

    def _load_hardware_config(self, path: str) -> Dict[str, Any]:
        """Load hardware configuration using the shared schema helpers."""
        if not path:
            raise FileNotFoundError(f"Cannot find config file at: {path}")

        return load_hardware_config(path)


    def _aggregate_widgets(self) -> List[str]:
        """
        Aggregate widget keys from root, camera entries, and encoder into a single list.
        """
        widgets: List[str] = []
        # Global widgets
        for w in self.yaml.get('widgets', []):
            if w not in widgets:
                widgets.append(w)
        # Camera-specific widgets
        for cam in self.yaml.get('cameras', []):
            for w in cam.get('widgets', []):
                if w not in widgets:
                    widgets.append(w)
        # Encoder-specific widgets
        for w in self.yaml.get('encoder', {}).get('widgets', []):
            if w not in widgets:
                widgets.append(w)
        # NIDAQ widgets
        if self.yaml.get('nidaq'):
            for w in self.yaml['nidaq'].get('widgets', []):
                if w not in widgets:
                    widgets.append(w)
        return widgets


    def _initialize_devices(self):
        """Initialize hardware devices from YAML configuration."""
        self.logger.info("Initializing hardware devices from YAML configuration...")
        self._initialize_cameras()
        self._initialize_encoder()
        self._initialize_daq()

    # ------------------------------------------------------------------
    def initialize(self, cfg) -> None:
        """Public wrapper to configure all devices and engines."""
        self._initialize_devices()
        self._configure_engines(cfg)


    def _initialize_daq(self):
        """Initialize NI-DAQ device from YAML configuration."""
        if self.yaml.get("nidaq"):
            params = self.yaml.get("nidaq")
            self.nidaq = Nidaq(
                device_name=params.get('device_name'),
                lines=params.get('lines'),
                io_type=params.get('io_type'),
                ctr=params.get('crt', 'ctr0'),
            )
            self.devices["nidaq"] = self.nidaq
        else:
            self.nidaq = None


    def _initialize_encoder(self):
        """Initialize encoder device from YAML configuration."""
        if self.yaml.get("encoder"):
            params = self.yaml.get("encoder")
            if params.get('type') == 'wheel':
                self.encoder = SerialWorker(
                    serial_port=params.get('port'),
                    baud_rate=params.get('baudrate'),
                    sample_interval=params.get('sample_interval_ms'),
                    wheel_diameter=params.get('diameter_mm'),
                    cpr=params.get('cpr'),
                    development_mode=params.get('development_mode')
                )
            elif params.get('type') == 'treadmill':
                self.encoder = EncoderSerialInterface(
                    port=params.get('port'),
                    baudrate=params.get('baudrate'),
                )
            output = params.get('output', {})
            self.encoder.path_args = {
                'suffix': output.get('suffix', 'encoder'),
                'extension': output.get('file_type', getattr(self.encoder, 'file_type', 'csv')),
                'bids_type': output.get('bids_type', getattr(self.encoder, 'bids_type', None))
            }
            self.encoder.file_type = self.encoder.path_args['extension']
            self.encoder.bids_type = self.encoder.path_args['bids_type']
            self.devices["encoder"] = self.encoder


    def _initialize_cameras(self):
        cams = []
        CameraClass = DeviceRegistry.get_class("camera")
        for cfg in self.yaml.get("cameras", []):
            cam = CameraClass(cfg)
            output = cfg.get('output', {})
            cam.path_args = {
                'suffix': output.get('suffix', cam.name),
                'extension': output.get('file_type', getattr(cam, 'file_type', 'ome.tiff')),
                'bids_type': output.get('bids_type', getattr(cam, 'bids_type', None))
            }
            cam.file_type = cam.path_args['extension']
            cam.bids_type = cam.path_args['bids_type']
            setattr(self, cam.id, cam)
            self.devices[cam.id] = cam
            cams.append(cam)
        self.cameras = tuple(cams)
        
    #TODO move to cameras.py
    def _configure_engines(self, cfg):
        """If using micromanager cameras, configure the engines."""
        from pymmcore_plus import CMMCorePlus
        for cam in self.cameras:
            if isinstance(cam.core, CMMCorePlus):
                cam.core.mda.engine.set_config(cfg)


    def cam_backends(self, backend):
        """Generator to iterate through cameras with a specific backend."""
        for cam in self.cameras:
            if cam.backend == backend:
                yield cam


    # Interface methods
    def get_device(self, device_id: str) -> Optional[HardwareDevice]:
        """Get a device by its ID."""
        return self.devices.get(device_id)


    def get_encoder(self) -> Optional[SerialWorker | EncoderSerialInterface]:
        """Get the encoder device."""
        return self.encoder

        


