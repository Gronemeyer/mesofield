VALID_BACKENDS = {"micromanager", "opencv"}
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Type, ClassVar, Tuple, Set
import importlib
import yaml
import nidaqmx.system
import nidaqmx
from pymmcore_plus import CMMCorePlus

from mesofield.engines import DevEngine, MesoEngine, PupilEngine
from mesofield.io.arducam import VideoThread
from mesofield.io.encoder import SerialWorker
from mesofield.protocols import HardwareDevice, AbstractHardwareManager, HardwareManagerProtocol


@dataclass
class Nidaq:
    """
    NIDAQ hardware control device.
    
    This class implements the ControlDevice protocol via duck typing,
    providing all the necessary methods and attributes without inheritance.
    """
    device_name: str
    lines: str
    io_type: str
    device_type: ClassVar[str] = "nidaq"
    device_id: str = "nidaq"
    config: Dict[str, Any] = None

    def __post_init__(self):
        if self.config is None:
            self.config = {
                "device_name": self.device_name,
                "lines": self.lines,
                "io_type": self.io_type
            }

    def initialize(self) -> bool:
        """Initialize the device."""
        return True

    def test_connection(self):
        """Test the connection to the NI-DAQ device."""
        print(f"Testing connection to NI-DAQ device: {self.device_name}")
        try:
            with nidaqmx.Task() as task:
                task.do_channels.add_do_chan(f'{self.device_name}/{self.lines}')
                task.write(True)
                time.sleep(3)
                task.write(False)
            print("Connection successful.")
            return True
        except nidaqmx.DaqError as e:
            print(f"NI-DAQ connection error: {e}")
            return False

    def reset(self):
        """Reset the NI-DAQ device."""
        print(f"Resetting NI-DAQ device: {self.device_name}")
        nidaqmx.system.Device(self.device_name).reset_device()

    def start(self) -> bool:
        """Start the device."""
        return True
    
    def stop(self) -> bool:
        """Stop the device."""
        return True
    
    def shutdown(self) -> None:
        """Close and clean up resources."""
        pass
    
    def status(self) -> Dict[str, Any]:
        """Get the status of the device."""
        return {"status": "ok"}
        
    def set_parameter(self, parameter: str, value: Any) -> bool:
        """Set a parameter on the device."""
        if parameter in self.config:
            self.config[parameter] = value
            return True
        return False
    
    def get_parameter(self, parameter: str) -> Any:
        """Get a parameter from the device."""
        return self.config.get(parameter)
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Return metadata about the hardware."""
        return {
            "device_type": self.device_type,
            "device_name": self.device_name,
            "lines": self.lines,
            "io_type": self.io_type
        }


class DeviceRegistry:
    """Registry for device classes."""
    
    _registry: Dict[str, Type[Any]] = {}
    
    @classmethod
    def register(cls, device_type: str) -> callable:
        """Register a device class for a specific device type."""
        def decorator(device_class: Type[Any]) -> Type[Any]:
            cls._registry[device_type] = device_class
            return device_class
        return decorator
    
    @classmethod
    def get_class(cls, device_type: str) -> Optional[Type[Any]]:
        """Get the device class for a specific device type."""
        return cls._registry.get(device_type)


class HardwareManager(AbstractHardwareManager):
    """
    High-level class that initializes all hardware (cameras, encoder, etc.)
    using a configuration file. Keeps references easily accessible and ensures
    proper type checking and logging.
    
    This class implements the HardwareManagerProtocol and extends the
    AbstractHardwareManager to provide concrete implementations of the abstract
    methods.
    """

    def __init__(self, config_file: str):
        """
        Initialize the hardware manager.
        
        Args:
            config_file: Path to the hardware configuration file (YAML).
        """
        # Initialize base class
        super().__init__(config_file)
        
        # Setup additional attributes
        self._config = self._load_configuration()
        self._viewer = self._config.get('viewer_type', 'static')
        
        # Initialize devices
        self._initialize_devices()
        
        self.logger.info(f"Hardware manager initialized with {len(self.devices)} devices")

    def __repr__(self):
        """Return a string representation of the hardware manager."""
        return (
            f"<{self.__class__.__name__}>\n"
            f"  Cameras: {[getattr(cam, 'device_id', str(cam)) for cam in self.cameras]}\n"
            f"  Devices: {list(self.devices.keys())}\n"
            f"  Config: {self._config}\n"
            f"</HardwareManager>"
        )
    
    def _load_configuration(self) -> Dict[str, Any]:
        """
        Load hardware configuration from a YAML file.
        
        Returns:
            Dict[str, Any]: The loaded configuration.
        """
        config_path = Path(self.config_file)
        if not config_path.exists():
            self.logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Cannot find config file at: {config_path}")
        
        self.logger.info(f"Loading hardware configuration from: {config_path}")
        
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file) or {}
                self.logger.debug(f"Loaded configuration: {config}")
                return config
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML configuration: {e}")
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _initialize_devices(self) -> None:
        """Initialize all hardware devices from the configuration."""
        self.logger.info("Initializing hardware devices")
        
        # Initialize cameras
        self._initialize_cameras()
        
        # Initialize encoder
        self._initialize_encoder()
        
        # Initialize NI-DAQ
        self._initialize_daq()
        
        # Initialize any other devices registered in the configuration
        self._initialize_additional_devices()
        
        self.logger.info(f"Initialized {len(self.devices)} devices")

    def _initialize_additional_devices(self) -> None:
        """Initialize additional devices from the configuration."""
        # Get all device types from the configuration that haven't been initialized yet
        initialized_devices = set(self.devices.keys())
        device_configs = {k: v for k, v in self._config.items() 
                         if isinstance(v, dict) and k not in initialized_devices 
                         and k not in ["cameras", "encoder", "nidaq"]}
        
        for device_type, config in device_configs.items():
            self.logger.debug(f"Looking for device class for type: {device_type}")
            device_class = DeviceRegistry.get_class(device_type)
            
            if device_class is not None:
                try:
                    self.logger.info(f"Initializing {device_type} device")
                    device = device_class(**config)
                    self.register_device(config.get('device_id', device_type), device)
                except Exception as e:
                    self.logger.error(f"Error initializing {device_type} device: {e}")
            else:
                self.logger.warning(f"No device class registered for type: {device_type}")

    def _initialize_daq(self) -> None:
        """Initialize NI-DAQ device from configuration."""
        if "nidaq" in self._config:
            self.logger.info("Initializing NI-DAQ device")
            params = self._config.get("nidaq", {})
            
            try:
                nidaq = Nidaq(
                    device_name=params.get('device_name'),
                    lines=params.get('lines'),
                    io_type=params.get('io_type')
                )
                
                # Add a direct reference for backward compatibility
                setattr(self, 'nidaq', nidaq)
                
                # Register the device
                self.register_device("nidaq", nidaq)
                
                self.logger.debug(f"NI-DAQ device initialized: {nidaq}")
            except Exception as e:
                self.logger.error(f"Error initializing NI-DAQ device: {e}")
        else:
            self.logger.debug("No NI-DAQ configuration found")
            setattr(self, 'nidaq', None)
            
    def _initialize_encoder(self) -> None:
        """Initialize encoder device from configuration."""
        if "encoder" in self._config:
            self.logger.info("Initializing encoder device")
            params = self._config.get("encoder", {})
            
            try:
                encoder = SerialWorker(
                    serial_port=params.get('port'),
                    baud_rate=params.get('baudrate'),
                    sample_interval=params.get('sample_interval_ms'),
                    wheel_diameter=params.get('diameter_mm'),
                    cpr=params.get('cpr'),
                    development_mode=params.get('development_mode')
                )
                
                # Add a direct reference for backward compatibility
                setattr(self, 'encoder', encoder)
                
                # Register the device
                self.register_device("encoder", encoder)
                
                self.logger.debug(f"Encoder device initialized: {encoder}")
            except Exception as e:
                self.logger.error(f"Error initializing encoder device: {e}")
        else:
            self.logger.debug("No encoder configuration found")
            setattr(self, 'encoder', None)
         
    def _initialize_cameras(self) -> None:
        """
        Initialize and configure camera objects based on configuration settings.
        
        This method reads the "cameras" section of the configuration,
        iterating over each camera definition. Depending on the specified backend
        (micromanager or opencv), it initializes and returns corresponding camera
        objects while applying any device-specific properties.
        """
        self.logger.info("Initializing camera devices")
        
        cams = []
        for camera_config in self._config.get("cameras", []):
            camera_id = camera_config.get("id")
            backend = camera_config.get("backend")
            
            self.logger.debug(f"Initializing camera: {camera_id} with backend: {backend}")
            
            try:
                if backend == "micromanager":
                    camera_object = self._init_micromanager_camera(camera_config)
                elif backend == 'opencv':
                    camera_object = VideoThread()
                else:
                    self.logger.warning(f"Unsupported camera backend: {backend}")
                    continue
                
                # Add the camera to the list
                cams.append(camera_object)
                
                # Add a direct reference for backward compatibility
                setattr(self, camera_id, camera_object)
                
                # Register the device
                self.register_device(camera_id, camera_object)
                
                self.logger.debug(f"Camera initialized: {camera_id}")
            except Exception as e:
                self.logger.error(f"Error initializing camera {camera_id}: {e}")
        
        # Store the tuple of cameras
        self.cameras = tuple(cams)
        self.logger.info(f"Initialized {len(cams)} cameras")
    
    def _init_micromanager_camera(self, config):
        """Initialize a Micro-Manager camera from config."""
        camera_id = config.get("id")
        
        # Get the Micro-Manager core object
        core = self._get_core_object(
            config.get("micromanager_path"),
            config.get("configuration_path"),
        )
        
        # Get the camera device object
        camera_object = core.getDeviceObject(camera_id)
        
        # Configure properties
        for device_id, props in config.get("properties", {}).items():
            if isinstance(props, dict):
                for property_id, value in props.items():
                    try:
                        if property_id == 'ROI':
                            self.logger.debug(f"Setting {device_id} ROI to {value}")
                            core.setROI(device_id, *value)
                        elif property_id == 'fps':
                            self.logger.debug(f"Setting {device_id} fps to {value}")
                            setattr(camera_object, 'fps', value)
                        elif property_id == 'viewer_type':
                            setattr(self, '_viewer', value)
                        else:
                            self.logger.debug(f"Setting {device_id} {property_id} to {value}")
                            core.setProperty(device_id, property_id, value)
                    except Exception as e:
                        self.logger.error(f"Error setting property {property_id} on {device_id}: {e}")
        
        # Set engine based on camera type
        if camera_id == 'ThorCam':
            engine = PupilEngine(core, use_hardware_sequencing=True)
            core.mda.set_engine(engine)
            self.logger.info(f"Using PupilEngine for camera: {camera_id}")
        elif camera_id == 'Dhyana':
            engine = MesoEngine(core, use_hardware_sequencing=True)
            core.mda.set_engine(engine)
            self.logger.info(f"Using MesoEngine for camera: {camera_id}")
        else:
            engine = DevEngine(core, use_hardware_sequencing=True)
            core.mda.set_engine(engine)
            self.logger.info(f"Using DevEngine for camera: {camera_id}")
        
        return camera_object
                
    def _get_core_object(self, mm_path, mm_cfg_path):
        """Get a Micro-Manager core object."""
        self.logger.debug(f"Creating CMMCorePlus instance with path: {mm_path}")
        
        core = CMMCorePlus(mm_path)
        
        if mm_path and mm_cfg_path is not None:
            self.logger.debug(f"Loading configuration from: {mm_cfg_path}")
            core.loadSystemConfiguration(mm_cfg_path)
        elif mm_cfg_path is None and mm_path:
            self.logger.debug("Loading default configuration")
            core.loadSystemConfiguration()
            
        return core
    
    @staticmethod
    def get_property_object(core: CMMCorePlus, device_id: str, property_id: str):
        """Get a property object from a Micro-Manager core."""
        return core.getPropertyObject(device_id, property_id)
    
    def configure_engines(self, cfg):
        """Configure engines for all cameras that have a core.mda.engine."""
        self.logger.info("Configuring camera engines")
        
        for cam in self.cameras:
            if hasattr(cam, 'core') and isinstance(cam.core, CMMCorePlus):
                try:
                    self.logger.debug(f"Configuring engine for camera: {getattr(cam, 'device_id', 'unknown')}")
                    cam.core.mda.engine.set_config(cfg)
                except Exception as e:
                    self.logger.error(f"Error configuring engine for camera {getattr(cam, 'device_id', 'unknown')}: {e}")
    
    def cam_backends(self, backend):
        """Generator to iterate through cameras with a specific backend."""
        for cam in self.cameras:
            if hasattr(cam, 'backend') and cam.backend == backend:
                yield cam
    
    def validate_camera_backends(self) -> bool:
        """Validate that all camera backends are supported."""
        self.logger.info("Validating camera backends")
        
        valid = True
        for cam in self.cameras:
            if hasattr(cam, 'backend'):
                if cam.backend not in VALID_BACKENDS:
                    self.logger.error(f"Invalid backend {cam.backend} for camera {getattr(cam, 'id', 'unknown')}")
                    valid = False
            else:
                self.logger.warning(f"Camera has no backend attribute: {cam}")
        
        return valid
    
    # Backward compatibility methods
    def get_camera(self, camera_id: str) -> Optional[Any]:
        """Get a camera device by its ID."""
        return self.get_device(camera_id)
    
    def get_encoder(self) -> Optional[SerialWorker]:
        """Get the encoder device."""
        return getattr(self, 'encoder', None)
    
    def has_camera(self) -> bool:
        """Check if at least one camera device exists."""
        return len(self.cameras) > 0
    
    def has_encoder(self) -> bool:
        """Check if the encoder device exists."""
        return hasattr(self, 'encoder') and self.encoder is not None
    
    @property
    def viewer_type(self) -> str:
        """Get the viewer type."""
        return self._viewer