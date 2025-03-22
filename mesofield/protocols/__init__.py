"""
Protocol definitions for hardware instruments and data management.

This module defines the core interfaces that standardize behavior across
the mesofield project, allowing for interoperability between different
hardware instruments, data producers, and data consumers.

Protocol Implementation Notes
----------------------------
When implementing these protocols, there are two approaches:

1. Direct inheritance (for regular classes without metaclass conflicts):
   ```python
   class MySensor(DataAcquisitionDevice):
       # Implement required methods and attributes
   ```

2. Duck typing (for classes with existing inheritance or metaclass conflicts, e.g., QThread):
   ```python
   class MyQThreadSensor(QThread):  # Cannot inherit from Protocol due to metaclass conflict
       # Implement all required methods and attributes
       device_type = "sensor"
       device_id = "my_sensor"
       # etc.
   ```

The second approach is necessary for Qt classes (QObject, QThread, QWidget) or 
any class that already uses a metaclass. Protocol checking uses duck typing 
internally, so both approaches will work with our system.
"""

import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Protocol, TypeVar, Generic, runtime_checkable, Tuple, Type, Set

T = TypeVar('T')

# These are the Protocol definitions - they are useful for static type checking
# and documentation, but should not be used for inheritance with classes that
# already have a metaclass (like QThread)

class Procedure(Protocol):
    """Protocol defining the standard interface for experiment procedures."""
    
    experiment_id: str
    experimentor: str
    hardware_yaml: str
    data_dir: str
    
    def initialize_hardware(self) -> bool:
        """Setup the experiment procedure.
        
        Returns:
            bool: True if setup was successful, False otherwise.
        """
    
    def setup_configuration(self, json_config: str) -> None:
        """Set up the configuration for the experiment procedure.
        
        Args:
            json_config: Path to a JSON configuration file (.json)
        """
        ...    
        
    def run(self) -> None:
        """Run the experiment procedure."""
        ...
        
    def save_data(self) -> None:
        """Save data from the experiment."""
        ...
        
    def cleanup(self) -> None:
        """Clean up after the experiment procedure."""
        ...

# Define configuration interface
class Configurator(Protocol):
    """Protocol defining the interface for configuration providers."""
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration value for the given key."""
        ...
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value for the given key."""
        ...
    
    def has(self, key: str) -> bool:
        """Check if the configuration contains the given key."""
        ...
    
    def keys(self) -> List[str]:
        """Get all configuration keys."""
        ...
    
    def items(self) -> Dict[str, Any]:
        """Get all configuration key-value pairs."""
        ...


@runtime_checkable
class HardwareDevice(Protocol):
    """Protocol defining the standard interface for all hardware devices."""
    
    device_type: str
    device_id: str
    config: Dict[str, Any]
    
    def initialize(self) -> bool:
        """Initialize the hardware device.
        
        Returns:
            bool: True if stopped successfully, False otherwise.
        """
        ...
    
    def shutdown(self) -> None:
        """Close and clean up resources."""
        ...
    
    def status(self) -> Dict[str, Any]:
        """Get the current status of the device.
        
        Returns:
        
            Dict[str, Any]: Dictionary containing device status information.
        """
        ...
        
    @property
    def metadata(self) -> Dict[str, Any]:
        """Return metadata about the hardware."""
        ...


@runtime_checkable
class DataProducer(HardwareDevice, Protocol):
    """Protocol defining the interface for data-producing components."""
    
    sampling_rate: float  # in Hz
    data_type: str
    is_active: bool
    
    def start(self) -> bool:
        """Start data acquisition or operation.
        
        Returns:
            bool: True if started successfully, False otherwise.
        """
        ...
    
    def stop(self) -> bool:
        """Stop data acquisition or operation.
        
        Returns:
            bool: True if stopped successfully, False otherwise.
        """
        ...
    
    def get_data(self) -> Optional[Any]:
        """Get the latest data from the producer.
        
        Returns:
            Optional[Any]: The latest data, or None if no data available.
        """
        ...


@runtime_checkable
class DataConsumer(Protocol):
    """Protocol defining the interface for data-consuming components."""
    
    @property
    def name(self) -> str:
        """Return the name of the data consumer."""
        ...
    
    @property
    def get_supported_data_types(self) -> List[str]:
        """Return the types of data this consumer can process."""
        ...
    
    def process_data(self, data: Any, metadata: Dict[str, Any]) -> bool:
        """Process data with metadata.
        
        Args:
            data: The data to process.
            metadata: Metadata about the data, including source, timestamp, etc.
            
        Returns:
            bool: True if data was processed successfully, False otherwise.
        """
        ...


@runtime_checkable
class HardwareManagerProtocol(Protocol):
    """Protocol defining the standard interface for hardware managers."""
    
    devices: Dict[str, HardwareDevice]
    cameras: Tuple[Any, ...]
    
    def __init__(self, config_file: str):
        """Initialize the hardware manager with the given configuration file."""
        ...
    
    def get_device(self, device_id: str) -> Optional[HardwareDevice]:
        """Get a device by its ID."""
        ...
    
    def get_devices_by_type(self, device_type: str) -> List[HardwareDevice]:
        """Get all devices of a specific type."""
        ...
    
    def has_device(self, device_id: str) -> bool:
        """Check if a device with the given ID exists."""
        ...
    
    def initialize_all(self) -> None:
        """Initialize all devices."""
        ...
    
    def close_all(self) -> None:
        """Close all devices."""
        ...
    
    def shutdown(self) -> None:
        """Shutdown all devices."""
        ...


# Helper functions for protocol checking

def is_hardware_device(obj) -> bool:
    """Check if an object implements the HardwareDevice interface."""
    required_attrs = ['device_id', 'device_type', 'config', 'initialize', 
                     'start', 'stop', 'close', 'get_status']
    return all(hasattr(obj, attr) for attr in required_attrs)

def is_data_acquisition_device(obj) -> bool:
    """Check if an object implements the DataAcquisitionDevice interface."""
    if not is_hardware_device(obj):
        return False
    return hasattr(obj, 'data_rate') and hasattr(obj, 'get_data')

def is_hardware_manager(obj) -> bool:
    """Check if an object implements the HardwareManager interface."""
    required_attrs = ['devices', 'cameras', 'get_device', 'has_device', 
                     'initialize_all', 'close_all', 'shutdown']
    return all(hasattr(obj, attr) for attr in required_attrs)


class AbstractHardwareManager(ABC):
    """
    Abstract base class implementing the HardwareManagerProtocol.
    
    This abstract class provides a foundation for hardware management with
    built-in logging and type checking. It handles device registration,
    initialization, and shutdown while ensuring operations are properly logged.
    """
    
    def __init__(self, config_file: str):
        """
        Initialize the hardware manager.
        
        Args:
            config_file: Path to the hardware configuration file.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_file = config_file
        self.devices: Dict[str, HardwareDevice] = {}
        self.cameras: Tuple[Any, ...] = ()
        
        # Validate config file exists
        if not Path(config_file).exists():
            self.logger.error(f"Configuration file not found: {config_file}")
            raise FileNotFoundError(f"Cannot find config file at: {config_file}")
        
        self.logger.info(f"Initializing hardware manager with config: {config_file}")
    
    def __repr__(self) -> str:
        """Return a string representation of the hardware manager."""
        return (
            f"<{self.__class__.__name__}>\n"
            f"  Cameras: {[getattr(cam, 'device_id', str(cam)) for cam in self.cameras]}\n"
            f"  Devices: {list(self.devices.keys())}\n"
            f"</HardwareManager>"
        )
    
    @abstractmethod
    def _load_configuration(self) -> Dict[str, Any]:
        """
        Load hardware configuration from file.
        
        Returns:
            Dict[str, Any]: The loaded configuration.
        """
        pass
    
    @abstractmethod
    def _initialize_devices(self) -> None:
        """Initialize all devices from the configuration."""
        pass
    
    def register_device(self, device_id: str, device: HardwareDevice) -> None:
        """
        Register a device with the hardware manager.
        
        Args:
            device_id: The ID of the device.
            device: The device object.
        """
        if not isinstance(device, HardwareDevice):
            self.logger.warning(f"Device {device_id} is not a HardwareDevice, attempting registration anyway")
        
        self.logger.debug(f"Registering device: {device_id}")
        self.devices[device_id] = device
    
    def unregister_device(self, device_id: str) -> bool:
        """
        Unregister a device from the hardware manager.
        
        Args:
            device_id: The ID of the device to unregister.
            
        Returns:
            bool: True if the device was unregistered, False otherwise.
        """
        if device_id not in self.devices:
            self.logger.warning(f"Device {device_id} not found, cannot unregister")
            return False
        
        self.logger.debug(f"Unregistering device: {device_id}")
        del self.devices[device_id]
        return True
    
    def get_device(self, device_id: str) -> Optional[HardwareDevice]:
        """
        Get a device by its ID.
        
        Args:
            device_id: The ID of the device to get.
            
        Returns:
            Optional[HardwareDevice]: The device if found, None otherwise.
        """
        device = self.devices.get(device_id)
        if device is None:
            self.logger.debug(f"Device {device_id} not found")
        return device
    
    def get_devices_by_type(self, device_type: str) -> List[HardwareDevice]:
        """
        Get all devices of a specific type.
        
        Args:
            device_type: The type of devices to get.
            
        Returns:
            List[HardwareDevice]: A list of devices of the specified type.
        """
        devices = [dev for dev in self.devices.values() 
                  if getattr(dev, 'device_type', None) == device_type]
        self.logger.debug(f"Found {len(devices)} devices of type {device_type}")
        return devices
    
    def has_device(self, device_id: str) -> bool:
        """
        Check if a device with the given ID exists.
        
        Args:
            device_id: The ID of the device to check.
            
        Returns:
            bool: True if the device exists, False otherwise.
        """
        return device_id in self.devices
    
    def initialize_all(self) -> None:
        """Initialize all devices."""
        self.logger.info("Initializing all devices")
        for device_id, device in self.devices.items():
            if hasattr(device, 'initialize'):
                try:
                    self.logger.debug(f"Initializing device: {device_id}")
                    device.initialize()
                except Exception as e:
                    self.logger.error(f"Error initializing device {device_id}: {e}")
    
    def close_all(self) -> None:
        """Close all devices."""
        self.logger.info("Closing all devices")
        for device_id, device in self.devices.items():
            if hasattr(device, 'shutdown'):
                try:
                    self.logger.debug(f"Closing device: {device_id}")
                    device.shutdown()
                except Exception as e:
                    self.logger.error(f"Error closing device {device_id}: {e}")
    
    def shutdown(self) -> None:
        """Shutdown all devices."""
        self.logger.info("Shutting down all devices")
        for device_id, device in self.devices.items():
            try:
                if hasattr(device, 'stop'):
                    self.logger.debug(f"Stopping device: {device_id}")
                    device.stop()
                if hasattr(device, 'shutdown'):
                    self.logger.debug(f"Shutting down device: {device_id}")
                    device.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down device {device_id}: {e}")
    
    def get_device_status(self, device_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a device.
        
        Args:
            device_id: The ID of the device to get the status of.
            
        Returns:
            Optional[Dict[str, Any]]: The device status if found, None otherwise.
        """
        device = self.get_device(device_id)
        if device is None:
            return None
        
        if hasattr(device, 'status'):
            try:
                return device.status()
            except Exception as e:
                self.logger.error(f"Error getting status for device {device_id}: {e}")
                return {"error": str(e)}
        
        return {"status": "unknown"}
    
    def get_all_device_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the status of all devices.
        
        Returns:
            Dict[str, Dict[str, Any]]: A dictionary mapping device IDs to their statuses.
        """
        return {device_id: self.get_device_status(device_id) or {"status": "unknown"} 
                for device_id in self.devices}

