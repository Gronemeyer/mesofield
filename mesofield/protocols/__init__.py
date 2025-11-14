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
       def __init__(self):
           self._init_logger()  # Initialize logging
           # Implement required methods and attributes
   ```

2. Duck typing (for classes with existing inheritance or metaclass conflicts, e.g., QThread):
   ```python
   class MyQThreadSensor(QThread):  # Cannot inherit from Protocol due to metaclass conflict
       def __init__(self):
           super().__init__()
           self._init_logger()  # Initialize logging manually
           # Implement all required methods and attributes
           self.device_type = "sensor"
           self.device_id = "my_sensor"
           # etc.
   ```

The second approach is necessary for Qt classes (QObject, QThread, QWidget) or 
any class that already uses a metaclass. Protocol checking uses duck typing 
internally, so both approaches will work with our system.
"""

from typing import Dict, List, Any, Optional, Protocol, TypeVar, Generic, runtime_checkable

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mesofield.hardware import HardwareManager

T = TypeVar('T')

# These are the Protocol definitions - they are useful for static type checking
# and documentation, but should not be used for inheritance with classes that
# already have a metaclass (like QThread)


# Define configuration interface
class Configurator(Protocol):
    """Protocol defining the interface for configuration providers."""
    
    hardware: "HardwareManager"
    
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

class Procedure(Protocol):
    """Protocol defining the standard interface for experiment procedures."""

    protocol: str
    experimenter: str
    config: Configurator
    data_dir: str

    def initialize_hardware(self) -> None:
        """Prepare hardware resources required for the experiment."""
        ...

    def setup_configuration(self, json_config: Optional[str]) -> None:
        """Apply configuration overrides from a JSON file if provided."""
        ...

    def prerun(self) -> None:
        """Perform any pre-experiment setup prior to running."""
        ...

    def run(self) -> None:
        """Run the experiment procedure."""
        ...

    def save_data(self) -> None:
        """Persist experiment outputs and metadata."""
        ...

    def add_note(self, note: str) -> None:
        """Record a note associated with the current experiment."""
        ...
        
@runtime_checkable
class HardwareDevice(Protocol):
    """Protocol defining the standard interface for all hardware devices."""
    
    device_type: str
    device_id: str
    #config: Dict[str, Any]
    
    def initialize(self) -> bool:
        """Initialize the hardware device.
        
        Returns:
            bool: True if stopped successfully, False otherwise.
        """
        ...
    
    def stop(self):
        """Stop the hardware device after starting it.
        
        This method will be called by the HardwareManager when a Procedure cleans up
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
    file_type: str
    bids_type: Optional[str] = None
    is_active: bool
    output_path: str
    metadata_path: Optional[str] = None
    
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
        
    def save_data(self, path: Optional[str] = None):
        """Save the device data captured during the recording"""
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


@runtime_checkable
class ProcedurePlugin(Protocol):
    """Standard API exposed by Procedure-level plugins."""

    name: str

    def begin_experiment(self) -> None:
        """Start any plugin-managed resources before hardware begins."""
        ...

    def end_experiment(self) -> None:
        """Clean up plugin resources after the run completes."""
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


