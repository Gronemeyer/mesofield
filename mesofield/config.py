import os
import json
import logging
import warnings
import datetime
import pathlib
from typing import Dict, Any, List, Optional, Type, TypeVar, Callable, Protocol, runtime_checkable, Tuple

import pandas as pd
import useq

from mesofield.hardware import HardwareManager
from mesofield.protocols import Configurator, HardwareManagerProtocol, Procedure


T = TypeVar('T')

# Configuration Registry pattern
class ConfigRegister:
    """A registry that maintains configuration values with optional type validation."""
    
    def __init__(self):
        self._registry: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._callbacks: Dict[str, List[Callable[[str, Any], None]]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def register(self, key: str, default: Any = None, 
                type_hint: Optional[Type] = None, 
                description: str = "", 
                category: str = "general") -> None:
        """Register a configuration parameter with metadata."""
        self._registry[key] = default
        self._metadata[key] = {
            "type": type_hint,
            "description": description,
            "category": category
        }
        self._callbacks[key] = []
        self.logger.debug(f"Registered parameter: {key} with default: {default}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._registry.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value with type validation."""
        # Register the key if it doesn't exist
        if key not in self._registry:
            self.logger.debug(f"Auto-registering new parameter: {key}")
            self.register(key, value)
        
        # Validate type if type hint exists
        type_hint = self._metadata.get(key, {}).get("type")
        if type_hint and not isinstance(value, type_hint):
            try:
                # Attempt type conversion
                value = type_hint(value)
                self.logger.debug(f"Converted parameter {key} to type {type_hint.__name__}")
            except (ValueError, TypeError):
                error_msg = f"Invalid type for {key}. Expected {type_hint.__name__}, got {type(value).__name__}"
                self.logger.error(error_msg)
                raise TypeError(error_msg)
        
        # Update value
        self._registry[key] = value
        
        # Trigger callbacks
        for callback in self._callbacks.get(key, []):
            try:
                callback(key, value)
            except Exception as e:
                self.logger.error(f"Error in callback for {key}: {e}")
    
    def has(self, key: str) -> bool:
        """Check if a key exists in the registry."""
        return key in self._registry
    
    def keys(self) -> List[str]:
        """Get all registered keys."""
        return list(self._registry.keys())
    
    def items(self) -> Dict[str, Any]:
        """Get all key-value pairs."""
        return self._registry.copy()
    
    def get_metadata(self, key: str) -> Dict[str, Any]:
        """Get metadata for a key."""
        return self._metadata.get(key, {})
    
    def register_callback(self, key: str, callback: Callable[[str, Any], None]) -> None:
        """Register a callback for when a key's value changes."""
        if key not in self._callbacks:
            self._callbacks[key] = []
        self._callbacks[key].append(callback)
        self.logger.debug(f"Registered callback for {key}")
    
    def unregister_callback(self, key: str, callback: Callable[[str, Any], None]) -> None:
        """Unregister a callback."""
        if key in self._callbacks and callback in self._callbacks[key]:
            self._callbacks[key].remove(callback)
            self.logger.debug(f"Unregistered callback for {key}")
    
    def clear(self) -> None:
        """Clear all configurations."""
        self._registry.clear()
        self.logger.debug("Cleared all configurations")


class ExperimentConfig(Configurator, Procedure):
    """## Generate and store parameters using a configuration registry. 
    
    This class implements both the Configurator and Procedure protocols, serving
    as the central configuration point for an experiment as well as providing
    methods to set up and run the experiment.
    
    The class uses a dynamic attribute system that allows access to any registered
    parameter as if it were a normal attribute, without requiring explicit property
    definitions for each parameter. This makes it easy to add and use new parameters
    without modifying the class definition.
    
    #### Example Usage:
    ```python
    config = ExperimentConfig('hardware.yaml')
    # create dict and pandas DataFrame from JSON file path:
    config.load_json('experiment_config.json')
        
    config.save_dir = './output'
    config.subject = '001'
    config.task = 'TestTask'
    config.notes.append('This is a test note.')

    # Add a custom parameter
    config.custom_parameter = 'custom value'
    
    # Access any parameter directly
    print(config.custom_parameter)  # Prints: custom value

    # Initialize hardware and run experiment
    config.initialize_hardware()
    config.run()
    config.save_data()
    config.cleanup()
    ```
    """
    
    def __getattr__(self, name):
        """
        Dynamic attribute access for parameters stored in the registry.
        
        This allows access to any parameter as if it were a normal attribute,
        without requiring explicit property definitions for each parameter.
        
        Args:
            name: The name of the attribute to get.
            
        Returns:
            The value of the parameter, if found in the registry.
            
        Raises:
            AttributeError: If the attribute is not found in the registry or 
                            is not a real attribute of the object.
        """
        # Avoid recursion for properties that don't exist
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
        
        # Check if the attribute exists in the registry
        if self._registry.has(name):
            return self._registry.get(name)
        
        # Check if it's in the legacy parameters
        if name in self._parameters:
            return self._parameters[name]
            
        # Not found in any parameter store
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
        
    def __setattr__(self, name, value):
        """
        Dynamic attribute setting for parameters stored in the registry.
        
        This allows setting any parameter as if it were a normal attribute,
        without requiring explicit property definitions for each parameter.
        
        Args:
            name: The name of the attribute to set.
            value: The value to set the attribute to.
        """
        # These are the attributes that should be set directly on the object
        # rather than going into the parameter registry
        direct_attributes = {
            'logger', '_registry', '_json_file_path', '_output_path', 
            '_save_dir', '_parameters', 'notes', 'hardware', 'data_manager',
            'experiment_id', 'experimentor', 'hardware_yaml', 'data_dir'
        }
        
        # Check if the attribute should be set directly
        if name in direct_attributes or name.startswith('_'):
            super().__setattr__(name, value)
        # Otherwise, store it in the registry and legacy parameters
        else:
            # If the attribute already exists as a property, use its setter
            if isinstance(getattr(self.__class__, name, None), property) and getattr(self.__class__, name).fset is not None:
                # Call the property's setter
                super().__setattr__(name, value)
            else:
                # Set as a dynamic parameter
                self.update_parameter(name, value)
                
    def __dir__(self):
        """
        Return all attributes and parameters available on this object.
        
        This makes auto-completion work correctly in IDEs and REPLs.
        
        Returns:
            A list of all attributes and parameters.
        """
        # Get standard attributes and methods
        standard_attrs = super().__dir__()
        
        # Add all registry keys
        registry_keys = self._registry.keys()
        
        # Add all legacy parameters
        legacy_params = list(self._parameters.keys())
        
        # Combine and remove duplicates
        return sorted(set(standard_attrs + registry_keys + legacy_params))

    def __init__(self, hardware_yaml: str):
        """
        Initialize the experiment configuration.
        
        Args:
            hardware_yaml: Path to the hardware YAML configuration file.
        """
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize the configuration registry
        self._registry = ConfigRegister()
        self._json_file_path = ''
        self._output_path = ''
        self._save_dir = ''
        self._parameters: dict = {}  # NOTE: For backward compatibility

        # Register common configuration parameters with defaults and types
        self._register_default_parameters()

        # Setup experiment attributes
        self.experiment_id = "default"
        self.experimentor = "default"
        self.hardware_yaml = hardware_yaml
        self.data_dir = self._save_dir

        # Initialize hardware
        self.logger.info(f"Initializing hardware from {hardware_yaml}")
        self.hardware = HardwareManager(hardware_yaml)
        
        # Initialize data manager
        self.logger.info("Initializing data manager")
        from mesofield.io.manager import DataManager
        self.data_manager = DataManager()
        
        # Register hardware devices with data manager
        self._register_hardware_devices()
        
        self.notes: list = []
        self.logger.info("ExperimentConfig initialized")

    def _register_hardware_devices(self):
        """Register hardware devices with the data manager."""
        for device_id, device in self.hardware.devices.items():
            if hasattr(device, 'device_type') and hasattr(device, 'get_data'):
                try:
                    self.logger.debug(f"Registering hardware device with data manager: {device_id}")
                    self.data_manager.register_hardware_device(device)
                except Exception as e:
                    self.logger.error(f"Error registering device {device_id} with data manager: {e}")

    def _register_default_parameters(self):
        """Register default parameters in the registry."""
        # Core experiment parameters
        self._registry.register("subject", "sub", str, "Subject identifier", "experiment")
        self._registry.register("session", "ses", str, "Session identifier", "experiment")
        self._registry.register("task", "task", str, "Task identifier", "experiment")
        self._registry.register("start_on_trigger", False, bool, "Whether to start acquisition on trigger", "hardware")
        self._registry.register("duration", 60, int, "Sequence duration in seconds", "experiment")
        self._registry.register("trial_duration", None, int, "Trial duration in seconds", "experiment")
        self._registry.register("led_pattern", ['4', '4', '2', '2'], list, "LED pattern sequence", "hardware")
        self._registry.register("psychopy_filename", "experiment.py", str, "PsychoPy experiment filename", "experiment")
        self.logger.debug("Registered default parameters")

    # Implement Procedure protocol methods
    def initialize_hardware(self) -> bool:
        """Initialize all hardware devices."""
        self.logger.info("Initializing all hardware devices")
        try:
            self.hardware.initialize_all()
            self.logger.info("Hardware initialization complete")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing hardware: {e}")
            return False
    
    def setup_configuration(self, json_config: str) -> None:
        """Set up the configuration from a JSON file."""
        self.logger.info(f"Loading configuration from {json_config}")
        self.load_json(json_config)
        
        # Set the data directory based on the save directory
        self.data_dir = self.save_dir
        
        # Configure camera engines
        try:
            self.logger.debug("Configuring camera engines")
            self.hardware.configure_engines(self)
        except Exception as e:
            self.logger.error(f"Error configuring camera engines: {e}")
    
    def run(self) -> None:
        """Run the experiment."""
        self.logger.info("Running experiment")
        # Start data manager
        try:
            self.data_manager.start_all()
            self.logger.info("Data manager started")
        except Exception as e:
            self.logger.error(f"Error starting data manager: {e}")
    
    def save_data(self) -> None:
        """Save all experiment data."""
        self.logger.info("Saving experiment data")
        try:
            self.save_configuration()
            self.logger.info("Configuration saved")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
    
    def cleanup(self) -> None:
        """Clean up after the experiment."""
        self.logger.info("Cleaning up experiment")
        # Stop data manager
        try:
            self.data_manager.stop_all()
            self.logger.info("Data manager stopped")
        except Exception as e:
            self.logger.error(f"Error stopping data manager: {e}")
        
        # Shutdown hardware
        try:
            self.hardware.shutdown()
            self.logger.info("Hardware shutdown complete")
        except Exception as e:
            self.logger.error(f"Error shutting down hardware: {e}")

    # Implement Configurator protocol methods
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._registry.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self._registry.set(key, value)
        # For backward compatibility
        self._parameters[key] = value
    
    def has(self, key: str) -> bool:
        """Check if a key exists in the configuration."""
        return self._registry.has(key)
    
    def keys(self) -> List[str]:
        """Get all configuration keys."""
        return self._registry.keys()
    
    def items(self) -> Dict[str, Any]:
        """Get all configuration key-value pairs."""
        return self._registry.items()

    # Helper properties
    @property
    def _cores(self) -> Tuple:
        """Return the tuple of CMMCorePlus instances from the hardware cameras."""
        return tuple(cam.core for cam in self.hardware.cameras if hasattr(cam, 'core'))

    @property
    def save_dir(self) -> str:
        """Get the save directory."""
        return os.path.join(self._save_dir, 'data')

    @save_dir.setter
    def save_dir(self, path: str):
        """Set the save directory."""
        if isinstance(path, str):
            self._save_dir = os.path.abspath(path)
            self.data_dir = self.save_dir
            self.logger.debug(f"Save directory set to {self._save_dir}")
        else:
            self.logger.error(f"Invalid save directory path: {path}")
            print(f"ExperimentConfig: \n Invalid save directory path: {path}")

    @property
    def subject(self) -> str:
        """Get the subject ID."""
        return self._registry.get("subject", self._parameters.get('subject', 'sub'))

    @subject.setter
    def subject(self, value: str):
        """Set the subject ID."""
        self.set("subject", value)

    @property
    def session(self) -> str:
        """Get the session ID."""
        return self._registry.get("session", self._parameters.get('session', 'ses'))

    @session.setter
    def session(self, value: str):
        """Set the session ID."""
        self.set("session", value)

    @property
    def task(self) -> str:
        """Get the task ID."""
        return self._registry.get("task", self._parameters.get('task', 'task'))

    @task.setter
    def task(self, value: str):
        """Set the task ID."""
        self.set("task", value)

    @property
    def start_on_trigger(self) -> bool:
        """Get whether to start on trigger."""
        return self._registry.get("start_on_trigger", self._parameters.get('start_on_trigger', False))
    
    @start_on_trigger.setter
    def start_on_trigger(self, value: bool):
        """Set whether to start on trigger."""
        self.set("start_on_trigger", value)
    
    @property
    def sequence_duration(self) -> int:
        """Get the sequence duration in seconds."""
        return int(self._registry.get("duration", self._parameters.get('duration', 60)))
    
    @sequence_duration.setter
    def sequence_duration(self, value: int):
        """Set the sequence duration in seconds."""
        self.set("duration", value)
    
    @property
    def trial_duration(self) -> int:
        """Get the trial duration in seconds."""
        return int(self._registry.get("trial_duration", self._parameters.get('trial_duration', None)))
    
    @trial_duration.setter
    def trial_duration(self, value: int):
        """Set the trial duration in seconds."""
        self.set("trial_duration", value)
        
    @property
    def num_trials(self) -> int:
        """Calculate the number of trials."""
        return int(self.sequence_duration / self.trial_duration)  
    
    @property
    def parameters(self) -> dict:
        """Get all parameters as a dictionary."""
        # Merge registry with legacy parameters for backward compatibility
        params = self._registry.items()
        params.update(self._parameters)
        return params
    
    @property
    def meso_sequence(self) -> useq.MDASequence:
        """Create a meso sequence configuration."""
        frames = int(self.hardware.Dhyana.fps * self.sequence_duration)
        return useq.MDASequence(time_plan={"interval": 0, "loops": frames})
    
    @property
    def pupil_sequence(self) -> useq.MDASequence:
        """Create a pupil sequence configuration."""
        frames = int((self.hardware.ThorCam.fps * self.sequence_duration)) + 100 
        return useq.MDASequence(time_plan={"interval": 0, "loops": frames})
    
    @property
    def bids_dir(self) -> str:
        """ Dynamic construct of BIDS directory path """
        bids = os.path.join(
            f"sub-{self.subject}",
            f"ses-{self.session}",
        )
        return os.path.abspath(os.path.join(self.save_dir, bids))

    @property
    def notes_file_path(self):
        """Get the notes file path."""
        return self.make_path(suffix="notes", extension="txt")
    
    @property
    def dataframe(self):
        """Convert parameters to a pandas DataFrame."""
        # Combine registry and legacy parameters
        combined_params = self._registry.items()
        combined_params.update(self._parameters)
        
        data = {'Parameter': list(combined_params.keys()),
                'Value': list(combined_params.values())}
        return pd.DataFrame(data)
    
    @property
    def psychopy_filename(self) -> str:
        """Get the PsychoPy experiment filename."""
        py_files = list(pathlib.Path(self._save_dir).glob('*.py'))
        if py_files:
            return py_files[0].name
        else:
            self.logger.warning(f'No Psychopy experiment file found in directory {pathlib.Path(self.save_dir).parent}.')
            warnings.warn(f'No Psychopy experiment file found in directory {pathlib.Path(self.save_dir).parent}.')
        return self._registry.get("psychopy_filename", self._parameters.get('psychopy_filename', 'experiment.py'))

    @property
    def psychopy_path(self) -> str:
        """Get the PsychoPy script path."""
        return os.path.join(self._save_dir, self.psychopy_filename)
    
    @property
    def psychopy_save_path(self) -> str:
        """Get the PsychoPy save path."""
        return os.path.join(self._save_dir, f"data/sub-{self.subject}/ses-{self.session}/beh/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_sub-{self.subject}_ses-{self.session}_task-{self.task}_psychopy")
    
    @property
    def psychopy_parameters(self) -> dict:
        """Get parameters for PsychoPy."""
        return {
            'subject': self.subject,
            'session': self.session,
            'save_dir': self.save_dir,
            'num_trials': self.num_trials,
            'filename': self.psychopy_save_path
        }
    
    @property
    def led_pattern(self) -> list[str]:
        """Get the LED pattern."""
        return self._registry.get("led_pattern", self._parameters.get('led_pattern', ['4', '4', '2', '2']))
    
    @led_pattern.setter
    def led_pattern(self, value: list) -> None:
        """Set the LED pattern."""
        if isinstance(value, str):
            try:
                value = json.loads(value)
                self.logger.debug(f"Converted LED pattern string to list: {value}")
            except json.JSONDecodeError:
                error_msg = "led_pattern string must be a valid JSON list"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        if isinstance(value, list):
            value_str = [str(item) for item in value]
            self._registry.set("led_pattern", value_str)
            self._parameters['led_pattern'] = value_str  # For backward compatibility
            self.logger.debug(f"Set LED pattern to {value_str}")
        else:
            error_msg = "led_pattern must be a list or a JSON string representing a list"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
    
    # Helper methods
    def make_path(self, suffix: str, extension: str, bids_type: str = None):
        """ Generate a unique file path following BIDS conventions.
        
        Example:
        ```py
            ExperimentConfig.make_path("images", "jpg", "func")
            print(unique_path)
        ```
        Output:
            C:/save_dir/data/sub-id/ses-id/func/20250110_123456_sub-001_ses-01_task-example_images.jpg
        """
        file = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_sub-{self.subject}_ses-{self.session}_task-{self.task}_{suffix}.{extension}"

        if bids_type is None:
            bids_path = self.bids_dir
        else:
            bids_path = os.path.join(self.bids_dir, bids_type)
            
        os.makedirs(bids_path, exist_ok=True)
        base, ext = os.path.splitext(file)
        counter = 1
        file_path = os.path.join(bids_path, file)
        while os.path.exists(file_path):
            file_path = os.path.join(bids_path, f"{base}_{counter}{ext}")
            counter += 1
        
        self.logger.debug(f"Generated file path: {file_path}")
        return file_path
        
    def load_json(self, file_path) -> None:
        """ Load parameters from a JSON configuration file into the config object. """
        self.logger.info(f"Loading parameters from: {file_path}")
        params = {}
        
        try:
            with open(file_path, 'r') as f:
                params = json.load(f)
                self.logger.debug(f"Loaded parameters: {params}")
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            print(f"File not found: {file_path}")
            return
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON: {e}")
            print(f"Error decoding JSON: {e}")
            return
            
        self._json_file_path = file_path  # store the json filepath
        
        # Update the registry and legacy parameters
        for key, value in params.items():
            self._registry.set(key, value)
            self._parameters[key] = value  # NOTE: For backward compatibility
                
    def update_parameter(self, key, value) -> None:
        """Update a parameter in both registry and legacy dictionary."""
        self.logger.debug(f"Updating parameter: {key} = {value}")
        self._registry.set(key, value)
        self._parameters[key] = value  # NOTE: For backward compatibility
        
    def list_parameters(self) -> pd.DataFrame:
        """ Create a DataFrame from the ExperimentConfig properties """
        properties = [prop for prop in dir(self.__class__) if isinstance(getattr(self.__class__, prop), property)]
        exclude_properties = {'dataframe', 'parameters', 'json_path', "_cores", "meso_sequence", "pupil_sequence", "psychopy_path", "encoder"}
        data = {prop: getattr(self, prop) for prop in properties if prop not in exclude_properties}
        self.logger.debug(f"Listed {len(data)} parameters")
        return pd.DataFrame(data.items(), columns=['Parameter', 'Value'])
                
    def save_wheel_encoder_data(self, data):
        """ Save the wheel encoder data to a CSV file """
        self.logger.info("Saving wheel encoder data")
        if isinstance(data, list):
            data = pd.DataFrame(data)
            
        encoder_path = self.make_path(suffix="encoder-data", extension="csv", bids_type='beh')
        try:
            data.to_csv(encoder_path, index=False)
            self.logger.info(f"Encoder data saved to {encoder_path}")
            print(f"Encoder data saved to {encoder_path}")
        except Exception as e:
            self.logger.error(f"Error saving encoder data: {e}")
            print(f"Error saving encoder data: {e}")
            
    def save_configuration(self):
        """ Save the configuration parameters from the registry to a CSV file """
        self.logger.info("Saving configuration data")
        params_path = self.make_path(suffix="configuration", extension="csv")
        try:
            # Get all parameters from the registry
            registry_items = self._registry.items()
            params_df = pd.DataFrame(list(registry_items.items()), columns=['Parameter', 'Value'])
            params_df.to_csv(params_path, index=False)
            self.logger.info(f"Configuration saved to {params_path}")
            print(f"Configuration saved to {params_path}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            print(f"Error saving configuration: {e}")
        
        notes_path = self.make_path(suffix="notes", extension="txt")
        try:
            with open(notes_path, 'w') as f:
                f.write('\n'.join(self.notes))
                self.logger.info(f"Notes saved to {notes_path}")
                print(f"Notes saved to {notes_path}")
        except Exception as e:
            self.logger.error(f"Error saving notes: {e}")
            print(f"Error saving notes: {e}")
    
    def save_parameters(self, file_path=None):
        """Save parameters to a file (JSON or YAML based on extension)."""
        if file_path is None:
            file_path = self._json_file_path
            
        if not file_path:
            self.logger.warning("No file path specified for saving parameters")
            print("No file path specified for saving parameters")
            return
            
        self.logger.info(f"Saving parameters to: {file_path}")
        try: # to save combined registry and legacy parameters
            combined_params = self._registry.items()
            combined_params.update(self._parameters)
            with open(file_path, 'w') as f:
                json.dump(combined_params, f, indent=2)
                self.logger.info(f"Parameters saved to {file_path}")
                print(f"Parameters saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving JSON: {e}")
            print(f"Error saving JSON: {e}")
            
    def get_parameter_metadata(self, key=None):
        """Get metadata for a parameter or all parameters.
        
        Args:
            key: Optional parameter key to get metadata for. If None, returns all metadata.
            
        Returns:
            Dictionary of parameter metadata including type, description, and category.
        """
        if key is not None:
            self.logger.debug(f"Getting metadata for parameter: {key}")
            return self._registry.get_metadata(key)
        else:
            # Return metadata for all parameters
            self.logger.debug("Getting metadata for all parameters")
            return {k: self._registry.get_metadata(k) for k in self._registry.keys()}
            
    def get_parameters_by_category(self, category=None):
        """Get parameters grouped by category.
        
        Args:
            category: Optional category to filter by. If None, returns all categories.
            
        Returns:
            Dictionary of parameters grouped by category.
        """
        self.logger.debug(f"Getting parameters by category: {category if category else 'all'}")
        result = {}
        for key in self._registry.keys():
            meta = self._registry.get_metadata(key)
            cat = meta.get('category', 'general')
            
            if category is not None and cat != category:
                continue
                
            if cat not in result:
                result[cat] = {}
                
            result[cat][key] = {
                'value': self._registry.get(key),
                'metadata': meta
            }
            
        return result
        
    def register_parameter(self, key, default=None, type_hint=None, description="", category="general", ui_widget=None, options=None, min_value=None, max_value=None, step=None, read_only=False):
        """Register a new parameter with metadata.
        
        This method registers a parameter with the config registry and provides 
        additional metadata that can be used by UI controllers to generate
        appropriate widgets for editing the parameter.
        
        Args:
            key: Parameter key
            default: Default value
            type_hint: Type of the parameter (str, int, float, bool, list, etc.)
            description: Description of the parameter
            category: Category for the parameter
            ui_widget: Suggested UI widget type for this parameter ('text', 'combo', 'check', 'spin', 'slider', etc.)
            options: If ui_widget is 'combo', a list of possible values for the parameter
            min_value: For numeric parameters, the minimum allowed value
            max_value: For numeric parameters, the maximum allowed value
            step: For numeric parameters, the step size for incrementing/decrementing
            read_only: Whether the parameter should be read-only in UIs
        """
        self.logger.debug(f"Registering parameter: {key} with default: {default}, type: {type_hint}")
        
        # Register the parameter with basic metadata
        self._registry.register(key, default, type_hint, description, category)
        
        # Add UI-specific metadata
        meta = self._registry.get_metadata(key)
        meta.update({
            "ui_widget": ui_widget,
            "options": options,
            "min_value": min_value,
            "max_value": max_value,
            "step": step,
            "read_only": read_only
        })
        
        # For backward compatibility
        if default is not None:
            self._parameters[key] = default
    
    def get_ui_schema(self, category=None):
        """
        Get a UI schema for all parameters or parameters in a specific category.
        
        This method returns a dictionary that describes all parameters in a format
        that can be used by UI controllers to generate appropriate widgets.
        
        Args:
            category: Optional category to filter by. If None, returns all parameters.
            
        Returns:
            Dict[str, Dict]: A dictionary mapping parameter keys to UI schema entries.
        """
        self.logger.debug(f"Getting UI schema for category: {category if category else 'all'}")
        
        schema = {}
        
        # Get all parameters
        params = self.get_parameters_by_category(category)
        
        # Flatten the category structure if no category is specified
        if category is None:
            all_params = {}
            for cat_params in params.values():
                all_params.update(cat_params)
            params = all_params
        else:
            params = params.get(category, {})
        
        # Process each parameter
        for key, param_data in params.items():
            value = param_data['value']
            metadata = param_data['metadata']
            
            # Default UI widget based on type if not specified
            ui_widget = metadata.get('ui_widget')
            if ui_widget is None:
                # Infer UI widget from type
                type_hint = metadata.get('type')
                if isinstance(value, bool) or type_hint == bool:
                    ui_widget = 'check'
                elif isinstance(value, (int, float)) or type_hint in (int, float):
                    ui_widget = 'spin'
                elif isinstance(value, list) or type_hint == list:
                    ui_widget = 'list'
                elif metadata.get('options'):
                    ui_widget = 'combo'
                else:
                    ui_widget = 'text'
            
            # Build the schema entry
            schema_entry = {
                'key': key,
                'value': value,
                'type': metadata.get('type').__name__ if metadata.get('type') else type(value).__name__,
                'description': metadata.get('description', ''),
                'category': metadata.get('category', 'general'),
                'ui_widget': ui_widget,
                'options': metadata.get('options'),
                'min_value': metadata.get('min_value'),
                'max_value': metadata.get('max_value'),
                'step': metadata.get('step'),
                'read_only': metadata.get('read_only', False)
            }
            
            schema[key] = schema_entry
        
        return schema

