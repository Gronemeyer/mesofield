import os
import json
import datetime
from typing import Dict, Any, List, Optional, Type, TypeVar, Callable

import pandas as pd
import useq
from useq import TIntervalLoops

from mesofield.hardware import HardwareManager
from mesofield.protocols import DataProducer
from mesofield.utils._logger import get_logger

T = TypeVar('T')

# Configuration Registry pattern
class ConfigRegister:
    """A registry that maintains configuration values with optional type validation."""
    
    def __init__(self):
        self._registry: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._callbacks: Dict[str, List[Callable[[str, Any], None]]] = {}
    
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
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._registry.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value with type validation."""
        # Register the key if it doesn't exist
        if key not in self._registry:
            self.register(key, value)
        
        # Validate type if type hint exists
        type_hint = self._metadata.get(key, {}).get("type")
        if type_hint and not isinstance(value, type_hint):
            try:
                # Attempt type conversion
                value = type_hint(value)
            except (ValueError, TypeError):
                raise TypeError(f"Invalid type for {key}. Expected {type_hint.__name__}, got {type(value).__name__}")
        
        # Update value
        self._registry[key] = value
        
        # Trigger callbacks
        for callback in self._callbacks.get(key, []):
            callback(key, value)
    
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
    
    
    def clear(self) -> None:
        """Clear all configurations."""
        self._registry.clear()


class ExperimentConfig(ConfigRegister):
    """## Generate and store parameters using a configuration registry. 
    
    #### Example Usage:
    ```python
    config = ExperimentConfig()
    # create dict and pandas DataFrame from JSON file path:
    config.load_parameters('path/to/json_file.json')
        
    config._save_dir = './output'
    config.subject = '001'
    config.task = 'TestTask'
    config.notes.append('This is a test note.')

    # Update a parameter
    config.update_parameter('new_param', 'test_value')

    # Save parameters and notes
    config.save_parameters()
    ```
    """

    def __init__(self, path: Optional[str] = None):
        super().__init__()
        # Initialize logging first
        self.logger = get_logger(__name__)
        self.logger.info(f"Initializing ExperimentConfig with hardware path: {path}")
        
        # Initialize the configuration registry
        self._json_file_path = ''
        self._save_dir = ''
        self.subjects: Dict[str, Dict[str, Any]] = {}
        self.selected_subject: str | None = None
        self.display_keys: List[str] | None = None

        # Register common configuration parameters with defaults and types
        self._register_default_parameters()
        self.logger.debug("Registered default parameters")

        # Initialize hardware
        self._hardware_yaml_path: Optional[str] = None
        self._hardware_path_locked: bool = False
        try:
            hardware_path, locked = self._resolve_hardware_path(path)
            self._hardware_yaml_path = hardware_path
            self._hardware_path_locked = locked
            self.hardware = HardwareManager(hardware_path)
        except Exception as e:
            self.logger.error(f"Failed to initialize hardware: {e}")
            raise
        
        self.notes: list = []

    def _register_default_parameters(self):
        """Register default parameters in the registry."""
        # Core experiment parameters
        self.register("subject", "sub", str, "Subject identifier", "experiment")
        self.register("session", "ses", str, "Session identifier", "experiment")
        self.register("task", "task", str, "Task identifier", "experiment")
        self.register("start_on_trigger", False, bool, "Whether to start acquisition on trigger", "hardware")
        self.register("duration", 60, int, "Sequence duration in seconds", "experiment")
        self.register("trial_duration", None, int, "Trial duration in seconds", "experiment")
        self.register("psychopy_filename", "experiment.py", str, "PsychoPy experiment filename", "experiment")

    def _resolve_hardware_path(self, path: Optional[str]) -> tuple[str, bool]:
        """Resolve the hardware YAML path.

        Returns (hardware_path, locked) where locked indicates an explicit file was provided.
        """
        if path:
            abs_path = os.path.abspath(path)
            if os.path.isdir(abs_path):
                self.experiment_dir = abs_path
                return os.path.join(abs_path, "hardware.yaml"), False
            # treat as explicit file path
            self.experiment_dir = os.path.dirname(abs_path)
            return abs_path, True

        # no path provided: resolve from experiment_dir (or cwd)
        if not self.experiment_dir:
            self.experiment_dir = os.getcwd()
        return os.path.join(self.experiment_dir, "hardware.yaml"), False

    @property
    def _cores(self):# -> tuple[CMMCorePlus, ...]:
        """Return the tuple of CMMCorePlus instances from the hardware cameras."""
        return tuple(cam.core for cam in self.hardware.cameras if hasattr(cam, 'core'))

    @property
    def experiment_dir(self) -> str:
        """Get the experiment directory (base directory)."""
        return self._save_dir

    @experiment_dir.setter
    def experiment_dir(self, path: str):
        """Set the experiment directory (base directory)."""
        if isinstance(path, str):
            self._save_dir = os.path.abspath(path)
        else:
            print(f"ExperimentConfig: \n Invalid experiment directory path: {path}")

    @property
    def data_dir(self) -> str:
        """Get the data directory (experiment_dir/data)."""
        return os.path.join(self._save_dir, 'data')

    @property
    def save_dir(self) -> str:
        """Get the save directory (legacy alias for data_dir)."""
        return self.data_dir

    @save_dir.setter
    def save_dir(self, path: str):
        """Set the save directory (base experiment directory)."""
        self.experiment_dir = path

    @property
    def subject(self) -> str:
        """Get the subject ID."""
        return self.get("subject")

    @property
    def session(self) -> str:
        """Get the session ID."""
        return self.get("session")

    @property
    def task(self) -> str:
        """Get the task ID."""
        return self.get("task")

    @property
    def start_on_trigger(self) -> bool:
        """Get whether to start on trigger."""
        return self.get("start_on_trigger")
    
    @property
    def sequence_duration(self) -> int:
        """Get the sequence duration in seconds."""
        return int(self.get("duration"))
    
    @property
    def trial_duration(self) -> int:
        """Get the trial duration in seconds."""
        trial_dur = self.get("trial_duration")
        return int(trial_dur) if trial_dur is not None else None
        
    @property
    def num_trials(self) -> int:
        """Calculate the number of trials."""
        return int(self.get("num_trials", 20))
    
    
    def build_sequence(self, camera: DataProducer) -> useq.MDASequence:
        if self.has('num_meso_frames'):
            loops = int(self.get('num_meso_frames'))
        else:
            try:
                loops = int(camera.sampling_rate * self.sequence_duration)
            except Exception:
                loops = 5
            metadata = self.hardware.__dict__

        # convert to a datetime.timedelta and build the time_plan
        time_plan = TIntervalLoops(
            interval=0,
            loops=loops,
            prioritize_duration=False
        )
        return useq.MDASequence(metadata=metadata, time_plan=time_plan)
    
    @property
    def bids_dir(self) -> str:
        """ Dynamic construct of BIDS directory path """
        bids = os.path.join(
            f"sub-{self.subject}",
            f"ses-{self.session}",
        )
        return os.path.abspath(os.path.join(self.save_dir, bids))

    
    @property
    def dataframe(self):
        """Convert parameters to a pandas DataFrame."""
        combined_params = self.items()
        data = {'Parameter': list(combined_params.keys()),
                'Value': list(combined_params.values())}
        return pd.DataFrame(data)
    
    @property
    def psychopy_filename(self) -> str:
        """Get the PsychoPy experiment filename."""
        return self.get("psychopy_filename")

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
            'save_path': self.psychopy_save_path
        }
    
    @property
    def led_pattern(self) -> list[str]:
        """Get the LED pattern."""
        return self.get("led_pattern")
    
    @led_pattern.setter
    def led_pattern(self, value: list) -> None:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("led_pattern string must be a valid JSON list")
        if isinstance(value, list):
            value_str = [str(item) for item in value]
            self.set("led_pattern", value_str)
        else:
            raise ValueError("led_pattern must be a list or a JSON string representing a list")
    
    # Helper method to generate a unique file path
    def make_path(self, suffix: str, extension: str, bids_type: Optional[str] = None, create_dir: bool = False):
        """ Example:
        ```py
            ExperimentConfig._generate_unique_file_path("images", "jpg", "func")
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

        if create_dir:
            os.makedirs(bids_path, exist_ok=True)
        base, ext = os.path.splitext(file)
        counter = 1
        file_path = os.path.join(bids_path, file)
        while os.path.exists(file_path):
            file_path = os.path.join(bids_path, f"{base}_{counter}{ext}")
            counter += 1
        return file_path
        
    def load_json(self, file_path) -> None:
        """ Load parameters from a JSON configuration file into the config object. 
        """
        self.logger.info(f"Loading configuration from: {file_path}")
        try:
            with open(file_path, 'r') as f:
                loaded_config = json.load(f)
            self.logger.info("Successfully loaded configuration JSON")
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {file_path}")
            return
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON from {file_path}: {e}")
            return

        self._json_file_path = file_path #store the json filepath
        json_dir = os.path.dirname(os.path.abspath(file_path))
        if json_dir:
            self.experiment_dir = json_dir
            if not self._hardware_path_locked:
                try:
                    hardware_path = os.path.join(self.experiment_dir, "hardware.yaml")
                    if hardware_path != self._hardware_yaml_path:
                        self._hardware_yaml_path = hardware_path
                        self.hardware = HardwareManager(hardware_path)
                except Exception as e:
                    self.logger.error(f"Failed to reinitialize hardware: {e}")
                    raise
        self.display_keys = loaded_config.get("DisplayKeys")
        # Detect new style JSON with 'Configuration' and 'Subjects'
        self.subjects = {}

        if "Configuration" in loaded_config and "Subjects" in loaded_config:
            config_params = loaded_config.get("Configuration", {})
            for key, value in config_params.items():
                self.set(key, value)
            if config_params.get("experiment_directory"):
                self.experiment_dir = config_params.get("experiment_directory")
            # We can register a parameter as a list, and the `ConfigFormWidget` will handle it as a dropdown
            # if config_params.get("task"):
            #     self.register_parameter("task", config_params.get("task"), list, "Task identifier", "experiment")
            #     # set the first task in the list as task
            #     self.set("task", config_params.get("task")[0])
            self.subjects = loaded_config.get("Subjects", {})
            if self.subjects:
                first = next(iter(self.subjects.keys()))
                self.select_subject(first)
        else:
            # legacy flat structure
            for key, value in loaded_config.items():
                self.set(key, value)
        
        if "Plugins" in loaded_config:
            self.plugins: dict = loaded_config.get("Plugins", {})
            for plugin in self.plugins:
                if self.plugins.get(plugin, {}).get('enabled') is True:
                    self.register(plugin, 
                                self.plugins.get(plugin, {}).get('config'), 
                                dict, 
                                f"{plugin} plugin configuration", 
                                "plugins")

    def _auto_increment_session(self) -> None:
        """Increment the session number in the config and persist it to the JSON file."""
        # get current session number
        curr = int(self.session)
        next_num = curr + 1
        session_str = f"{next_num:02d}"

        # update in-memory config
        self.set("session", session_str)

        # persist back to the JSON file if available
        path = getattr(self, "_json_file_path", "")
        if path and os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)

                # new-style JSON
                if "Subjects" in data and self.selected_subject in data["Subjects"]:
                    data["Subjects"][self.selected_subject]["session"] = session_str
                # configuration block
                elif "Configuration" in data:
                    data["Configuration"]["session"] = session_str
                # legacy flat structure
                else:
                    data["session"] = session_str

                with open(path, "w") as f:
                    json.dump(data, f, indent=4)
            except Exception as e:
                self.logger.error(f"Failed to update session in JSON file: {e}")
        else:
            self.logger.warning("No JSON file to update; _json_file_path not set or file missing")

    def save_json(self, path: Optional[str] = None) -> None:
        """Persist displayed configuration values back to the JSON file."""
        path = path or getattr(self, "_json_file_path", "")
        if not path or not os.path.isfile(path):
            self.logger.warning("No JSON file to update; _json_file_path not set or file missing")
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)

            display = self.display_keys or []
            subject_vals = data.get("Subjects", {}).get(self.selected_subject, {})

            if "Configuration" in data:
                cfg_block = data.get("Configuration", {})
                for k in display:
                    if k in subject_vals:
                        continue  # subject-specific key
                    if k in cfg_block:
                        cfg_block[k] = self.get(k)
                data["Configuration"] = cfg_block
            else:
                for k in display:
                    if k in subject_vals:
                        continue
                    if k in data:
                        data[k] = self.get(k)

            if subject_vals:
                for k in display:
                    if k in subject_vals and self.has(k):
                        subject_vals[k] = self.get(k)
                if "Subjects" not in data:
                    data["Subjects"] = {}
                data["Subjects"][self.selected_subject] = subject_vals

            with open(path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            self.logger.error(f"Failed to update configuration JSON: {e}")

    def select_subject(self, subject_id: str) -> None:
        """Apply subject-specific parameters from ``self.subjects``."""
        subj = self.subjects.get(subject_id)
        if not subj:
            raise ValueError(f"Subject {subject_id} not found")
        self.selected_subject = subject_id
        self.set("subject", subject_id)
        for key, val in subj.items():
            try:
                self.set(key, val)
            except Exception as e:
                self.logger.error(f"Failed to update session in JSON file: {e}")


