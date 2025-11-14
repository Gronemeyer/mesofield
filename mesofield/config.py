import os
import json
import datetime
from pathlib import Path, PureWindowsPath
from typing import Dict, Any, List, Optional, Type, TypeVar, Callable

import pandas as pd
import useq
from useq import TIntervalLoops

from mesofield.hardware import HardwareManager
from mesofield.protocols import DataProducer
from mesofield.utils._logger import get_logger
from mesofield.plugins import PluginManager
from mesofield.utils.config import load_experiment_config

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
        if value is not None and type_hint and not isinstance(value, type_hint):
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

    def __init__(self, hardware_config_path: Optional[str] = None):
        super().__init__()
        # Initialize logging first
        self.logger = get_logger(__name__)
        if hardware_config_path:
            self.logger.info("Initializing ExperimentConfig with hardware path: %s", hardware_config_path)
        else:
            self.logger.info("Initializing ExperimentConfig without immediate hardware binding")
        
        # Initialize the configuration registry
        self._json_file_path = ''
        self._config_file_path = ''
        self._config_directory = ''
        self._save_dir = ''
        self.subjects: Dict[str, Dict[str, Any]] = {}
        self.selected_subject: str | None = None
        self.display_keys: List[str] | None = None
        self.plugins = PluginManager()
        self._attached_plugins: list[Any] = []
        self.hardware: HardwareManager

        # Register common configuration parameters with defaults and types
        self._register_default_parameters()
        self.logger.debug("Registered default parameters")

        # Initialize hardware
        if hardware_config_path:
            self._set_hardware_config(hardware_config_path)

        self.notes: list = []

    @classmethod
    def from_file(cls, config_path: str, overrides: Optional[Dict[str, Any]] = None) -> "ExperimentConfig":
        path = Path(config_path).expanduser().resolve()
        config_data = load_experiment_config(path)
        instance = cls()
        instance._apply_config_data(config_data, path)

        if overrides:
            for key, value in overrides.items():
                instance.set(key, value)

        return instance

    def _set_hardware_config(self, hardware_path: str) -> None:
        resolved = Path(hardware_path).expanduser()
        if not resolved.is_absolute():
            resolved = resolved.resolve()
        resolved_str = str(resolved)

        existing_path = getattr(getattr(self, "hardware", None), "config_file", None)
        if existing_path == resolved_str:
            return

        try:
            self.hardware = HardwareManager(resolved_str)
        except Exception as exc:
            self.logger.error("Failed to initialize hardware from %s: %s", resolved_str, exc)
            raise

        self.set("hardware_config_file", resolved_str)

    def _apply_config_data(self, config_data: Dict[str, Any], source_path: Optional[Path]) -> Dict[str, Any]:
        if source_path:
            self._config_file_path = str(source_path)
            self._config_directory = str(source_path.parent)
            self._json_file_path = str(source_path) if source_path.suffix.lower() == ".json" else ""
        else:
            self._json_file_path = ""

        base_dir = source_path.parent if source_path else None
        flattened, plugins, subjects, display_keys = self._normalize_config_sections(config_data)
        available_display_keys = list(flattened.keys())

        if isinstance(display_keys, list):
            self.display_keys = display_keys
        else:
            self.display_keys = available_display_keys

        self.plugins.clear()
        if plugins:
            self.plugins.configure_from_mapping(plugins)
            for plugin_name, entry in plugins.items():
                if not isinstance(entry, dict) or not entry.get("enabled"):
                    continue
                self.register(
                    plugin_name,
                    entry.get("config", {}),
                    dict,
                    f"{plugin_name} plugin configuration",
                    "plugins",
                )

        experiment_dir = flattened.pop("experiment_directory", None)
        if experiment_dir:
            resolved_exp_path = self._resolve_path(experiment_dir, base_dir)
            self.save_dir = resolved_exp_path
            self.set("experiment_directory", resolved_exp_path)

        hardware_value = flattened.pop("hardware_config_file", None)
        if hardware_value:
            resolved_hw = self._resolve_path(hardware_value, base_dir)
            self._set_hardware_config(resolved_hw)
        else:
            existing = getattr(getattr(self, "hardware", None), "config_file", None)
            if existing:
                self.set("hardware_config_file", existing)
            else:
                default_hw = self.get("hardware_config_file")
                if default_hw:
                    resolved_default = self._resolve_path(default_hw, base_dir)
                    self._set_hardware_config(resolved_default)

        for key, value in flattened.items():
            self.set(key, value)

        if subjects:
            self.subjects = {
                str(sub_id): dict(values)
                for sub_id, values in subjects.items()
                if isinstance(values, dict)
            }
        else:
            self.subjects = {}

        if self.subjects:
            preferred = flattened.get("subject")

            def _apply_subject(sub_id: str) -> None:
                values = self.subjects.get(sub_id, {})
                if not isinstance(values, dict):
                    return
                self.selected_subject = sub_id
                self.set("subject", sub_id)
                for key, val in values.items():
                    try:
                        self.set(key, val)
                    except Exception as exc:
                        self.logger.error("Failed to apply subject parameter %s: %s", key, exc)

            if preferred and str(preferred) in self.subjects:
                _apply_subject(str(preferred))
            else:
                first_subject = next(iter(self.subjects))
                _apply_subject(first_subject)
        else:
            self.selected_subject = None

        return dict(self.items())

    def _normalize_config_sections(
        self, config_data: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, Any]], Optional[List[str]]]:
        """Flatten the external configuration into simple sections."""
        special_keys = {"Configuration", "Plugins", "DisplayKeys", "Subjects"}
        flattened = {
            str(key): value for key, value in config_data.items() if key not in special_keys
        }

        config_block = config_data.get("Configuration")
        if isinstance(config_block, dict):
            flattened.update(config_block)

        plugins = config_data.get("Plugins")
        plugin_block = plugins if isinstance(plugins, dict) else {}

        subjects = config_data.get("Subjects")
        subject_block = subjects if isinstance(subjects, dict) else {}

        display_keys = config_data.get("DisplayKeys")
        display = display_keys if isinstance(display_keys, list) else None

        return flattened, plugin_block, subject_block, display

    def _resolve_path(self, value: Any, base_dir: Optional[Path]) -> str:
        """Resolve a user-specified path relative to the config file when needed."""
        if value is None:
            return ""

        path_str = str(value)
        if self._looks_like_windows_absolute(path_str):
            return path_str

        candidate = Path(path_str).expanduser()
        if candidate.is_absolute():
            return str(candidate.resolve(strict=False))

        if base_dir:
            return str((base_dir / candidate).resolve(strict=False))

        return str(candidate.resolve(strict=False))

    @staticmethod
    def _looks_like_windows_absolute(path_str: str) -> bool:
        """Detect Windows-style absolute paths even when running under POSIX."""
        try:
            return bool(PureWindowsPath(path_str).drive)
        except Exception:
            return False

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
        self.register("protocol", "experiment", str, "Protocol identifier", "experiment")
        self.register("experimenter", "researcher", str, "Experimenter name", "experiment")
        self.register("experiment_directory", "./data", str, "Base directory for experiment output", "experiment")
        self.register("hardware_config_file", "hardware.yaml", str, "Hardware configuration file", "hardware")

    @property
    def _cores(self):# -> tuple[CMMCorePlus, ...]:
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
        else:
            print(f"ExperimentConfig: \n Invalid save directory path: {path}")

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
    def protocol(self) -> str:
        """Get the protocol identifier."""
        return self.get("protocol")

    @property
    def experimenter(self) -> str:
        """Get the experimenter name."""
        return self.get("experimenter")

    @property
    def start_on_trigger(self) -> bool:
        """Get whether to start on trigger."""
        return self.get("start_on_trigger")
    
    @property
    def sequence_duration(self) -> int:
        """Get the sequence duration in seconds."""
        return int(self.get("duration"))
    
    @property
    def trial_duration(self) -> Optional[int]:
        """Get the trial duration in seconds."""
        trial_dur = self.get("trial_duration")
        return int(trial_dur) if trial_dur is not None else None
        
    @property
    def num_trials(self) -> int:
        """Calculate the number of trials."""
        return int(self.get("num_trials", 1))
    
    
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
            interval=datetime.timedelta(seconds=0),
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
        
    def load_config_file(self, file_path: str) -> Dict[str, Any]:
        """Load an experiment configuration from a modern JSON/YAML file."""
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        self.logger.info("Loading experiment parameters from %s", file_path)

        try:
            config = load_experiment_config(path)
        except Exception as exc:
            self.logger.error("Failed to load experiment configuration from %s: %s", file_path, exc)
            raise

        result = self._apply_config_data(config, path)

        self.logger.debug("Loaded experiment parameters: %s", result)
        return result

    def load_parameters(self, file_path: str) -> Dict[str, Any]:
        """Compatibility shim returning the loaded configuration mapping."""
        return self.load_config_file(file_path)

    def load_json(self, file_path: str) -> Dict[str, Any]:
        """Backward compatible wrapper for legacy calls."""
        return self.load_config_file(file_path)

    # ------------------------------------------------------------------
    def attach_plugins(self, data_manager: Any) -> List[Any]:
        """Instantiate all enabled plugins and attach them to the data layer."""

        self._attached_plugins = list(
            self.plugins.attach_all(experiment_config=self, data_manager=data_manager)
        )
        return self._attached_plugins

    def shutdown_plugins(self) -> None:
        """Stop any running plugins."""

        self.plugins.shutdown()
        self._attached_plugins = []

    def get_plugin_config(self, name: str) -> Dict[str, Any]:
        """Return the configuration block for a given plugin name."""

        return self.plugins.get_config(name)

    def auto_increment_session(self) -> None:
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
        # else:
        #     self.logger.warning("No JSON file to update; _json_file_path not set or file missing")


