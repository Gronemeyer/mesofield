"""Experiment configuration registry.

This module defines two classes:

:class:`ConfigRegister`
    A generic key/value registry with optional type validation and
    per-key change callbacks. Used as a building block by
    :class:`ExperimentConfig`.

:class:`ExperimentConfig`
    Experiment-aware extension of ``ConfigRegister`` with default
    parameters (subject / session / task / LED pattern / duration), a
    BIDS-style path layout (``experiment_dir`` → ``data_dir`` →
    ``bids_dir``), JSON load/save, and an attached
    :class:`~mesofield.hardware.HardwareManager`.

Typical lifecycle:

.. code-block:: python

    cfg = ExperimentConfig("path/to/hardware.yaml")
    cfg.load_json("path/to/experiment.json")
    cfg.set("subject", "001")
    seq = cfg.build_sequence(cfg.hardware.primary)
"""

import os
import json
import dataclasses
import datetime
from typing import Dict, Any, List, Optional, Type, TypeVar, Callable

import pandas as pd
import useq
from useq import TIntervalLoops

from mesofield.hardware import HardwareManager
from mesofield.protocols import DataProducer
from mesofield.utils._logger import get_logger, hyperlink

T = TypeVar('T')


class ConfigRegister:
    """A registry that maintains configuration values with optional type validation."""
    
    def __init__(self):
        self._registry: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._callbacks: Dict[str, List[Callable[[str, Any], None]]] = {}
        self._choices: Dict[str, List[Any]] = {}
    
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
    
    
    def register_choices(self, key: str, choices: List[Any]) -> None:
        """Register a list of selectable choices for a configuration key."""
        self._choices[key] = list(choices)

    def get_choices(self, key: str) -> Optional[List[Any]]:
        """Return the list of choices for *key*, or ``None`` if none are registered."""
        return self._choices.get(key)

    def clear(self) -> None:
        """Clear all configurations."""
        self._registry.clear()
        self._choices.clear()


class ExperimentConfig(ConfigRegister):
    """Generate and store experiment parameters using a configuration registry.

    ``ExperimentConfig`` extends :class:`ConfigRegister` with experiment-aware
    defaults (subject / session / task, LED pattern, duration, etc.), a
    BIDS-style path layout (``experiment_dir / data_dir / bids_dir``), and
    integration with a :class:`~mesofield.hardware.HardwareManager`.

    Example:
        .. code-block:: python

            from mesofield.config import ExperimentConfig

            config = ExperimentConfig("path/to/hardware.yaml")
            # Populate from a JSON config file:
            config.load_json("path/to/experiment.json")

            config.experiment_dir = "./output"
            config.set("subject", "001")
            config.set("task", "TestTask")
            config.notes.append("This is a test note.")

            # Persist parameters and notes back to JSON:
            config.save_json("path/to/experiment.json")
    """

    def __init__(self, path: Optional[str] = None):
        super().__init__()
        # Initialize logging first
        self.logger = get_logger(__name__)
        if path:
            self.logger.info(
                "Initializing ExperimentConfig with hardware path: "
                f"{hyperlink(path, os.path.basename(os.path.normpath(path)))}"
            )
        else:
            self.logger.info("Initializing ExperimentConfig with hardware path: None")
        
        # Initialize the configuration registry
        self._json_file_path = ''
        self._save_dir = ''
        self.subjects: Dict[str, Dict[str, Any]] = {}
        self.selected_subject: str | None = None
        self.display_keys: List[str] | None = None

        # Register common configuration parameters with defaults and types
        self._register_default_parameters()
        self.logger.debug("Registered default parameters")

        # Data output directory defaults to the current working directory;
        # an `experiment_directory` config key, load_json, or the GUI picker
        # overrides it. It is intentionally NOT derived from the hardware
        # file's location -- a rig YAML may live in a shared rig store, and
        # data must never land there. Assigned directly (not via the property
        # setter) so `experiment_dir_is_set` stays False until a caller
        # chooses a directory explicitly.
        self._experiment_dir_set = False
        self._save_dir = os.path.abspath(os.getcwd())

        # Initialize hardware. The rig (hardware.yaml) is the anchor; experiment
        # params are loaded separately and never touch hardware state.
        self.hardware: HardwareManager
        try:
            self.hardware = HardwareManager(self._resolve_hardware_path(path))
        except Exception as e:
            self.logger.warning(f"Hardware config not available: {e}. Starting in default state.")
            self.hardware = HardwareManager()
        
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
        self.register("led_pattern", ["4", "4"], list, "Arduino LED sequence pattern", "hardware")
        self.register("psychopy_filename", "experiment.py", str, "PsychoPy experiment filename", "experiment")

    def set(self, key: str, value: Any) -> None:
        """Set config values with field-specific normalization where needed."""
        if key == "led_pattern":
            value = self._normalize_led_pattern(value)
        super().set(key, value)

    def _resolve_hardware_path(self, path: Optional[str]) -> Optional[str]:
        """Normalize a hardware path to a YAML file (or ``None``).

        A file path is used as-is; a directory resolves to
        ``<dir>/hardware.yaml``; ``None`` stays ``None``. This never touches
        ``experiment_dir`` -- the rig location does not dictate where data is
        written.
        """
        if not path:
            return None
        abs_path = os.path.abspath(path)
        if os.path.isdir(abs_path):
            return os.path.join(abs_path, "hardware.yaml")
        return abs_path

    def load_hardware(self, yaml_path: str) -> None:
        """Load (or reload) a hardware YAML configuration.

        This replaces the current :class:`HardwareManager` with a new one
        pointed at *yaml_path*.  Devices are **not** initialised until
        :meth:`HardwareManager.initialize` is called (which is normally done
        by :class:`~mesofield.base.Procedure.initialize_hardware`).
        """
        abs_path = os.path.abspath(yaml_path)
        self.hardware = HardwareManager(abs_path)
        self.logger.info(
            "Loaded hardware config from: "
            f"{hyperlink(abs_path, os.path.basename(abs_path))}"
        )

    @property
    def _cores(self):# -> tuple[CMMCorePlus, ...]:
        """Return the tuple of CMMCorePlus instances from the hardware cameras."""
        return tuple(cam.core for cam in self.hardware.cameras if hasattr(cam, 'core'))

    @property
    def experiment_dir(self) -> str:
        """Get the experiment directory (base directory)."""
        return self._save_dir

    @experiment_dir.setter
    def experiment_dir(self, path):
        """Set the experiment directory (base directory)."""
        if isinstance(path, (str, os.PathLike)):
            self._save_dir = os.path.abspath(os.fspath(path))
            self._experiment_dir_set = True
        else:
            print(f"ExperimentConfig: \n Invalid experiment directory path: {path}")

    @property
    def experiment_dir_is_set(self) -> bool:
        """``True`` once a caller has explicitly chosen ``experiment_dir``.

        Lets launchers (e.g. ``load_procedure_from_config``) apply a fallback
        directory only when the user/config never picked one.
        """
        return self._experiment_dir_set

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
        """Get the session ID as a zero-padded BIDS string (e.g. "01").

        Formatting is enforced here so paths/filenames are always padded
        regardless of how the raw value was entered (GUI, JSON, etc.).
        """
        raw = self.get("session")
        try:
            return f"{int(raw):02d}"
        except (TypeError, ValueError):
            return "" if raw is None else str(raw)

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
        """Build a :class:`useq.MDASequence` sized to this experiment.

        The loop count is derived from ``num_meso_frames`` when set, or
        from ``camera.sampling_rate * sequence_duration`` otherwise.  All
        ``HardwareManager`` fields are attached as sequence metadata so
        downstream engines (e.g. :class:`~mesofield.engines.MesoEngine`)
        can resolve the LED pattern and NI-DAQ at setup time.

        Args:
            camera: The primary :class:`DataProducer` whose
                ``sampling_rate`` drives the default loop count.

        Returns:
            A ready-to-run ``MDASequence`` with a zero-interval time plan.
        """
        if self.has('num_meso_frames'):
            loops = int(self.get('num_meso_frames'))
        else:
            try:
                loops = int(camera.sampling_rate * self.sequence_duration)
            except Exception:
                loops = 5

        metadata = dict(self.hardware.__dict__)
        metadata["led_sequence"] = self.led_pattern

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
        value = self.get("led_pattern")
        return self._normalize_led_pattern(value)
    
    @led_pattern.setter
    def led_pattern(self, value: Any) -> None:
        """Set the LED pattern, normalising strings / JSON lists to ``list[str]``."""
        self.set("led_pattern", value)

    @staticmethod
    def _normalize_led_pattern(value: Any) -> list[str]:
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                raise ValueError("led_pattern must not be empty")

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = raw

            if isinstance(parsed, list):
                value = parsed
            elif isinstance(parsed, str):
                value = list(parsed)
            else:
                # Numeric tokens like "422222442" parse as int in json.loads.
                # Treat the original string as compact LED sequence shorthand.
                value = list(raw)

        if isinstance(value, list):
            normalized = [str(item) for item in value]
            if not normalized:
                raise ValueError("led_pattern must not be empty")
            return normalized

        raise ValueError("led_pattern must be a list or a JSON string representing a list")
    
    def make_path(self, suffix: str, extension: str, bids_type: Optional[str] = None, create_dir: bool = False):
        """Build a unique BIDS-style output file path.

        The returned path follows the layout
        ``<bids_dir>/[<bids_type>/]<timestamp>_sub-<id>_ses-<id>_task-<id>_<suffix>.<ext>``.
        If a file with that name already exists, ``_<n>`` is appended to keep
        the path unique.

        Args:
            suffix: Trailing tag added to the filename, e.g. ``"images"``.
            extension: File extension without the leading dot, e.g. ``"jpg"``.
            bids_type: Optional BIDS modality subdirectory under ``bids_dir``
                (e.g. ``"func"``). When ``None``, the file is placed directly
                under ``bids_dir``.
            create_dir: When ``True``, parent directories are created.

        Returns:
            Absolute path to the generated file.

        Example:
            .. code-block:: python

                cfg.make_path("images", "jpg", "func")
                # -> 'C:/save_dir/data/sub-001/ses-01/func/'
                #    '20250110_123456_sub-001_ses-01_task-example_images.jpg'
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
        file_path_str = os.fspath(file_path)
        file_link = hyperlink(
            file_path_str,
            os.path.basename(os.path.normpath(file_path_str)),
        )
        self.logger.info(f"Loading configuration from: {file_link}")
        try:
            with open(file_path_str, 'r') as f:
                loaded_config = json.load(f)
            self.logger.info("Successfully loaded configuration JSON")
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {file_link}")
            return
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON from {file_link}: {e}")
            return

        self._json_file_path = file_path_str #store the json filepath
        json_dir = os.path.dirname(os.path.abspath(file_path_str))
        if json_dir:
            # Experiment params live beside their JSON; point data output there.
            # Hardware is owned by the constructor/load_hardware and is never
            # rebuilt as a side effect of loading parameters.
            self.experiment_dir = json_dir
        self._apply_config(loaded_config)

    def load_dict(self, data: Any) -> None:
        """Load parameters from a dataclass instance or plain mapping.

        This is the programmatic counterpart to :meth:`load_json` used by
        scripted procedures (see :meth:`Procedure.define_config`). Unlike
        :meth:`load_json` it does not touch the hardware YAML path -- scripted
        hardware is supplied directly via :meth:`Procedure.define_hardware`.

        *data* may be a ``@dataclass`` instance or any mapping. Both the flat
        and the ``Configuration``/``Subjects`` shapes are accepted.
        """
        if dataclasses.is_dataclass(data) and not isinstance(data, type):
            loaded_config = dataclasses.asdict(data)
        else:
            loaded_config = dict(data)
        self.logger.info("Loading configuration from in-memory mapping")
        self._apply_config(loaded_config)

    def _apply_config(self, loaded_config: dict) -> None:
        """Apply a parsed configuration mapping to the registry."""
        self.display_keys = loaded_config.get("DisplayKeys")
        # Detect new style JSON with 'Configuration' and 'Subjects'
        self.subjects = {}

        if "Configuration" in loaded_config and "Subjects" in loaded_config:
            config_params = loaded_config.get("Configuration", {})
            for key, value in config_params.items():
                if isinstance(value, list) and key != "led_pattern":
                    # Lists in Configuration are treated as selectable choices.
                    # Store the full list as choices and default to the first item.
                    self.register_choices(key, value)
                    if value:
                        self.set(key, value[0])
                else:
                    self.set(key, value)
            if config_params.get("experiment_directory"):
                self.experiment_dir = config_params.get("experiment_directory")
            self.subjects = loaded_config.get("Subjects", {})
            if self.subjects:
                first = next(iter(self.subjects.keys()))
                self.select_subject(first)
        else:
            # flat structure (legacy JSON or scripted define_config mappings)
            for key, value in loaded_config.items():
                self.set(key, value)
            if loaded_config.get("experiment_directory"):
                self.experiment_dir = loaded_config["experiment_directory"]

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

    def save_json_as(self, path: str) -> None:
        """Write the current configuration to a *new* JSON file and adopt it.

        Unlike :meth:`save_json` (which edits an existing file in place), this
        serializes the full in-memory state -- registry values, subjects, and
        DisplayKeys -- into the ``Configuration`` / ``Subjects`` / ``DisplayKeys``
        shape, then points ``_json_file_path`` at the new file so later saves
        land there. Lets the GUI author an experiment.json from a hardware-only
        session.
        """
        data = {
            "Configuration": self.items(),
            "Subjects": self.subjects,
            "DisplayKeys": self.display_keys or [],
        }
        abs_path = os.path.abspath(path)
        try:
            with open(abs_path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            self.logger.error(f"Failed to write configuration JSON: {e}")
            raise
        self._json_file_path = abs_path
        self.logger.info(
            "Wrote experiment configuration to: "
            f"{hyperlink(abs_path, os.path.basename(abs_path))}"
        )

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

    def _read_json_file(self) -> Optional[dict]:
        path = getattr(self, "_json_file_path", "")
        if not path or not os.path.isfile(path):
            self.logger.warning("No JSON file to update; _json_file_path not set or file missing")
            return None
        with open(path, "r") as f:
            return json.load(f)

    def _write_json_file(self, data: dict) -> None:
        path = getattr(self, "_json_file_path", "")
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def add_subject(self, subject_id: str) -> None:
        """Add a new subject, seeding parameters from existing subjects.

        The new subject's parameter dict is the union of keys from existing
        subjects with blank string values, so all subjects share a consistent
        parameter set and :meth:`select_subject` will accept the new entry.
        Persists to ``experiment.json``.
        """
        subject_id = (subject_id or "").strip()
        if not subject_id:
            raise ValueError("Subject ID must not be empty")
        if subject_id in self.subjects:
            raise ValueError(f"Subject '{subject_id}' already exists")

        seed_keys: set = set()
        for params in self.subjects.values():
            seed_keys.update(params.keys())
        if not seed_keys and self.display_keys:
            seed_keys = {k for k in self.display_keys if k != "subject"}
        if not seed_keys:
            seed_keys = {"session"}

        new_params = {k: "" for k in seed_keys}
        self.subjects[subject_id] = new_params

        data = self._read_json_file()
        if data is not None:
            data.setdefault("Subjects", {})
            data["Subjects"][subject_id] = new_params
            self._write_json_file(data)

    def add_parameter(self, name: str, default: Any, type_hint: Type) -> None:
        """Add a subject-scoped parameter to every subject and DisplayKeys.

        Registers the parameter in the config registry, fills it on every
        subject in :attr:`subjects`, appends it to :attr:`display_keys`, and
        persists the additions to ``experiment.json``.
        """
        name = (name or "").strip()
        if not name:
            raise ValueError("Parameter name must not be empty")
        if self.has(name):
            raise ValueError(f"Parameter '{name}' already exists")
        if type_hint not in (str, int, bool):
            raise ValueError("type_hint must be str, int, or bool")

        self.register(name, default, type_hint, "User-added subject parameter", "subject")

        for params in self.subjects.values():
            if name not in params:
                params[name] = default

        if self.selected_subject and self.selected_subject in self.subjects:
            self.set(name, self.subjects[self.selected_subject].get(name, default))
        else:
            self.set(name, default)

        if self.display_keys is None:
            self.display_keys = []
        if name not in self.display_keys:
            self.display_keys.append(name)

        data = self._read_json_file()
        if data is not None:
            subjects_block = data.setdefault("Subjects", {})
            for subj_id in subjects_block:
                subjects_block[subj_id].setdefault(name, default)
            display = data.setdefault("DisplayKeys", [])
            if name not in display:
                display.append(name)
            self._write_json_file(data)


