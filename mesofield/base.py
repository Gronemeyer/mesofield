"""
Base procedure classes for implementing experimental workflows in Mesofield.

This module provides base classes that implement the Procedure protocol and integrate
with the Mesofield configuration and hardware management systems.
"""
import os
from datetime import datetime

from typing import Dict, Any, Optional, Type

from mesofield.config import ExperimentConfig
from mesofield.hardware import HardwareManager
from mesofield.data.manager import DataManager
from mesofield.utils._logger import get_logger
from mesofield.protocols import ProcedurePlugin
from mesofield.protocols import ProcedurePlugin
from PyQt6.QtCore import QObject, pyqtSignal

class ProcedureSignals(QObject):
    """All procedure-level signals that a Qt GUI can connect to."""
    procedure_started      = pyqtSignal()
    hardware_initialized   = pyqtSignal(bool)     # success
    data_saved             = pyqtSignal()
    procedure_error        = pyqtSignal(str)      # emits error message
    procedure_finished     = pyqtSignal()


class Procedure:
    """High level class describing an experiment run in Mesofield."""

    def __init__(self, experiment_config: ExperimentConfig, *, overrides: Optional[Dict[str, Any]] = None):
        self.events = ProcedureSignals()
        self.config = experiment_config

        if overrides:
            for key, value in overrides.items():
                self.config.set(key, value)

        self.protocol = self.config.protocol if hasattr(self.config, "protocol") else self.config.get("protocol")
        if not self.protocol:
            self.protocol = "experiment"
            self.config.set("protocol", self.protocol)

        self.experimenter = self.config.experimenter if hasattr(self.config, "experimenter") else self.config.get("experimenter", "researcher")
        self.config.set("experimenter", self.experimenter)

        experiment_dir = self.config.get("experiment_directory") or "./data"
        self.data_dir = os.path.abspath(experiment_dir)
        self.config.set("experiment_directory", self.data_dir)
        self.config.save_dir = self.data_dir

        self.h5_path = os.path.join(self.data_dir, f"{self.protocol}.h5")

        self.logger = get_logger(f"PROCEDURE.{self.protocol}")
        self.logger.info(f"Initialized procedure: {self.protocol}")

        self.data: Optional[DataManager] = None
        self.active_plugins: list[ProcedurePlugin] = []
        self.mouse_portal_plugin: Any | None = None

        self.initialize_hardware()
    # ------------------------------------------------------------------
    # Convenience accessors

    @property
    def paths(self):
        if self.data is None:
            raise RuntimeError("Data manager not initialized")
        return self.data.base.read('datapaths')

    @property
    def hardware(self) -> HardwareManager:
        return self.config.hardware
    
    # ------------------------------------------------------------------
    # Core business logic
    def initialize_hardware(self) -> None:
        """Boot up hardware and a `DataManager`.
        
        The core logic here is to have a `Procedure` with instance attributes of: 
            | `ExperimentConfig` | `HardwareManager` | `DataManager` |

        Hardware is ininitialized via the `ExperimentConfig.HardwareManager` instance.
        This is partially leftover from a legacy design, but remains convenient to pass an `ExperimentConfig`
        object in order to provide stateful access to the hardware configuration and management without 
        passing the entire `Procedure` instance itself.
        
        The `DataManager` singleton is initialized here, too, as an attribute of the `Procedure` instance. 
        """
        try:
            self.config.hardware.initialize(self.config)
            self.data = DataManager(self.h5_path)
            self.active_plugins = list(self.config.attach_plugins(self.data))
            self.logger.info("Hardware initialized successfully")
            
        except RuntimeError as e:  # pragma: no cover - initialization failures
            self.logger.error(f"Failed to initialize hardware: {e}")
            
    # ------------------------------------------------------------------
    #TODO: Connect an update event from the GUI controller with this method
    def setup_configuration(self, json_config: Optional[str]) -> None:
        """ This method loads ExperimentConfig instance with a JSON configuration file.
        
        It then sends this ExperimentConfig object to the HardwareManager, relaying it to MicroManager mda engines
        
        NOTE: This method is called by the ConfigController in the GUI whenever a json configuration file is picked
        """
        if json_config:
            self.config.load_json(json_config)
            self.config.hardware._configure_engines(self.config)

    # ------------------------------------------------------------------
    
    def prerun(self) -> None:
        """Run any pre-experiment setup logic."""
        self.logger.info("Running pre-experiment setup")

        if self.data is None:
            raise RuntimeError("Data manager not initialized")

        self.data.setup(self.config)
        if not self.data.devices:
            self.data.register_devices(self.config.hardware.devices.values())

        self.data.start_queue_logger()

        for cam in self.hardware.cameras:
            cam.set_writer(self.config.make_path)
            cam.set_sequence(self.config.build_sequence)
    
    def run(self) -> None:
        """Run the standard Mesofield workflow."""
        self.logger.info("================= Starting experiment ===================")

        self.prerun()
        if self.data is not None:
            self.data.start_queue_logger()
        self._start_plugins()
        self._start_plugins()
        
        try:
            self.hardware.cameras[0].core.mda.events.sequenceFinished.connect(self._cleanup_procedure) #type: ignore

            if self.config.get("start_on_trigger", False):
                self.psychopy_process = self._launch_psychopy()
                self.psychopy_process.start()

            self.start_time = datetime.now()
            self.hardware.encoder.start_recording()
            for cam in self.hardware.cameras:
                cam.start()
        except Exception as e:  # pragma: no cover - hardware errors
            self.logger.error(f"Error during experiment: {e}")
            raise

    def _start_plugins(self) -> None:
        for plugin in self.active_plugins:
            if not isinstance(plugin, ProcedurePlugin):
                continue
            try:
                plugin.begin_experiment()
            except Exception as exc:
                name = getattr(plugin, "name", repr(plugin))
                self.logger.error("Failed to start plugin '%s': %s", name, exc)
                raise

    def _start_plugins(self) -> None:
        for plugin in self.active_plugins:
            if not isinstance(plugin, ProcedurePlugin):
                continue
            try:
                plugin.begin_experiment()
            except Exception as exc:
                self.logger.error("Failed to start plugin '%s': %s", plugin.name, exc)
                raise

    # ------------------------------------------------------------------
    def save_data(self) -> None:
        mgr = self.data
        if mgr is None:
            self.logger.warning("Data manager not initialized; skipping save")
            return

        # stop any ongoing camera acquisitions if supported
        cameras = getattr(self.hardware, "cameras", [])
        if len(cameras) > 1:
            core = getattr(cameras[1], "core", None)
            if core is not None and hasattr(core, "stopSequenceAcquisition"):
                core.stopSequenceAcquisition()  # type: ignore[attr-defined]

        for cam in cameras:
            if hasattr(cam, "stop"):
                cam.stop()

        mgr.stop_queue_logger()

        saver = getattr(mgr, "save", None)
        if saver is None:
            self.logger.warning("Data saver not initialized; skipping save")
            return
        saver.configuration()
        saver.all_notes()
        saver.all_hardware()

        start_time = getattr(self, "start_time", datetime.now())
        stop_time = getattr(self, "stopped_time", datetime.now())
        saver.save_timestamps(self.protocol, start_time, stop_time)

        mgr.update_database()

        # persist session increment and configuration values back to the JSON file
        if hasattr(self.config, "auto_increment_session"):
            self.config.auto_increment_session()
        self.config.save_json()
        self.logger.info("Data saved successfully")


    # ------------------------------------------------------------------
    def _launch_psychopy(self):
        from mesofield.subprocesses.psychopy import PsychoPyProcess
        return PsychoPyProcess(self.config)


    def _cleanup_procedure(self):
        self.logger.info("Cleanup Procedure")
        try:
            cameras = getattr(self.hardware, "cameras", [])
            if len(cameras) > 1:
                core = getattr(cameras[1], "core", None)
                if core is not None and hasattr(core, "stopSequenceAcquisition"):
                    core.stopSequenceAcquisition()  # type: ignore[attr-defined]

            if cameras:
                core0 = getattr(cameras[0], "core", None)
                mda = getattr(core0, "mda", None)
                events = getattr(getattr(mda, "events", None), "sequenceFinished", None)
                if events is not None and hasattr(events, "disconnect"):
                    events.disconnect(self._cleanup_procedure)

            if hasattr(self.hardware, "stop"):
                self.hardware.stop()

            if self.data is not None:
                self.data.stop_queue_logger()
            self.stopped_time = datetime.now()
            self.save_data()
            self.config.shutdown_plugins()
            self.active_plugins = []
            self.mouse_portal_plugin = None
        except Exception as e:  # pragma: no cover - cleanup failure
            self.logger.error(f"Error during cleanup: {e}")

    # ------------------------------------------------------------------

    def add_note(self, note: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.config.notes.append(f"{timestamp}: {note}")
        self.logger.info(f"Added note: {note}")

    def load_database(self, key: str = "datapaths"):
        """Return a DataFrame with all sessions stored for this Procedure."""
        if self.data is not None:
            return self.data.read_database(key)
        return None


# Factory function for creating procedures
def create_procedure(
    procedure_class: Type[Procedure],
    *,
    config: Optional[ExperimentConfig] = None,
    config_path: Optional[str] = None,
    hardware_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Procedure:
    """Instantiate a procedure using an existing or newly loaded configuration."""

    if config is None:
        if config_path:
            config = ExperimentConfig.from_file(config_path, overrides)
            overrides = None
        else:
            hardware = hardware_path or "hardware.yaml"
            config = ExperimentConfig(hardware)

    return procedure_class(config, overrides=overrides)


# Legacy constants for backward compatibility
NAME = "mesofield"
