"""
Base procedure classes for implementing experimental workflows in Mesofield.

This module provides base classes that implement the Procedure protocol and integrate
with the Mesofield configuration and hardware management systems.
"""

import os
from datetime import datetime

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Type

from mesofield.config import ExperimentConfig
from mesofield.hardware import HardwareManager
from mesofield.data.manager import DataManager
from mesofield.utils._logger import get_logger
from mesofield.data.writer import CustomWriter
from PyQt6.QtCore import QObject, pyqtSignal

class ProcedureSignals(QObject):
    """All procedure-level signals that a Qt GUI can connect to."""
    procedure_started      = pyqtSignal()
    hardware_initialized   = pyqtSignal(bool)     # success
    data_saved             = pyqtSignal()
    procedure_error        = pyqtSignal(str)      # emits error message
    procedure_finished     = pyqtSignal()
    
    
@dataclass 
class ProcedureConfig:
    """Configuration container for procedures."""
    experiment_id: str = "default_experiment"
    experimentor: str = "researcher"
    hardware_yaml: str = "hardware.yaml"
    data_dir: str = "./data"
    json_config: Optional[str] = None
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


class Procedure:
    """High level class describing an experiment run in Mesofield."""

    def __init__(self, procedure_config: ProcedureConfig):
        self.events = ProcedureSignals()

        # Default parameters for a typical Mesofield experiment
        defaults = {"duration": 60, "start_on_trigger": False}
        procedure_config.custom_parameters = {
            **defaults,
            **procedure_config.custom_parameters,
        }

        self.experiment_id = procedure_config.experiment_id
        self.experimentor = procedure_config.experimentor
        self.hardware_yaml = procedure_config.hardware_yaml
        self.data_dir = procedure_config.data_dir
        self.h5_path = os.path.join(self.data_dir, f"{self.experiment_id}.h5")

        # Initialize configuration and apply custom parameters
        self._config = ExperimentConfig(self.hardware_yaml)
        for key, value in procedure_config.custom_parameters.items():
            self._config.set(key, value)

        self._config.set("experiment_id", self.experiment_id)
        self._config.set("experimentor", self.experimentor)

        self.logger = get_logger(f"PROCEDURE.{self.experiment_id}")
        self.logger.info(f"Initialized procedure: {self.experiment_id}")
        self.initialize_hardware()
        
        if procedure_config.json_config:
            self.setup_configuration(procedure_config.json_config)
    # ------------------------------------------------------------------
    # Convenience accessors
    @property
    def config(self) -> ExperimentConfig:
        return self._config

    @property
    def hardware(self) -> HardwareManager:
        return self._config.hardware
    
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
            if hasattr(self._config.hardware, "initialize"):
                self._config.hardware.initialize(self._config)

            self.data = DataManager()
            self.data.setup(
                self._config,
                self.h5_path,
                self._config.hardware.devices.values(),
            )
            self.logger.info("Hardware initialized successfully")
            
        except RuntimeError as e:  # pragma: no cover - initialization failures
            self.logger.error(f"Failed to initialize hardware: {e}")
            
    # ------------------------------------------------------------------
    #TODO: Connect an update event from the GUI controller with this method
    def setup_configuration(self, json_config: Optional[str]) -> None:
        """ This method loads ExperimentConfig instance with a JSON configuration file.
        
        It then sends this ExperimentConfig object to the HardwareManager, relaying it to MicroManager mda engines
        
        NOTE: This method is called by the ConfigController in the GUI whenever a json configuration file is picked,
        thus, it is necessary to call the Procedure.DataManager.setup() method to ensure that the data manager
        is aware of the new configuration and can set up the data storage accordingly.
        """
        if json_config:
            self._config.load_json(json_config)
            self._config.hardware._configure_engines(self._config)
        
        self.data.setup(
                self._config,
                os.path.join(self._config.save_dir, f"{self.experiment_id}.h5"),
                self._config.hardware.devices.values(),
            )
        

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Run the standard Mesofield workflow."""
        self.logger.info("================= Starting experiment ===================")
        self.data.start_queue_logger()
        try:
            recorders = []
            for cam in self.hardware.cameras:
                writer = self.data.save.writer_for(cam)
                recorders.append((cam.core, self._config.build_sequence(cam), writer))
                
            self.hardware.cameras[0].core.mda.events.sequenceFinished.connect(self._cleanup_procedure)

            if self._config.get("start_on_trigger", False):
                self.psychopy_process = self._launch_psychopy()
                self.psychopy_process.start()

            self.start_time = datetime.now()
            self.hardware.encoder.start_recording()
            for mmc, sequence, writer in recorders:
                mmc.run_mda(sequence, output=writer, block=False)
        except Exception as e:  # pragma: no cover - hardware errors
            self.logger.error(f"Error during experiment: {e}")
            raise

    # ------------------------------------------------------------------
    def save_data(self) -> None:
        mgr = getattr(self, "data_manager", self.data)

        mgr.save.configuration()
        mgr.save.all_notes()
        mgr.save.all_hardware()
        mgr.save.save_timestamps(self.experiment_id, self.start_time, self.stopped_time)
        mgr.update_database()
        self.logger.info("Data saved successfully")


    # ------------------------------------------------------------------
    def _launch_psychopy(self):
        from mesofield.subprocesses.psychopy import PsychoPyProcess
        return PsychoPyProcess(self._config)


    def _cleanup_procedure(self):
        self.logger.info("Cleanup Procedure")
        try:
            #self.hardware.cameras[1].core.stopSequenceAcquisition()
            self.hardware.cameras[0].core.mda.events.sequenceFinished.disconnect(self._cleanup_procedure)
            self.hardware.stop()
            self.data.stop_queue_logger()
            self.stopped_time = datetime.now()
            self.save_data()
            if hasattr(self, "data_manager"):
                self.data.update_database()
        except Exception as e:  # pragma: no cover - cleanup failure
            self.logger.error(f"Error during cleanup: {e}")

    # ------------------------------------------------------------------

    def add_note(self, note: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._config.notes.append(f"{timestamp}: {note}")
        self.logger.info(f"Added note: {note}")

    def load_database(self, key: str = "datapaths"):
        """Return a DataFrame with all sessions stored for this Procedure."""
        if hasattr(self, "data_manager"):
            return self.data.read_database(key)
        return None




# Factory function for creating procedures
def create_procedure(procedure_class: Type[Procedure],
                    experiment_id: str = "default",
                    experimentor: str = "researcher",
                    hardware_yaml: str = "hardware.yaml",
                    data_dir: str = "./data",
                    json_config: Optional[str] = None,
                    **custom_parameters) -> Procedure:
    """
    Factory function to create procedure instances.
    
    Args:
        procedure_class: The procedure class to instantiate
        experiment_id: Unique identifier for the experiment
        experimentor: Name of the person running the experiment
        hardware_yaml: Path to hardware configuration file
        data_dir: Directory for saving data
        json_config: Optional JSON configuration file
        **custom_parameters: Additional custom parameters
    
    Returns:
        Instance of the specified procedure class
    """
    config = ProcedureConfig(
        experiment_id=experiment_id,
        experimentor=experimentor,
        hardware_yaml=hardware_yaml,
        data_dir=data_dir,
        json_config=json_config,
        custom_parameters=custom_parameters
    )
    
    return procedure_class(config)


# Legacy constants for backward compatibility
NAME = "mesofield"
