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
    
    
class Procedure:
    """High level class describing an experiment run in Mesofield."""

    def __init__(self, config_path: Optional[str]):
        self.events = ProcedureSignals()

        experiment_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else None
        self.config = ExperimentConfig(experiment_dir)
        if config_path:
            self.config.load_json(config_path)

        # Default parameters for a typical Mesofield experiment
        defaults = {"duration": 60, "start_on_trigger": False}
        for key, value in defaults.items():
            if not self.config.has(key):
                self.config.set(key, value)

        self.protocol = self.config.get("protocol", "default_experiment")
        self.experimenter = self.config.get("experimenter", "researcher")

        self.data_dir = self.config.data_dir
        self.h5_path = os.path.join(self.data_dir, f"{self.protocol}.h5")

        self.logger = get_logger(f"PROCEDURE.{self.protocol}")
        self.logger.info(f"Initialized procedure: {self.protocol}")
        self.initialize_hardware()
        
        self.config.hardware._configure_engines(self.config)
    # ------------------------------------------------------------------
    # Convenience accessors

    @property
    def paths(self):
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
        self.data.start_queue_logger()
        
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

    # ------------------------------------------------------------------
    def save_data(self) -> None:
        mgr = getattr(self, "data_manager", self.data)
        self.hardware.cameras[1].core.stopSequenceAcquisition() #type: ignore
        for cam in self.hardware.cameras:
            cam.stop()
        mgr.save.configuration()
        mgr.save.all_notes()
        mgr.save.all_hardware()
        mgr.save.save_timestamps(self.protocol, self.start_time, self.stopped_time)
        mgr.update_database()
        #self.config.auto_increment_session()
        # persist any modified configuration values back to the JSON file
        self.config.save_json()
        self.logger.info("Data saved successfully")


    # ------------------------------------------------------------------
    def _launch_psychopy(self):
        from mesofield.subprocesses.psychopy import PsychoPyProcess
        return PsychoPyProcess(self.config)


    def _cleanup_procedure(self):
        self.logger.info("Cleanup Procedure")
        try:
            self.hardware.cameras[1].core.stopSequenceAcquisition()
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
        self.config.notes.append(f"{timestamp}: {note}")
        self.logger.info(f"Added note: {note}")

    def load_database(self, key: str = "datapaths"):
        """Return a DataFrame with all sessions stored for this Procedure."""
        if hasattr(self, "data_manager"):
            return self.data.read_database(key)
        return None




# Factory function for creating procedures
def create_procedure(
    procedure_class: Type[Procedure],
    config_path: Optional[str],
    **custom_parameters,
) -> Procedure:
    """Factory function to create procedure instances."""
    procedure = procedure_class(config_path)
    for key, value in custom_parameters.items():
        procedure.config.set(key, value)
    return procedure


# Legacy constants for backward compatibility
NAME = "mesofield"
