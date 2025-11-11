"""
Base procedure classes for implementing experimental workflows in Mesofield.

This module provides base classes that implement the Procedure protocol and integrate
with the Mesofield configuration and hardware management systems.
"""

from __future__ import annotations

import os
import time
from datetime import datetime

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from mesofield.config import ExperimentConfig
from mesofield.hardware import HardwareManager
from mesofield.data.manager import DataManager
from mesofield.utils._logger import get_logger
from mesofield.subprocesses.mouseportal import MousePortal
from mesofield.protocols.experiment_logic import StructuredTrial
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
        self.config = ExperimentConfig(self.hardware_yaml)
        for key, value in procedure_config.custom_parameters.items():
            self.config.set(key, value)

        self.config.set("experiment_id", self.experiment_id)
        self.config.set("experimentor", self.experimentor)

        self.logger = get_logger(f"PROCEDURE.{self.experiment_id}")
        self.logger.info(f"Initialized procedure: {self.experiment_id}")

        self.initialize_hardware()

        if procedure_config.json_config:
            self.setup_configuration(procedure_config.json_config)
        else:
            self._refresh_experiment_plan()
    # ------------------------------------------------------------------
    # Convenience accessors

    @property
    def paths(self):
        return self.data.base.read('datapaths')

    @property
    def hardware(self) -> HardwareManager:
        return self.config.hardware

    @property
    def experiment_definition(self) -> Optional[Dict[str, Any]]:
        definition = self.config.experiment_definition
        return definition if isinstance(definition, dict) else None

    @property
    def experiment_trials(self) -> List[Any]:
        trials = self.config.experiment_trials
        return trials if isinstance(trials, list) else []

    @property
    def experiment_metadata(self) -> Dict[str, Any]:
        metadata = self.config.experiment_metadata
        return metadata if isinstance(metadata, dict) else {}

    @property
    def experiment_plan_payload(self) -> Dict[str, Any]:
        payload = self.config.experiment_plan_payload
        return payload if isinstance(payload, dict) else {}

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
            self.data.setup(self.config)
            self.config.plugins.refresh(data_manager=self.data)
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
        if not json_config:
            return
        self.config.load_json(json_config)
        self.config.hardware._configure_engines(self.config)
        self._refresh_experiment_plan()

    def _refresh_experiment_plan(self) -> None:
        self.config.refresh_experiment_plan()
        data_manager = getattr(self, "data", None)
        self.config.plugins.refresh(data_manager=data_manager)

    # ------------------------------------------------------------------

    def prerun(self) -> None:
        """Run any pre-experiment setup logic."""
        self.logger.info("Running pre-experiment setup")

        if getattr(self.data, "save", None) is None:
            self.data.setup(self.config)
        if not self.data.devices:
            self.data.register_devices(self.config.hardware.devices.values())

        self.data.start_queue_logger()

        self.config.plugins.refresh(data_manager=self.data)
        self.config.plugins.start()

        for cam in self.hardware.cameras:
            cam.set_writer(self.config.make_path)
            cam.set_sequence(self.config.build_sequence)
    
    def run(self) -> None:
        """Run the standard Mesofield workflow."""
        self.logger.info("================= Starting experiment ===================")

        self.prerun()
        self.data.start_queue_logger()

        self._log_experiment_plan()

        portal = self.config.plugins.get("mouseportal")
        portal_ready = False
        if isinstance(portal, MousePortal):
            self.logger.info("Waiting for MousePortal connection...")
            portal_ready = self._wait_for_mouseportal_connection(portal)
            if portal_ready:
                self.logger.info("MousePortal connection established.")
            else:
                self.logger.warning("MousePortal did not report ready within the timeout; continuing without synchronization.")
        else:
            portal = None

        manual_trial = self._first_manual_trial()
        should_wait_for_manual = bool(portal and portal_ready and manual_trial)
        if manual_trial and not should_wait_for_manual:
            self.logger.info(
                "Manual trial '%s' detected but skipping synchronized start.",
                manual_trial.label,
            )
        
        try:
            self.hardware.cameras[0].core.mda.events.sequenceFinished.connect(self._cleanup_procedure) #type: ignore

            if self.config.get("start_on_trigger", False):
                self.psychopy_process = self._launch_psychopy()
                self.psychopy_process.start()

            if should_wait_for_manual and portal and manual_trial:
                portal.drain_messages()
                self.config.plugins.drive()
                required_keys = self.experiment_metadata.get("required_keys") or ["space"]
                self.logger.info(
                    "Awaiting manual trigger '%s' (keys: %s)...",
                    manual_trial.label,
                    ", ".join(required_keys),
                )
                if not self._wait_for_manual_trial_completion(portal, manual_trial):
                    self.logger.warning(
                        "Manual trigger '%s' was not observed before timeout; proceeding with acquisition.",
                        manual_trial.label,
                    )
                self.start_time = datetime.now()
                for cam in self.hardware.cameras:
                    cam.start()
            else:
                self.start_time = datetime.now()
                for cam in self.hardware.cameras:
                    cam.start()
                self.config.plugins.drive()
        except Exception as e:  # pragma: no cover - hardware errors
            self.logger.error(f"Error during experiment: {e}")
            raise

    # ------------------------------------------------------------------
    def save_data(self) -> None:
        mgr = getattr(self, "data_manager", self.data)
        if len(self.hardware.cameras) > 1:
            self.hardware.cameras[1].core.stopSequenceAcquisition()  # type: ignore[attr-defined]
        for cam in self.hardware.cameras:
            cam.stop()
        mgr.save.configuration()
        mgr.save.all_notes()
        mgr.save.all_hardware()
        mgr.save.save_timestamps(self.experiment_id, self.start_time, self.stopped_time)
        mgr.update_database()
        self.config.auto_increment_session()
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
            if len(self.hardware.cameras) > 1:
                self.hardware.cameras[1].core.stopSequenceAcquisition()  # type: ignore[attr-defined]
            if self.hardware.cameras:
                self.hardware.cameras[0].core.mda.events.sequenceFinished.disconnect(self._cleanup_procedure)  # type: ignore[attr-defined]
            self.hardware.stop()
            self.data.stop_queue_logger()
            self.stopped_time = datetime.now()
            self.save_data()
            if hasattr(self, "data_manager"):
                self.data.update_database()
            self.config.plugins.shutdown()
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

    # ------------------------------------------------------------------
    # MousePortal integration
    # ------------------------------------------------------------------
    def ensure_mouseportal(self) -> Optional[MousePortal]:
        data_manager = getattr(self, "data", None)
        portal = self.config.plugins.ensure("mouseportal", data_manager=data_manager)
        return portal if isinstance(portal, MousePortal) else None

    @property
    def mouseportal(self) -> Optional[MousePortal]:
        controller = self.config.plugins.get("mouseportal")
        if isinstance(controller, MousePortal):
            return controller
        return self.ensure_mouseportal()

    def start_mouseportal(self) -> bool:
        portal = self.ensure_mouseportal()
        if portal is None:
            self.logger.warning("MousePortal plugin not enabled; cannot start handler")
            return False
        try:
            return bool(portal.start())
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error(f"MousePortal failed to start: {exc}")
            return False

    def stop_mouseportal(self) -> None:
        portal = self.config.plugins.get("mouseportal")
        if isinstance(portal, MousePortal):
            self.config.plugins.shutdown_one("mouseportal")

    def mouseportal_running(self) -> bool:
        portal = self.config.plugins.get("mouseportal")
        return bool(isinstance(portal, MousePortal) and portal.is_running)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _log_experiment_plan(self) -> None:
        trials = self.experiment_trials
        if not trials:
            self.logger.info("No compiled experiment trials available.")
            return

        self.logger.info("Experiment plan contains %d trial(s):", len(trials))
        for trial in trials:
            duration_text = "manual" if trial.duration is None else f"{trial.duration:.2f}s"
            mode_text = trial.mode or "unspecified"
            self.logger.info(
                "  #%d %s (mode=%s, duration=%s)",
                trial.sequence_index,
                trial.label,
                mode_text,
                duration_text,
            )

        required_keys = self.experiment_metadata.get("required_keys")
        if required_keys:
            self.logger.info("Manual trigger keys required: %s", ", ".join(required_keys))

    def _wait_for_mouseportal_connection(self, portal: MousePortal, timeout: float = 20.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = portal.status()
            socket_info = status.get("socket") or {}
            if str(socket_info.get("status", "")).lower() == "connected":
                return True
            if not status.get("process_running") and not portal.is_running:
                break
            time.sleep(0.2)
        return False

    def _first_manual_trial(self) -> Optional[StructuredTrial]:
        for trial in self.experiment_trials:
            if getattr(trial, "duration", None) is None:
                return trial
        return None

    def _wait_for_manual_trial_completion(
        self,
        portal: MousePortal,
        trial: StructuredTrial,
        *,
        timeout: float = 300.0,
    ) -> bool:
        label = (trial.label or "").strip()
        if not label:
            return False

        target_label = label.lower()
        target_start = f"trial_start:{target_label}"
        target_end = f"trial_end:{target_label}"
        deadline = time.time() + timeout
        seen_start = False

        def status_implies_completion(status_message: Dict[str, Any]) -> bool:
            trial_info = status_message.get("trial")
            current_label = str(trial_info.get("label") or "").lower() if isinstance(trial_info, dict) else ""
            trial_mode = str(status_message.get("trial_mode") or "").lower()
            plan_state = str(status_message.get("plan_state") or "").lower()
            state_field = str(status_message.get("state") or "").lower()
            remaining_field = status_message.get("remaining_trials")
            try:
                remaining_trials = int(remaining_field) if remaining_field is not None else None
            except (TypeError, ValueError):
                remaining_trials = None

            if current_label == target_label and trial_mode == "idle":
                return True
            if remaining_trials == 0 and plan_state in {"idle", "loaded", "complete", "completed"}:
                return True
            if state_field == "idle" and plan_state in {"", "idle", "loaded"} and remaining_trials in (None, 0):
                return True
            return False

        while time.time() < deadline:
            entry = portal.get_message()
            if entry is None:
                if seen_start:
                    snapshot = portal.status().get("last_status") or {}
                    if isinstance(snapshot, dict) and status_implies_completion(snapshot):
                        self.logger.info(
                            "MousePortal manual trial '%s' completed (status snapshot).",
                            label,
                        )
                        return True
                time.sleep(0.1)
                continue

            while entry is not None:
                message = entry.get("message")
                if isinstance(message, dict):
                    msg_type = str(message.get("type") or "").lower()
                    if msg_type == "event":
                        name = str(message.get("name") or "").lower()
                        if name == target_start:
                            seen_start = True
                            self.logger.info("MousePortal manual trial '%s' started.", label)
                        elif name == target_end or (seen_start and "space" in name):
                            self.logger.info("MousePortal manual trial '%s' completed.", label)
                            return True
                    elif msg_type == "status":
                        trial_info = message.get("trial")
                        if isinstance(trial_info, dict):
                            current_label = str(trial_info.get("label") or "").lower()
                            trial_mode = str(message.get("trial_mode") or "").lower()
                            if current_label == target_label and trial_mode == "idle" and seen_start:
                                self.logger.info("MousePortal manual trial '%s' completed (status).", label)
                                return True
                        if seen_start and status_implies_completion(message):
                            self.logger.info(
                                "MousePortal manual trial '%s' completed (plan state).",
                                label,
                            )
                            return True
                entry = portal.get_message()

        return False




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
