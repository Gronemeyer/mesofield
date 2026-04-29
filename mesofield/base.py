"""
Base procedure classes for implementing experimental workflows in Mesofield.

This module defines a *generic* :class:`Procedure` orchestrator that contains
zero device-specific logic.  Custom experiment subclasses live outside the
package (typically under ``experiments/<name>/procedure.py``) and are
discovered via :func:`load_procedure_from_config`, which reads the optional
``procedure_file`` and ``procedure_class`` fields from ``experiment.json``.

Lifecycle (subclass hooks shown in **bold**):

1. ``initialize_hardware``  -- bring devices up
2. ``prerun``               -- **subclass hook** (default: no-op)
3. ``hardware.arm_all``     -- per-run prep on every device
4. connect ``hardware.primary.signals.finished`` -> ``_cleanup_procedure``
5. ``on_started``           -- **subclass hook** (default: no-op)
6. ``hardware.start_all``
7. ``on_finished``          -- **subclass hook** (default: no-op)
8. ``save_data`` + cleanup
"""

import importlib.util
import json
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Optional, Type

from PyQt6.QtCore import QObject, pyqtSignal

from mesofield.config import ExperimentConfig
from mesofield.data.manager import DataManager
from mesofield.hardware import HardwareManager
from mesofield.protocols import Configurator
from mesofield.utils._logger import get_logger


class ProcedureSignals(QObject):
    """All procedure-level signals that a Qt GUI can connect to."""
    procedure_started      = pyqtSignal()
    hardware_initialized   = pyqtSignal(bool)     # success
    data_saved             = pyqtSignal()
    procedure_error        = pyqtSignal(str)      # emits error message
    procedure_finished     = pyqtSignal()


class Procedure:
    """Generic orchestrator for a Mesofield experiment.

    Subclass this in ``experiments/<name>/procedure.py`` and override the
    extension hooks (:meth:`prerun`, :meth:`on_started`, :meth:`on_finished`)
    and/or the lifecycle methods (:meth:`run`, :meth:`save_data`,
    :meth:`cleanup`) as needed.  The base class never references a specific
    device type -- multi-camera sync is driven by the YAML ``primary: true``
    flag and ``HardwareManager.start_all/stop_all``.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.events = ProcedureSignals()

        self.config: Configurator
        experiment_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else None
        self.config = ExperimentConfig(experiment_dir)
        if config_path:
            self.config.load_json(config_path)

        self.protocol = self.config.get("protocol", "default_experiment")
        self.experimenter = self.config.get("experimenter", "researcher")

        self.data_dir = self.config.data_dir
        self.h5_path = os.path.join(self.data_dir, f"{self.protocol}.h5")

        self.start_time: Optional[datetime] = None
        self.stopped_time: Optional[datetime] = None

        self.logger = get_logger(f"PROCEDURE.{self.protocol}")
        self.logger.info(f"Initialized procedure: {self.protocol}")

        if self.config.hardware.is_configured:
            self.initialize_hardware()
            self.config.hardware._configure_engines(self.config)
        else:
            self.logger.info(
                "Hardware not configured yet -- launch in default state. "
                "Use load_config() to apply a configuration."
            )

    # ------------------------------------------------------------------
    # Convenience accessors

    @property
    def paths(self):
        return self.data.base.read('datapaths')

    @property
    def hardware(self) -> HardwareManager:
        return self.config.hardware

    # ------------------------------------------------------------------
    # Hardware bring-up

    def initialize_hardware(self) -> None:
        """Boot up hardware and a :class:`DataManager`."""
        try:
            self.config.hardware.initialize(self.config)
            self.data = DataManager(self.h5_path)
            self.logger.info("Hardware initialized successfully")
        except RuntimeError as e:
            self.logger.error(f"Failed to initialize hardware: {e}")
            self.config.hardware.deinitialize()
            raise

    def setup_configuration(self, json_config: Optional[str]) -> None:
        """Load a JSON configuration into the existing :class:`ExperimentConfig`."""
        if json_config:
            self.config.load_json(json_config)
            self.config.hardware._configure_engines(self.config)

    def load_config(self, json_path: Optional[str] = None,
                    hardware_yaml_path: Optional[str] = None) -> None:
        """Hot-load an experiment configuration and/or hardware YAML."""
        if hardware_yaml_path:
            self.config.load_hardware(hardware_yaml_path)
        elif json_path:
            candidate = os.path.join(
                os.path.dirname(os.path.abspath(json_path)), "hardware.yaml"
            )
            if os.path.isfile(candidate):
                self.config.load_hardware(candidate)

        if json_path:
            self.config.load_json(json_path)

        self.protocol = self.config.get("protocol", "default_experiment")
        self.experimenter = self.config.get("experimenter", "researcher")
        self.data_dir = self.config.data_dir
        self.h5_path = os.path.join(self.data_dir, f"{self.protocol}.h5")

        if self.config.hardware.is_configured:
            self.initialize_hardware()
            self.config.hardware._configure_engines(self.config)

        self.events.hardware_initialized.emit(self.config.hardware.is_configured)
        self.logger.info("Configuration hot-loaded successfully")

    # ------------------------------------------------------------------
    # Subclass extension hooks (no-op defaults)

    def prerun(self) -> None:
        """Subclass hook called before arming devices.  Override as needed."""
        return None

    def on_started(self) -> None:
        """Subclass hook called immediately after ``start_all``."""
        return None

    def on_finished(self) -> None:
        """Subclass hook called immediately after the primary device finishes."""
        return None

    # ------------------------------------------------------------------
    # Core lifecycle

    def run(self) -> None:
        """Drive a standard experiment run.

        Subclasses may override, but the default body is generic and handles
        any combination of devices declared in ``hardware.yaml``.
        """
        self.logger.info("================= Starting experiment ===================")

        # 1. DataManager / queue logger setup
        self.data.setup(self.config)
        if not self.data.devices:
            self.data.register_devices(self.config.hardware.devices.values())
        self.data.start_queue_logger()

        # 2. Subclass pre-run hook
        self.prerun()

        try:
            # 3. Per-run device prep
            self.hardware.arm_all(self.config)

            # 4. Wire termination: primary device's finished signal triggers cleanup
            self.hardware.primary.signals.finished.connect(self._cleanup_procedure)

            # 5. Start everything
            self.start_time = datetime.now()
            self.events.procedure_started.emit()
            self.hardware.start_all()

            # 6. Subclass post-start hook
            self.on_started()
        except Exception as e:
            self.logger.error(f"Error during experiment: {e}")
            self.events.procedure_error.emit(str(e))
            raise

    def save_data(self) -> None:
        mgr = getattr(self, "data_manager", self.data)
        mgr.save.configuration()
        mgr.save.all_notes()
        mgr.save.all_hardware()
        mgr.save.save_timestamps(self.protocol, self.start_time, self.stopped_time)
        mgr.update_database()
        self.config.save_json()
        self.events.data_saved.emit()
        self.logger.info("Data saved successfully")

    def cleanup(self) -> None:
        """Public cleanup entry-point (manual stop)."""
        self._cleanup_procedure()

    def _cleanup_procedure(self):
        self.logger.info("Cleanup Procedure")
        try:
            try:
                self.hardware.primary.signals.finished.disconnect(self._cleanup_procedure)
            except Exception:
                pass
            self.hardware.stop_all()
            self.data.stop_queue_logger()
            self.stopped_time = datetime.now()
            self.save_data()
            self.on_finished()
            self.events.procedure_finished.emit()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            self.events.procedure_error.emit(str(e))

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


# ----------------------------------------------------------------------
# Procedure discovery

def load_procedure_from_config(config_path: str) -> "Procedure":
    """Instantiate the right :class:`Procedure` subclass for a given JSON.

    The experiment JSON may declare two optional fields:

    ``procedure_file``
        Path to a Python file containing the subclass.  Relative paths are
        resolved against the JSON's directory.
    ``procedure_class``
        Name of the class to import from ``procedure_file``.

    When either field is missing, a base :class:`Procedure` is returned.
    The user file is loaded via :func:`importlib.util.spec_from_file_location`
    so ``experiments/`` does not need to be on ``sys.path``.
    """
    if not config_path or not os.path.isfile(config_path):
        return Procedure(config_path)

    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception:
        return Procedure(config_path)

    proc_file = cfg.get("procedure_file")
    proc_class = cfg.get("procedure_class")
    if not proc_file or not proc_class:
        return Procedure(config_path)

    # Resolve relative paths against the JSON's directory
    json_dir = os.path.dirname(os.path.abspath(config_path))
    if not os.path.isabs(proc_file):
        proc_file = os.path.join(json_dir, proc_file)

    if not os.path.isfile(proc_file):
        raise FileNotFoundError(
            f"procedure_file declared in {config_path} not found: {proc_file}"
        )

    mod_name = f"mesofield_user_procedure_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, proc_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load procedure_file: {proc_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

    cls = getattr(module, proc_class, None)
    if cls is None:
        raise AttributeError(
            f"Class '{proc_class}' not found in {proc_file}"
        )
    if not (isinstance(cls, type) and issubclass(cls, Procedure)):
        raise TypeError(
            f"{proc_class} in {proc_file} must be a subclass of mesofield.base.Procedure"
        )

    return cls(config_path)


# Factory function for creating procedures
def create_procedure(
    procedure_class: Type[Procedure],
    config_path: Optional[str],
    **custom_parameters: Any,
) -> Procedure:
    """Factory function to create procedure instances."""
    procedure = procedure_class(config_path)
    for key, value in custom_parameters.items():
        procedure.config.set(key, value)
    return procedure


# Legacy constants for backward compatibility
NAME = "mesofield"

