"""
Minimal custom procedures demonstrating how to create user-defined experimental workflows.

This module shows a minimal subclass template aligned with mesofield.base.Procedure,
so users can copy and adapt it for their own lab procedures.
"""

import time
from datetime import datetime
from typing import Type, Union

from mesofield.base import Procedure, ProcedureConfig, resolve_procedure_class


class MinimalProcedure(Procedure):
    """A minimal example of a Procedure subclass.

    Behavior:
    - Initializes hardware and data manager via base constructor
    - Starts encoder and all cameras
    - Runs for `duration` seconds (from the JSON config or custom_parameters)
    - Stops devices, persists data, and updates the database
    """

    def __init__(self, procedure_config: ProcedureConfig):
        super().__init__(procedure_config)

    def run(self) -> None:
        self.logger.info("=== Starting MinimalProcedure ===")
        # Prepare writers, sequences, and queue logging
        self.prerun()

        # Start acquisition
        self.start_time = datetime.now()
        try:
            if hasattr(self.hardware, "encoder") and self.hardware.encoder:
                self.hardware.encoder.start_recording()
            for cam in self.hardware.cameras:
                cam.start()

            duration = int(self.config.get("duration", 60))
            self.logger.info(f"Running for {duration} seconds")
            time.sleep(duration)

        finally:
            # Stop devices regardless of errors
            try:
                if hasattr(self.hardware, "encoder") and self.hardware.encoder:
                    self.hardware.encoder.stop()
            except Exception:
                pass
            for cam in self.hardware.cameras:
                try:
                    cam.stop()
                except Exception:
                    pass

            self.stopped_time = datetime.now()
            self.save_data()
            self.logger.info("=== MinimalProcedure completed ===")


def create_custom_procedure(
    procedure: Union[str, Type[Procedure]],
    **procedure_config_kwargs,
) -> Procedure:
    """Factory to create a custom Procedure from a class or import string.

    Example:
        create_custom_procedure(
            "mesofield.examples.custom_procedures:MinimalProcedure",
            experiment_id="exp_001",
            experimentor="researcher",
            hardware_yaml="hardware.yaml",
            data_dir="./data",
            json_config="./config.json",
            duration=10,  # passed via custom_parameters
        )
    """
    ProcClass = resolve_procedure_class(procedure)
    # ProcedureConfig forwards extra kwargs to custom_parameters
    pcfg = ProcedureConfig(**{
        k: v for k, v in procedure_config_kwargs.items() if k in {
            "experiment_id", "experimentor", "hardware_yaml", "data_dir", "json_config"
        }
    })
    # collect remaining keys as custom parameters
    extra = {k: v for k, v in procedure_config_kwargs.items() if k not in {
        "experiment_id", "experimentor", "hardware_yaml", "data_dir", "json_config"
    }}
    if extra:
        pcfg.custom_parameters.update(extra)
    return ProcClass(pcfg)
