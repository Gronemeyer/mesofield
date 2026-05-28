"""Headless demo procedure for the mesokit-schema pipeline test.

Reuses the legacy SampleProcedure's wall-clock duration cap so the run
terminates cleanly without any GUI or real hardware. The
AcquisitionManifest is written by the base `Procedure._cleanup_procedure`
hook -- subclasses do not have to import or know about mesokit-schema.

Registers two synthetic device types so `hardware.yaml` can reference
them without touching real hardware:

  - mock_wheel  : MockEncoderDevice (CSV samples)
  - mock_camera : MockFrameProducer (OME-TIFF + frame metadata JSON)
"""

from __future__ import annotations

import threading

from mesofield import DeviceRegistry
from mesofield.base import Procedure
from mesofield.devices.mocks import MockFrameProducer
from mesofield.devices.mocks import MockEncoderDevice


DeviceRegistry._registry.setdefault("mock_wheel", MockEncoderDevice)
DeviceRegistry._registry.setdefault("mock_camera", MockFrameProducer)


class PipelineDemoProcedure(Procedure):
    """Mock-encoder-only procedure with duration-gated cleanup."""

    def on_started(self) -> None:
        duration = self.config.get("duration")
        if duration:
            self.logger.info(f"Duration cap armed: {duration}s")
            self._duration_timer = threading.Timer(float(duration), self.cleanup)
            self._duration_timer.daemon = True
            self._duration_timer.start()

    def on_finished(self) -> None:
        super().on_finished()
        timer = getattr(self, "_duration_timer", None)
        if timer is not None:
            timer.cancel()
            self._duration_timer = None

def main():
    """Run the procedure."""
    procedure = PipelineDemoProcedure("/Users/jakegronemeyer/dev/mesofield/experiments/pipeline_demo/experiment.json")
    procedure.run()