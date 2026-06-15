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

from pathlib import Path

from mesofield import DeviceRegistry
from mesofield.base import Procedure
from mesofield.devices.mocks import MockFrameProducer
from mesofield.devices.mocks import MockEncoderDevice


DeviceRegistry._registry.setdefault("mock_wheel", MockEncoderDevice)
DeviceRegistry._registry.setdefault("mock_camera", MockFrameProducer)


class PipelineDemoProcedure(Procedure):
    """Mock-encoder-only procedure.

    The base Procedure stops the run after `duration` seconds (set in
    experiment.json); no on_started/on_finished timer needed.
    """


def main():
    """Run the procedure headlessly until the duration cap stops it."""
    cfg = Path(__file__).parent / "experiment.json"
    procedure = PipelineDemoProcedure(str(cfg))
    procedure.run_until_finished()