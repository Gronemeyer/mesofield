"""Two-camera + encoder demo: 10-second headless acquisition.

Subclasses :class:`mesofield.base.Procedure` with the standard duration-
timer pattern so the run terminates cleanly without a primary device's
hardware finished-signal. The AcquisitionManifest is written
automatically by the base Procedure's cleanup hook.

Run:

    python procedure.py            # headless (no Qt)
    mesofield launch experiment.json   # GUI

The headless path is what `mesofield init` scaffolds; cf. TUTORIAL.md.
"""

from __future__ import annotations

from mesofield import DeviceRegistry
from mesofield.base import Procedure
from mesofield.devices.mocks import MockFrameProducer
from mesofield.devices.mocks import MockEncoderDevice
from mesofield.processors import FrameMean


DeviceRegistry._registry.setdefault("mock_wheel", MockEncoderDevice)
DeviceRegistry._registry.setdefault("mock_camera", MockFrameProducer)


class TwoCamDemoProcedure(Procedure):
    """Mock 2-camera + encoder procedure with duration-gated cleanup."""

    experiment = "experiment.json"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Worked example of the procedure-authored processor API:
        # construct, assign, done. Auto-attached, auto-registered with the
        # DataManager, and auto-plotted in the GUI when `plot=True`.
        self.frame_mean = FrameMean(
            camera=self.hardware.primary,
            plot=True,
            label="Frame Mean",
            value_label="Mean intensity",
            y_range=(0, 255),
        )

    # The base Procedure stops the run after `duration` seconds (set in
    # experiment.json); no on_started/on_finished timer needed.


def main() -> int:
    import sys
    from pathlib import Path

    proc = TwoCamDemoProcedure()
    finished = proc.run_until_finished(timeout=30.0)
    if not finished:
        print("Procedure did not finish in time.", file=sys.stderr)
        return 1

    session_dir = (
        Path(proc.data_dir)
        / f"sub-{proc.config.subject}"
        / f"ses-{proc.config.session}"
    )
    print(f"\nAcquisition complete.")
    print(f"  Session dir: {session_dir}")
    print(f"  Manifest:    {session_dir / 'manifest.json'}")
    print(f"\nNext: python load_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
