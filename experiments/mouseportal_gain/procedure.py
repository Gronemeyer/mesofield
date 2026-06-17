"""Widefield + pupil + treadmill + MousePortal gain-trial experiment.

mesofield owns the cameras and treadmill and launches MousePortal (the Panda3D
corridor stimulus) as a subprocess stimulus device. Live treadmill velocity is
forwarded to MousePortal over UDP; the treadmill ``device_us`` clock ties the
corridor frames to the camera timeline (see
``mesofield.datakit.sources.behavior.mouseportal.MousePortalSource``).

The primary widefield camera drives session timing; a duration timer ends the
run cleanly (the mock camera has no intrinsic finished-signal), matching the
``two_cam_demo`` pattern.

Run:

    python procedure.py                                   # headless
    mesofield launch experiment.json                      # GUI
"""

from __future__ import annotations

import threading

from mesofield.base import Procedure
from mesofield.devices.mocks import MockFrameProducer, MockTreadmillDevice
from mesofield import DeviceRegistry

# Register the mock device types used by hardware.yaml (the real `camera` /
# `encoder` / `opencv_camera` types are always registered; these mocks are not).
DeviceRegistry._registry.setdefault("mock_camera", MockFrameProducer)
DeviceRegistry._registry.setdefault("mock_treadmill", MockTreadmillDevice)

# Importing the device module runs its @DeviceRegistry.register("mouseportal").
import mesofield.devices.mouseportal_device  # noqa: F401,E402


class MousePortalGainProcedure(Procedure):
    """Mock widefield + pupil + treadmill + MousePortal, duration-gated.

    MousePortal drives the run length: the recording ``duration`` is derived
    from the MousePortal experiment plus a tail buffer, so the cameras
    preallocate enough frames and run a few seconds past the last trial (an
    early cutoff would leave black frames in the OME-TIFF).
    """

    def prerun(self) -> None:
        # Runs before arm_all, so the cameras pick up the derived duration when
        # they build their MDA sequence.
        mp = self.hardware.devices.get("mouseportal")
        if mp is None or not hasattr(mp, "expected_experiment_duration"):
            return
        mp_seconds = mp.expected_experiment_duration()
        tail = float(getattr(mp, "tail_seconds", 5.0))
        total = int(round(mp_seconds + tail))
        self.config.set("duration", total)
        self.logger.info(
            f"Recording duration = MousePortal experiment ({mp_seconds:.1f}s) "
            f"+ tail ({tail:.1f}s) = {total}s"
        )

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


def main() -> int:
    import sys
    from pathlib import Path

    cfg = Path(__file__).parent / "experiment.json"
    proc = MousePortalGainProcedure(str(cfg))
    duration = float(proc.config.get("duration", 60))
    finished = proc.run_until_finished(timeout=duration + 30.0)
    if not finished:
        print("Procedure did not finish in time.", file=sys.stderr)
        return 1

    session_dir = (
        Path(proc.data_dir)
        / f"sub-{proc.config.subject}"
        / f"ses-{proc.config.session}"
    )
    print("\nAcquisition complete.")
    print(f"  Session dir: {session_dir}")
    print(f"  Manifest:    {session_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
