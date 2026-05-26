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

import threading

from mesofield import DeviceRegistry
from mesofield.base import Procedure
from mesofield.devices.mocks import MockFrameProducer
from mesofield.devices.mocks import MockEncoderDevice


DeviceRegistry._registry.setdefault("mock_wheel", MockEncoderDevice)
DeviceRegistry._registry.setdefault("mock_camera", MockFrameProducer)


class TwoCamDemoProcedure(Procedure):
    """Mock 2-camera + encoder procedure with duration-gated cleanup."""

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
    proc = TwoCamDemoProcedure(str(cfg))
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
