"""pymmcore-plus integration -- MMCamera engine selection + MDA signal wiring.

These exercise the real ``MMCamera`` (micromanager backend) on the MicroManager
**demo** config: the custom ``MDAEngine`` subclasses in ``mesofield.engines`` are
selected by camera id, and ``MMCamera._wire_signals`` bridges MDA events
(sequenceStarted / frameReady / sequenceFinished) onto the standard
``DeviceSignals``. Skipped when the MM demo adapters are not installed. Nothing
here touches physical hardware.
"""

from __future__ import annotations

import pytest
import useq

from mesofield.devices.cameras import MMCamera

pytestmark = pytest.mark.integration


def _make_camera(cam_id: str = "dev_cam") -> MMCamera:
    try:
        return MMCamera({"id": cam_id, "backend": "micromanager"})
    except Exception as exc:  # demo adapters not installed
        pytest.skip(f"MicroManager demo unavailable: {exc}")


@pytest.mark.parametrize(
    "cam_id,engine_name",
    [
        ("dev_cam", "DevEngine"),    # default
        ("ThorCam", "PupilEngine"),  # pupil rig
        ("Dhyana", "MesoEngine"),    # mesoscope rig
    ],
)
def test_engine_selection_by_camera_id(cam_id, engine_name):
    cam = _make_camera(cam_id)
    try:
        assert type(cam._engine).__name__ == engine_name
        assert cam.core.mda.engine is cam._engine
    finally:
        cam.shutdown()


@pytest.mark.slow
def test_mda_events_emit_device_signals():
    """A short MDA run drives signals.started / .frame / .finished."""
    cam = _make_camera("dev_cam")
    # Force the simple per-event path so the demo camera streams reliably.
    cam._engine.use_hardware_sequencing = False

    started, finished, frames = [], [], []
    cam.signals.started.connect(lambda *a: started.append(1))
    cam.signals.finished.connect(lambda *a: finished.append(1))
    cam.signals.frame.connect(lambda *a: frames.append(a))

    try:
        cam.core.mda.run(useq.MDASequence(time_plan={"interval": 0.0, "loops": 3}))
        assert started, "sequenceStarted did not reach signals.started"
        assert finished, "sequenceFinished did not reach signals.finished"
        assert len(frames) >= 1, "frameReady did not reach signals.frame"
        # frame payload is (img, idx, ts)
        assert len(frames[0]) == 3
    finally:
        cam.shutdown()
