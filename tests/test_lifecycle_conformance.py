"""Lifecycle conformance suite for every device class.

Asserts the invariants the rest of mesofield depends on (see
``mesofield/devices/base.py``):

  * ``signals.finished`` is emitted **exactly once** per run, on every exit
    path: natural end (planned duration reached / source exhausted),
    ``stop()``, or an unrecoverable error.
  * ``signals.error`` is emitted (once, before ``finished``) when the run
    terminates because of a failure, and ``device.error`` holds the exception.
  * ``stop()`` is idempotent and safe to call from any thread — including
    from a ``finished`` callback running on the device's own acquisition
    thread (the Procedure-cleanup re-entrancy path).
  * A device flagged ``primary: true`` self-terminates once the planned
    duration from ``arm(config)`` elapses, which is what ends a Procedure.

Parametrized over the concrete producers so a new backend earns these
guarantees by passing this file, not by its author re-reading docstrings.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

import pytest

from mesofield.devices.base import BaseSerialDevice
from mesofield.devices.mocks import MockEncoderDevice, MockFrameProducer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubConfig:
    """Minimal stand-in for ExperimentConfig: duration + path factory."""

    def __init__(self, tmp_path, duration: float = 0.25):
        self._tmp = tmp_path
        self.sequence_duration = duration

    def make_path(self, name, ext, bids_type=None, *_args):
        return str(self._tmp / f"{name}.{ext}")

    @staticmethod
    def build_sequence(_device):
        return None


class SignalCounter:
    """Counts DeviceSignals emissions and exposes wait-able events."""

    def __init__(self, device):
        self.started = 0
        self.finished = 0
        self.errors: list[BaseException] = []
        self.finished_event = threading.Event()
        # Keep refs: psygnal holds strong refs to non-method callables,
        # but be explicit about lifetime anyway.
        self._cbs = [self._on_started, self._on_finished, self._on_error]
        device.signals.started.connect(self._on_started)
        device.signals.finished.connect(self._on_finished)
        device.signals.error.connect(self._on_error)

    def _on_started(self):
        self.started += 1

    def _on_finished(self):
        self.finished += 1
        self.finished_event.set()

    def _on_error(self, exc):
        self.errors.append(exc)

    def wait_finished(self, timeout: float = 5.0) -> bool:
        return self.finished_event.wait(timeout)


class CrashingEncoder(MockEncoderDevice):
    """Records a couple of samples then dies."""

    def _acquire_loop(self) -> None:
        self.record(1, ts=0.0)
        raise RuntimeError("synthetic acquisition failure")


class CrashingFrameProducer(MockFrameProducer):
    def _acquire_loop(self) -> None:
        raise RuntimeError("synthetic acquisition failure")


class FakeSerial:
    """In-memory pyserial stand-in."""

    def __init__(self, line: Optional[bytes] = b"5\n", fail: bool = False):
        self._line = line
        self._fail = fail

    def readline(self) -> bytes:
        if self._fail:
            raise IOError("synthetic serial failure")
        time.sleep(0.005)
        return self._line or b""

    def close(self) -> None:
        pass


class FakeSerialDevice(BaseSerialDevice):
    max_consecutive_failures = 3

    def parse_line(self, line: bytes):
        return int(line.strip()), None


# ---------------------------------------------------------------------------
# Factories (each returns a started-ready, armed device)
# ---------------------------------------------------------------------------


def make_mock_encoder(tmp_path, cfg: dict[str, Any], crashing=False):
    cls = CrashingEncoder if crashing else MockEncoderDevice
    dev = cls({"id": "enc", "sample_interval_ms": 10, **cfg})
    dev.arm(StubConfig(tmp_path))
    return dev


def make_mock_camera(tmp_path, cfg: dict[str, Any], crashing=False):
    cls = CrashingFrameProducer if crashing else MockFrameProducer
    dev = cls({"id": "cam", "width": 8, "height": 8, "frame_interval_ms": 10, **cfg})
    dev.arm(StubConfig(tmp_path))
    return dev


def make_fake_serial(tmp_path, cfg: dict[str, Any], crashing=False):
    dev = FakeSerialDevice({"id": "ser", "development_mode": True, **cfg})
    dev._serial = FakeSerial(fail=crashing)
    dev.arm(StubConfig(tmp_path))
    return dev


FACTORIES = [
    pytest.param(make_mock_encoder, id="mock_encoder"),
    pytest.param(make_mock_camera, id="mock_camera"),
    pytest.param(make_fake_serial, id="serial"),
]


# ---------------------------------------------------------------------------
# Conformance invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", FACTORIES)
def test_stop_emits_finished_exactly_once(tmp_path, factory):
    dev = factory(tmp_path, {})
    counter = SignalCounter(dev)
    assert dev.start() is True
    assert dev.is_active
    time.sleep(0.05)
    dev.stop()
    assert counter.wait_finished()
    assert counter.started == 1
    assert counter.finished == 1
    assert counter.errors == []
    assert dev.error is None
    assert not dev.is_active


@pytest.mark.parametrize("factory", FACTORIES)
def test_stop_is_idempotent(tmp_path, factory):
    dev = factory(tmp_path, {})
    counter = SignalCounter(dev)
    dev.start()
    time.sleep(0.03)
    dev.stop()
    dev.stop()
    dev.shutdown()
    assert counter.finished == 1


@pytest.mark.parametrize("factory", FACTORIES)
def test_primary_self_terminates_at_planned_duration(tmp_path, factory):
    dev = factory(tmp_path, {"primary": True})
    assert dev._planned_duration == pytest.approx(0.25)
    counter = SignalCounter(dev)
    dev.start()
    # No stop() call anywhere: the device must end the run on its own.
    assert counter.wait_finished(timeout=5.0), "primary device never self-terminated"
    assert counter.finished == 1
    assert counter.errors == []
    assert dev.error is None


@pytest.mark.parametrize("factory", FACTORIES)
def test_crash_emits_error_then_finished_once(tmp_path, factory):
    dev = factory(tmp_path, {}, crashing=True)
    counter = SignalCounter(dev)
    dev.start()
    assert counter.wait_finished(timeout=5.0), "crashed device never finalized"
    assert counter.finished == 1
    assert len(counter.errors) == 1
    assert dev.error is counter.errors[0]
    assert not dev.is_active
    # A late stop() (e.g. Procedure stop_all) must not re-emit.
    dev.stop()
    assert counter.finished == 1


@pytest.mark.parametrize("factory", FACTORIES)
def test_stop_from_finished_callback_does_not_deadlock(tmp_path, factory):
    """Procedure cleanup runs stop_all from the primary's `finished` emission,
    which (headless) executes on the device's own acquisition thread."""
    dev = factory(tmp_path, {"primary": True})
    counter = SignalCounter(dev)
    dev.signals.finished.connect(lambda: dev.stop())
    dev.start()
    assert counter.wait_finished(timeout=5.0)
    # Give the re-entrant stop a beat, then confirm single emission.
    time.sleep(0.1)
    assert counter.finished == 1


@pytest.mark.parametrize("factory", FACTORIES)
def test_rearm_allows_second_run(tmp_path, factory):
    dev = factory(tmp_path, {})
    counter = SignalCounter(dev)
    dev.start()
    time.sleep(0.03)
    dev.stop()
    assert counter.finished == 1

    dev.arm(StubConfig(tmp_path))
    assert dev.error is None
    dev.start()
    time.sleep(0.03)
    dev.stop()
    assert counter.finished == 2


def test_serial_dead_port_reaches_error_not_spin(tmp_path):
    """A serial source that raises on every read must transition to ERROR
    after the consecutive-failure budget instead of logging forever."""
    dev = make_fake_serial(tmp_path, {}, crashing=True)
    counter = SignalCounter(dev)
    dev.start()
    assert counter.wait_finished(timeout=5.0)
    assert dev.error is counter.errors[0]
    assert len(counter.errors) == 1
    assert "consecutive" in str(counter.errors[0])


# ---------------------------------------------------------------------------
# OpenCVCamera (QThread-based) — same invariants, faked capture hardware
# ---------------------------------------------------------------------------

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from mesofield.devices.cameras import OpenCVCamera  # noqa: E402


class FakeCapture:
    def __init__(self, *_a, **_k):
        self._props = {
            cv2.CAP_PROP_FRAME_WIDTH: 32.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 24.0,
            cv2.CAP_PROP_FPS: 100.0,
        }

    def isOpened(self):
        return True

    def set(self, prop, val):
        self._props[prop] = float(val)

    def get(self, prop):
        return self._props.get(prop, 0.0)

    def read(self):
        return True, np.zeros((24, 32, 3), dtype=np.uint8)

    def release(self):
        pass


class FakeCV2Writer:
    def __init__(self, fail_begin: bool = False):
        self.fail_begin = fail_begin
        self.begin_calls = 0
        self.frames = 0
        self.finish_calls = 0

    def begin(self, *_a, **_k):
        self.begin_calls += 1
        if self.fail_begin:
            raise RuntimeError("synthetic codec failure")

    def add_frame(self, _frame):
        self.frames += 1

    def finish(self, extra_metadata=None):
        self.finish_calls += 1


@pytest.fixture()
def fake_cv_camera(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    cam = OpenCVCamera(
        {"id": "webcam", "backend": "opencv", "device_index": 0, "fps": 100}
    )
    yield cam
    cam._stop = True
    cam.requestInterruption()
    cam.wait(2000)


def _arm_for_recording(cam, writer, expected_frames=5, primary=True):
    cam.is_primary = primary
    cam.writer = writer
    cam._expected_frames = expected_frames
    with cam._finalize_lock:
        cam._finalized = False


def test_opencv_primary_stops_at_expected_frames(fake_cv_camera):
    cam = fake_cv_camera
    writer = FakeCV2Writer()
    _arm_for_recording(cam, writer, expected_frames=5)
    counter = SignalCounter(cam)
    cam.start()
    assert counter.wait_finished(timeout=5.0), "camera never self-terminated"
    cam.wait(2000)
    assert counter.finished == 1
    assert counter.errors == []
    assert writer.frames == 5
    assert writer.finish_calls == 1
    assert cam.error is None


def test_opencv_stop_finalizes_writer_once(fake_cv_camera):
    cam = fake_cv_camera
    writer = FakeCV2Writer()
    _arm_for_recording(cam, writer, expected_frames=0, primary=False)
    counter = SignalCounter(cam)
    cam.start()
    time.sleep(0.1)
    cam.stop()
    cam.stop()
    assert counter.finished == 1
    assert writer.finish_calls == 1
    assert writer.frames > 0


def test_opencv_begin_failure_aborts_with_error(fake_cv_camera):
    """The Linux 'add_frame called before begin()' regression: a writer that
    fails to open must abort the recording with a single error + finished,
    not spin emitting per-frame exceptions."""
    cam = fake_cv_camera
    writer = FakeCV2Writer(fail_begin=True)
    _arm_for_recording(cam, writer, expected_frames=50)
    counter = SignalCounter(cam)
    cam.start()
    assert counter.wait_finished(timeout=5.0)
    cam.wait(2000)
    assert counter.finished == 1
    assert len(counter.errors) == 1
    assert writer.frames == 0, "frames were written despite begin() failing"
    assert cam.error is counter.errors[0]


def test_opencv_preview_does_not_touch_lifecycle(fake_cv_camera):
    cam = fake_cv_camera
    counter = SignalCounter(cam)
    cam.start_live()
    time.sleep(0.1)
    cam.stop_live()
    assert counter.started == 0
    assert counter.finished == 0
