"""Integration test for :class:`mesofield.io.devices.OpenCVCamera`.

Records a few seconds from the local OpenCV camera, verifies that:
  * Protocol attributes are present (HardwareDevice / DataProducer).
  * An MP4 is produced and is non-empty.
  * Frames flowed through DataQueue (data_event signal -> DataManager).

Skipped automatically if no OpenCV-accessible camera is available.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

pytest.importorskip("cv2")
from PyQt6.QtCore import QCoreApplication  # noqa: E402

from mesofield.data.manager import DataManager  # noqa: E402
from mesofield.io.devices.opencv_camera import OpenCVCamera, _resolve_cv_backend  # noqa: E402
from mesofield.protocols import DataProducer, HardwareDevice  # noqa: E402


CAMERA_INDEX = int(os.environ.get("MESOFIELD_TEST_CAMERA_INDEX", "0"))
CV_BACKEND = os.environ.get("MESOFIELD_TEST_CV_BACKEND", "MSMF")


def _camera_available() -> bool:
    import cv2

    cap = cv2.VideoCapture(CAMERA_INDEX, _resolve_cv_backend(CV_BACKEND))
    try:
        if not cap.isOpened():
            return False
        ok, frame = cap.read()
        return bool(ok and frame is not None)
    finally:
        cap.release()


pytestmark = pytest.mark.skipif(
    not _camera_available(), reason="No OpenCV camera available for integration test"
)


@pytest.fixture(scope="module")
def qapp():
    app = QCoreApplication.instance() or QCoreApplication([])
    yield app


def test_protocol_compliance(qapp):
    cam = OpenCVCamera(
        {
            "id": "test_cam",
            "name": "test_cam",
            "backend": "opencv",
            "device_index": CAMERA_INDEX,
            "cv_backend": CV_BACKEND,
            "fps": 15,
        }
    )
    try:
        assert isinstance(cam, HardwareDevice)
        assert isinstance(cam, DataProducer)
        assert cam.device_type == "camera"
        assert cam.file_type == "mp4"
        assert cam.device_id == "test_cam"
        assert cam.metadata["backend"] == "opencv"
    finally:
        cam.shutdown()


def test_capture_and_queue(tmp_path: Path, qapp):
    cam = OpenCVCamera(
        {
            "id": "test_cam",
            "name": "test_cam",
            "backend": "opencv",
            "device_index": CAMERA_INDEX,
            "cv_backend": CV_BACKEND,
            "fps": 15,
        }
    )

    out_path = tmp_path / "test_cam.mp4"

    def make_path(suffix, ext, bids_type, _flag):
        return str(out_path)

    cam.set_writer(make_path)
    cam.set_sequence(lambda _device: None)

    # Wire into a DataManager so frames flow into the queue.
    dm = DataManager(str(tmp_path / "scratch.h5"))
    dm.register_hardware_device(cam)
    assert cam in dm.devices

    cam.start()
    # Capture for ~1.5 s
    deadline = time.perf_counter() + 1.5
    while time.perf_counter() < deadline:
        qapp.processEvents()
        time.sleep(0.05)
    cam.stop()
    # Drain any queued events
    qapp.processEvents()

    assert out_path.exists(), "MP4 was not written"
    assert out_path.stat().st_size > 0, "MP4 is empty"
    assert cam._frame_index > 0, "No frames captured"
    assert not dm.queue.empty(), "Nothing was pushed to DataQueue"

    # Spot-check a packet
    pkt = dm.queue.pop(block=False)
    assert pkt.device_id == "test_cam"

    cam.shutdown()
