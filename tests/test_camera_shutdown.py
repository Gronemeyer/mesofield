"""Contract test for the consolidated BaseCamera.shutdown() template.

`shutdown()` is now a final template on BaseCamera: stop -> _release ->
disconnect the psygnal bundle -> disconnect owned Qt signals, guarded so it's
idempotent. After it returns, nothing the camera emits can reach a subscriber —
the invariant that lets a hot-reload tear a rig down without firing signals into
deleted GUI widgets.

Uses MockFrameProducer, which exercises the template headlessly (no real
hardware, no QApplication needed — QObject signals work without an event loop).
"""

from __future__ import annotations

import pytest

from mesofield.devices.mocks import MockFrameProducer


def _make_cam() -> MockFrameProducer:
    return MockFrameProducer({"id": "mock", "width": 32, "height": 32, "fps": 10})


def test_shutdown_disconnects_signals_and_is_idempotent() -> None:
    cam = _make_cam()
    calls = {"data": 0, "img": 0}
    cam.signals.data.connect(lambda *a: calls.__setitem__("data", calls["data"] + 1))
    if cam.image_ready is not None:
        cam.image_ready.connect(lambda *a: calls.__setitem__("img", calls["img"] + 1))

    # Connected: an emit reaches the subscriber.
    cam.signals.data.emit(0, 0.0)
    assert calls["data"] == 1

    cam.shutdown()
    assert cam._shutdown_done is True
    # _release nulls the mock's Qt image adapter.
    assert cam.image_ready is None

    # Severed at the source: further emissions reach nobody.
    cam.signals.data.emit(1, 1.0)
    assert calls["data"] == 1, "signals.data not disconnected by shutdown()"

    # Idempotent — a second shutdown (e.g. deinitialize + a teardown path) is a no-op.
    cam.shutdown()


def test_shutdown_before_any_run_is_safe() -> None:
    # A camera that was built but never armed/started must still shut down clean.
    cam = _make_cam()
    cam.shutdown()
    assert cam._shutdown_done is True
