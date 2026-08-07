"""GUI camera live-viewing signals -- ImagePreview frame delivery + cleanup.

The non-mmcore frame path (used by OpenCVCamera / MockFrameProducer) delivers
numpy frames into ImagePreview via a queued ``image_ready`` signal. A frame is
stashed under a lock and rendered on the next timer tick (all draws on the GUI
thread), and ``cleanup()`` severs the connection so a late frame can't land in a
deleted widget. These are the behaviors that previously caused "wrapped C/C++
object deleted" crashes, so they are worth pinning down.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gui


@pytest.fixture
def preview_and_adapter(qtbot):
    from mesofield.gui.qt_device_adapter import QtImageAdapter
    from mesofield.gui.viewer import ImagePreview

    adapter = QtImageAdapter()
    preview = ImagePreview(image_payload=adapter.image_ready)
    qtbot.addWidget(preview)
    return preview, adapter


def _frame():
    return np.full((16, 16), 1000, dtype=np.uint16)


def test_requires_mmcore_or_payload(qapp):
    # qapp: ImagePreview is a QWidget; constructing one needs a QApplication
    # even though __init__ raises before the widget is fully built.
    from mesofield.gui.viewer import ImagePreview

    with pytest.raises(ValueError):
        ImagePreview()


def test_external_frame_is_stashed_then_displayed(preview_and_adapter, qtbot):
    preview, adapter = preview_and_adapter

    adapter.emit_frame(_frame())
    # Queued delivery: pump the event loop until the frame lands.
    qtbot.waitUntil(lambda: preview._current_frame is not None, timeout=1000)

    # The timer tick pops the stashed frame and draws it (clearing the stash).
    preview._on_streaming_timeout()
    assert preview._current_frame is None


def test_cleanup_disconnects_payload(preview_and_adapter, qtbot):
    preview, adapter = preview_and_adapter

    preview.cleanup()
    assert preview._cleaned is True

    # After cleanup the queued connection is gone: a late frame is not stashed.
    adapter.emit_frame(_frame())
    qtbot.wait(50)
    assert preview._current_frame is None

    # Idempotent: a second cleanup must not raise.
    preview.cleanup()


def test_streaming_timer_starts_on_external_frame(preview_and_adapter, qtbot):
    preview, adapter = preview_and_adapter

    assert not preview.streaming_timer.isActive()
    adapter.emit_frame(_frame())
    qtbot.waitUntil(lambda: preview.streaming_timer.isActive(), timeout=1000)
