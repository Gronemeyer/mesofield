"""Real-time, per-frame processors for camera DataProducers.

A :class:`FrameProcessor` subscribes to a camera's ``signals.frame``,
runs ``compute(img, idx, ts)`` on a daemon worker thread, and emits the
result on both ``signals.data`` (for the DataQueue / CSV logger) and a
Qt-compatible ``valueUpdated(time, value)`` signal (for the existing
:class:`~mesofield.gui.speedplotter.SerialWidget` plotter).
"""

from .base import FrameProcessor
from .frame_mean import FrameMean

__all__ = ["FrameProcessor", "FrameMean"]
