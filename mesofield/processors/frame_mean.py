"""Reference :class:`FrameProcessor` that emits per-frame mean intensity."""

from __future__ import annotations

from typing import Any, Optional

from .base import FrameProcessor
from .registry import register_processor


@register_processor("frame_mean")
class FrameMean(FrameProcessor):
    data_type = "frame_mean"

    def compute(self, img: Any, idx: Any, ts: Any) -> Optional[float]:
        if img is None:
            return None
        try:
            return float(img.mean())
        except Exception:
            return None
