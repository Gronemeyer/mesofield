from pathlib import Path

import numpy as np
import tifffile

from mesofield.playback.launch import discover_playback_context
from mesofield.playback.media import (
    TiffFrameSource,
    _normalize_to_uint8,
    load_treadmill_trace,
)


def _write_minimal_dataqueue(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """queue_elapsed,packet_ts,device_ts,device_id,payload
0.0,2024-01-01T00:00:00,0.0,test_device,"0"
0.5,2024-01-01T00:00:00,0.5,test_device,"1"
"""
    )


def test_discover_playback_collects_media(tmp_path: Path) -> None:
    dq_path = tmp_path / "sub" / "ses" / "dataqueue.csv"
    _write_minimal_dataqueue(dq_path)

    meso_path = tmp_path / "sub" / "ses" / "meso.ome.tiff"
    tifffile.imwrite(meso_path, np.zeros((2, 2, 2), dtype=np.uint16))
    pupil_path = tmp_path / "sub" / "ses" / "pupil.mp4"
    pupil_path.touch()
    treadmill_path = tmp_path / "sub" / "ses" / "treadmill.csv"
    treadmill_path.touch()

    context = discover_playback_context(tmp_path)

    assert context.dataqueue_path == dq_path
    assert context.meso_path == meso_path
    assert context.pupil_path == pupil_path
    assert context.treadmill_path == treadmill_path


def test_tiff_frame_source_maps_fraction(tmp_path: Path) -> None:
    tiff_path = tmp_path / "stack.ome.tiff"
    frames = np.stack([np.full((2, 2), idx, dtype=np.uint8) for idx in range(4)])
    tifffile.imwrite(tiff_path, frames)

    source = TiffFrameSource(tiff_path, duration_hint=2.0)

    assert source.frame_at_fraction(0.0)[0, 0] == 0
    assert source.frame_at_fraction(0.49)[0, 0] == 1
    assert source.frame_at_fraction(0.99)[0, 0] == 3


def test_tiff_frame_source_scaling_preserves_contrast() -> None:
    frame = np.array([[0, 65535], [32768, 16384]], dtype=np.uint16)

    scaled = _normalize_to_uint8(frame)

    assert scaled.dtype == np.uint8
    assert scaled.min() == 0
    assert scaled.max() == 255


def test_load_treadmill_trace_parses_iso_and_numeric(tmp_path: Path) -> None:
    trace = tmp_path / "treadmill.csv"
    trace.write_text(
        """timestamp,speed
2025-12-05T17:09:41.742987,0.0
2025-12-05T17:09:42.742987,12.5
"""
    )

    times, vals = load_treadmill_trace(trace)

    assert times.tolist() == [0.0, 1.0]
    assert vals.tolist() == [0.0, 12.5]
