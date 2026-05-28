"""Stress harness for :class:`mesofield.processors.FrameProcessor`.

Emits synthetic frames through ``DeviceSignals.frame`` at a target rate
with a configurable image size and (optional) artificial compute load,
then prints throughput, drop %, and EWMA / max compute time.

No real camera or Qt is required.  Example::

    python scripts/bench_frame_processor.py --size 512 --fps 60 --seconds 3
    python scripts/bench_frame_processor.py --size 2048 --fps 120 \\
        --extra-compute-ms 5 --seconds 5
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Optional

import numpy as np

from mesofield.processors import FrameMean
from mesofield.signals import DeviceSignals


class _FakeCamera:
    def __init__(self, fps: float) -> None:
        self.signals = DeviceSignals()
        self.sampling_rate = fps


class _BenchMean(FrameMean):
    """FrameMean with an optional artificial compute delay."""

    def __init__(self, *args: Any, extra_compute_ms: float = 0.0, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self._extra_s = extra_compute_ms / 1000.0

    def compute(self, img: Any, idx: Any, ts: Any) -> Optional[float]:
        v = super().compute(img, idx, ts)
        if self._extra_s > 0:
            time.sleep(self._extra_s)
        return v


def run(size: int, fps: float, seconds: float, extra_compute_ms: float) -> dict:
    cam = _FakeCamera(fps=fps)
    proc = _BenchMean(name="bench", camera=cam, extra_compute_ms=extra_compute_ms)
    proc.attach(cam)

    period = 1.0 / fps if fps > 0 else 0.0
    n_target = int(seconds * fps)
    img_buf = np.random.default_rng(0).integers(
        0, 255, size=(size, size), dtype=np.uint16
    )

    emit_t0 = time.perf_counter()
    next_t = emit_t0
    for i in range(n_target):
        # Mutate a couple of pixels so the mean changes each frame.
        img_buf[i % size, 0] = (i * 13) & 0xFF
        cam.signals.frame.emit(img_buf, i, time.perf_counter() - emit_t0)
        if period:
            next_t += period
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
    emit_elapsed = time.perf_counter() - emit_t0

    # Let the worker drain the last frame.
    time.sleep(max(0.1, (proc.compute_ms_ewma * 3) / 1000.0))
    proc.detach()

    s = proc.status()
    s["emit_elapsed_s"] = round(emit_elapsed, 3)
    s["target_fps"] = fps
    s["actual_emit_fps"] = round(n_target / emit_elapsed, 1) if emit_elapsed else 0.0
    s["frame_size"] = f"{size}x{size}"
    s["extra_compute_ms"] = extra_compute_ms
    return s


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=512, help="square frame edge (px)")
    p.add_argument("--fps", type=float, default=60.0, help="target emission rate")
    p.add_argument("--seconds", type=float, default=3.0, help="run duration")
    p.add_argument(
        "--extra-compute-ms",
        type=float,
        default=0.0,
        help="extra artificial compute load per frame (ms)",
    )
    args = p.parse_args()

    s = run(args.size, args.fps, args.seconds, args.extra_compute_ms)
    width = max(len(k) for k in s)
    for k, v in s.items():
        print(f"  {k:<{width}} : {v}")


if __name__ == "__main__":
    main()
