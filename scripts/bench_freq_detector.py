"""Stress harness for :class:`FrequencyBandDetector`.

Emits synthetic frames whose mean intensity is a sinusoid at a chosen
``--signal-hz`` (plus white noise) into ``DeviceSignals.frame``.  Drives
a ``FrequencyBandDetector(target_hz=--target-hz)`` and reports per-channel
behavior plus the processor's compute-load counters.

Examples::

    # Matched: signal and target both at 4 Hz -> detected should latch.
    python scripts/bench_freq_detector.py --signal-hz 4 --target-hz 4 \\
        --fps 30 --seconds 6

    # Mismatched: signal at 10 Hz vs target 4 Hz -> detected stays 0.
    python scripts/bench_freq_detector.py --signal-hz 10 --target-hz 4 \\
        --fps 30 --seconds 6

    # Onset latency: silent for 3s then start oscillating at the target.
    python scripts/bench_freq_detector.py --signal-hz 4 --target-hz 4 \\
        --fps 30 --seconds 8 --onset-s 3
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from mesofield.processors import FrequencyBandDetector
from mesofield.signals import DeviceSignals


class _FakeCamera:
    def __init__(self, fps: float) -> None:
        self.signals = DeviceSignals()
        self.sampling_rate = fps


def run(args: argparse.Namespace) -> dict:
    cam = _FakeCamera(fps=args.fps)
    proc = FrequencyBandDetector(
        camera=cam,
        target_hz=args.target_hz,
        window_s=args.window_s,
        warmup_s=args.warmup_s,
        high_k=args.high_k,
        low_k=args.low_k,
    )

    # Collect emissions through ``signals.data`` (psygnal, fires same-
    # thread). The Qt cross-thread pyqtSignals would need a QApplication
    # to fire; we only care about correctness of the data path here.
    rows: list[tuple[float, float, float]] = []
    proc.signals.data.connect(
        lambda payload, ts: rows.append(
            (float(ts), float(payload["power"]), float(payload["detected"]))
        )
    )

    period = 1.0 / args.fps
    n_frames = int(args.seconds * args.fps)
    rng = np.random.default_rng(0)

    # Synthetic frames: 64x64 with a base mean that becomes sinusoidal
    # at signal_hz once t >= onset_s. We mutate just the [0,0] pixel
    # — Goertzel only sees the mean, so we adjust the base level.
    base = np.full((64, 64), 128, dtype=np.uint16)
    emit_t0 = time.perf_counter()
    next_t = emit_t0
    for i in range(n_frames):
        t = i * period
        noise = float(rng.standard_normal()) * args.noise
        if t >= args.onset_s:
            amp = args.amplitude
            s = amp * np.sin(2 * np.pi * args.signal_hz * t)
        else:
            s = 0.0
        # Cheap mean injection: add (s + noise) by setting one pixel.
        # mean(img) shifts by delta / N^2; we want a sinusoid in the
        # mean, so scale accordingly.
        delta = (s + noise) * base.size  # so mean shifts by exactly (s + noise)
        img = base.copy()
        # Clip to uint16 range to avoid overflow
        new_px = int(np.clip(base[0, 0] + delta, 0, 65535))
        img[0, 0] = new_px
        cam.signals.frame.emit(img, i, t)
        next_t += period
        sleep_for = next_t - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)

    # Give the worker a beat to drain.
    time.sleep(max(0.2, 3 * proc.compute_ms_ewma / 1000.0))
    proc.detach()

    # Summarize.
    if not rows:
        return {"error": "no rows emitted (warmup never completed?)"}
    arr = np.array(rows)
    ts_arr, power_arr, det_arr = arr[:, 0], arr[:, 1], arr[:, 2]
    n = len(rows)
    n_detected = int(det_arr.sum())
    # Quiet-period reference: average power in the first 1s of post-
    # warmup output (or first 20% if onset > 0).
    quiet_mask = ts_arr < max(args.onset_s, args.warmup_s + 0.2)
    active_mask = (ts_arr >= args.onset_s + args.warmup_s) & ~quiet_mask
    quiet_mean = float(power_arr[quiet_mask].mean()) if quiet_mask.any() else float("nan")
    active_mean = float(power_arr[active_mask].mean()) if active_mask.any() else float("nan")
    snr = active_mean / quiet_mean if quiet_mean > 0 else float("inf")

    # Onset latency: first frame index after onset_s where detected flips to 1.
    onset_latency_s = float("nan")
    after_onset = ts_arr >= args.onset_s + args.warmup_s
    cand = np.where(after_onset & (det_arr > 0.5))[0]
    if len(cand) > 0:
        onset_latency_s = float(ts_arr[cand[0]] - args.onset_s)

    status = proc.status()
    return {
        "n_frames_emitted": n_frames,
        "n_rows_output": n,
        "n_detected_frames": n_detected,
        "detected_fraction": n_detected / n if n else 0.0,
        "power_quiet_mean": round(quiet_mean, 2),
        "power_active_mean": round(active_mean, 2),
        "snr_active_over_quiet": round(snr, 2),
        "onset_latency_s": round(onset_latency_s, 3),
        "thresholds_high_low": tuple(round(t, 2) for t in proc.thresholds()),
        "compute_ms_ewma": status["compute_ms_ewma"],
        "compute_ms_max": status["compute_ms_max"],
        "drop_ratio": status["drop_ratio"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--signal-hz", type=float, default=4.0,
                   help="Hz of the sinusoid injected into the frame mean")
    p.add_argument("--target-hz", type=float, default=4.0,
                   help="Hz the detector is tuned to")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--window-s", type=float, default=2.0)
    p.add_argument("--warmup-s", type=float, default=4.0)
    p.add_argument("--high-k", type=float, default=4.0)
    p.add_argument("--low-k", type=float, default=2.5)
    p.add_argument("--amplitude", type=float, default=8.0,
                   help="sinusoid amplitude in mean-intensity units")
    p.add_argument("--noise", type=float, default=1.0,
                   help="white-noise stddev added to the mean each frame")
    p.add_argument("--onset-s", type=float, default=0.0,
                   help="seconds before the sinusoid starts (silent before)")
    args = p.parse_args()

    s = run(args)
    width = max(len(k) for k in s)
    for k, v in s.items():
        print(f"  {k:<{width}} : {v}")


if __name__ == "__main__":
    main()
