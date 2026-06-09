"""Stream a real OME-TIFF stack through :class:`FrequencyBandDetector`.

Lazy-loads frames via :func:`tifffile.memmap` (no full-stack load) and
emits them into a fake camera's ``signals.frame`` at a chosen pacing
rate, so the detector experiences the same per-frame timing it would
during a live acquisition.

Outputs per run:
  - CSV with one row per emitted (post-warmup) frame:
        frame_idx, t_s, mean_intensity, power, detected
  - Optional PNG with three stacked panels (mean intensity, power,
    detected) for a quick visual sanity check.

Example:

    python scripts/bench_freq_detector_tiff.py \
        --tiff /Volumes/Untitled/Projects/RO1_ETOH/sub-GS28/\
20251215_160036_sub-GS28_ses-01_task-spont_mesoscope.ome.tiff \
        --fps 50 --target-hz 4 --window-s 2 --warmup-s 4 \
        --max-frames 6000 --plot

A pacing rate of 0 disables the sleep loop, so the detector runs as
fast as the SSD + worker thread will allow -- useful for measuring
the *throughput ceiling* of the compute step.  Any non-zero ``--fps``
matches that rate (sleeps between emissions) so ``compute_ms_ewma``
reports realistic per-frame latency.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tifffile

from mesofield.processors import FrequencyBandDetector
from mesofield.signals import DeviceSignals


class _FakeCamera:
    def __init__(self, fps: float) -> None:
        self.signals = DeviceSignals()
        self.sampling_rate = fps


def _open_memmap(path: Path) -> np.ndarray:
    """Memory-map the TIFF and return a (T, H, W[, C]) view.

    Falls back to ``tifffile.imread`` if memmap is unavailable for the
    particular OME layout (e.g. tiled with compression).
    """
    try:
        arr = tifffile.memmap(str(path))[1:]
        if arr.ndim == 2:
            arr = arr[None, ...]  # single-frame file
        return arr
    except Exception:
        # imread will materialize the stack; only acceptable for small files.
        arr = tifffile.imread(str(path))
        if arr.ndim == 2:
            arr = arr[None, ...]
        return arr


def run(args: argparse.Namespace) -> dict:
    tiff_path = Path(args.tiff)
    if not tiff_path.is_file():
        raise FileNotFoundError(tiff_path)

    print(f"Opening {tiff_path}")
    t_open = time.perf_counter()
    stack = _open_memmap(tiff_path)
    open_elapsed = time.perf_counter() - t_open
    n_total, H, W = stack.shape[0], stack.shape[1], stack.shape[2]
    n_frames = min(n_total, args.max_frames) if args.max_frames else n_total
    print(f"  {n_total} frames, {H}x{W}, dtype={stack.dtype}, "
          f"opened in {open_elapsed * 1000:.1f} ms")
    print(f"  Streaming {n_frames} frames at {args.fps:.1f} fps "
          f"(target {args.target_hz} Hz)")

    cam = _FakeCamera(fps=args.fps)
    proc = FrequencyBandDetector(
        camera=cam,
        target_hz=args.target_hz,
        window_s=args.window_s,
        warmup_s=args.warmup_s,
        high_k=args.high_k,
        low_k=args.low_k,
    )

    rows: list[tuple[int, float, float, float, float]] = []

    def _on_payload(payload, ts):
        # We don't know the frame_idx from the payload alone; use the
        # processor's own counter — it strictly increments once per
        # post-warmup emission. We'll back-fill below using the emit
        # order.
        if isinstance(payload, dict):
            rows.append((
                len(rows),                       # placeholder; rewritten below
                float(ts),
                float("nan"),                    # mean filled in by emitter loop
                float(payload.get("power", 0.0)),
                float(payload.get("detected", 0.0)),
            ))

    proc.signals.data.connect(_on_payload)

    # ------------------------------------------------------------------
    # Emit frames at the target pacing rate.
    period = 1.0 / args.fps if args.fps > 0 else 0.0
    emit_t0 = time.perf_counter()
    next_t = emit_t0
    means: list[float] = []  # one entry per emitted frame, in emit order
    for i in range(n_frames):
        frame = stack[i]
        # Compute mean here too so we can write it alongside power in
        # the CSV; the detector recomputes its own (cheap, but symmetric).
        m = float(frame.mean())
        means.append(m)
        cam.signals.frame.emit(frame, i, time.perf_counter() - emit_t0)
        if period:
            next_t += period
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
    emit_elapsed = time.perf_counter() - emit_t0

    # Let the worker drain.
    time.sleep(max(0.3, 5 * proc.compute_ms_ewma / 1000.0))
    proc.detach()

    # ------------------------------------------------------------------
    # The processor swallows the warmup window — there are fewer rows
    # than means. Align them by tail: the *last* len(rows) means correspond
    # to the rows in order (Goertzel emits one row per frame after the
    # ring buffer fills).
    n_rows = len(rows)
    offset = len(means) - n_rows
    rebuilt: list[tuple[int, float, float, float, float]] = []
    for i, (_, ts, _, power, det) in enumerate(rows):
        frame_idx = offset + i
        m = means[frame_idx] if 0 <= frame_idx < len(means) else float("nan")
        rebuilt.append((frame_idx, ts, m, power, det))
    rows = rebuilt

    # ------------------------------------------------------------------
    # Write CSV.
    out_path = Path(args.output) if args.output else (
        tiff_path.with_suffix("")
            .with_name(tiff_path.stem + f".freq{args.target_hz:g}.csv")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "t_s", "mean_intensity", "power", "detected"])
        for row in rows:
            w.writerow(row)
    print(f"  Wrote {len(rows)} rows -> {out_path}")

    # ------------------------------------------------------------------
    # Optional plot.
    if args.plot and rows:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"  (skipping plot: matplotlib not available: {exc})")
        else:
            arr = np.array(rows, dtype=float)
            ts, m, power, det = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
            fig, (ax0, ax1, ax2) = plt.subplots(
                3, 1, figsize=(12, 6), sharex=True,
                gridspec_kw={"height_ratios": [2, 2, 1], "hspace": 0.1},
                layout="constrained",
            )
            ax0.plot(ts, m, lw=0.6, color="0.3")
            ax0.set_ylabel("Mean intensity")
            ax0.set_title(
                f"{tiff_path.name}  —  target {args.target_hz} Hz, "
                f"window {args.window_s}s, k={args.high_k}/{args.low_k}",
                fontsize=10, loc="left",
            )
            ax1.plot(ts, power, lw=0.6, color="#1D3557")
            ax1.axhline(proc._high_th, color="#E63946", ls="--", lw=0.8,
                        label=f"high_th={proc._high_th:.1f}")
            ax1.axhline(proc._low_th, color="#457B9D", ls="--", lw=0.8,
                        label=f"low_th={proc._low_th:.1f}")
            ax1.set_ylabel(f"Power @ {args.target_hz:g} Hz")
            ax1.legend(loc="upper right", fontsize=8)
            ax2.fill_between(ts, det, step="mid", color="#2A9D8F", alpha=0.6, lw=0)
            ax2.set_ylim(-0.1, 1.1); ax2.set_yticks([0, 1])
            ax2.set_ylabel("detected"); ax2.set_xlabel("t (s)")
            for ax in (ax0, ax1, ax2):
                ax.grid(True, alpha=0.3)
            png_path = out_path.with_suffix(".png")
            fig.savefig(png_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  Wrote plot -> {png_path}")

    # ------------------------------------------------------------------
    # Summary.
    status = proc.status()
    summary: dict[str, Any] = {
        "tiff": str(tiff_path),
        "frames_streamed": n_frames,
        "frames_post_warmup": len(rows),
        "emit_elapsed_s": round(emit_elapsed, 3),
        "actual_fps": round(n_frames / emit_elapsed, 1) if emit_elapsed else 0.0,
        "target_hz": args.target_hz,
        "window_s": args.window_s,
        "compute_ms_ewma": status["compute_ms_ewma"],
        "compute_ms_max": status["compute_ms_max"],
        "drop_ratio": status["drop_ratio"],
        "thresholds_high_low": tuple(round(t, 2) for t in proc.thresholds()),
    }
    if rows:
        det_arr = np.array([r[4] for r in rows])
        pow_arr = np.array([r[3] for r in rows])
        summary["detected_fraction"] = float(det_arr.mean())
        summary["power_mean"] = round(float(pow_arr.mean()), 2)
        summary["power_p50"] = round(float(np.median(pow_arr)), 2)
        summary["power_p95"] = round(float(np.percentile(pow_arr, 95)), 2)
        summary["power_p99"] = round(float(np.percentile(pow_arr, 99)), 2)
        summary["csv"] = str(out_path)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tiff", required=True, help="Path to .ome.tiff stack")
    p.add_argument("--fps", type=float, default=50.0,
                   help="Pacing rate at which frames are fed to the detector "
                        "(0 = free-run, no sleep)")
    p.add_argument("--target-hz", type=float, default=4.0)
    p.add_argument("--window-s", type=float, default=2.0)
    p.add_argument("--warmup-s", type=float, default=4.0)
    p.add_argument("--high-k", type=float, default=4.0)
    p.add_argument("--low-k", type=float, default=2.5)
    p.add_argument("--max-frames", type=int, default=0,
                   help="Cap on frames streamed (0 = whole stack)")
    p.add_argument("--output", default=None,
                   help="CSV output path (default: <stem>.freq<hz>.csv next to TIFF)")
    p.add_argument("--plot", action="store_true",
                   help="Also write a PNG with mean / power / detected panels")
    args = p.parse_args()

    s = run(args)
    width = max(len(k) for k in s)
    print()
    for k, v in s.items():
        print(f"  {k:<{width}} : {v}")


if __name__ == "__main__":
    main()
