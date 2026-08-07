"""Headless acquisition overhead benchmark for a rig + experiment config.

This script runs a normal Mesofield :class:`Procedure` without the GUI and logs
camera-side pressure over time so you can separate GUI/rendering effects from
camera/writer throughput limits.

What it records (per camera, sampled periodically):
- MMCore remaining image count (circular-buffer backlog)
- MMCore buffer total/free capacity (frames) -- how full the ring actually is
- MMCore overflow flag
- write throughput estimate (MB/s) derived from the emitted frame count, NOT
  from on-disk file size (the OME-TIFF is pre-allocated to full size on frame 0,
  so file size never grows during acquisition and cannot measure throughput)
- process RSS and system-available RAM (to see slow RAM growth objectively)
- allocated output size on disk (kept for reference only; not a throughput proxy)

Outputs:
- telemetry.csv  : time-series samples for each camera
- summary.csv    : one row per camera with peak backlog, peak RSS and write rate

Use --null-writer to run with a disk-free NullWriter (sets MESOFIELD_NULL_WRITER):
comparing that run against a normal run isolates how much backlog/RAM the write
path itself contributes versus the engine drain / oversized circular buffer.

Example:
    python scripts/bench_camera_overhead.py \
      --rig "C:/Users/SIPE_LAB/AppData/Local/mesofield/rigs/widefield-pupil-treadmill.yaml" \
      --experiment "E:/jgronemeyer/260703_Sox9-widefield/experiment.json" \
      --duration 120 \
      --outdir bench_out/camera_overhead
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TelemetryRow:
    t_s: float
    camera_id: str
    sequence_running: bool
    remaining_images: int
    buffer_total_capacity: int
    buffer_free_capacity: int
    overflow: bool
    frame_count: int
    write_mbps: float
    process_rss_mb: float
    system_available_mb: float
    alloc_bytes_on_disk: int


try:  # optional; degrades gracefully if psutil is not installed
    import psutil  # type: ignore

    _PROC = psutil.Process(os.getpid())
except Exception:  # pragma: no cover - psutil is optional
    psutil = None  # type: ignore
    _PROC = None


def _rss_and_available_mb() -> tuple[float, float]:
    """Return (process RSS MB, system available RAM MB); (0, 0) without psutil."""
    if psutil is None or _PROC is None:
        return 0.0, 0.0
    try:
        rss = _PROC.memory_info().rss / 1e6
        avail = psutil.virtual_memory().available / 1e6
        return rss, avail
    except Exception:
        return 0.0, 0.0


def _bytes_for_output(path: Optional[str]) -> int:
    if not path:
        return 0
    total = 0
    for p in glob.glob(path + "*"):
        if os.path.isfile(p):
            total += os.path.getsize(p)
    return total


def _bytes_per_frame(core: Any) -> int:
    """Bytes for one full frame from MMCore image geometry (0 if unavailable)."""
    if core is None:
        return 0
    try:
        w = int(core.getImageWidth())
        h = int(core.getImageHeight())
        depth = int(core.getImageBitDepth())
        bpp = max(1, (depth + 7) // 8)
        return max(0, w * h * bpp)
    except Exception:
        return 0


def run_benchmark(
    rig_path: str,
    experiment_path: str,
    *,
    outdir: str,
    duration_s: Optional[float],
    sample_period_s: float,
    timeout_s: Optional[float],
) -> tuple[List[TelemetryRow], Dict[str, Any]]:
    from mesofield.base import Procedure

    proc = Procedure(config=experiment_path, hardware=rig_path)
    proc.config.experiment_dir = outdir
    proc.data_dir = proc.config.data_dir
    if duration_s is not None:
        proc.config.set("duration", int(round(duration_s)))

    cameras = list(getattr(proc.hardware, "cameras", ()) or ())
    frame_counts: Dict[str, int] = {getattr(cam, "id", "camera"): 0 for cam in cameras}

    # Use synchronous device signals as a cheap frame counter tap.
    for cam in cameras:
        cam_id = getattr(cam, "id", "camera")
        sig = getattr(getattr(cam, "signals", None), "data", None)
        if sig is not None and hasattr(sig, "connect"):
            sig.connect(lambda _payload, _ts=None, _id=cam_id: frame_counts.__setitem__(_id, frame_counts[_id] + 1))

    stop_evt = threading.Event()
    rows: List[TelemetryRow] = []
    lock = threading.Lock()

    # Track previous frame count + timestamp to estimate write throughput from
    # frames actually emitted (pre-allocated OME-TIFF file size cannot).
    last_frames: Dict[str, int] = {getattr(cam, "id", "camera"): 0 for cam in cameras}
    last_t: Dict[str, float] = {getattr(cam, "id", "camera"): 0.0 for cam in cameras}
    bytes_per_frame: Dict[str, int] = {getattr(cam, "id", "camera"): 0 for cam in cameras}

    t0 = time.perf_counter()

    def monitor() -> None:
        while not stop_evt.is_set():
            now = time.perf_counter()
            t_rel = now - t0
            rss_mb, avail_mb = _rss_and_available_mb()
            for cam in cameras:
                cam_id = getattr(cam, "id", "camera")
                core = getattr(cam, "core", None)
                output_path = getattr(cam, "output_path", None)

                remaining = 0
                total_cap = 0
                free_cap = 0
                running = False
                overflow = False
                if core is not None:
                    try:
                        remaining = int(core.getRemainingImageCount())
                    except Exception:
                        remaining = 0
                    try:
                        total_cap = int(core.getBufferTotalCapacity())
                    except Exception:
                        total_cap = 0
                    try:
                        free_cap = int(core.getBufferFreeCapacity())
                    except Exception:
                        free_cap = 0
                    try:
                        running = bool(core.isSequenceRunning())
                    except Exception:
                        running = False
                    try:
                        overflow = bool(core.isBufferOverflowed())
                    except Exception:
                        overflow = False

                # Resolve bytes/frame once the camera geometry is available.
                if not bytes_per_frame.get(cam_id):
                    bytes_per_frame[cam_id] = _bytes_per_frame(core)

                frames_now = frame_counts.get(cam_id, 0)
                prev_frames = last_frames.get(cam_id, 0)
                prev_t = last_t.get(cam_id, now)
                dt = max(1e-9, now - prev_t)
                dframes = max(0, frames_now - prev_frames)
                write_mbps = (dframes * bytes_per_frame.get(cam_id, 0)) / 1e6 / dt
                last_frames[cam_id] = frames_now
                last_t[cam_id] = now

                row = TelemetryRow(
                    t_s=round(t_rel, 3),
                    camera_id=cam_id,
                    sequence_running=running,
                    remaining_images=remaining,
                    buffer_total_capacity=total_cap,
                    buffer_free_capacity=free_cap,
                    overflow=overflow,
                    frame_count=frames_now,
                    write_mbps=round(write_mbps, 3),
                    process_rss_mb=round(rss_mb, 1),
                    system_available_mb=round(avail_mb, 1),
                    alloc_bytes_on_disk=_bytes_for_output(output_path),
                )
                with lock:
                    rows.append(row)

            stop_evt.wait(sample_period_s)

    mon = threading.Thread(target=monitor, name="bench-camera-overhead-monitor", daemon=True)
    mon.start()
    try:
        timeout = timeout_s
        if timeout is None and duration_s is not None:
            timeout = max(duration_s * 3.0, duration_s + 30.0)
        proc.run_until_finished(timeout=timeout)
    finally:
        stop_evt.set()
        mon.join(timeout=2.0)
        try:
            proc.config.hardware.deinitialize()
        except Exception:
            pass

    summary: Dict[str, Any] = {}
    by_cam: Dict[str, List[TelemetryRow]] = {}
    for row in rows:
        by_cam.setdefault(row.camera_id, []).append(row)
    for cam_id, samples in by_cam.items():
        peak_remaining = max(s.remaining_images for s in samples) if samples else 0
        overflow_seen = any(s.overflow for s in samples)
        # Average only over samples where frames were actually being written, so
        # idle head/tail padding doesn't dilute the steady-state throughput.
        active = [s for s in samples if s.write_mbps > 0]
        avg_write = (
            sum(s.write_mbps for s in active) / len(active) if active else 0.0
        )
        peak_rss = max((s.process_rss_mb for s in samples), default=0.0)
        min_avail = min((s.system_available_mb for s in samples), default=0.0)
        peak_buffer_used = max(
            (s.buffer_total_capacity - s.buffer_free_capacity for s in samples),
            default=0,
        )
        final_frames = samples[-1].frame_count if samples else 0
        final_bytes = samples[-1].alloc_bytes_on_disk if samples else 0
        summary[cam_id] = {
            "peak_remaining_images": peak_remaining,
            "peak_buffer_used_frames": peak_buffer_used,
            "overflow_seen": overflow_seen,
            "avg_write_mbps": round(avg_write, 3),
            "peak_rss_mb": round(peak_rss, 1),
            "min_system_available_mb": round(min_avail, 1),
            "final_frame_count": final_frames,
            "final_alloc_bytes_on_disk": final_bytes,
        }

    return rows, summary


def _write_outputs(outdir: str, rows: List[TelemetryRow], summary: Dict[str, Any]) -> None:
    Path(outdir).mkdir(parents=True, exist_ok=True)

    telemetry_csv = os.path.join(outdir, "telemetry.csv")
    with open(telemetry_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "t_s",
            "camera_id",
            "sequence_running",
            "remaining_images",
            "buffer_total_capacity",
            "buffer_free_capacity",
            "overflow",
            "frame_count",
            "write_mbps",
            "process_rss_mb",
            "system_available_mb",
            "alloc_bytes_on_disk",
        ])
        for r in rows:
            writer.writerow([
                r.t_s,
                r.camera_id,
                int(r.sequence_running),
                r.remaining_images,
                r.buffer_total_capacity,
                r.buffer_free_capacity,
                int(r.overflow),
                r.frame_count,
                r.write_mbps,
                r.process_rss_mb,
                r.system_available_mb,
                r.alloc_bytes_on_disk,
            ])

    summary_csv = os.path.join(outdir, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "camera_id",
            "peak_remaining_images",
            "peak_buffer_used_frames",
            "overflow_seen",
            "avg_write_mbps",
            "peak_rss_mb",
            "min_system_available_mb",
            "final_frame_count",
            "final_alloc_bytes_on_disk",
        ])
        for cam_id, s in summary.items():
            writer.writerow([
                cam_id,
                s["peak_remaining_images"],
                s["peak_buffer_used_frames"],
                int(bool(s["overflow_seen"])),
                s["avg_write_mbps"],
                s["peak_rss_mb"],
                s["min_system_available_mb"],
                s["final_frame_count"],
                s["final_alloc_bytes_on_disk"],
            ])

    print(f"Wrote telemetry: {telemetry_csv}")
    print(f"Wrote summary:   {summary_csv}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig", required=True, help="Path to hardware YAML.")
    parser.add_argument("--experiment", required=True, help="Path to experiment JSON.")
    parser.add_argument(
        "--outdir",
        default=os.path.join("bench_out", time.strftime("%Y%m%d_%H%M%S")),
        help="Output directory for benchmark CSV files.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Override experiment duration (seconds). Uses JSON value when omitted.",
    )
    parser.add_argument(
        "--sample-period",
        type=float,
        default=0.2,
        help="Telemetry sampling period in seconds.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Hard timeout for run_until_finished in seconds.",
    )
    parser.add_argument(
        "--null-writer",
        action="store_true",
        help=(
            "Run with a disk-free NullWriter (sets MESOFIELD_NULL_WRITER=1). "
            "Compare against a normal run to isolate the write path's cost."
        ),
    )
    args = parser.parse_args(argv)

    if not os.path.isfile(args.rig):
        raise FileNotFoundError(f"Rig file not found: {args.rig}")
    if not os.path.isfile(args.experiment):
        raise FileNotFoundError(f"Experiment file not found: {args.experiment}")

    if args.null_writer:
        os.environ["MESOFIELD_NULL_WRITER"] = "1"

    print("Starting headless camera-overhead benchmark")
    print(f"  rig        : {args.rig}")
    print(f"  experiment : {args.experiment}")
    print(f"  outdir     : {args.outdir}")
    if args.duration is not None:
        print(f"  duration   : {args.duration}s (override)")
    print(f"  sample     : {args.sample_period}s")
    print(f"  writer     : {'NULL (disk-free)' if args.null_writer else 'real (OME-TIFF)'}")
    if psutil is None:
        print("  note       : psutil not installed -> RSS/available-RAM columns will be 0")

    rows, summary = run_benchmark(
        args.rig,
        args.experiment,
        outdir=args.outdir,
        duration_s=args.duration,
        sample_period_s=args.sample_period,
        timeout_s=args.timeout,
    )
    _write_outputs(args.outdir, rows, summary)

    print("Summary:")
    for cam_id, s in summary.items():
        print(
            f"  {cam_id}: peak_remaining={s['peak_remaining_images']} "
            f"peak_buf_used={s['peak_buffer_used_frames']} "
            f"overflow={s['overflow_seen']} "
            f"avg_write={s['avg_write_mbps']} MB/s "
            f"peak_rss={s['peak_rss_mb']} MB "
            f"min_avail={s['min_system_available_mb']} MB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
