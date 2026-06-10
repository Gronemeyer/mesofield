"""Probe OpenCV-accessible cameras.

Iterates through device indices and the available capture backends,
reporting which combinations successfully open a stream and what frame
properties they report. Saves a JPEG snapshot from each working
combination so you can visually identify which physical camera maps to
which (index, backend) pair.

Usage
-----
    python -m tests.detect_opencv_camera
or  python tests/detect_opencv_camera.py

Snapshots are written to ``tests/_camera_snapshots/`` by default; override
with ``--out``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Silence opencv/ffmpeg before import
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")

import cv2  # noqa: E402

# Backend name -> OpenCV CAP_* enum. Not every backend is built into every
# wheel; missing ones are silently skipped.
_BACKEND_NAMES = [
    "CAP_MSMF",      # Windows Media Foundation
    "CAP_DSHOW",     # DirectShow
    "CAP_ANY",       # let opencv decide
    "CAP_V4L2",      # Linux
    "CAP_AVFOUNDATION",  # macOS
]


def _backends() -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for name in _BACKEND_NAMES:
        val = getattr(cv2, name, None)
        if isinstance(val, int):
            found.append((name, val))
    return found


def probe(
    index: int,
    backend_name: str,
    backend_id: int,
    *,
    grab_frames: int = 5,
    snapshot_dir: Path | None = None,
) -> dict | None:
    cap = cv2.VideoCapture(index, backend_id)
    if not cap.isOpened():
        cap.release()
        return None

    info: dict = {
        "index": index,
        "backend": backend_name,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)),
        "frames_read": 0,
        "snapshot": None,
    }
    # Warm up + grab a few frames to confirm streaming works. Keep the
    # last successfully-decoded frame to write to disk as a snapshot.
    last_frame = None
    t0 = time.perf_counter()
    for _ in range(grab_frames):
        ok, frame = cap.read()
        if ok and frame is not None:
            info["frames_read"] += 1
            info["last_shape"] = tuple(frame.shape)
            last_frame = frame
    info["read_time_s"] = round(time.perf_counter() - t0, 3)

    if last_frame is not None and snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snap_path = snapshot_dir / f"index{index}_{backend_name}.jpg"
        # Annotate the snapshot with the (index, backend) pair so it's
        # obvious which file belongs to which device.
        annotated = last_frame.copy()
        label = f"index={index} backend={backend_name}"
        cv2.putText(
            annotated, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            annotated, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
        )
        if cv2.imwrite(str(snap_path), annotated):
            info["snapshot"] = str(snap_path)
        else:
            info["snapshot"] = f"<imwrite failed: {snap_path}>"

    cap.release()
    return info


def main(max_index: int = 5, snapshot_dir: Path | None = None) -> int:
    backends = _backends()
    if not backends:
        print("No supported OpenCV capture backends found.")
        return 1

    print(f"OpenCV version: {cv2.__version__}")
    print(f"Probing indices 0..{max_index - 1} across backends: "
          f"{[b[0] for b in backends]}")
    if snapshot_dir is not None:
        print(f"Snapshots -> {snapshot_dir}")
    print()

    hits: list[dict] = []
    for idx in range(max_index):
        for name, bid in backends:
            try:
                result = probe(idx, name, bid, snapshot_dir=snapshot_dir)
            except Exception as exc:  # pragma: no cover - hardware/driver errors
                print(f"  index={idx} backend={name}: ERROR {exc}")
                continue
            if result is None:
                continue
            if result["frames_read"] == 0:
                # Opened but no frames -> probably a phantom/disconnected device
                print(f"  index={idx} backend={name}: opened but read 0 frames")
                continue
            hits.append(result)
            snap = result.get("snapshot") or "(no snapshot)"
            print(
                f"  index={idx:>2} backend={name:<16} "
                f"{result['width']}x{result['height']} @ {result['fps']:.1f} fps "
                f"frames={result['frames_read']} shape={result.get('last_shape')} "
                f"snap={snap}"
            )

    print()
    if not hits:
        print("No working camera found via OpenCV.")
        return 2
    print(f"Detected {len(hits)} working (index, backend) combination(s).")
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-index", type=int, default=5,
        help="Highest device index to probe (exclusive). Default: 5",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "_camera_snapshots",
        help="Directory to write per-camera JPEG snapshots into. "
             "Pass an empty string to disable snapshotting.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    snapshot_dir: Path | None = args.out if str(args.out) else None
    sys.exit(main(max_index=args.max_index, snapshot_dir=snapshot_dir))
