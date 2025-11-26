"""Quick helper to visualize Mesofield *_dataqueue.json logs.

Usage
-----
python -m mesofield.examples.plot_queue_log path/to/log.json [--device DEVICE_ID]

The script renders two plots:
1. A scatter plot showing when packets arrived per device.
2. A numeric payload trace when values can be inferred.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def _load_samples(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("samples", [])


def _extract_numeric(payload: Any) -> float | None:
    if isinstance(payload, (int, float)):
        return float(payload)
    if isinstance(payload, dict):
        for key in ("value", "abs_time", "rel_time", "numeric", "reading"):
            val = payload.get(key)
            if isinstance(val, (int, float)):
                return float(val)
    return None


def _iso_to_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _filter_samples(samples: Iterable[dict[str, Any]], device_id: str | None) -> list[dict[str, Any]]:
    if not device_id:
        return list(samples)
    return [sample for sample in samples if sample.get("device_id") == device_id]


def plot_queue(log_path: Path, device_id: str | None = None) -> None:
    samples = _filter_samples(_load_samples(log_path), device_id)
    if not samples:
        raise SystemExit("No samples available for plotting. Did you filter too aggressively?")

    queue_start: float | None = None
    times: list[float] = []
    for sample in samples:
        since_start = sample.get("since_start")
        if isinstance(since_start, (int, float)):
            times.append(float(since_start))
            continue
        queue_elapsed = sample.get("queue_elapsed")
        if isinstance(queue_elapsed, (int, float)):
            q_val = float(queue_elapsed)
            if queue_start is None:
                queue_start = q_val
            times.append(q_val - queue_start)
        else:
            times.append(float(len(times)))
    devices = sorted({sample.get("device_id", "") for sample in samples})
    device_to_idx = {dev_id: idx for idx, dev_id in enumerate(devices)}
    device_positions = [device_to_idx.get(sample.get("device_id", ""), 0) for sample in samples]

    numeric_payloads = [
        (time, _extract_numeric(sample.get("payload")))
        for time, sample in zip(times, samples)
    ]
    numeric_payloads = [(t, v) if v is not None else None for t, v in numeric_payloads]
    numeric_payloads = [nv for nv in numeric_payloads if nv is not None]

    fig, axes = plt.subplots(2 if numeric_payloads else 1, 1, figsize=(10, 6), sharex=bool(numeric_payloads))
    if isinstance(axes, Axes):
        axes_list = [axes]
    else:
        axes_list = list(axes) if isinstance(axes, Iterable) else [axes]

    axes_list[0].scatter(times, device_positions, c=device_positions, cmap="tab20", s=12)
    axes_list[0].set_yticks(range(len(devices)))
    axes_list[0].set_yticklabels(devices)
    axes_list[0].set_ylabel("Device")
    axes_list[0].set_title(f"Queue timeline for {log_path.name}")

    packet_range = (
        _iso_to_datetime(samples[0].get("packet_ts")),
        _iso_to_datetime(samples[-1].get("packet_ts")),
    )
    if packet_range[0] and packet_range[1]:
        axes_list[0].text(
            0.01,
            0.95,
            f"Packets: {packet_range[0]} – {packet_range[1]}",
            transform=axes_list[0].transAxes,
            fontsize=9,
            verticalalignment="top",
        )

    if numeric_payloads:
        payload_ax = axes_list[1]
        payload_ax.plot(
            [entry[0] for entry in numeric_payloads],
            [entry[1] for entry in numeric_payloads],
            marker="o",
            linestyle="-",
            linewidth=1,
            markersize=3,
        )
        payload_ax.set_ylabel("Payload value")
        payload_ax.set_xlabel("Seconds since start")
        payload_ax.set_title("Numeric payload trace")
    else:
        axes_list[0].set_xlabel("Seconds since start")

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Mesofield queue logs")
    parser.add_argument("log", type=Path, help="Path to *_dataqueue.json log")
    parser.add_argument(
        "--device",
        "-d",
        dest="device_id",
        help="Optional device id to visualize",
    )
    args = parser.parse_args()
    plot_queue(args.log, args.device_id)


if __name__ == "__main__":
    main()
