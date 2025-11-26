"""Schema helpers for Mesofield output artifacts.

This module centralizes the structure of JSON documents written by
:class:`mesofield.data.manager.DataSaver`. Moving the schema here keeps
`DataSaver` lean and gives downstream consumers a single import to rely on
when they need to read or validate queue logs.

Usage
-----
>>> from mesofield.schema.output import build_queue_log
>>> payload = build_queue_log(config, queue_rows)
>>> json.dump(payload, open(path, "w"), indent=2)

The helpers intentionally return simple ``dict`` objects so the resulting
structures stay human-readable and easy to serialize without extra
dependencies. The current queue schema is version ``2.0``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Iterable, Sequence, TYPE_CHECKING
QUEUE_SCHEMA_VERSION = "2.0"
QUEUE_SCHEMA_ID = f"mesofield.dataqueue/{QUEUE_SCHEMA_VERSION}"

if TYPE_CHECKING:  # pragma: no cover
    from mesofield.config import ExperimentConfig


@dataclass
class QueueLog:
    """Lightweight data container for a Mesofield queue log."""

    schema: str
    created: str
    recording: dict[str, Any]
    hardware: list[dict[str, Any]]
    samples: list[dict[str, Any]]
    summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_rows(
        cls,
        cfg: "ExperimentConfig",
        rows: Sequence[Sequence[Any]],
    ) -> "QueueLog":
        created = datetime.now().astimezone().isoformat()
        samples, summary = _format_queue_samples(rows)
        return cls(
            schema=QUEUE_SCHEMA_ID,
            created=created,
            recording=_recording_header(cfg, created),
            hardware=_hardware_summary(cfg),
            samples=samples,
            summary=summary,
        )


def build_queue_log(cfg: "ExperimentConfig", rows: Sequence[Sequence[Any]]) -> dict[str, Any]:
    """Build a JSON-serializable dictionary describing the queue log."""

    return QueueLog.from_rows(cfg, rows).as_dict()


# ---------------------------------------------------------------------------
# Internal helpers.


def _recording_header(cfg: "ExperimentConfig", created: str) -> dict[str, Any]:
    header = {
        "subject": cfg.subject,
        "session": cfg.session,
        "task": cfg.task,
        "experimenter": getattr(cfg, "experimenter", ""),
        "protocol": cfg.protocol,
        "duration_s": getattr(cfg, "sequence_duration", None),
        "start_on_trigger": getattr(cfg, "start_on_trigger", False),
        "notes": list(getattr(cfg, "notes", [])),
        "save_dir": getattr(cfg, "save_dir", ""),
        "config_file": getattr(cfg, "_config_file_path", ""),
        "hardware_config_file": cfg.get("hardware_config_file"),
    }
    return {key: _json_safe_value(value) for key, value in header.items()}


def _hardware_summary(cfg: "ExperimentConfig") -> list[dict[str, Any]]:
    devices = getattr(getattr(cfg, "hardware", None), "devices", {}) or {}
    summary: list[dict[str, Any]] = []
    for dev_id, device in devices.items():
        info = {
            "id": dev_id,
            "device_type": getattr(device, "device_type", ""),
            "class_name": device.__class__.__name__,
            "description": getattr(device, "description", ""),
            "sampling_rate": getattr(device, "sampling_rate", None),
            "output_path": getattr(device, "output_path", None),
        }
        summary.append({key: _json_safe_value(value) for key, value in info.items()})
    return summary


def _format_queue_samples(rows: Sequence[Sequence[Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    device_counts: Counter[str] = Counter()
    queue_values: list[float] = []
    packet_times: list[datetime] = []
    queue_start = _safe_float(rows[0][0]) if rows else None

    for idx, row in enumerate(rows):
        if len(row) < 5:
            continue
        queue_elapsed = _safe_float(row[0])
        packet_ts = _to_iso(row[1])
        if isinstance(row[1], datetime):
            packet_times.append(row[1])
        device_ts = _normalize_device_ts(row[2])
        device_id = str(row[3]) if row[3] is not None else ""
        payload = _json_safe_value(row[4])

        if queue_elapsed is not None:
            queue_values.append(queue_elapsed)
        device_counts[device_id] += 1

        samples.append(
            {
                "index": idx,
                "queue_elapsed": queue_elapsed,
                "since_start": (queue_elapsed - queue_start)
                if (queue_elapsed is not None and queue_start is not None)
                else None,
                "packet_ts": packet_ts,
                "device_ts": device_ts,
                "device_id": device_id,
                "payload": payload,
            }
        )

    summary = _queue_summary(queue_values, packet_times, device_counts)
    return samples, summary


def _queue_summary(
    queue_values: Sequence[float],
    packet_times: Sequence[datetime],
    device_counts: Counter[str],
) -> dict[str, Any]:
    queue_range = _range(queue_values)
    packet_range = _range(packet_times)
    duration = None
    if queue_range[0] is not None and queue_range[1] is not None:
        duration = queue_range[1] - queue_range[0]

    return {
        "total_samples": sum(device_counts.values()),
        "duration_seconds": duration,
        "devices": dict(device_counts),
        "queue_elapsed_range": queue_range,
        "packet_ts_range": [
            packet_range[0].isoformat() if packet_range[0] else None,
            packet_range[1].isoformat() if packet_range[1] else None,
        ],
    }


def _range(values: Sequence[Any]) -> tuple[Any | None, Any | None]:
    if not values:
        return (None, None)
    return (min(values), max(values))


def _to_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_device_ts(value: Any) -> Any:
    if value is None:
        return None
    numeric = _safe_float(value)
    if numeric is not None:
        return numeric
    if isinstance(value, datetime):
        return value.isoformat()
    return _json_safe_value(value)


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # pragma: no cover - best effort
            pass
    if hasattr(value, "item"):
        try:
            return _json_safe_value(value.item())
        except Exception:  # pragma: no cover - best effort
            pass
    return repr(value)


__all__ = ["QueueLog", "build_queue_log"]
