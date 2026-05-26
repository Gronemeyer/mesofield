"""Session timing markers data source.

Transforms a CSV of per-device ``started``/``stopped`` timestamps into a static
stream that downstream tooling can inspect alongside other metadata sources.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mesofield.datakit.sources.register import IntervalSeriesSource, LoadContext


class SessionTimestampsSource(IntervalSeriesSource):
    """Load per-device start/stop timestamps as intervals."""

    tag = "timestamps"
    patterns = ("**/*_timestamps.csv",)
    camera_tag = None

    required_columns = ("device_id", "started", "stopped")

    def build_intervals(
        self,
        path: Path,
        *,
        context: LoadContext | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        frame = pd.read_csv(path)

        missing = [col for col in self.required_columns if col not in frame.columns]
        if missing:
            raise ValueError(f"Timestamps CSV missing required columns {missing}: {path}")

        normalized = frame.copy()
        for column in ("started", "stopped"):
            normalized[column] = pd.to_datetime(normalized[column], errors="coerce")

        if normalized["started"].notna().any():
            origin = normalized["started"].min()
        else:
            origin = pd.Timestamp(0)

        normalized["start_s"] = (normalized["started"] - origin).dt.total_seconds()
        normalized["stop_s"] = (normalized["stopped"] - origin).dt.total_seconds()

        meta = {
            "source_file": str(path),
            "devices": normalized["device_id"].dropna().astype(str).tolist(),
            "time_basis": "timestamps_relative",
        }

        return normalized, meta


__all__ = ["SessionTimestampsSource"]
