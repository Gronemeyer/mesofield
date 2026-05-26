"""Wheel encoder behavioral data source with dda synchronized timeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..._utils._logger import get_logger

from mesofield.datakit.sources.register import LoadContext, TimeseriesSource
from mesofield.datakit.timeline import DataqueueIndex

logger = get_logger(__name__)


@dataclass(frozen=True)
class _WheelSummary:
    """Aggregate metrics extracted from a wheel run."""

    duration_s: float
    total_distance_mm: float
    total_click_delta: int
    start_ts: Optional[str]
    stop_ts: Optional[str]


class WheelEncoder(TimeseriesSource):
    """Load wheel encoder streams recorded alongside nidaq pulses.

    The raw CSV emitted by the behavioral rig contains incremental click counts,
    elapsed time in seconds, and instantaneous speed estimates. The loader
    converts this information into a strictly increasing timeline anchored to
    acquisition start, computes cumulative distance, and exposes rich metadata
    for downstream alignment against the nidaq-driven master clock.
    """

    tag = "wheel"
    patterns = ("**/*_wheel.csv",)
    camera_tag = None  # Not bound to camera
    requires = ("dataqueue",)

    required_columns = ("Clicks", "Time", "Speed")
    time_column = "Time"
    click_column = "Clicks"
    speed_column = "Speed"
    cumulative_column = "click_delta"
    anchor_filter_pattern = "encoder"
    queue_elapsed_column = "queue_elapsed"
    dataqueue_payload_column = "payload"
    alignment_min_points = 2
    alignment_time_basis_dataqueue = "dataqueue"
    alignment_time_basis_wheel = "wheel_clock"
    alignment_poly_degree = 1
    distance_integration_method = "trapezoid"
    absolute_device_id = "encoder"
    alignment_origin_device_id = "ThorCam"
    def build_timeseries(
        self,
        path: Path,
        *,
        context: LoadContext | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame, dict]:
        context = self._require_context(context)
        raw = pd.read_csv(path)

        if not set(self.required_columns).issubset(raw.columns):
            raise ValueError(
                f"Wheel file is missing required columns {self.required_columns}: {path}"
            )

        df = self._prepare_frame(raw)
        df_aligned, alignment_meta = self._align_to_dataqueue(df, context)
        summary = self._summarize(df_aligned, raw)
        df_aligned = df_aligned.rename(columns={"time_s": "time_elapsed_s", "time_raw_s": "time_reference_s"})

        meta = {
            "source_file": str(path),
            "n_samples": int(len(df_aligned)),
            "start_time": summary.start_ts,
            "stop_time": summary.stop_ts,
            "duration_s": summary.duration_s,
            "total_distance_mm": summary.total_distance_mm,
            "total_click_delta": summary.total_click_delta,
            "source_method": "wheel_csv_v2",
            **alignment_meta,
        }

        return df_aligned["time_elapsed_s"].to_numpy(dtype=np.float64), df_aligned, meta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Type cast and derive the canonical wheel dataframe."""

        frame = pd.DataFrame()
        frame["time_s"] = pd.to_numeric(raw[self.time_column], errors="coerce")
        frame["click_delta"] = pd.to_numeric(raw[self.click_column], errors="coerce")
        frame["speed_mm"] = pd.to_numeric(raw[self.speed_column], errors="coerce")

        frame = frame.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)

        if frame.empty:
            frame = pd.DataFrame(
            {
                "time_s": [0.0, 0.0],
                "click_delta": [0, 0],
                "speed_mm": [0.0, 0.0],
            })

        # Ensure speed/clicks missing entries become zeros
        frame["click_delta"] = frame["click_delta"].fillna(0)
        frame["speed_mm"] = frame["speed_mm"].fillna(0.0)

        # Re-zero the timeline relative to the first available sample
        frame["time_raw_s"] = frame["time_s"].to_numpy(dtype=np.float64)
        t = frame["time_raw_s"].copy()
        t0 = float(t[0])
        t -= t0
        frame["time_s"] = t

        # Derive cumulative click position for quick sanity checks
        frame["click_position"] = np.cumsum(frame["click_delta"].to_numpy(dtype=np.int64))

        # Integrate speed to distance using trapezoidal rule
        speed = frame["speed_mm"].to_numpy(dtype=np.float64)
        distance = np.zeros_like(speed)
        if speed.size > 1:
            dt = np.diff(t)
            trapezoids = 0.5 * (speed[:-1] + speed[1:]) * dt
            distance[1:] = np.cumsum(trapezoids)
        frame["distance_mm"] = distance

        return frame[["time_s", "time_raw_s", "click_delta", "click_position", "speed_mm", "distance_mm"]]

    def _summarize(self, frame: pd.DataFrame, raw: pd.DataFrame) -> _WheelSummary:
        """Build aggregate metadata for diagnostics and alignment hints."""

        duration = float(frame["time_s"].iloc[-1]) if len(frame) else 0.0
        total_distance = float(frame["distance_mm"].iloc[-1]) if len(frame) else 0.0
        total_clicks = int(frame["click_delta"].sum())

        start_ts = self._extract_timestamp(raw.get("Started"))
        stop_ts = self._extract_timestamp(raw.get("Stopped"))

        return _WheelSummary(
            duration_s=duration,
            total_distance_mm=total_distance,
            total_click_delta=total_clicks,
            start_ts=start_ts,
            stop_ts=stop_ts,
        )

    # ------------------------------------------------------------------
    # Alignment helpers
    # ------------------------------------------------------------------
    def _align_to_dataqueue(self, frame: pd.DataFrame, context: LoadContext) -> tuple[pd.DataFrame, dict]:
        """Map wheel timestamps onto the nidaq master clock via dataqueue anchors."""
        dq_path = context.path_for("dataqueue")
        dataqueue_file = str(dq_path) if dq_path is not None else None
        timeline = None

        if context.dataqueue_frame is not None:
            encoder_times = self._extract_times(context.dataqueue_frame, self.anchor_filter_pattern)
            origin_times = self._extract_times(context.dataqueue_frame, self.alignment_origin_device_id)
        else:
            if dq_path is None:
                raise FileNotFoundError("WheelEncoder: dataqueue path not available")
            timeline = DataqueueIndex.from_path(dq_path)
            encoder_slice = timeline.slice(
                lambda ids: ids.str.contains(self.anchor_filter_pattern, case=False, na=False, regex=False)
            )
            encoder_times = encoder_slice.queue_elapsed().to_numpy(dtype=np.float64)
            origin_slice = timeline.slice(
                lambda ids: ids.str.contains(self.alignment_origin_device_id, case=False, na=False, regex=False)
            )
            origin_times = origin_slice.queue_elapsed().to_numpy(dtype=np.float64)
            dataqueue_file = str(timeline.source_path)
        if encoder_times.size < 2:
            logger.warning(
                "WheelEncoder: insufficient encoder anchors (%d). "
                "Returning unaligned wheel timeline.",
                encoder_times.size,
            )
            return frame.copy(), {
                "time_basis": self.alignment_time_basis_wheel,
                "dataqueue_file": dataqueue_file,
                "dataqueue_alignment": "insufficient_anchors",
                "dataqueue_anchors": int(encoder_times.size),
                "alignment_device": self.anchor_filter_pattern,
            }
        if origin_times.size == 0:
            logger.warning(
                "WheelEncoder: %s anchors missing for alignment. "
                "Returning unaligned wheel timeline.",
                self.alignment_origin_device_id,
            )
            return frame.copy(), {
                "time_basis": self.alignment_time_basis_wheel,
                "dataqueue_file": dataqueue_file,
                "dataqueue_alignment": "missing_origin_anchors",
                "dataqueue_anchors": int(encoder_times.size),
                "alignment_device": self.anchor_filter_pattern,
                "alignment_origin_device": self.alignment_origin_device_id,
            }

        n_samples = len(frame)
        n_anchors = int(encoder_times.size)
        if n_anchors == n_samples:
            aligned_times = encoder_times
            method = "direct"
        else:
            anchor_index = np.arange(n_anchors, dtype=np.float64)
            frame_index = np.linspace(0.0, float(n_anchors - 1), num=n_samples, dtype=np.float64)
            aligned_times = np.interp(frame_index, anchor_index, encoder_times)
            method = "resampled"

        origin = float(origin_times[0])
        aligned = frame.copy()
        aligned["time_s"] = aligned_times - origin

        encoder_start = float(encoder_times[0])
        origin_offset = encoder_start - origin

        if timeline is not None:
            elapsed_abs = timeline.absolute_for_device(self.absolute_device_id)
        else:
            elapsed_abs = None
        if elapsed_abs:
            elapsed, absolute = elapsed_abs
            elapsed = elapsed + origin_offset
            aligned_t = aligned["time_s"].to_numpy(dtype=np.float64)
            aligned["time_absolute"] = np.interp(aligned_t, elapsed, np.arange(len(absolute))).astype(int)
            aligned["time_absolute"] = aligned["time_absolute"].map(
                lambda i: absolute[i] if 0 <= i < len(absolute) else None
            )

        return aligned, {
            "time_basis": self.alignment_time_basis_dataqueue,
            "dataqueue_file": dataqueue_file,
            "dataqueue_alignment": method,
            "dataqueue_anchors": n_anchors,
            "alignment_device": self.anchor_filter_pattern,
            "alignment_origin_device": self.alignment_origin_device_id,
            "alignment_origin_queue_elapsed": origin,
            "alignment_origin_offset": float(origin_offset),
        }

    def _extract_times(self, frame: pd.DataFrame, pattern: str) -> np.ndarray:
        if self.queue_elapsed_column not in frame.columns or "device_id" not in frame.columns:
            return np.array([], dtype=np.float64)
        device_series = frame["device_id"].astype(str)
        mask = device_series.str.contains(pattern, case=False, na=False, regex=False)
        times = pd.to_numeric(frame.loc[mask, self.queue_elapsed_column], errors="coerce").dropna()
        return times.to_numpy(dtype=np.float64)

    @staticmethod
    def _extract_timestamp(series: Optional[pd.Series]) -> Optional[str]:
        """Return an ISO8601 string if the provided column encodes a timestamp."""

        if series is None:
            return None

        first = series.dropna().astype(str).head(1)
        if first.empty:
            return None

        try:
            ts = pd.to_datetime(first.iloc[0], utc=False, errors="raise")
        except (TypeError, ValueError):
            return None

        return ts.isoformat()