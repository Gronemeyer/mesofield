"""Minimal treadmill loader aligned to dataqueue window (MVP)."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from mesofield.datakit.config import settings
from mesofield.datakit.sources.register import SourceContext, TimeseriesSource


class TreadmillSource(TimeseriesSource):
    """Load treadmill samples aligned to the experiment window."""
    tag = "treadmill"
    patterns = ("**/*_treadmill.csv", "**/*_treadmill_data.csv")
    camera_tag = None

    timestamp_column = "timestamp"
    distance_columns = ("distance_mm", "distance")
    speed_columns = ("speed_mm", "speed_mm_s", "speed")
    output_distance_column = "distance_mm"
    output_speed_column = "speed_mm"

    dataqueue_queue_column = settings.timeline.queue_column
    dataqueue_device_id_column = "device_id"
    dataqueue_payload_column = "payload"

    dataqueue_payload_prefix = "EncoderData"
    timeline_master_patterns = ("dhyana", "mesoscope")
    _encoder_ts_re = re.compile(r"timestamp\s*=\s*(\d+)", re.IGNORECASE)
    def build_timeseries(
        self,
        path: Path,
        *,
        context: SourceContext | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame, dict]:
        context = self._require_context(context)
        df = pd.read_csv(path)
        dist = next((c for c in self.distance_columns if c in df.columns), None)
        speed = next((c for c in self.speed_columns if c in df.columns), None)
        if self.timestamp_column not in df.columns or dist is None or speed is None:
            raise ValueError(f"Expected timestamp/distance/speed columns in {path}")
        df = df.rename(columns={dist: self.output_distance_column, speed: self.output_speed_column})

        dq_path = context.path_for("dataqueue")
        aligned, meta = self.extract_treadmill_aligned(
            df,
            dq_path,
            window=context.experiment_window,
            dataqueue_frame=context.dataqueue_frame,
        )
        t = aligned["time_elapsed_s"].to_numpy(dtype=np.float64)
        meta = {
            "source_file": str(path),
            "dataqueue_file": str(dq_path) if dq_path is not None else None,
            "n_samples": len(aligned),
            **meta,
        }
        return t, aligned, meta

    # --- alignment --------------------------------

    def extract_treadmill_aligned(
        self,
        treadmill_df: pd.DataFrame,
        dq_path: Path | None,
        *,
        window: tuple[float, float] | None = None,
        dataqueue_frame: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        t0, t1 = window or self._window(dq_path, dataqueue_frame)
        duration = float(t1 - t0)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid camera window duration from dataqueue: start={t0}, end={t1}")

        enc = self._encoder_rows(dq_path, dataqueue_frame)
        a, b = self._fit(enc["encoder_ts"].to_numpy(), enc["queue_elapsed"].to_numpy())

        ts = pd.to_numeric(treadmill_df[self.timestamp_column], errors="coerce").to_numpy(dtype=np.float64)
        time_s = a * ts + b - t0

        mask = np.isfinite(time_s) & (time_s >= 0.0) & (time_s <= duration)
        if not np.any(mask):
            raise ValueError("No treadmill samples fall inside the Dhyana window")

        aligned = treadmill_df.loc[mask].copy()
        aligned["time_elapsed_s"] = time_s[mask]
        aligned.sort_values("time_elapsed_s", inplace=True)
        aligned.reset_index(drop=True, inplace=True)

        return aligned, {
            "source_method": "dataqueue_align_mvp",
            "experiment_window": {"start": float(t0), "end": float(t1)},
            "alignment": {"a": float(a), "b": float(b)},
        }

    # --- dataqueue utilities -----------------------------------------------------


    def _window(
        self,
        dq_path: Path | None,
        dataqueue_frame: pd.DataFrame | None,
    ) -> tuple[float, float]:
        dq = dataqueue_frame
        if dq is None:
            if dq_path is None:
                raise FileNotFoundError("TreadmillSource: dataqueue path not available")
            dq = pd.read_csv(
                dq_path,
                usecols=[self.dataqueue_queue_column, self.dataqueue_device_id_column],
                low_memory=False,
            )
        device = dq.get(self.dataqueue_device_id_column, pd.Series(dtype=str)).astype(str)

        mask = np.zeros(len(dq), dtype=bool)
        for pattern in self.timeline_master_patterns:
            mask |= device.str.contains(pattern, case=False, na=False, regex=False).to_numpy()

        rows = pd.to_numeric(dq.loc[mask, self.dataqueue_queue_column], errors="coerce").dropna()
        if len(rows) < 2:
            raise ValueError("Could not find >=2 master camera rows in dataqueue")
        return float(rows.iloc[0]), float(rows.iloc[-1])

    def _encoder_rows(
        self,
        dq_path: Path | None,
        dataqueue_frame: pd.DataFrame | None,
    ) -> pd.DataFrame:
        dq = dataqueue_frame
        if dq is None:
            if dq_path is None:
                raise FileNotFoundError("TreadmillSource: dataqueue path not available")
            dq = pd.read_csv(
                dq_path,
                usecols=[self.dataqueue_queue_column, self.dataqueue_payload_column],
                low_memory=False,
            )
        payload = dq[self.dataqueue_payload_column].astype(str)

        # strict parse only from EncoderData payloads with timestamp=...
        mask = payload.str.contains(self.dataqueue_payload_prefix, na=False)
        if not np.any(mask):
            raise ValueError("No EncoderData payloads found in dataqueue")

        enc = dq.loc[mask, [self.dataqueue_queue_column, self.dataqueue_payload_column]].copy()
        enc["encoder_ts"] = enc[self.dataqueue_payload_column].apply(self._parse_encoder_ts)

        enc = enc.dropna(subset=["encoder_ts", self.dataqueue_queue_column])
        enc = enc.rename(columns={self.dataqueue_queue_column: "queue_elapsed"})
        enc["queue_elapsed"] = pd.to_numeric(enc["queue_elapsed"], errors="coerce")
        enc = enc.dropna(subset=["queue_elapsed"])

        if len(enc) < 2:
            raise ValueError("Insufficient encoder samples for alignment")
        return enc

    def _parse_encoder_ts(self, payload: str) -> int | None:
        m = self._encoder_ts_re.search(payload or "")
        return int(m.group(1)) if m else None

    def _fit(self, encoder_ts: np.ndarray, queue_elapsed: np.ndarray) -> tuple[float, float]:
        x = np.asarray(encoder_ts, dtype=np.float64)
        y = np.asarray(queue_elapsed, dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size < 2:
            raise ValueError("Insufficient encoder samples for alignment")

        # Centered affine fit (stable for large microsecond timestamps)
        x0 = float(x.mean())
        y0 = float(y.mean())
        xc = x - x0
        yc = y - y0
        denom = float(np.dot(xc, xc))
        if denom <= 0:
            raise ValueError("Degenerate encoder timestamps for alignment")

        a = float(np.dot(xc, yc) / denom)
        b = float(y0 - a * x0)

        if not np.isfinite(a) or not np.isfinite(b):
            raise ValueError("Encoder alignment fit returned non-finite coefficients")
        return a, b