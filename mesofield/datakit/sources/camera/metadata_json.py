"""Shared JSON metadata loader for camera sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json

import numpy as np
import pandas as pd

from mesofield.datakit.sources.register import LoadContext, TableSource


class MetadataJSON(TableSource):
    """Load camera metadata JSON into a normalized table."""

    json_entry_key = "p0"
    allow_fallback_entry_key = False
    metadata_column = "camera_metadata"
    timestamp_preference = ("TimeReceivedByCore", "ElapsedTime-ms", "runner_time_ms")
    millisecond_columns = {"ElapsedTime-ms", "runner_time_ms"}
    drop_columns: tuple[str, ...] = ()
    device_column = "camera_device"

    def build_table(
        self,
        path: Path,
        *,
        context: LoadContext | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
        """Normalize metadata rows and build a timeline."""
        with open(path, "r") as handle:
            data = json.load(handle)
        entry_key, payload = self._resolve_entry(data, path)

        df = pd.DataFrame(payload)

        device_id = None
        if self.device_column in df.columns and len(df):
            device_id = str(df[self.device_column].iloc[0])

        if self.metadata_column in df.columns:
            camera_metadata_df = pd.json_normalize(df[self.metadata_column].tolist())
            non_overlapping = [
                col for col in camera_metadata_df.columns if col not in df.columns
            ]
            df = df.join(camera_metadata_df[non_overlapping])

        existing_columns = [col for col in self.drop_columns if col in df.columns]
        if existing_columns:
            df = df.drop(columns=existing_columns)

        t, absolute = self._resolve_timeline(df, path)
        df = df.copy()
        df["time_elapsed_s"] = t
        if absolute is not None:
            df["time_absolute"] = absolute

        # Memory optimizations for full-length per-row arrays in the materialized
        # output: downcast numeric counters and convert low-cardinality string
        # columns to categoricals.
        for col in ("ImageNumber",):
            if col in df.columns:
                numeric = pd.to_numeric(df[col], errors="coerce")
                if numeric.notna().all() and (numeric >= 0).all() and numeric.max() < np.iinfo(np.uint32).max:
                    df[col] = numeric.astype(np.uint32)
        for col in ("Temperature",):
            if col in df.columns:
                numeric = pd.to_numeric(df[col], errors="coerce")
                if numeric.notna().any():
                    df[col] = numeric.astype(np.float32)
        for col in ("Camera", self.device_column):
            if col in df.columns:
                df[col] = df[col].astype("category")

        meta = {
            "source_file": str(path),
            "n_frames": int(len(df)),
            "device_id": device_id,
            "json_entry_key": entry_key,
        }

        return t.astype(np.float64), df, meta

    def _resolve_entry(self, data: dict[str, Any], path: Path) -> tuple[str, Any]:
        if not isinstance(data, dict):
            raise ValueError(f"Unsupported metadata format in {path}")

        payload = data.get(self.json_entry_key)
        if payload is None and self.allow_fallback_entry_key:
            if data:
                entry_key, payload = next(iter(data.items()))
                return entry_key, payload
        if payload is None:
            raise KeyError(
                f"Missing expected entry '{self.json_entry_key}' in {path}; found keys: {list(data.keys())}"
            )
        return self.json_entry_key, payload

    def _resolve_timeline(
        self, df: pd.DataFrame, path: Path
    ) -> tuple[np.ndarray, pd.Series | None]:
        t: np.ndarray | None = None
        absolute: pd.Series | None = None

        for column in self.timestamp_preference:
            if column not in df.columns:
                continue
            if column == self.timestamp_preference[0]:
                timestamps = pd.to_datetime(df[column], errors="coerce", utc=True)
                valid = timestamps.dropna()
                if not valid.empty:
                    origin = valid.iloc[0]
                    t = (timestamps - origin).dt.total_seconds().to_numpy(dtype=np.float64)
                    # Keep as datetime64[ns, UTC] (8 B/value) rather than ISO strings (~30 B/value).
                    absolute = timestamps
            else:
                scale = 1000.0 if column in self.millisecond_columns else 1.0
                values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)
                if values.size:
                    t = (values - values[0]) / scale
            if t is not None:
                break

        if t is None:
            raise ValueError(f"No recognized timestamp column found in {path}")

        return t.astype(np.float64), absolute
