"""Dataqueue synchronization data source.

This loader ingests the central ``*_dataqueue.csv`` log produced by the rig and
returns a stream describing every message with absolute and relative timing.
It is commonly used to align per-device clocks (treadmill, cameras, nidaq) by
extracting anchor pairs for downstream timeline fitting.
"""

from typing import Optional
import pandas as pd
import numpy as np
from pathlib import Path

from mesofield.datakit.sources.register import SourceContext, TimeseriesSource


class DataqueueSource(TimeseriesSource):
    """Load the dataqueue CSV as a time-indexed table."""
    tag = "dataqueue"
    patterns = ("**/*_dataqueue.csv",)
    camera_tag = None
    time_column = "queue_elapsed"
    device_id_column = "device_id"
    device_timestamp_column = "device_ts"
    payload_column = "payload"
    master_device_priority: tuple[str, ...] = ("dhyana", "mesoscope")
    device_alias_patterns: tuple[tuple[str, str], ...] = (
        ("dhyana", "meso"),
        ("mesoscope", "meso"),
        ("thorcam", "pupil"),
    )
    
    def build_timeseries(
        self,
        path: Path,
        *,
        context: SourceContext | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame, dict]:
        """Read ``*_dataqueue.csv`` and return a time-indexed table."""
        raw = pd.read_csv(path, low_memory=False)

        if self.time_column not in raw.columns:
            raise ValueError(f"Dataqueue file missing required time column '{self.time_column}': {path}")

        queue_numeric = pd.to_numeric(raw[self.time_column], errors="coerce")
        queue_elapsed = queue_numeric.to_numpy(dtype=np.float64)
        if queue_elapsed.size == 0:
            raise ValueError("Dataqueue has no timing samples")

        queue_start = float(queue_elapsed[0])
        t = queue_elapsed - queue_start

        master_id = self._select_master_device(raw)
        master_elapsed = self._master_elapsed(raw, master_id, queue_start)
        if master_elapsed.size == 0:
            master_elapsed = t

        device_elapsed = self._build_device_elapsed(raw, queue_start)
        device_ts = self._build_device_ts(raw)
        device_aliases = self._build_device_aliases(raw)
        affine = self._fit_master_affine(raw, master_id, queue_start)
        device_aligned = self._apply_affine(device_elapsed, affine)

        frame = raw.copy()
        frame["time_elapsed_s"] = t.astype(np.float64)
        if self.device_id_column in frame.columns:
            per_device_start = (
                frame.groupby(self.device_id_column)[self.time_column]
                .transform("min")
                .astype(np.float64)
            )
            frame["device_elapsed_s"] = pd.to_numeric(frame[self.time_column], errors="coerce") - per_device_start

        meta = {
            "source_file": str(path),
            "n_entries": len(raw),
            "master_device_id": master_id,
            "master_elapsed": master_elapsed.astype(np.float64),
            "device_ts": device_ts,
            "device_elapsed": device_elapsed,
            "device_aligned_abs": device_aligned,
            "device_sample_rate_hz": self._estimate_device_rates(device_elapsed),
            "device_aliases": device_aliases,
            "affine": affine,
            "time_basis": self.time_column,
        }

        return t.astype(np.float64), frame, meta

    def _select_master_device(self, df: pd.DataFrame) -> str:
        """Pick the device used as the master time basis."""

        if self.device_id_column not in df.columns:
            return "master"

        ids = df[self.device_id_column].astype(str).fillna("")
        for pattern in self.master_device_priority:
            mask = ids.str.contains(pattern, case=False, regex=False)
            if mask.any():
                return str(ids[mask].iloc[0])

        non_empty = ids[ids != ""]
        if not non_empty.empty:
            return str(non_empty.iloc[0])
        return "master"

    def _master_elapsed(self, df: pd.DataFrame, master_id: str, queue_start: float) -> np.ndarray:
        """Return elapsed seconds for the master device."""

        device_series = df.get(self.device_id_column)
        if device_series is None:
            return np.array([], dtype=np.float64)

        mask = device_series.astype(str) == master_id
        master_rows = df.loc[mask]
        if master_rows.empty:
            return np.array([], dtype=np.float64)

        master_queue = pd.to_numeric(master_rows[self.time_column], errors="coerce").dropna()
        if master_queue.empty:
            return np.array([], dtype=np.float64)

        master_queue = master_queue - float(master_queue.iloc[0])
        return master_queue.to_numpy(dtype=np.float64)

    def _estimate_rate(self, timeline: np.ndarray) -> float:
        if timeline.size < 2:
            return 0.0
        diffs = np.diff(timeline)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            return 0.0
        return float(1.0 / np.median(diffs))

    def _build_device_elapsed(self, df: pd.DataFrame, queue_start: float) -> dict[str, np.ndarray]:
        """Return per-device elapsed seconds from queue timestamps."""

        device_elapsed: dict[str, np.ndarray] = {}
        if self.device_id_column not in df.columns:
            return device_elapsed

        for device_id, group in df.groupby(self.device_id_column):
            device_key = str(device_id)
            queue_rel = pd.to_numeric(group[self.time_column], errors="coerce").to_numpy(dtype=np.float64)
            queue_rel = queue_rel - float(queue_start)
            order = np.argsort(queue_rel)
            device_elapsed[device_key] = queue_rel[order]
        return device_elapsed

    def _build_device_ts(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return per-device raw timestamps as ISO strings when possible."""

        device_ts: dict[str, np.ndarray] = {}
        if self.device_id_column not in df.columns or self.device_timestamp_column not in df.columns:
            return device_ts

        for device_id, group in df.groupby(self.device_id_column):
            device_key = str(device_id)
            parsed = pd.to_datetime(group[self.device_timestamp_column], errors="coerce", utc=True)
            if parsed.isna().all():
                device_ts[device_key] = group[self.device_timestamp_column].to_numpy()
            else:
                device_ts[device_key] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ").to_numpy()
        return device_ts

    def _build_device_aliases(self, df: pd.DataFrame) -> dict[str, str]:
        """Map device identifiers to configured aliases (e.g., Dhyana → meso)."""

        aliases: dict[str, str] = {}
        if self.device_id_column not in df.columns:
            return aliases

        ids = df[self.device_id_column].dropna().astype(str).unique().tolist()
        for device_id in ids:
            alias = self._alias_for_device(device_id)
            if alias is not None:
                aliases[device_id] = alias
        return aliases

    def _alias_for_device(self, device_id: str) -> Optional[str]:
        lowered = device_id.lower()
        for pattern, alias in self.device_alias_patterns:
            if pattern.lower() in lowered:
                return alias
        return None

    def _fit_master_affine(
        self,
        df: pd.DataFrame,
        master_id: str,
        queue_start: float,
    ) -> dict[str, float] | None:
        """Fit affine map from elapsed seconds to absolute time using master device."""

        if self.device_id_column not in df.columns or self.device_timestamp_column not in df.columns:
            return None

        master_rows = df.loc[df[self.device_id_column].astype(str) == master_id]
        if master_rows.empty:
            return None

        elapsed = pd.to_numeric(master_rows[self.time_column], errors="coerce").to_numpy(dtype=np.float64)
        elapsed = elapsed - float(queue_start)
        absolute = pd.to_datetime(master_rows[self.device_timestamp_column], errors="coerce", utc=True)
        if elapsed.size < 2 or absolute.isna().all():
            return None

        valid_mask = np.isfinite(elapsed) & ~absolute.isna()
        if valid_mask.sum() < 2:
            return None

        elapsed = elapsed[valid_mask]
        absolute = absolute[valid_mask]

        e0 = float(elapsed[0])
        e1 = float(elapsed[-1])
        a0 = float(absolute.iloc[0].value) / 1e9
        a1 = float(absolute.iloc[-1].value) / 1e9
        if e1 == e0:
            return None

        a = (a1 - a0) / (e1 - e0)
        b = a0 - a * e0
        return {"a": float(a), "b": float(b)}

    def _apply_affine(
        self,
        device_elapsed: dict[str, np.ndarray],
        affine: dict[str, float] | None,
    ) -> dict[str, np.ndarray]:
        """Project elapsed times into aligned absolute seconds (UTC)."""

        if affine is None:
            return {}
        a = affine.get("a")
        b = affine.get("b")
        if a is None or b is None:
            return {}

        aligned: dict[str, np.ndarray] = {}
        for device_id, elapsed in device_elapsed.items():
            aligned[device_id] = (a * elapsed + b).astype(np.float64)
        return aligned

    def _estimate_device_rates(self, device_elapsed: dict[str, np.ndarray]) -> dict[str, float]:
        return {device_id: self._estimate_rate(values) for device_id, values in device_elapsed.items()}