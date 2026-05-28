"""Treadmill loader.

Data is reconstructed entirely from the central ``*_dataqueue.csv`` log: each
``EncoderData(timestamp=..., distance=..., speed=...)`` payload row carries the
full sample, already published on the master clock via ``queue_elapsed``. The
per-session ``*_treadmill.csv`` is only consulted as a fallback when no
dataqueue is available for that session (e.g. legacy fixtures used by the
``treadmill_csv_fallback`` test).
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from mesofield.datakit.config import settings
from mesofield.datakit.sources.register import LoadContext, TimeseriesSource
from mesofield.datakit.timeline import DataqueueIndex


class TreadmillSource(TimeseriesSource):
    """Load treadmill samples aligned to the experiment window."""

    tag = "treadmill"
    # Discovered from either the dataqueue (primary, parsed from EncoderData
    # payloads) or — when no dataqueue exists for the session — the standalone
    # treadmill CSV.
    patterns = (
        "**/*_dataqueue.csv",
        "**/*_treadmill.csv",
        "**/*_treadmill_data.csv",
    )
    camera_tag = None
    requires = ("dataqueue",)

    timestamp_column = "timestamp"
    distance_columns = ("distance_mm", "distance")
    speed_columns = ("speed_mm", "speed_mm_s", "speed")
    output_distance_column = "distance_mm"
    output_speed_column = "speed_mm"
    timestamp_to_seconds = 1e-6

    dataqueue_queue_column = settings.timeline.queue_column
    dataqueue_device_id_column = "device_id"
    dataqueue_payload_column = "payload"

    dataqueue_payload_prefix = "EncoderData"
    timeline_master_patterns = ("dhyana", "mesoscope")
    _encoder_ts_re = re.compile(r"timestamp\s*=\s*(\d+)", re.IGNORECASE)
    _encoder_payload_re = re.compile(
        r"EncoderData\s*\("
        r"\s*timestamp\s*=\s*(?P<ts>\d+)\s*,"
        r"\s*distance\s*=\s*(?P<dist>-?\d+(?:\.\d+)?)"
        r"(?:\s*[A-Za-z/]+)?\s*,"
        r"\s*speed\s*=\s*(?P<speed>-?\d+(?:\.\d+)?)"
        r"(?:\s*[A-Za-z/]+)?\s*\)",
        re.IGNORECASE,
    )

    def build_timeseries(
        self,
        path: Path,
        *,
        context: LoadContext | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame, dict]:
        context = self._require_context(context)
        dq_path = context.path_for("dataqueue")

        # Primary path: build from the dataqueue alone — the per-session
        # treadmill CSV (if it exists) is ignored.
        if dq_path is not None or context.dataqueue_frame is not None:
            aligned, meta = self._build_from_dataqueue(
                dq_path=dq_path,
                dataqueue_frame=context.dataqueue_frame,
                window=context.experiment_window,
            )
            t = aligned["time_elapsed_s"].to_numpy(dtype=np.float64)
            meta = {
                "source_file": str(dq_path) if dq_path is not None else None,
                "dataqueue_file": str(dq_path) if dq_path is not None else None,
                "n_samples": int(len(aligned)),
                **meta,
            }
            return t, aligned, meta

        # Fallback path: standalone treadmill CSV with no dataqueue available.
        if path is None or not Path(path).exists() or Path(path).name.endswith("_dataqueue.csv"):
            raise FileNotFoundError(
                "TreadmillSource: no dataqueue available and no treadmill CSV path"
            )
        df = pd.read_csv(path)
        dist = next((c for c in self.distance_columns if c in df.columns), None)
        speed = next((c for c in self.speed_columns if c in df.columns), None)
        if self.timestamp_column not in df.columns or dist is None or speed is None:
            raise ValueError(f"Expected timestamp/distance/speed columns in {path}")
        df = df.rename(columns={dist: self.output_distance_column, speed: self.output_speed_column})

        aligned, meta = self._build_from_csv_fallback(df)
        t = aligned["time_elapsed_s"].to_numpy(dtype=np.float64)
        meta = {
            "source_file": str(path),
            "dataqueue_file": None,
            "n_samples": int(len(aligned)),
            **meta,
        }
        return t, aligned, meta

    # --- dataqueue path -------------------------------------------------------

    def _build_from_dataqueue(
        self,
        *,
        dq_path: Path | None,
        dataqueue_frame: pd.DataFrame | None,
        window: tuple[float, float] | None,
    ) -> tuple[pd.DataFrame, dict]:
        t0, t1 = window or self._window(dq_path, dataqueue_frame)
        duration = float(t1 - t0)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid camera window duration from dataqueue: start={t0}, end={t1}")

        samples = self._encoder_samples(dq_path, dataqueue_frame)
        queue_elapsed = samples["queue_elapsed"]
        distance = samples["distance_mm"]
        speed = samples["speed_mm"]
        encoder_ts = samples["encoder_ts"]

        time_s = queue_elapsed - t0
        finite_vals = np.isfinite(distance) & np.isfinite(speed)
        mask = (
            np.isfinite(time_s)
            & finite_vals
            & (time_s >= 0.0)
            & (time_s <= duration)
        )
        if not np.any(mask):
            # No treadmill samples in this session: return empty DataFrame and meta
            aligned = pd.DataFrame({
                self.timestamp_column: np.array([], dtype=np.float64),
                self.output_distance_column: np.array([], dtype=np.float64),
                self.output_speed_column: np.array([], dtype=np.float64),
                "time_elapsed_s": np.array([], dtype=np.float64),
            })
            meta = {
                "source_method": "dataqueue_encoder_payload",
                "experiment_window": {"start": float(t0), "end": float(t1)},
                "n_encoder_payloads": int(queue_elapsed.size),
                "n_kept": 0,
            }
            return aligned, meta

        aligned = pd.DataFrame(
            {
                self.timestamp_column: encoder_ts[mask],
                self.output_distance_column: distance[mask],
                self.output_speed_column: speed[mask],
                "time_elapsed_s": time_s[mask],
            }
        )
        aligned.sort_values("time_elapsed_s", inplace=True)
        aligned.reset_index(drop=True, inplace=True)

        meta = {
            "source_method": "dataqueue_encoder_payload",
            "experiment_window": {"start": float(t0), "end": float(t1)},
            "n_encoder_payloads": int(queue_elapsed.size),
            "n_kept": int(np.count_nonzero(mask)),
        }
        return aligned, meta

    def _encoder_samples(
        self,
        dq_path: Path | None,
        dataqueue_frame: pd.DataFrame | None,
    ) -> dict[str, np.ndarray]:
        """Return parallel arrays parsed from EncoderData payloads.

        Treadmill owns this parsing — the ``dataqueue`` source intentionally
        treats payloads as opaque. We use the upstream frame only if the
        payload column is still in its original string dtype (which is no
        longer the case once ``DataqueueSource`` has run), otherwise we
        re-read the raw CSV via ``dq_path``.
        """

        dq = dataqueue_frame
        if dq is not None:
            payload = dq.get(self.dataqueue_payload_column)
            if payload is not None and (
                pd.api.types.is_object_dtype(payload) or pd.api.types.is_string_dtype(payload)
            ):
                return self._parse_dataqueue_payloads(dq)
        if dq_path is None:
            raise ValueError(
                "TreadmillSource: dataqueue payload is not string-typed and no path is "
                "available to re-read EncoderData strings"
            )
        raw = pd.read_csv(
            dq_path,
            usecols=[self.dataqueue_queue_column, self.dataqueue_payload_column],
            low_memory=False,
        )
        return self._parse_dataqueue_payloads(raw)

    def _parse_dataqueue_payloads(self, dq: pd.DataFrame) -> dict[str, np.ndarray]:
        if (
            self.dataqueue_payload_column not in dq.columns
            or self.dataqueue_queue_column not in dq.columns
        ):
            raise ValueError("Dataqueue frame missing payload or queue_elapsed column")

        payload = dq[self.dataqueue_payload_column].astype(str)
        mask = payload.str.contains(self.dataqueue_payload_prefix, na=False)
        if not np.any(mask):
            raise ValueError("No EncoderData payloads found in dataqueue")

        sub_payloads = payload.loc[mask].to_numpy()
        queue_full = pd.to_numeric(dq.loc[mask, self.dataqueue_queue_column], errors="coerce").to_numpy(dtype=np.float64)

        ts_list: list[float] = []
        dist_list: list[float] = []
        speed_list: list[float] = []
        q_list: list[float] = []
        for value, q in zip(sub_payloads, queue_full):
            if not np.isfinite(q):
                continue
            m = self._encoder_payload_re.search(value or "")
            if m is None:
                continue
            ts_list.append(float(int(m.group("ts"))))
            dist_list.append(float(m.group("dist")))
            speed_list.append(float(m.group("speed")))
            q_list.append(float(q))

        if not ts_list:
            raise ValueError("No parseable EncoderData payloads in dataqueue")

        return {
            "queue_elapsed": np.asarray(q_list, dtype=np.float64),
            "encoder_ts": np.asarray(ts_list, dtype=np.float64),
            "distance_mm": np.asarray(dist_list, dtype=np.float64),
            "speed_mm": np.asarray(speed_list, dtype=np.float64),
        }

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

    # --- CSV-only fallback ----------------------------------------------------

    def _build_from_csv_fallback(
        self, treadmill_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict]:
        ts = pd.to_numeric(treadmill_df[self.timestamp_column], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(ts)
        if not np.any(finite):
            raise ValueError("No valid treadmill timestamp samples")

        aligned = treadmill_df.loc[finite].copy()
        ts = ts[finite]
        ts_i64 = np.rint(ts).astype(np.int64)

        unwrapped = DataqueueIndex.fix_32bit_wraparound(ts_i64).astype(np.float64)
        # If the file contains a wrap, samples after the wrap may be out of order.
        # Sort by unwrapped timestamp before computing elapsed time.
        sort_idx = np.argsort(unwrapped)
        unwrapped = unwrapped[sort_idx]
        aligned = aligned.iloc[sort_idx].reset_index(drop=True)
        elapsed = DataqueueIndex.relative(unwrapped) * self.timestamp_to_seconds
        aligned["_elapsed"] = elapsed

        aligned[self.output_distance_column] = pd.to_numeric(
            aligned[self.output_distance_column], errors="coerce"
        )
        aligned[self.output_speed_column] = pd.to_numeric(
            aligned[self.output_speed_column], errors="coerce"
        )
        aligned = aligned.dropna(
            subset=[self.output_distance_column, self.output_speed_column, "_elapsed"]
        )
        if aligned.empty:
            raise ValueError("Treadmill fallback produced no finite elapsed samples")

        elapsed = aligned.pop("_elapsed").to_numpy(dtype=np.float64)
        if elapsed.size > 1:
            keep = np.concatenate(([True], np.diff(elapsed) > 0))
            aligned = aligned.iloc[keep].copy()
            elapsed = elapsed[keep]
        if elapsed.size == 0:
            raise ValueError("Treadmill fallback produced no usable samples after cleaning")

        aligned["time_elapsed_s"] = elapsed
        aligned.sort_values("time_elapsed_s", inplace=True)
        aligned.reset_index(drop=True, inplace=True)

        return aligned, {
            "source_method": "treadmill_csv_fallback",
            "experiment_window": None,
            "alignment": None,
        }
