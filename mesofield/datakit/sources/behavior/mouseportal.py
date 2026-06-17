"""MousePortal corridor loader.

Aligns the MousePortal per-frame CSV (``*_mouseportal.csv``) to the camera timeline
using the treadmill microsecond clock as the shared master.

MousePortal logs the treadmill ``device_us`` of the most recent forwarded sample
on every frame.  The same ``device_us`` values are recorded in the central
``*_dataqueue.csv`` (fanned out into a ``device_us`` column for the treadmill
device, alongside ``queue_elapsed``).  We fit an affine map
``device_us -> queue_elapsed`` from those dataqueue rows, apply it to each
corridor frame, and shift onto the experiment window so corridor position /
gain condition land on the same clock as the widefield + pupil frames.

This mirrors the dataqueue alignment strategy of
:class:`mesofield.datakit.sources.behavior.treadmill.TreadmillSource`, but reads
the treadmill's ``device_us`` *column* rather than parsing ``EncoderData(...)``
payload strings -- the producer pushes a dict payload, so the queue logger
stores its fields as columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mesofield.datakit.config import settings
from mesofield.datakit.sources.register import (
    IntervalSeriesSource,
    LoadContext,
    TimeseriesSource,
)


class MousePortalSource(TimeseriesSource):
    """Load MousePortal corridor frames aligned to the experiment window."""

    tag = "mouseportal"
    patterns = ("**/*_mouseportal.csv",)
    camera_tag = None
    requires = ("dataqueue",)

    device_us_column = "treadmill_device_us"

    dataqueue_queue_column = settings.timeline.queue_column  # "queue_elapsed"
    dataqueue_device_id_column = "device_id"
    dataqueue_device_us_column = "device_us"
    treadmill_device_match = "treadmill"
    timeline_master_patterns = settings.timeline.window_device_patterns

    def build_timeseries(
        self,
        path: Path,
        *,
        context: LoadContext | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame, dict]:
        context = self._require_context(context)
        frames = pd.read_csv(path)
        if self.device_us_column not in frames.columns:
            raise ValueError(
                f"MousePortal CSV missing '{self.device_us_column}' column: {path}"
            )

        dq_path = context.path_for("dataqueue")
        if dq_path is None and context.dataqueue_frame is None:
            raise FileNotFoundError(
                "MousePortalSource requires a dataqueue for alignment"
            )

        a, b = self._device_us_to_queue_fit(dq_path, context.dataqueue_frame)
        t0, t1 = context.experiment_window or self._window(dq_path, context.dataqueue_frame)
        duration = float(t1 - t0)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid experiment window: start={t0}, end={t1}")

        device_us = pd.to_numeric(frames[self.device_us_column], errors="coerce").to_numpy(dtype=np.float64)
        time_s = a * device_us + b - t0

        # Drop frames logged before the first forwarded treadmill sample
        # (device_us == 0) and any falling outside the camera window.
        mask = (
            np.isfinite(time_s)
            & (device_us > 0)
            & (time_s >= 0.0)
            & (time_s <= duration)
        )
        if not np.any(mask):
            raise ValueError("No MousePortal frames fall inside the experiment window")

        aligned = frames.loc[mask].copy()
        aligned["time_elapsed_s"] = time_s[mask]
        aligned.sort_values("time_elapsed_s", inplace=True)
        aligned.reset_index(drop=True, inplace=True)

        t = aligned["time_elapsed_s"].to_numpy(dtype=np.float64)
        meta = {
            "source_file": str(path),
            "dataqueue_file": str(dq_path) if dq_path is not None else None,
            "n_frames_total": int(len(frames)),
            "n_frames_aligned": int(len(aligned)),
            "experiment_window": {"start": float(t0), "end": float(t1)},
            "alignment": {"a": float(a), "b": float(b)},
            "source_method": "dataqueue_device_us_affine",
        }
        return t, aligned, meta

    # -- dataqueue helpers ----------------------------------------------
    def _read_dataqueue(
        self, dq_path: Path | None, dataqueue_frame: pd.DataFrame | None
    ) -> pd.DataFrame:
        if dq_path is not None:
            return pd.read_csv(dq_path, low_memory=False)
        return dataqueue_frame

    def _device_us_to_queue_fit(
        self, dq_path: Path | None, dataqueue_frame: pd.DataFrame | None
    ) -> tuple[float, float]:
        """Fit ``queue_elapsed = a * device_us + b`` from treadmill dataqueue rows."""
        dq = self._read_dataqueue(dq_path, dataqueue_frame)
        for col in (self.dataqueue_device_us_column, self.dataqueue_queue_column):
            if col not in dq.columns:
                raise ValueError(f"Dataqueue missing '{col}' column for alignment")

        device = dq.get(self.dataqueue_device_id_column, pd.Series(dtype=str)).astype(str)
        mask = device.str.contains(self.treadmill_device_match, case=False, na=False, regex=False)
        rows = dq.loc[mask]
        x = pd.to_numeric(rows[self.dataqueue_device_us_column], errors="coerce").to_numpy(dtype=np.float64)
        y = pd.to_numeric(rows[self.dataqueue_queue_column], errors="coerce").to_numpy(dtype=np.float64)
        return self._fit(x, y)

    def _window(
        self, dq_path: Path | None, dataqueue_frame: pd.DataFrame | None
    ) -> tuple[float, float]:
        """Experiment window from master-camera rows, else the full queue span."""
        dq = self._read_dataqueue(dq_path, dataqueue_frame)
        queue = pd.to_numeric(dq.get(self.dataqueue_queue_column), errors="coerce")
        device = dq.get(self.dataqueue_device_id_column, pd.Series(dtype=str)).astype(str)

        mask = np.zeros(len(dq), dtype=bool)
        for pattern in self.timeline_master_patterns:
            mask |= device.str.contains(pattern, case=False, na=False, regex=False).to_numpy()
        master = queue[mask].dropna()
        if len(master) >= 2:
            return float(master.iloc[0]), float(master.iloc[-1])

        # Fallback: span the whole dataqueue (mock rigs whose camera ids don't
        # match the master patterns still align against session start/end).
        all_q = queue.dropna()
        if len(all_q) < 2:
            raise ValueError("Could not determine an experiment window from dataqueue")
        return float(all_q.min()), float(all_q.max())

    @staticmethod
    def _fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Centered affine least-squares fit (stable for large microsecond x)."""
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size < 2:
            raise ValueError("Insufficient treadmill samples in dataqueue for alignment")
        x0 = float(x.mean())
        y0 = float(y.mean())
        xc = x - x0
        denom = float(np.dot(xc, xc))
        if denom <= 0:
            raise ValueError("Degenerate device_us values for alignment")
        a = float(np.dot(xc, y - y0) / denom)
        b = float(y0 - a * x0)
        if not (np.isfinite(a) and np.isfinite(b)):
            raise ValueError("Alignment fit returned non-finite coefficients")
        return a, b


class MousePortalTrials(IntervalSeriesSource):
    """Per-trial intervals from the MousePortal log, on the camera clock.

    Collapses the per-frame ``state == "TRIAL_RUNNING"`` runs into one row per
    ``(block, trial)`` with ``start_s``/``stop_s`` already aligned to the master
    (camera) timeline via :class:`MousePortalSource`. This makes
    event-triggered analyses trivial: each row's ``start_s`` is a trial onset on
    the same clock as the widefield/pupil frames, so a peri-event window is a
    direct time slice (see ``mesofield.datakit.epoch.event_triggered_average``).

    Emits the inter-trial intervals too (``phase`` column: ``trial`` | ``iti``)
    so stops/ITIs are available as events as well.
    """

    tag = "mouseportal_trials"
    patterns = ("**/*_mouseportal.csv",)
    camera_tag = None
    requires = ("dataqueue",)

    # Map MousePortal state names -> the interval "phase" we expose.
    _phase_states = {"TRIAL_RUNNING": "trial", "INTER_TRIAL_INTERVAL": "iti"}

    def build_intervals(
        self,
        path: Path,
        *,
        context: LoadContext | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        context = self._require_context(context)
        # Reuse the corridor source's device_us -> camera-time alignment.
        _t, aligned, src_meta = MousePortalSource().build_timeseries(path, context=context)

        rows: list[dict] = []
        for state, phase in self._phase_states.items():
            sub = aligned[aligned.get("state") == state]
            if sub.empty:
                continue
            for (block, trial), grp in sub.groupby(["block", "trial"], sort=True):
                te = pd.to_numeric(grp["time_elapsed_s"], errors="coerce").dropna()
                if te.empty:
                    continue
                start, stop = float(te.min()), float(te.max())
                cond = grp["condition"].iloc[0] if "condition" in grp.columns else ""
                rows.append({
                    "phase": phase,
                    "block": int(block),
                    "trial": int(trial),
                    "condition": cond,
                    "start_s": start,
                    "stop_s": stop,
                    "duration_s": stop - start,
                })

        intervals = pd.DataFrame(
            rows,
            columns=["phase", "block", "trial", "condition", "start_s", "stop_s", "duration_s"],
        )
        if not intervals.empty:
            intervals = intervals.sort_values("start_s").reset_index(drop=True)

        meta = {
            "source_file": str(path),
            "n_trials": int((intervals["phase"] == "trial").sum()) if not intervals.empty else 0,
            "alignment": src_meta.get("alignment"),
            "experiment_window": src_meta.get("experiment_window"),
        }
        return intervals, meta
