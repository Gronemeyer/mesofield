"""PsychoPy stimulus device.

Wraps :class:`mesofield.subprocesses.psychopy.PsychoPyProcess` behind the
standard :class:`~mesofield.protocols.StimulusDevice` interface so the
:class:`~mesofield.base.Procedure` can drive it through the same
``arm/start/stop/shutdown`` lifecycle as any other hardware device.

PsychoPy is *not* a :class:`~mesofield.protocols.DataProducer`: it never
emits on ``signals.data`` and does not implement ``save_data`` /
``get_data``.  Its ``signals.started`` fires when the subprocess prints
``PSYCHOPY_READY``; ``signals.finished`` fires when ``QProcess`` exits.

The producer (`PsychoPyDevice`) does not write its own data file --
PsychoPy writes its own CSV directly. The matching ingest-side parser
(`Psychopy`, an IntervalSeriesSource) lives at the bottom of this
module so producer and parser sit in the same file.
`PsychoPyDevice.Parser` resolves to `Psychopy` for manifest-driven
dispatch.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

import numpy as np
import pandas as pd

from mesofield import DeviceRegistry
from mesofield.signals import DeviceSignals
from mesofield.utils._logger import get_logger
from mesofield.datakit.sources.register import IntervalSeriesSource, SourceContext
from mesofield.datakit.timeline import DataqueueIndex


@DeviceRegistry.register("psychopy")
class PsychoPyDevice:
    """Stimulus device that launches a PsychoPy script as a subprocess."""

    device_type: ClassVar[str] = "stimulus"
    file_type: str = ""
    bids_type: Optional[str] = None

    def __init__(self, cfg: Dict[str, Any]):
        self.signals = DeviceSignals()
        self.cfg = dict(cfg)
        self.device_id: str = cfg.get("id", "psychopy")
        self.is_primary: bool = bool(cfg.get("primary", False))
        self.logger = get_logger(f"{__name__}.PsychoPyDevice[{self.device_id}]")
        self._process = None  # mesofield.subprocesses.psychopy.PsychoPyProcess
        self._config = None
        self._started: Optional[datetime] = None
        self._stopped: Optional[datetime] = None

    # -- lifecycle ------------------------------------------------------
    def initialize(self) -> bool:
        return True

    def arm(self, config) -> None:
        """Per-run prep: stash the experiment config so ``start`` can launch."""
        self._config = config

    def start(self) -> bool:
        from mesofield.devices.subprocesses.psychopy import PsychoPyProcess

        if self._config is None:
            self.logger.error("PsychoPyDevice.start called before arm()")
            return False

        self._process = PsychoPyProcess(self._config)
        self._process.ready.connect(self._on_ready)
        self._process.finished.connect(self._on_finished)
        self._started = datetime.now()
        self._process.start()
        return True

    def stop(self) -> bool:
        self._stopped = datetime.now()
        if self._process is not None and self._process.process is not None:
            try:
                self._process.process.terminate()
                if not self._process.process.waitForFinished(2000):
                    self._process.process.kill()
            except Exception as exc:
                self.logger.warning(f"PsychoPy stop failed: {exc}")
        self.signals.finished.emit()
        return True

    def shutdown(self) -> None:
        self.stop()
        self._process = None

    # -- callbacks ------------------------------------------------------
    def _on_ready(self) -> None:
        self.signals.started.emit()

    def _on_finished(self, *_args) -> None:
        self.signals.finished.emit()

    # -- introspection --------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "running": self._process is not None,
            "started": self._started,
            "stopped": self._stopped,
        }

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
        }


# ---------------------------------------------------------------------------
# Parser (the ingest-side counterpart). Loads PsychoPy trial windows as an
# interval table, aligns to nidaq pulses via the dataqueue. Bound below
# as `PsychoPyDevice.Parser`.


_TASK_PATTERN = re.compile(r"task[-_]?([A-Za-z0-9]+)", re.IGNORECASE)
_TASK_ALIASES = {
    "gratings": "task-gratings",
    "movies": "task-movies",
}
_TIME_SUFFIXES = (".started", ".stopped", ".rt", ".t")


class Psychopy(IntervalSeriesSource):
    """Load Psychopy trial windows as an interval table."""

    tag = "psychopy"
    patterns = ("**/*_psychopy.csv",)
    task_pattern = _TASK_PATTERN
    task_aliases = _TASK_ALIASES
    time_suffixes = _TIME_SUFFIXES
    keypress_rt_pattern = re.compile(r"key_resp.*\.rt$")
    time_source_prefix = "display_"
    time_source_suffix = ".started"
    gratings_columns = (
        "trials.thisN",
        "stim_grayScreen.started",
        "stim_grayScreen.stopped",
        "stim_grating.started",
        "stim_grating.stopped",
        "display_gratings.stopped",
    )
    movies_columns = (
        "trials_2.thisN",
        "trials_2.thisIndex",
        "trials2.thisN",
        "trials.thisIndex",
        "description",
        "thisRow.t",
        "display_mov.started",
        "natural_mov.started",
        "display_mov.stopped",
        "grey.started",
        "grey.stopped",
    )
    dataqueue_elapsed_column = "queue_elapsed"
    dataqueue_device_column = "device_id"
    dataqueue_device_match = "nidaq"
    dataqueue_payload_column = "payload"
    dataqueue_payload_value = 1

    def build_intervals(
        self,
        path: Path,
        *,
        context: SourceContext | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        context = self._require_context(context)
        # 1) Read raw psychopy CSV.
        raw = pd.read_csv(path, low_memory=False)

        # 2) Infer task and choose trial table.
        task = self._infer_task(path)
        if task == "task-gratings":
            trial_table = self._extract_gratings(raw)
            task_branch = "gratings"
        elif task == "task-movies":
            trial_table = self._extract_movies(raw)
            task_branch = "movies"
        else:
            trial_table = raw
            task_branch = "raw"

        # 3) Resolve timeline and keypress offset.
        time_source = trial_table if any(
            c.startswith(self.time_source_prefix) and c.endswith(self.time_source_suffix) for c in trial_table.columns
        ) else raw
        row_time, row_col, row_meta = self._resolve_row_time(time_source)
        key_rt, key_col = self._resolve_keypress_rt(raw)

        # 4) Parse experiment start time (optional).
        exp_start = None
        exp_start_raw = raw.get("expStart")
        if exp_start_raw is not None and exp_start_raw.dropna().any():
            exp_start = self._parse_exp_start(exp_start_raw.dropna().iloc[0])

        # 5) Apply dataqueue offset and align times.
        dq_start, dq_meta = self._dataqueue_offset(context)
        exp_start_offset = None
        exp_start_keypress_offset = None
        exp_start_offset = self._calc_exp_start_offset(exp_start, dq_meta.get("abs_start_time"))
        # Offset into psychopy-zero time (exp_start + keypress RT).
        exp_start_keypress_offset = exp_start_offset - float(key_rt)
        dq_start = float(exp_start_keypress_offset)
        aligned_t = self._align_time(row_time, key_rt, dq_start)
        aligned_table = self._align_table(trial_table, key_rt, dq_start)

        # 6) For gratings, build windows once from the aligned table.
        if task_branch == "gratings":
            if "window_grating_start" in aligned_table.columns:
                aligned_table["gratings_windows"] = [
                    self._build_window_list(aligned_table, "window_grating_start", "window_grating_stop")
                ] * len(aligned_table)
            if "window_gray_start" in aligned_table.columns:
                aligned_table["gray_windows"] = [
                    self._build_window_list(aligned_table, "window_gray_start", "window_gray_stop")
                ] * len(aligned_table)

        # 7) Build interval table and metrics.
        interval_start = None
        interval_stop = None
        if "window_grating_start" in aligned_table.columns and "window_grating_stop" in aligned_table.columns:
            interval_start = aligned_table["window_grating_start"]
            interval_stop = aligned_table["window_grating_stop"]
        elif "window_gray_start" in aligned_table.columns and "window_gray_stop" in aligned_table.columns:
            interval_start = aligned_table["window_gray_start"]
            interval_stop = aligned_table["window_gray_stop"]
        else:
            interval_start = pd.Series(aligned_t, index=aligned_table.index)
            interval_stop = pd.Series(aligned_t, index=aligned_table.index)

        intervals = aligned_table.rename(
            columns={c: f"{task_branch}_{c}" for c in aligned_table.columns}
        ).copy()
        intervals["start_s"] = pd.to_numeric(interval_start, errors="coerce")
        intervals["stop_s"] = pd.to_numeric(interval_stop, errors="coerce")
        intervals = self._apply_experiment_window(intervals, context, dq_start)
        metrics = {
            "source_file": str(path),
            "n_rows": int(len(raw)),
            "task_inferred": task,
            "task_branch": task_branch,
            "row_time_column": row_col,
            "key_resp_rt": key_rt,
            "key_resp_column": key_col,
            "exp_start": exp_start.isoformat() if exp_start is not None else None,
            "start_offset_s": dq_start,
        }
        if exp_start_offset is not None:
            metrics["exp_start_offset_s"] = float(exp_start_offset)
        if exp_start_keypress_offset is not None:
            metrics["exp_start_keypress_offset_s"] = float(exp_start_keypress_offset)
        metrics.update(row_meta)
        metrics.update(dq_meta)
        metrics["time_basis"] = "psychopy_zero_based" if dq_start is not None else "psychopy_native"
        metrics["alignment_method"] = "keypress_relative"

        return intervals, metrics

    def _parse_exp_start(self, value: Any) -> Optional[pd.Timestamp]:
        text = str(value).strip()
        text = text.replace("h", ":", 1)
        text = re.sub(r":(\d{2})\.(\d{2}\.\d+)", r":\1:\2", text)
        parsed = pd.to_datetime(text, errors="coerce", utc=False)
        return None if pd.isna(parsed) else parsed

    def _strip_tz(self, value: pd.Timestamp) -> pd.Timestamp:
        return value.tz_localize(None) if getattr(value, "tzinfo", None) else value

    def _calc_exp_start_offset(self, exp_start: pd.Timestamp, abs_start_time: Any) -> Optional[float]:
        if not abs_start_time:
            return None
        abs_start = pd.to_datetime(abs_start_time, errors="coerce", utc=False)
        if pd.isna(abs_start):
            return None
        abs_start = self._strip_tz(abs_start)
        exp_start = self._strip_tz(exp_start)
        return (abs_start - exp_start).total_seconds()

    def _infer_task(self, file_path: Path) -> str | None:
        name = file_path.as_posix().lower()
        match = self.task_pattern.search(name)
        if match:
            return self.task_aliases.get(match.group(1).lower())
        for key, label in self.task_aliases.items():
            if key in name:
                return label
        return None

    def _select_columns(self, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        available = [col for col in columns if col in frame.columns]
        if not available:
            raise ValueError("Psychopy required columns are missing from the CSV.")
        return frame.loc[:, available].copy()

    def _coalesce_column(self, frame: pd.DataFrame, columns: list[str]) -> pd.Series:
        available = [col for col in columns if col in frame.columns]
        if not available:
            raise ValueError(f"Psychopy required columns missing: {columns}")
        return frame.loc[:, available].bfill(axis=1).iloc[:, 0]

    def _resolve_keypress_rt(self, frame: pd.DataFrame) -> tuple[float, str]:
        candidates = [col for col in frame.columns if self.keypress_rt_pattern.match(str(col))]
        for col in candidates:
            series = pd.to_numeric(frame[col], errors="coerce")
            if series.notna().any():
                return float(series.dropna().iloc[0]), str(col)
        raise ValueError("Psychopy keypress RT column not found (key_resp*.rt).")

    def _resolve_row_time(self, frame: pd.DataFrame) -> tuple[np.ndarray, str, dict]:
        display_candidates = [
            c for c in frame.columns if c.startswith(self.time_source_prefix) and c.endswith(self.time_source_suffix)
        ]
        for col in display_candidates:
            series = pd.to_numeric(frame[col], errors="coerce")
            if series.notna().sum() >= 2:
                filled = series.interpolate(limit_direction="both")
                time_cols = [c for c in frame.columns if any(c.endswith(suf) for suf in self.time_suffixes)]
                max_times = [pd.to_numeric(frame[c], errors="coerce").max() for c in time_cols]
                max_time = max([t for t in max_times if pd.notna(t)], default=filled.max())
                if max_time > filled.max():
                    last_valid_idx = filled.last_valid_index()
                    if last_valid_idx is not None and last_valid_idx < len(filled) - 1:
                        filled.loc[last_valid_idx + 1:] = max_time
                meta = {
                    "row_time_column": col,
                    "row_time_extended_to": float(max_time),
                }
                return filled.to_numpy(dtype=np.float64), col, meta
        raise ValueError("Psychopy requires a display_*.started column with at least 2 numeric values.")

    def _dataqueue_offset(self, context: SourceContext) -> tuple[Optional[float], dict]:
        #TODO: If a dataqueue does not have nidaq payload==1 but may start at nidaq==2,
        # treat the first pulse as the start and log a warning.
        dq_path = context.path_for("dataqueue")
        if context.dataqueue_frame is not None:
            frame = context.dataqueue_frame
            queue_all = pd.to_numeric(frame.get(self.dataqueue_elapsed_column), errors="coerce").dropna()
            if queue_all.empty:
                raise ValueError("Dataqueue frame contains no queue_elapsed samples")
            queue_start = float(queue_all.iloc[0])
            device_series = frame.get(self.dataqueue_device_column, pd.Series(dtype=str)).astype(str)
            mask = device_series.str.contains(self.dataqueue_device_match, case=False, na=False, regex=False)
            device_rows = frame.loc[mask]
            pulses = pd.to_numeric(device_rows.get(self.dataqueue_elapsed_column), errors="coerce")
            abs_start = pd.to_datetime(device_rows.get("device_ts"), errors="coerce", utc=True)
            abs_start = abs_start.dropna().dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ").to_numpy(dtype=str)
        else:
            if dq_path is None:
                raise FileNotFoundError("Psychopy: dataqueue path not available")
            timeline = DataqueueIndex.from_path(dq_path)
            queue_all = timeline.queue_series()
            queue_start = float(queue_all.iloc[0])
            device_slice = timeline.slice(self.dataqueue_device_match)
            abs_start = device_slice.packet_absolute().to_numpy(dtype=str)
            pulses = pd.to_numeric(device_slice.rows.get(self.dataqueue_elapsed_column), errors="coerce")
        pulses = pulses.dropna()
        if pulses.empty:
            raise ValueError(f"Dataqueue timeline exists {timeline.source_path} but contains no nidaq payload==1 pulses.")

        start_rel = float(pulses.iloc[0]) - 0
        meta = {
            "dataqueue_file": str(dq_path) if dq_path is not None else None,
            "dataqueue_pulses": int(pulses.size),
            "dataqueue_queue_start": queue_start,
            "abs_start_time": abs_start[0] if len(abs_start) else None,
        }
        return start_rel, meta

    def _apply_experiment_window(
        self,
        intervals: pd.DataFrame,
        context: SourceContext,
        dq_start: float | None,
    ) -> pd.DataFrame:
        if context.experiment_window is None or dq_start is None:
            return intervals
        start, stop = context.experiment_window
        window_start = float(start) - float(dq_start)
        window_stop = float(stop) - float(dq_start)

        intervals = intervals.copy()
        intervals["start_s"] = intervals["start_s"].clip(lower=window_start, upper=window_stop)
        intervals["stop_s"] = intervals["stop_s"].clip(lower=window_start, upper=window_stop)
        mask = (intervals["stop_s"] >= window_start) & (intervals["start_s"] <= window_stop)
        return intervals.loc[mask].reset_index(drop=True)

    def _align_time(self, values: np.ndarray, key_rt: float, offset: Optional[float]) -> np.ndarray:
        aligned = values.astype(np.float64, copy=True) - float(key_rt)
        if offset is not None:
            aligned = aligned - float(offset)
        return aligned

    def _align_table(self, frame: pd.DataFrame, key_rt: float, offset: Optional[float]) -> pd.DataFrame:
        aligned = frame.copy()
        for col in aligned.columns:
            name = str(col)
            if not (name.lower().endswith(self.time_suffixes) or name.startswith("window_")):
                continue
            series = pd.to_numeric(aligned[col], errors="coerce")
            aligned[col] = self._align_time(series.to_numpy(dtype=np.float64), key_rt, offset)
        return aligned.dropna(axis=1, how="all")

    def _build_window_list(self, frame: pd.DataFrame, start_col: str, stop_col: str) -> list[tuple[float, float]]:
        if start_col not in frame.columns or stop_col not in frame.columns:
            return []
        starts = pd.to_numeric(frame[start_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        stops = pd.to_numeric(frame[stop_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        return [
            (float(start), float(stop))
            for start, stop in zip(starts, stops)
            if np.isfinite(start) and np.isfinite(stop) and stop > start
        ]

    def _extract_gratings(self, frame: pd.DataFrame) -> pd.DataFrame:
        selected = self._select_columns(frame, list(self.gratings_columns))
        result = selected.loc[selected.get("trials.thisN").notna()].copy()
        if result.empty:
            raise ValueError("Psychopy gratings trials are empty.")

        gray_start = self._coalesce_column(result, ["stim_grayScreen.started"])
        gray_stop = self._coalesce_column(result, ["stim_grating.started"])
        grating_start = self._coalesce_column(result, ["stim_grating.started"])
        grating_stop = self._coalesce_column(result, ["display_gratings.stopped"])

        result["window_grating_start"] = pd.to_numeric(grating_start, errors="coerce")
        result["window_grating_stop"] = pd.to_numeric(grating_stop, errors="coerce")
        result["window_gray_start"] = pd.to_numeric(gray_start, errors="coerce")
        result["window_gray_stop"] = pd.to_numeric(gray_stop, errors="coerce")

        return result

    def _extract_movies(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._select_columns(frame, list(self.movies_columns))


# Manifest-driven dispatch: SOURCE_REGISTRY["psychopy"] resolves to
# PsychoPyDevice.Parser.
PsychoPyDevice.Parser = Psychopy
