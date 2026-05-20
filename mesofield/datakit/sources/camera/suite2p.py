"""Suite2p loader with nidaq-aligned timeline and table payloads."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from mesofield.datakit.sources.register import LoadContext, TimeseriesSource
from mesofield.datakit.datamodel import StreamPayload



class Suite2pV2(TimeseriesSource):
    """Load Suite2p outputs using nidaq pulse alignment."""

    tag = "suite2p"
    patterns = ("**/*_suite2p/**/F.npy",)
    requires = ("dataqueue",)

    required_files: Tuple[str, ...] = ("Fneu.npy", "iscell.npy")
    optional_files: Tuple[str, ...] = ("ops.npy", "stat.npy")

    COLUMN_MAP: Dict[Tuple[str, str], str] = {
        ("raw", "cell_identifier"): "cell_identifier",
        ("raw", "stat"): "stat",
        ("raw", "ops"): "ops",
        ("raw", "plane_directory"): "plane_directory",
        ("raw", "cell_mask"): "cell_mask",
        ("processed", "roi_fluorescence"): "roi_fluorescence",
        ("processed", "neuropil_fluorescence"): "neuropil_fluorescence",
        ("processed", "deltaf_f"): "deltaf_f",
        ("toolkit", "time_native_s"): "time_native_s",
        ("toolkit", "time_elapsed_s"): "time_elapsed_s",
    }

    NEUROPIL_SCALE = 0.3
    BASELINE_PERCENTILE = 3.0
    PULSES_PER_FRAME: Optional[int] = 2

    def build_timeseries(
        self,
        path: Path,
        *,
        context: LoadContext | None = None,
    ) -> tuple[np.ndarray, StreamPayload, dict]:
        """Load Suite2p outputs and return timeline, payload, and metadata."""
        plane_dir = path.parent
        session_dir, subject, session, task = self._find_session_dir(plane_dir)
        dq_path = None
        if context is not None:
            dq_path = context.path_for("dataqueue")

        f_traces = self._ensure_float32(self._load_array(path))
        ancillary = self._load_companions(plane_dir)

        n_cells, n_frames = f_traces.shape

        timeline, timing_meta = self._nidaq_timeline(
            session_dir,
            n_frames,
            dq_path=dq_path,
            dq_frame=context.dataqueue_frame if context else None,
            subject=subject,
            session=session,
            task=task,
        )
        timeline = timeline - timeline[0]

        cell_mask = ancillary["iscell"][:, 0].astype(bool)
        accepted_cells = int(cell_mask.sum())

        sections, processing_meta = self._process_traces(
            raw_traces=f_traces,
            neuropil_traces=ancillary["Fneu"],
            cell_mask=cell_mask,
            timeline=timeline,
        )

        raw_section = {
            "cell_identifier": ancillary["iscell"],
            "stat": ancillary["stat"],
            "ops": ancillary["ops"],
            "plane_directory": str(plane_dir),
            "cell_mask": cell_mask,
        }

        sections = {"raw": raw_section, **sections}

        table = self._build_payload_table(sections)
        payload = StreamPayload.table(table)

        meta = {
            "source_file": str(path),
            "plane": plane_dir.name,
            "session_dir": str(session_dir) if session_dir else None,
            "n_rois": int(n_cells),
            "n_cells": accepted_cells,
            "n_frames": int(n_frames),
        }
        meta.update(timing_meta)
        meta.update(processing_meta)

        return timeline, payload, meta

    # ------------------------------------------------------------------
    # Processing helpers
    # ------------------------------------------------------------------
    def _process_traces(
        self,
        *,
        raw_traces: np.ndarray,
        neuropil_traces: np.ndarray,
        cell_mask: np.ndarray,
        timeline: np.ndarray,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        timeline = np.asarray(timeline, dtype=np.float64)

        defaults = {
            "neuropil_scale": self.NEUROPIL_SCALE,
            "baseline_percentile": self.BASELINE_PERCENTILE,
        }

        if raw_traces.size == 0 or neuropil_traces.size == 0:
            raise ValueError("Suite2pV2: empty fluorescence arrays")

        if timeline.shape[0] != raw_traces.shape[1]:
            raise ValueError(
                "Suite2pV2: timeline length mismatch; expected one timestamp per frame"
            )

        roi_fluo = raw_traces[cell_mask].astype(np.float32, copy=False)
        neuropil_fluo = neuropil_traces[cell_mask].astype(np.float32, copy=False)

        filtered_meta = {
            "total_rois": int(raw_traces.shape[0]),
            "filtered_cells": int(roi_fluo.shape[0]),
        }

        if roi_fluo.size == 0:
            raise ValueError("Suite2pV2: no accepted cells after masking")

        # Step 1: baseline and native $\Delta F / F$
        baseline = np.percentile(
            roi_fluo,
            self.BASELINE_PERCENTILE,
            axis=1,
            keepdims=True,
        ).astype(np.float32, copy=False)
        baseline_mask = np.abs(baseline) > np.finfo(np.float64).eps
        deltaf_f = np.full_like(roi_fluo, np.nan, dtype=np.float32)
        np.divide(
            roi_fluo - baseline,
            baseline,
            out=deltaf_f,
            where=baseline_mask,
        )

        sections = {
            "processed": {
                "roi_fluorescence": roi_fluo,
                "neuropil_fluorescence": neuropil_fluo,
                "deltaf_f": deltaf_f,
            },
            "toolkit": {
                "time_native_s": self._ensure_float32(timeline),
                "time_elapsed_s": self._ensure_float32(timeline),
            },
        }

        meta = {
            **filtered_meta,
            "processing_status": "ok",
            **defaults,
        }

        if baseline.size:
            meta["baseline_percentile_mean"] = float(np.nanmean(baseline))
        return sections, meta

    def _build_payload_table(self, sections: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        flat_data: Dict[str, list[Any]] = {}

        for section, items in sections.items():
            if not items:
                continue

            for field, value in items.items():
                column_name = self.COLUMN_MAP.get((section, field))
                if column_name is None:
                    raise KeyError(f"Suite2pV2 column mapping missing for section '{section}' field '{field}'")

                row_labels: list[str] | None = None
                data_value = value

                if isinstance(value, tuple) and len(value) == 2:
                    candidate, labels = value
                    if isinstance(labels, Iterable) and not isinstance(labels, (str, bytes)):
                        row_labels = []
                        for idx, lbl in enumerate(labels):
                            label_text = str(lbl).strip().replace(" ", "_")
                            row_labels.append(label_text or f"component_{idx}")
                        data_value = candidate
                    else:
                        data_value = candidate

                coerced = self._coerce_table_value(data_value)

                if row_labels:
                    components = self._expand_components(coerced, len(row_labels))
                    for idx, label in enumerate(row_labels):
                        flat_data[f"{column_name}_{label}"] = [components[idx]]
                else:
                    flat_data[column_name] = [coerced]

        frame = pd.DataFrame(flat_data)
        frame.index = pd.Index([0], name="record")
        return frame

    @staticmethod
    def _coerce_table_value(value: Any) -> Any:
        if isinstance(value, pd.Series):
            return Suite2pV2._coerce_table_value(value.to_numpy(copy=False))
        if isinstance(value, pd.Index):
            return Suite2pV2._coerce_table_value(value.to_numpy(copy=False))
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                return [Suite2pV2._coerce_table_value(item) for item in value.tolist()]
            return Suite2pV2._ensure_numeric_dtype(value)
        if isinstance(value, (list, tuple)):
            if not value:
                return np.asarray(value)
            try:
                array = np.asarray(value)
            except Exception:
                array = None
            if array is not None and array.dtype != object:
                return Suite2pV2._ensure_numeric_dtype(array)
            return [Suite2pV2._coerce_table_value(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _expand_components(value: Any, expected: int) -> list[Any]:
        if isinstance(value, list) and len(value) >= expected:
            return value
        if isinstance(value, list) and value:
            first = value[0]
            return [first for _ in range(expected)]
        if isinstance(value, np.ndarray):
            if value.ndim >= 1 and value.shape[0] >= expected:
                return [value[idx] for idx in range(expected)]
            return [value for _ in range(expected)]
        return [value for _ in range(expected)]

    @staticmethod
    def _ensure_numeric_dtype(array: np.ndarray) -> np.ndarray:
        if array.dtype.kind in {"i", "u"}:
            return array.astype(np.int32, copy=False)
        if array.dtype.kind in {"f"} and array.dtype != np.float32:
            return array.astype(np.float32, copy=False)
        return array

    @staticmethod
    def _ensure_float32(array: np.ndarray | Iterable[float]) -> np.ndarray:
        arr = np.asarray(array)
        if arr.dtype == np.float32:
            return arr
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _cast_nested_numeric(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: Suite2pV2._cast_nested_numeric(val) for key, val in value.items()}
        if isinstance(value, list):
            return [Suite2pV2._cast_nested_numeric(val) for val in value]
        if isinstance(value, tuple):
            return tuple(Suite2pV2._cast_nested_numeric(val) for val in value)
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                return [Suite2pV2._cast_nested_numeric(item) for item in value.tolist()]
            return Suite2pV2._ensure_numeric_dtype(value)
        if isinstance(value, np.generic):
            kind = value.dtype.kind
            if kind == "f" and value.dtype != np.float32:
                return np.float32(value)
            if kind in {"i", "u"} and value.itemsize > np.dtype(np.int32).itemsize:
                return np.int32(value)
            return value.item()
        return value

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_array(path: Path) -> np.ndarray:
        arr = np.load(path, allow_pickle=False)
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"Expected numpy array in {path}, got {type(arr)}")
        return arr

    def _load_companions(self, plane_dir: Path) -> Dict[str, Any]:
        required = tuple(self.required_files)

        missing = [name for name in required if not (plane_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"Suite2p plane missing companion files {missing} in {plane_dir}")

        companions: Dict[str, Any] = {}
        companions["Fneu"] = self._ensure_float32(self._load_array(plane_dir / "Fneu.npy"))
        companions["iscell"] = self._ensure_numeric_dtype(
            np.load(plane_dir / "iscell.npy", allow_pickle=False)
        )

        stat_file = "stat.npy"
        stat_path = plane_dir / stat_file
        if stat_path.exists():
            stat_raw = np.load(stat_path, allow_pickle=True)
            stat_payload = stat_raw.tolist() if hasattr(stat_raw, "tolist") else stat_raw
            companions["stat"] = self._cast_nested_numeric(stat_payload)
        else:
            companions["stat"] = []

        ops_file = "ops.npy"
        ops_path = plane_dir / ops_file
        if ops_path.exists():
            ops_raw = np.load(ops_path, allow_pickle=True)
            try:
                ops_payload = ops_raw.item()
            except (ValueError, AttributeError):
                ops_payload = ops_raw
            companions["ops"] = self._cast_nested_numeric(ops_payload)
        else:
            companions["ops"] = {}

        return companions

    def _nidaq_timeline(
        self,
        session_dir: Optional[Path],
        n_frames: int,
        *,
        dq_path: Optional[Path] = None,
        dq_frame: Optional[pd.DataFrame] = None,
        subject: Optional[str] = None,
        session: Optional[str] = None,
        task: Optional[str] = None,
    ) -> tuple[np.ndarray, dict]:
        """Build timeline using nidaq pulse timestamps from dataqueue."""
        if dq_frame is None and dq_path is None:
            if session_dir is None:
                raise FileNotFoundError("Suite2pV2: session directory not found for nidaq timeline")
            dq_path = self._find_dataqueue(session_dir, subject=subject, session=session, task=task)
        if dq_path is None:
            raise FileNotFoundError("Suite2pV2: dataqueue file not found for nidaq timeline")
        if dq_frame is None:
            df = pd.read_csv(dq_path)
        else:
            df = dq_frame

        if not {"device_id", "queue_elapsed", "payload"}.issubset(df.columns):
            raise ValueError("Suite2pV2: dataqueue missing device_id/queue_elapsed/payload columns")

        nidaq = df.loc[df["device_id"].astype(str).str.contains("nidaq", case=False, na=False)]
        if nidaq.empty:
            raise ValueError("Suite2pV2: no nidaq rows found in dataqueue")

        nidaq_times = pd.to_numeric(nidaq["queue_elapsed"], errors="coerce")
        nidaq_payload = pd.to_numeric(nidaq["payload"], errors="coerce")

        valid_mask = nidaq_times.notna() & nidaq_payload.notna()
        nidaq_times = nidaq_times[valid_mask].to_numpy(dtype=np.float64)
        nidaq_payload = nidaq_payload[valid_mask].to_numpy(dtype=np.float64)

        if nidaq_times.size == 0:
            raise ValueError("Suite2pV2: nidaq timestamps empty after filtering")

        nidaq_times_rel = nidaq_times - nidaq_times[0]

        order = np.argsort(nidaq_payload)
        pulse_counter = nidaq_payload[order]
        pulse_times = nidaq_times_rel[order]

        if np.any(np.diff(pulse_counter) <= 0):
            raise ValueError("Suite2pV2: nidaq payload must be strictly increasing")

        pulse_span = pulse_counter[-1] - pulse_counter[0] + 1
        pulses_per_frame = self.PULSES_PER_FRAME
        if pulses_per_frame is None:
            inferred = pulse_span / n_frames
            if not np.isclose(inferred, round(inferred)):
                raise ValueError("Suite2pV2: nidaq payload does not match expected frame count")
            pulses_per_frame = int(round(inferred))

        if pulses_per_frame < 1:
            raise ValueError("Suite2pV2: pulses_per_frame must be >= 1")

        expected_span = int(pulses_per_frame * n_frames)
        if pulse_counter.size < 2:
            raise ValueError("Suite2pV2: nidaq payload requires at least two pulses for interpolation")

        expected_pulses = np.arange(expected_span, dtype=np.float64) + pulse_counter[0]
        expected_times = self._interpolate_pulse_times(pulse_counter, pulse_times, expected_pulses)
        timeline = expected_times[::pulses_per_frame]

        if timeline.size != n_frames:
            raise ValueError("Suite2pV2: nidaq interpolation produced unexpected frame count")

        derived_frames = int(n_frames)

        meta = {
            "time_basis": "nidaq_interpolated",
            "dataqueue_file": str(dq_path),
            "nidaq_pulses_received": int(len(nidaq_times)),
            "pulses_per_frame": int(pulses_per_frame),
            "pulses_per_frame_source": "configured" if self.PULSES_PER_FRAME is not None else "inferred",
            "dq_nidaq_pulses": int(len(nidaq_times)),
            "dq_nidaq_frames_derived": derived_frames,
            "dq_pulse_span": int(pulse_span),
            "dq_expected_pulse_span": int(expected_span),
            "dq_pulse_count": int(pulse_counter.size),
            "dq_pulse_span_match": bool(int(pulse_span) == expected_span),
            "dq_pulses_per_frame": int(pulses_per_frame),
        }

        return timeline, meta

    @staticmethod
    def _interpolate_pulse_times(
        pulse_counter: np.ndarray,
        pulse_times: np.ndarray,
        expected_pulses: np.ndarray,
    ) -> np.ndarray:
        if pulse_counter.size < 2:
            raise ValueError("Suite2pV2: insufficient nidaq pulses for interpolation")

        expected_times = np.interp(expected_pulses, pulse_counter, pulse_times)

        start_slope = (pulse_times[1] - pulse_times[0]) / (pulse_counter[1] - pulse_counter[0])
        end_slope = (pulse_times[-1] - pulse_times[-2]) / (pulse_counter[-1] - pulse_counter[-2])

        left_mask = expected_pulses < pulse_counter[0]
        right_mask = expected_pulses > pulse_counter[-1]

        if left_mask.any():
            expected_times[left_mask] = pulse_times[0] + start_slope * (
                expected_pulses[left_mask] - pulse_counter[0]
            )
        if right_mask.any():
            expected_times[right_mask] = pulse_times[-1] + end_slope * (
                expected_pulses[right_mask] - pulse_counter[-1]
            )

        return expected_times

    def _find_session_dir(self, plane_dir: Path) -> tuple[Optional[Path], Optional[str], Optional[str], Optional[str]]:
        """Walk up from plane_dir to find session directory or use inventory hint."""
        subject = None
        session = None
        task = None

        suite2p_dir = plane_dir.parent if plane_dir.name.startswith("plane") else plane_dir
        suite2p_name = suite2p_dir.name
        if suite2p_name:
            subject_match = re.search(r"sub-([A-Za-z0-9]+)", suite2p_name)
            session_match = re.search(r"ses-([A-Za-z0-9]+)", suite2p_name)
            task_match = re.search(r"task-([A-Za-z0-9]+)", suite2p_name)
            if subject_match:
                subject = subject_match.group(1)
            if session_match:
                session = session_match.group(1)
            if task_match:
                task = task_match.group(1)

        # Try walking up from plane_dir
        current = plane_dir
        max_levels = 5
        for _ in range(max_levels):
            if current == current.parent:
                break
            if (current / "beh").is_dir() and (current / "func").is_dir():
                return current, subject, session, task
            current = current.parent

        # If suite2p is in processed/, try to find the data/ sibling
        if "processed" in plane_dir.parts:
            # Navigate to experiment root and check data/
            idx = plane_dir.parts.index("processed")
            exp_root = Path(*plane_dir.parts[:idx])
            data_dir = exp_root / "data"
            if data_dir.is_dir():
                subject_dir = None
                session_dir = None

                if subject:
                    subject_dir = data_dir / f"sub-{subject}"
                    if not subject_dir.is_dir():
                        subject_dir = None

                if subject_dir is not None and session:
                    session_dir = subject_dir / f"ses-{session}"
                    if not session_dir.is_dir():
                        session_dir = None

                if session_dir is not None and (session_dir / "beh").is_dir():
                    return session_dir, subject, session, task

                # Look for subject/session structure
                for subject_dir in data_dir.iterdir():
                    if subject_dir.is_dir() and subject_dir.name.startswith("sub-"):
                        for session_dir in subject_dir.iterdir():
                            if session_dir.is_dir() and session_dir.name.startswith("ses-"):
                                if (session_dir / "beh").is_dir():
                                    return session_dir, subject, session, task
        return None, subject, session, task

    def _find_dataqueue(
        self,
        session_dir: Path,
        *,
        subject: Optional[str] = None,
        session: Optional[str] = None,
        task: Optional[str] = None,
    ) -> Optional[Path]:
        """Find dataqueue CSV in session/beh directory."""
        candidates = sorted((session_dir / "beh").glob("*_dataqueue.csv"))
        if not candidates:
            return None

        def matches(candidate: Path) -> bool:
            name = candidate.name
            if subject and f"sub-{subject}" not in name:
                return False
            if session and f"ses-{session}" not in name:
                return False
            if task and f"task-{task}" not in name:
                return False
            return True

        filtered = [path for path in candidates if matches(path)]
        return filtered[0] if filtered else candidates[0]

