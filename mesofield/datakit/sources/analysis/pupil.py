"""Pupil DeepLabCut analysis data source."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..._utils._logger import get_logger
from mesofield.datakit.sources.camera.pupil import PupilMetadataSource
from mesofield.datakit.sources.register import LoadContext, TimeseriesSource
from mesofield.datakit.timeline import DataqueueIndex

logger = get_logger(__name__)


class PupilDLCSource(TimeseriesSource):
    """Load DeepLabCut HDF5 output and compute pupil diameters."""

    tag = "pupil_dlc"
    patterns = (
        "**/*_pupilDLC_*.h5",
        "**/*_pupilDLC_*.hdf5",
        "**/*_pupil.omeDLC_*.h5"
    )
    camera_tag = "pupil_metadata"
    flatten_payload = True
    requires = ("dataqueue", "pupil_metadata")
    default_frame_rate_hz = 20.0
    confidence_threshold = 0.7
    pixel_to_mm = 53.6
    dpi = 300
    landmark_pairs = ((0, 1), (2, 3), (4, 5), (6, 7))
    warn_on_low_confidence = True
    alignment_device_id = "thorcam"

    def build_timeseries(
        self,
        path: Path,
        *,
        context: LoadContext | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame, dict]:
        """Load DeepLabCut output and return a time-indexed table."""
        context = self._require_context(context)
        analyzed_df = self._analyze_pupil_h5(path)

        n_frames = len(analyzed_df)
        timing = self._aligned_timeline(context, n_frames)
        if timing is None:
            metadata_path = context.path_for("pupil_metadata")
            if metadata_path is None:
                logger.warning(
                    "PupilDLCSource: ThorCam alignment and pupil metadata missing; falling back to assumed frame-rate timeline."
                )
                t = np.arange(n_frames, dtype=np.float64) / float(self.default_frame_rate_hz)
                timing_meta = {
                    "time_basis": "assumed_frame_rate",
                    "assumed_frame_rate_hz": float(self.default_frame_rate_hz),
                    "pupil_metadata_file": None,
                    "pupil_metadata_alignment": None,
                    "pupil_metadata_frames": None,
                }
            else:
                logger.warning(
                    "PupilDLCSource: ThorCam alignment missing; falling back to metadata timeline."
                )
                metadata_stream = PupilMetadataSource().load(metadata_path, context=context)
                metadata_t = metadata_stream.t
                if metadata_t.size == 0:
                    raise ValueError("PupilDLCSource: pupil metadata contains no timeline samples")
                metadata_t = metadata_t - metadata_t[0]
                n_meta = int(metadata_t.size)
                if n_meta == n_frames:
                    t = metadata_t
                    method = "metadata_direct"
                else:
                    anchor_index = np.arange(n_meta, dtype=np.float64)
                    frame_index = np.linspace(0.0, float(n_meta - 1), num=n_frames, dtype=np.float64)
                    t = np.interp(frame_index, anchor_index, metadata_t)
                    method = "metadata_resampled"

                timing_meta = {
                    "time_basis": "pupil_metadata",
                    "pupil_metadata_file": str(metadata_path),
                    "pupil_metadata_alignment": method,
                    "pupil_metadata_frames": n_meta,
                }
        else:
            t, timing_meta = timing

        analyzed_df = analyzed_df.copy()
        analyzed_df["time_elapsed_s"] = t

        meta = {
            "source_file": str(path),
            "n_frames": n_frames,
            "class_name": self.__class__.__name__,
            "confidence_threshold": self.confidence_threshold,
            "pixel_to_mm": self.pixel_to_mm,
            "landmark_pairs": self.landmark_pairs,
            **timing_meta,
        }

        return t, analyzed_df, meta

    def _aligned_timeline(self, context: LoadContext, n_frames: int) -> tuple[np.ndarray, dict] | None:
        if context.dataqueue_frame is not None:
            times = self._aligned_from_frame(context.dataqueue_frame)
            dq_path = context.path_for("dataqueue")
        else:
            dq_path = context.path_for("dataqueue")
            if dq_path is None:
                # No dataqueue available for this session (e.g. older
                # STREHAB02/03/05). Signal the caller to fall back to the
                # pupil-metadata timeline instead of raising.
                return None
            timeline = DataqueueIndex.from_path(dq_path)
            device_slice = timeline.slice(
                lambda ids: ids.str.contains(self.alignment_device_id, case=False, na=False, regex=False)
            )
            times = device_slice.queue_elapsed().to_numpy(dtype=np.float64)
        if times.size == 0:
            return None
        n_anchors = int(times.size)
        if n_anchors == n_frames:
            aligned = times
            method = "direct"
        else:
            anchor_index = np.arange(n_anchors, dtype=np.float64)
            frame_index = np.linspace(0.0, float(n_anchors - 1), num=n_frames, dtype=np.float64)
            aligned = np.interp(frame_index, anchor_index, times)
            method = "resampled"

        aligned = aligned - aligned[0]

        meta = {
            "time_basis": "dataqueue",
            "dataqueue_file": str(dq_path) if dq_path is not None else None,
            "dataqueue_device": self.alignment_device_id,
            "dataqueue_anchors": n_anchors,
            "dataqueue_alignment": method,
        }
        return aligned, meta

    def _aligned_from_frame(self, frame: pd.DataFrame) -> np.ndarray:
        device_col = "device_id"
        time_col = "queue_elapsed"
        if device_col not in frame.columns or time_col not in frame.columns:
            return np.array([], dtype=np.float64)
        device_series = frame[device_col].astype(str)
        mask = device_series.str.contains(self.alignment_device_id, case=False, na=False, regex=False)
        times = pd.to_numeric(frame.loc[mask, time_col], errors="coerce").dropna()
        return times.to_numpy(dtype=np.float64)

    def _analyze_pupil_h5(
        self,
        filepath: Path,
        *,
        confidence_threshold: float | None = None,
        pixel_to_mm: float | None = None,
        dpi: int | None = None,
    ) -> pd.DataFrame:
        """Compute pupil diameters from DeepLabCut HDF5 output."""
        threshold = self.confidence_threshold if confidence_threshold is None else confidence_threshold
        px_to_mm = self.pixel_to_mm if pixel_to_mm is None else pixel_to_mm
        dpi_value = self.dpi if dpi is None else dpi

        with pd.HDFStore(filepath, mode="r") as store:
            keys = store.keys()
            if not keys:
                raise ValueError(f"No keys found in HDF5 file: {filepath}")
            key = "/df_with_missing" if "/df_with_missing" in keys else keys[0]
            frame = store.get(key)

        if not isinstance(frame.columns, pd.MultiIndex) or frame.columns.nlevels < 3:
            raise ValueError(f"Unexpected HDF5 column format for {filepath}")

        scorer = str(frame.columns.get_level_values(0)[0])
        sub = frame[scorer]

        bodyparts = sorted(set(sub.columns.get_level_values(0)))
        coords_list: list[np.ndarray] = []
        conf_list: list[np.ndarray] = []
        used_bodyparts: list[str] = []

        for bodypart in bodyparts:
            try:
                x = sub[(bodypart, "x")].to_numpy(dtype=np.float64)
                y = sub[(bodypart, "y")].to_numpy(dtype=np.float64)
                likelihood = sub[(bodypart, "likelihood")].to_numpy(dtype=np.float64)
            except KeyError:
                continue
            coords_list.append(np.stack([x, y], axis=1))
            conf_list.append(likelihood)
            used_bodyparts.append(bodypart)

        if not coords_list:
            raise ValueError(f"No coordinate columns found in {filepath}")

        coords = np.stack(coords_list, axis=1)
        conf = np.stack(conf_list, axis=1)

        any_confident = np.any(conf >= threshold)
        if self.warn_on_low_confidence and not any_confident:
            logger.warning(
                "PupilDLCSource: no confidence values above threshold",
                extra={
                    "phase": "pupil_dlc_analysis",
                    "threshold": threshold,
                    "dpi": dpi_value,
                    "bodyparts": used_bodyparts,
                },
            )

        n_points = coords.shape[1]
        pairs = [(a, b) for a, b in self.landmark_pairs if a < n_points and b < n_points]
        if not pairs:
            diameters = np.full(coords.shape[0], np.nan, dtype=np.float64)
        else:
            dists = []
            for a, b in pairs:
                diff = coords[:, a, :] - coords[:, b, :]
                dist = np.linalg.norm(diff, axis=1)
                valid = (conf[:, a] >= threshold) & (conf[:, b] >= threshold)
                dist = np.where(valid, dist, np.nan)
                dists.append(dist)
            stacked = np.vstack(dists).T
            valid_counts = np.sum(np.isfinite(stacked), axis=1)
            diameters = np.full(stacked.shape[0], np.nan, dtype=np.float64)
            valid_mask = valid_counts > 0
            if not np.any(valid_mask) and self.warn_on_low_confidence:
                logger.warning(
                    "PupilDLCSource: no valid landmark pairs after confidence filtering",
                    extra={
                        "phase": "pupil_dlc_analysis",
                        "threshold": threshold,
                        "bodyparts": used_bodyparts,
                    },
                )
            if np.any(valid_mask):
                with np.errstate(all="ignore"):
                    diameters[valid_mask] = np.nanmean(stacked[valid_mask], axis=1)

        pupil_series = pd.Series(diameters, index=frame.index).interpolate() / px_to_mm
        return pd.DataFrame({"pupil_diameter_mm": pupil_series})