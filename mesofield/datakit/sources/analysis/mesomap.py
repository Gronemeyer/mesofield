"""Mesomap trace loader.

Parses wide-field mesomap traces exported as ``*_mesoscope.ome_traces.csv``
files. The accompanying ``.mask.npy`` and ``.regions.csv`` files are treated
as session-level metadata so they can be stored once per subject/session
instead of repeating for every task.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from mesofield.datakit.config import settings
from mesofield.datakit.sources.register import SourceContext, TimeseriesSource


class MesoMapSource(TimeseriesSource):
	"""Load mesomap traces and attach optional mask/region metadata."""

	tag = "mesomap"
	patterns = ("**/*_mesoscope.ome_traces.csv",)
	camera_tag = "meso_metadata"
	flatten_payload = True
 
	mask_suffix = ".mask.npy"
	regions_suffix = ".regions.csv"

	frame_column = "frame"
	time_basis = "mesomap_frame"

	def build_timeseries(
		self,
		path: Path,
		*,
		context: SourceContext | None = None,
	) -> tuple[np.ndarray, pd.DataFrame, dict]:
		trace_df = pd.read_csv(path)

		if self.frame_column not in trace_df.columns:
			raise ValueError(f"Mesomap traces missing required column '{self.frame_column}' in {path}")

		trace_df = trace_df.copy()
		trace_df.sort_values(by=self.frame_column, inplace=True)

		trace_columns = [col for col in trace_df.columns if col != self.frame_column]
		timeline = trace_df[self.frame_column].to_numpy(dtype=np.float64)
		timeline_meta = {"time_basis": self.time_basis}
		aligned = self._aligned_timeline(context, len(trace_df))
		if aligned is not None:
			timeline, timeline_meta = aligned

		payload_df = trace_df.loc[:, trace_columns].copy()
		payload_df["time_elapsed_s"] = timeline

		regions_df, roi_to_mask, missing_regions = self._load_regions(path, trace_columns)
		mask_info = self._load_mask(path)

		meta = {
			"source_file": str(path),
			"n_frames": len(trace_df),
			"n_rois": len(trace_columns),
			"trace_columns": trace_columns,
			"frame_column": self.frame_column,
			"frame_index": trace_df[self.frame_column].to_numpy(dtype=np.float64),
			"roi_to_mask_label": roi_to_mask,
			"roi_missing_region_metadata": missing_regions,
			**timeline_meta,
			"scope": settings.dataset.session_scope,
			**mask_info,
		}

		if regions_df is not None:
			meta.update(
				{
					"regions_path": str(self._regions_path(path)),
					"region_count": len(regions_df),
					"regions_table": regions_df,
				}
			)

		return timeline, payload_df, meta

	def _aligned_timeline(
		self,
		context: SourceContext | None,
		n_frames: int,
	) -> tuple[np.ndarray, dict] | None:
		if context is None:
			return None

		if context.dataqueue_frame is not None:
			aligned = self._aligned_from_frame(context.dataqueue_frame, n_frames)
			if aligned is not None:
				return aligned

		if context.master_timeline is not None and context.master_timeline.shape[0] == n_frames:
			return context.master_timeline, {
				"time_basis": "master_timeline",
				"dataqueue_alignment": "direct",
				"dataqueue_anchors": int(n_frames),
			}

		return None

	def _aligned_from_frame(
		self,
		frame: pd.DataFrame,
		n_frames: int,
	) -> tuple[np.ndarray, dict] | None:
		device_col = "device_id"
		queue_col = settings.timeline.queue_column
		if device_col not in frame.columns or queue_col not in frame.columns:
			return None

		device_series = frame[device_col].astype(str)
		mask = pd.Series(False, index=frame.index)
		for pattern in settings.timeline.window_device_patterns:
			mask |= device_series.str.contains(pattern, case=False, na=False, regex=False)

		if not mask.any():
			return None

		times = pd.to_numeric(frame.loc[mask, queue_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
		if times.size == 0:
			return None

		times = times - times[0]
		n_anchors = int(times.size)
		if n_anchors == n_frames:
			return times, {
				"time_basis": "dataqueue",
				"dataqueue_alignment": "direct",
				"dataqueue_anchors": n_anchors,
			}

		anchor_index = np.arange(n_anchors, dtype=np.float64)
		frame_index = np.linspace(0.0, float(n_anchors - 1), num=n_frames, dtype=np.float64)
		aligned = np.interp(frame_index, anchor_index, times)
		return aligned, {
			"time_basis": "dataqueue",
			"dataqueue_alignment": "resampled",
			"dataqueue_anchors": n_anchors,
		}

	def _mask_path(self, path: Path) -> Path:
		return path.with_suffix(self.mask_suffix)

	def _regions_path(self, path: Path) -> Path:
		return path.with_suffix(self.regions_suffix)

	def _load_mask(self, path: Path) -> dict:
		mask_path = self._mask_path(path)
		if not mask_path.exists():
			return {
				"mask_path": str(mask_path),
				"mask_missing": True,
			}

		mask_array = np.load(mask_path)
		unique_labels = np.unique(mask_array)
		return {
			"mask_path": str(mask_path),
			"mask_shape": tuple(int(dim) for dim in mask_array.shape),
			"mask_dtype": str(mask_array.dtype),
			"mask_labels": unique_labels.tolist(),
			"mask_missing": False,
		}

	def _load_regions(
		self, path: Path, trace_columns: Iterable[str]
	) -> Tuple[pd.DataFrame | None, dict[str, int], list[str]]:
		regions_path = self._regions_path(path)
		if not regions_path.exists():
			return None, {}, list(trace_columns)

		regions_df = pd.read_csv(regions_path)

		required_cols = {"acronym", "mask_label"}
		if not required_cols.issubset(regions_df.columns):
			missing = ", ".join(sorted(required_cols - set(regions_df.columns)))
			raise ValueError(f"Regions file {regions_path} missing required columns: {missing}")

		region_lookup = regions_df.set_index("acronym")["mask_label"]
		roi_to_mask = {
			roi: int(region_lookup[roi]) for roi in trace_columns if roi in region_lookup
		}
		missing_regions = sorted(set(trace_columns) - set(region_lookup.index))

		return regions_df, roi_to_mask, missing_regions
