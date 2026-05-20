"""Shared, data-agnostic timeline utilities backed by the dataqueue master clock.

Expects a dataqueue.csv DataSource loader. The dataqueue file contains this structure:

```python
['queue_elapsed', 'packet_ts', 'device_ts', 'device_id', 'payload']
   queue_elapsed                   packet_ts  ...     device_id payload
0  976929.429661  2025-07-08 16:58:03.627918  ...  encoder_COM3       0
1  976929.439462  2025-07-08 16:58:03.637494  ...  encoder_COM3       0
2  976929.470421  2025-07-08 16:58:03.668444  ...  encoder_COM3       0
3  976929.503343  2025-07-08 16:58:03.701397  ...  encoder_COM3       0
4  976929.503767  2025-07-08 16:58:03.701888  ...  encoder_COM3       0
```

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._utils._logger import get_logger
from mesofield.datakit.config import settings

logger = get_logger(__name__)

DeviceSelector = Union[str, Sequence[str], Callable[[pd.Series], pd.Series]]


@dataclass(frozen=True)
class TimelineSlice:
	"""Lightweight view over a subset of the dataqueue.

	It simply holds the filtered rows plus the selector that produced them so
	callers can reapply the same logic or log what slice they were analysing.
	"""

	rows: pd.DataFrame
	selector: Optional[DeviceSelector] = None

	def queue_elapsed(self) -> pd.Series:
		queue_col = settings.timeline.queue_column
		column = self.rows.get(queue_col)
		if column is None:
			return pd.Series(dtype=np.float64)

		series = pd.to_numeric(column, errors="coerce")
		return series.dropna().reset_index(drop=True).astype(np.float64)

	def packet_absolute(self) -> pd.Series:
		"""Return packet_ts as UTC ISO8601 strings."""
		column = self.rows.get("packet_ts")
		if column is None:
			return pd.Series(dtype=str)
		parsed = pd.to_datetime(column, errors="coerce", utc=True)
		valid_mask = parsed.notna()
		result = pd.Series(index=self.rows.index, dtype=str)
		result.loc[valid_mask] = parsed.loc[valid_mask].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
		return result.reset_index(drop=True)


class DataqueueIndex:
	"""Cacheable facade over a session's dataqueue CSV.

	The class deliberately avoids hard-coding knowledge of specific hardware.
	Consumers supply selectors or parsing callbacks that describe which device
	streams they are interested in. The index exposes common utilities for
	retrieving filtered views and fitting time mappings onto the shared clock.
	"""

	_cache: ClassVar[dict[Path, Optional["DataqueueIndex"]]] = {}

	def __init__(self, directory: Path, dataqueue_path: Path, dataqueue: pd.DataFrame):
		self._directory = directory
		self._dataqueue_path = dataqueue_path
		self._frame = dataqueue
		queue_col = settings.timeline.queue_column
		self._queue_all = self._prepare_queue_series(dataqueue.get(queue_col))

	# ------------------------------------------------------------------
	# Construction helpers
	# ------------------------------------------------------------------
	@classmethod
	def for_directory(cls, directory: Optional[Path]) -> Optional["DataqueueIndex"]:
		if directory is None:
			return None

		resolved = Path(directory).resolve()
		if resolved in cls._cache:
			return cls._cache[resolved]

		dq_path = cls._find_dataqueue(resolved)
		if dq_path is None:
			cls._cache[resolved] = None
			return None

		frame = cls._load_dataqueue(dq_path)
		if frame is None or frame.empty:
			cls._cache[resolved] = None
			return None

		index = cls(resolved, dq_path, frame)
		cls._cache[resolved] = index
		logger.debug(
			"DataqueueIndex initialised",
			extra={
				"phase": "dataqueue_index_init",
				"directory": str(resolved),
				"dataqueue": str(dq_path),
				"rows": int(len(frame)),
			},
		)
		return index

	@classmethod
	def from_path(cls, dataqueue_path: Path) -> "DataqueueIndex":
		resolved = Path(dataqueue_path).resolve()
		frame = cls._load_dataqueue(resolved)
		if frame is None or frame.empty:
			raise FileNotFoundError(f"DataqueueIndex: no readable dataqueue at {resolved}")
		return cls(resolved.parent, resolved, frame)

	@staticmethod
	def _find_dataqueue(directory: Path) -> Optional[Path]:
		candidates = sorted(directory.glob(settings.timeline.dataqueue_glob))
		return candidates[0] if candidates else None

	@staticmethod
	def _load_dataqueue(path: Path) -> Optional[pd.DataFrame]:
		try:
			frame = pd.read_csv(path, low_memory=False)
		except Exception as exc:  # pragma: no cover - defensive
			logger.warning(
				"DataqueueIndex: failed to read dataqueue",
				extra={"phase": "dataqueue_index_load", "path": str(path), "error": repr(exc)},
			)
			return None

		queue_col = settings.timeline.queue_column
		if queue_col not in frame.columns:
			logger.warning(
				"DataqueueIndex: dataqueue missing queue_elapsed column",
				extra={"phase": "dataqueue_index_load", "path": str(path)},
			)
			return None

		frame = frame.copy()
		frame[queue_col] = pd.to_numeric(frame[queue_col], errors="coerce")
		frame.dropna(subset=[queue_col], inplace=True)
		frame.reset_index(drop=True, inplace=True)
		return frame

	@staticmethod
	def _prepare_queue_series(series: Optional[pd.Series]) -> pd.Series:
		if series is None:
			return pd.Series(dtype=np.float64)
		prepared = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
		return prepared.astype(np.float64)

	# ------------------------------------------------------------------
	# Public accessors
	# ------------------------------------------------------------------
	@property
	def source_path(self) -> Path:
		return self._dataqueue_path

	@property
	def directory(self) -> Path:
		return self._directory

	def dataframe(self) -> pd.DataFrame:
		"""Return a shallow copy of the underlying dataqueue frame."""

		return self._frame.copy()

	def device_ids(self) -> Sequence[str]:
		if "device_id" not in self._frame.columns:
			return ()
		ids = self._frame["device_id"].dropna().astype(str)
		# Preserve order of first appearance
		return tuple(dict.fromkeys(ids))

	def queue_series(self, selector: Optional[DeviceSelector] = None) -> pd.Series:
		"""Return queue_elapsed samples filtered by an optional device selector."""

		if selector is None:
			return self._queue_all.copy()

		mask = self._resolve_selector(selector)
		if mask is None:
			return pd.Series(dtype=np.float64)
		queue_col = settings.timeline.queue_column
		series = pd.to_numeric(self._frame.loc[mask, queue_col], errors="coerce")
		series = series.dropna().reset_index(drop=True)
		return series.astype(np.float64)

	def slice(self, selector: DeviceSelector) -> TimelineSlice:
		"""Return a filtered subset of the dataqueue rows matching the selector."""

		mask = self._resolve_selector(selector)
		if mask is None:
			rows = self._frame.iloc[0:0].copy()
		else:
			rows = self._frame.loc[mask].copy()
		rows.reset_index(drop=True, inplace=True)
		return TimelineSlice(rows=rows, selector=selector)

	def absolute_for_device(self, device_pattern: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
		"""Return (elapsed_s, absolute_iso) arrays for devices matching pattern, or None if empty."""
		device_slice = self.slice(
			lambda ids: ids.str.contains(device_pattern, case=False, na=False, regex=False)
		)
		elapsed = device_slice.queue_elapsed().to_numpy(dtype=np.float64)
		absolute = device_slice.packet_absolute().to_numpy(dtype=str)
		if elapsed.size == 0 or absolute.size == 0:
			return None
		origin = elapsed[0] if elapsed.size else 0.0
		return elapsed - origin, absolute

	# ------------------------------------------------------------------
	# General utilities
	# ------------------------------------------------------------------
	@staticmethod
	def fit_affine(x: Iterable[float], y: Iterable[float]) -> tuple[float, float, float]:
		"""Fit an affine transform y ≈ a * x + b and return (a, b, r²)."""

		x_arr = np.asarray(list(x), dtype=np.float64)
		y_arr = np.asarray(list(y), dtype=np.float64)

		if x_arr.size < 2 or y_arr.size < 2:
			raise ValueError("At least two points are required to fit an affine transform")

		X = np.vstack([x_arr, np.ones_like(x_arr)]).T
		slope, intercept = np.linalg.lstsq(X, y_arr, rcond=None)[0]

		predicted = slope * x_arr + intercept
		residual = y_arr - predicted
		ss_res = float(np.sum(residual ** 2))
		ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2)) if len(y_arr) > 1 else 0.0
		r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

		return float(slope), float(intercept), float(r2)

	@staticmethod
	def relative(series: Iterable[float]) -> np.ndarray:
		"""Produce a series relative to its first finite value."""

		values = np.asarray(list(series), dtype=np.float64)
		finite = values[np.isfinite(values)]
		if finite.size == 0:
			return values
		origin = float(finite[0])
		return values - origin

	@staticmethod
	def fix_32bit_wraparound(values: Iterable[int]) -> np.ndarray:
		"""Correct unsigned 32-bit wraparound for monotonically increasing counters.

		A wrap is only registered when the backward step exceeds half the
		counter range (2**31). Small backward steps caused by clock jitter,
		duplicate or slightly out-of-order samples are left alone — otherwise
		every µs of jitter would falsely add 2**32 (~4295 s) to the tail of
		the series.
		"""

		arr = np.asarray(list(values), dtype=np.int64)
		if arr.size == 0:
			return arr

		wrap_value = 2 ** 32
		wrap_threshold = 2 ** 31
		corrected = arr.copy()
		offset = 0
		for idx in range(1, len(arr)):
			if arr[idx - 1] - arr[idx] > wrap_threshold:
				offset += wrap_value
			corrected[idx] = arr[idx] + offset
		return corrected

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------
	def _resolve_selector(self, selector: DeviceSelector) -> Optional[pd.Series]:
		if "device_id" not in self._frame.columns:
			return None

		device_ids = self._frame["device_id"].astype(str)

		if callable(selector):
			raw_mask = selector(device_ids)
			if isinstance(raw_mask, pd.Series):
				mask = raw_mask.reindex(device_ids.index, fill_value=False)
			else:
				mask = pd.Series(raw_mask, index=device_ids.index)
		elif isinstance(selector, str):
			mask = device_ids == selector
		else:
			candidates = [str(item) for item in selector]
			mask = device_ids.isin(candidates)

		mask = mask.fillna(False).astype(bool)
		return mask
