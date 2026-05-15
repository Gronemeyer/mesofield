"""Base classes for loading datakit sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from mesofield.datakit.config import settings
from ..datamodel import LoadedStream


@dataclass(frozen=True)
class SourceContext:
    """Context for a DataSource load call.

    Provides quick access to sibling paths (e.g., dataqueue) based on the
    inventory row for the same subject/session/task.
    """

    subject: str
    session: str
    task: str | None
    inventory_row: Mapping[str, str]
    master_timeline: np.ndarray | None = None
    experiment_window: tuple[float, float] | None = None
    dataqueue_frame: pd.DataFrame | None = None
    dataqueue_meta: Mapping[str, Any] | None = None

    def path_for(self, tag: str) -> Path | None:
        value = self.inventory_row.get(tag)
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        return Path(str(value))

    def require_path(self, tag: str) -> Path:
        path = self.path_for(tag)
        if path is None:
            raise FileNotFoundError(
                f"Missing '{tag}' path for ({self.subject}, {self.session}, {self.task})"
            )
        return path


class DataSource:
    """Base class for a file-backed data source."""

    tag: ClassVar[str]
    patterns: ClassVar[Iterable[str]]
    camera_tag: ClassVar[str | None] = None
    is_timeseries: ClassVar[bool] = True
    flatten_payload: ClassVar[bool] = True

    def load(self, path: Path, *, context: SourceContext | None = None) -> LoadedStream:
        """Load data from the given path."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement load()")

    def _require_context(self, context: SourceContext | None) -> SourceContext:
        if context is None:
            raise ValueError(
                f"{self.__class__.__name__} requires SourceContext; call through ExperimentData"
            )
        return context

    def _decorate_meta(self, meta: Optional[Dict[str, Any]] = None, *, is_interval: bool = False) -> Dict[str, Any]:
        meta_dict: Dict[str, Any] = dict(meta or {})
        if self.camera_tag is not None:
            meta_dict.setdefault(settings.sources.meta_camera_key, self.camera_tag)
        meta_dict.setdefault(settings.sources.meta_timeseries_key, self.is_timeseries)
        meta_dict.setdefault(settings.sources.meta_source_key, self.tag)
        if is_interval:
            meta_dict.setdefault(settings.sources.meta_interval_key, True)
        return meta_dict


class TimeseriesSource(DataSource):
    """Base class for time-indexed sources."""

    is_timeseries: ClassVar[bool] = True

    def build_timeseries(
        self,
        path: Path,
        *,
        context: SourceContext | None = None,
    ) -> tuple[np.ndarray, Any, Dict[str, Any]]:
        """Return (timeline, value, meta)."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement build_timeseries()")

    def load(self, path: Path, *, context: SourceContext | None = None) -> LoadedStream:
        t, value, meta = self.build_timeseries(path, context=context)
        timeline = np.asarray(t, dtype=np.float64)
        return LoadedStream(tag=self.tag, t=timeline, value=value, meta=self._decorate_meta(meta))


class TableSource(DataSource):
    """Base class for static table sources."""

    is_timeseries: ClassVar[bool] = False

    def build_table(
        self,
        path: Path,
        *,
        context: SourceContext | None = None,
    ) -> tuple[np.ndarray, Any, Dict[str, Any]]:
        """Return (timeline, value, meta)."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement build_table()")

    def load(self, path: Path, *, context: SourceContext | None = None) -> LoadedStream:
        t, value, meta = self.build_table(path, context=context)
        timeline = np.asarray(t, dtype=np.float64)
        return LoadedStream(tag=self.tag, t=timeline, value=value, meta=self._decorate_meta(meta))


class IntervalSeriesSource(DataSource):
    """Base class for interval-based sources."""

    is_timeseries: ClassVar[bool] = True

    def build_intervals(
        self,
        path: Path,
        *,
        context: SourceContext | None = None,
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Return an intervals table with start/stop columns and meta."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement build_intervals()")

    def load(self, path: Path, *, context: SourceContext | None = None) -> LoadedStream:
        intervals, meta = self.build_intervals(path, context=context)
        if "start_s" not in intervals.columns or "stop_s" not in intervals.columns:
            raise ValueError("Intervals must include 'start_s' and 'stop_s' columns")
        timeline = pd.to_numeric(intervals["start_s"], errors="coerce").to_numpy(dtype=np.float64)
        return LoadedStream(
            tag=self.tag,
            t=timeline,
            value=intervals,
            meta=self._decorate_meta(meta, is_interval=True),
        )