"""Base classes for loading datakit sources."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from mesofield.datakit.config import settings
from ..datamodel import LoadedStream


_DOCUMENT_FLAG = "__datakit_document__"


def document(method):
    """Mark a method so its docstring is captured into stream ``meta['docs']``."""
    setattr(method, _DOCUMENT_FLAG, True)
    return method


@dataclass(frozen=True)
class LoadContext:
    """Context object passed to every ``DataSource.load`` call.

    Carries identity (subject/session/task), the inventory row for that cell
    (so sources can locate sibling files via :meth:`path_for`), and any
    upstream sources that were loaded for the same cell as declared on
    :attr:`DataSource.requires`.

    For backward parity with previous releases, when ``"dataqueue"`` is
    present in :attr:`dependencies`, the convenience attributes
    :attr:`dataqueue_frame`, :attr:`dataqueue_meta`, :attr:`master_timeline`,
    and :attr:`experiment_window` are populated from it. New sources should
    prefer reading from :attr:`dependencies` directly.
    """

    subject: str
    session: str
    task: str | None
    inventory_row: Mapping[str, Any]
    dependencies: Mapping[str, "LoadedStream | None"] = field(default_factory=dict)
    master_timeline: np.ndarray | None = None
    experiment_window: tuple[float, float] | None = None
    dataqueue_frame: pd.DataFrame | None = None
    dataqueue_meta: Mapping[str, Any] | None = None

    def path_for(self, tag: str) -> Path | None:
        """Return the path stored in the inventory row for ``tag``, or None."""
        value = self.inventory_row.get(tag)
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return Path(str(value))

    def require_path(self, tag: str) -> Path:
        """Like :meth:`path_for` but raises ``FileNotFoundError`` when missing."""
        path = self.path_for(tag)
        if path is None:
            raise FileNotFoundError(
                f"Missing '{tag}' path for ({self.subject}, {self.session}, {self.task})"
            )
        return path

    def get_dependency(self, tag: str) -> "LoadedStream | None":
        """Return a previously-loaded dependency stream, or None if unavailable."""
        return self.dependencies.get(tag)

    def require_dependency(self, tag: str) -> "LoadedStream":
        """Like :meth:`get_dependency` but raises if the dependency is missing."""
        dep = self.dependencies.get(tag)
        if dep is None:
            raise RuntimeError(
                f"Missing required dependency '{tag}' for "
                f"({self.subject}, {self.session}, {self.task})"
            )
        return dep


class DataSource:
    """Base class for a file-backed data source."""

    tag: ClassVar[str]
    patterns: ClassVar[Iterable[str]]
    camera_tag: ClassVar[str | None] = None
    is_timeseries: ClassVar[bool] = True
    flatten_payload: ClassVar[bool] = True
    requires: ClassVar[Tuple[str, ...]] = ()
    """Tag names of upstream sources whose loaded streams should be made
    available via ``LoadContext.dependencies``. Soft contract: a missing or
    failed dependency yields ``None`` in ``dependencies[tag]``; sources are
    responsible for either degrading gracefully or raising."""

    def load(self, path: Path, *, context: LoadContext | None = None) -> LoadedStream:
        """Load data from the given path."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement load()")

    def _require_context(self, context: LoadContext | None) -> LoadContext:
        if context is None:
            raise ValueError(
                f"{self.__class__.__name__} requires a LoadContext; call through Dataset"
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
        docs = {
            name: inspect.getdoc(attr)
            for name, attr in inspect.getmembers(type(self), callable)
            if getattr(attr, _DOCUMENT_FLAG, False)
        }
        if docs:
            meta_dict.setdefault("docs", docs)
        return meta_dict


class TimeseriesSource(DataSource):
    """Base class for time-indexed sources."""

    is_timeseries: ClassVar[bool] = True

    def build_timeseries(
        self,
        path: Path,
        *,
        context: LoadContext | None = None,
    ) -> tuple[np.ndarray, Any, Dict[str, Any]]:
        """Return (timeline, value, meta)."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement build_timeseries()")

    def load(self, path: Path, *, context: LoadContext | None = None) -> LoadedStream:
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
        context: LoadContext | None = None,
    ) -> tuple[np.ndarray, Any, Dict[str, Any]]:
        """Return (timeline, value, meta)."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement build_table()")

    def load(self, path: Path, *, context: LoadContext | None = None) -> LoadedStream:
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
        context: LoadContext | None = None,
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Return an intervals table with start/stop columns and meta."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement build_intervals()")

    def load(self, path: Path, *, context: LoadContext | None = None) -> LoadedStream:
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


__all__ = [
    "LoadContext",
    "DataSource",
    "TimeseriesSource",
    "TableSource",
    "IntervalSeriesSource",
]
