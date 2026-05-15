"""Common data structures shared across the ``datakit`` pipeline.

The module groups the frozen dataclasses that represent discovered files,
aligned timelines, and high-level dataset bundles so that explorers can see
"what flows through the system" in one place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


StreamKind = str


@dataclass(frozen=True)
class StreamPayload:
    """Typed wrapper around arbitrary stream values.

    Each payload records its ``kind`` (table/array/mapping/sequence/scalar), the
    raw ``data`` object, and lightweight ``attrs`` that describe shape or column
    metadata. Persisting this metadata makes it easier to reconstruct the
    original Python object on reload.
    """

    kind: StreamKind
    data: Any
    attrs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attrs", dict(self.attrs or {}))

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def table(cls, frame: pd.DataFrame, attrs: Optional[Dict[str, Any]] = None) -> "StreamPayload":
        base_attrs = {
            "columns": list(frame.columns),
            "index_name": frame.index.name,
        }
        if attrs:
            base_attrs.update(attrs)
        return cls(kind="table", data=frame, attrs=base_attrs)

    @classmethod
    def array(cls, array: np.ndarray, attrs: Optional[Dict[str, Any]] = None) -> "StreamPayload":
        base_attrs = {
            "dtype": str(array.dtype),
            "shape": tuple(int(x) for x in array.shape),
        }
        if attrs:
            base_attrs.update(attrs)
        return cls(kind="array", data=array, attrs=base_attrs)

    @classmethod
    def mapping(cls, mapping: Mapping[str, Any], attrs: Optional[Dict[str, Any]] = None) -> "StreamPayload":
        base_attrs = {
            "keys": list(mapping.keys()),
        }
        if attrs:
            base_attrs.update(attrs)
        return cls(kind="mapping", data=dict(mapping), attrs=base_attrs)

    @classmethod
    def sequence(cls, seq: Sequence[Any], attrs: Optional[Dict[str, Any]] = None) -> "StreamPayload":
        base_attrs = {
            "length": len(seq),
        }
        if attrs:
            base_attrs.update(attrs)
        return cls(kind="sequence", data=list(seq), attrs=base_attrs)

    @classmethod
    def scalar(cls, value: Any, attrs: Optional[Dict[str, Any]] = None) -> "StreamPayload":
        return cls(kind="scalar", data=value, attrs=dict(attrs or {}))

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    @property
    def is_table(self) -> bool:
        return self.kind == "table"

    @property
    def is_array(self) -> bool:
        return self.kind == "array"

    @property
    def is_mapping(self) -> bool:
        return self.kind == "mapping"

    @property
    def is_sequence(self) -> bool:
        return self.kind == "sequence"

    @classmethod
    def ensure(cls, value: Any, *, kind: Optional[StreamKind] = None, attrs: Optional[Dict[str, Any]] = None) -> "StreamPayload":
        if isinstance(value, StreamPayload):
            if attrs:
                merged = dict(value.attrs)
                merged.update(attrs)
                return StreamPayload(kind=value.kind, data=value.data, attrs=merged)
            return value

        detected_kind = kind

        if detected_kind is None:
            if isinstance(value, pd.DataFrame):
                detected_kind = "table"
            elif isinstance(value, np.ndarray):
                detected_kind = "array"
            elif isinstance(value, Mapping):
                detected_kind = "mapping"
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                detected_kind = "sequence"
            else:
                detected_kind = "scalar"

        if detected_kind == "table":
            frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
            return cls.table(frame, attrs)
        if detected_kind == "array":
            array = value if isinstance(value, np.ndarray) else np.asarray(value)
            return cls.array(array, attrs)
        if detected_kind == "mapping":
            if not isinstance(value, Mapping):
                raise TypeError("Cannot coerce value to mapping payload")
            return cls.mapping(dict(value), attrs)
        if detected_kind == "sequence":
            sequence = value if isinstance(value, Sequence) else list(value)
            return cls.sequence(sequence, attrs)
        return cls.scalar(value, attrs)

    @property
    def data_view(self) -> Any:
        """Return the canonical data representation (DataFrame, ndarray, etc)."""

        if self.kind == "table" and not isinstance(self.data, pd.DataFrame):
            return pd.DataFrame(self.data)
        if self.kind == "array" and not isinstance(self.data, np.ndarray):
            return np.asarray(self.data)
        if self.kind == "mapping" and not isinstance(self.data, dict):
            return dict(self.data)
        if self.kind == "sequence" and not isinstance(self.data, list):
            return list(self.data)
        return self.data


@dataclass(frozen=True)
class ManifestEntry:
    """Single file discovered during manifest building."""

    tag: str          # e.g. "meso_metadata"
    path: str
    origin: str       # "data" | "processed"
    subject: str      # e.g. "STREHAB07"
    session: str      # e.g. "05"
    task: str | None = None  # e.g. "widefield"


@dataclass(frozen=True)
class LoadedStream:
    """Hydrated data stream with timestamps and metadata."""

    tag: str
    t: np.ndarray  # seconds, float64, strictly increasing
    value: object  # array or domain object
    meta: dict
    _payload: StreamPayload = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        payload = StreamPayload.ensure(self.value)
        object.__setattr__(self, "t", np.asarray(self.t, dtype=np.float64))
        object.__setattr__(self, "meta", dict(self.meta or {}))
        object.__setattr__(self, "_payload", payload)
        object.__setattr__(self, "value", payload.data_view)

    @property
    def payload(self) -> StreamPayload:
        return self._payload

    @property
    def data(self) -> Any:
        return self._payload.data_view


@dataclass(frozen=True)
class AlignedStream:
    """Stream aligned to the master clock with a normalized payload."""

    t: np.ndarray
    payload: StreamPayload

    def __post_init__(self) -> None:
        object.__setattr__(self, "t", np.asarray(self.t, dtype=np.float64))

    @property
    def data(self) -> Any:
        return self.payload.data_view

    @property
    def kind(self) -> StreamKind:
        return self.payload.kind


@dataclass(frozen=True)
class CameraTimeline:
    """Timeline for a specific camera including frame indexes."""
    t_cam: np.ndarray
    frame_index: np.ndarray


@dataclass(frozen=True)
class AlignmentReport:
    """Small summary of how well a stream aligned to the master clock."""
    a: float           # slope of linear fit
    b: float           # intercept of linear fit
    r2: float          # coefficient of determination
    max_resid_s: float # maximum residual in seconds
    anchors: int       # number of anchor points used


# ---------------------------------------------------------------------------
# Procedural workflow helpers for the simplified API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Manifest:
    """Serializable manifest of discovered files and where they live."""

    root: Path
    entries: list[ManifestEntry]
    manifest_path: Path | None = None

    def grouped(
        self,
        *,
        include_task: bool = False,
    ) -> Dict[tuple[str, str] | tuple[str, str, str | None], list[ManifestEntry]]:
        """Group manifest entries by (subject, session[, task])."""
        grouped: Dict[tuple[str, str] | tuple[str, str, str | None], list[ManifestEntry]] = {}
        for entry in self.entries:
            key: tuple[str, str] | tuple[str, str, str | None]
            if include_task:
                key = (entry.subject, entry.session, entry.task)
            else:
                key = (entry.subject, entry.session)
            grouped.setdefault(key, []).append(entry)
        return grouped

    def select(
        self,
        subjects: Iterable[str] | None = None,
        sessions: Iterable[str] | None = None,
        tasks: Iterable[str] | None = None,
    ) -> "Manifest":
        """Return a filtered manifest copy limited to the requested subjects/sessions/tasks."""
        if not subjects and not sessions and not tasks:
            return self

        subject_set = {s for s in subjects or []}
        session_set = {s for s in sessions or []}
        task_set = {t for t in tasks or []}

        filtered: list[ManifestEntry] = []
        for entry in self.entries:
            subject_match = not subject_set or entry.subject in subject_set
            session_match = not session_set or entry.session in session_set
            task_match = not task_set or entry.task in task_set
            if subject_match and session_match and task_match:
                filtered.append(entry)

        return Manifest(root=self.root, entries=filtered, manifest_path=self.manifest_path)


