"""Centralized configuration for ``datakit``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SourceMetaDefaults:
    """Keys used to annotate stream metadata for all data sources."""

    meta_camera_key: str = "camera_tag"
    meta_timeseries_key: str = "is_timeseries"
    meta_source_key: str = "source_tag"
    meta_interval_key: str = "is_interval"


@dataclass
class DatasetLayout:
    """Shape and naming expectations for materialized experiment datasets."""

    index_names: Tuple[str, str, str] = ("Subject", "Session", "Task")
    scope_key: str = "scope"
    session_scope: str = "session"
    experiment_scope: str = "experiment"


@dataclass
class TimelineDefaults:
    """Options that describe how we discover and parse timeline CSV files."""

    dataqueue_glob: str = "*_dataqueue.csv"
    queue_column: str = "queue_elapsed"
    window_device_patterns: Tuple[str, ...] = ("dhyana", "mesoscope")


@dataclass
class Settings:
    """Container aggregating all configuration namespaces for callers."""

    sources: SourceMetaDefaults = field(default_factory=SourceMetaDefaults)
    dataset: DatasetLayout = field(default_factory=DatasetLayout)
    timeline: TimelineDefaults = field(default_factory=TimelineDefaults)


settings = Settings()


__all__ = ["settings"]
