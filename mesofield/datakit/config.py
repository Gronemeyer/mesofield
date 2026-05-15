"""Centralized configuration for ``datakit`` infrastructure.

The file gathers all tweakable defaults (source metadata keys, dataset layout,
timeline parsing, debug helpers, etc.) into a single, human-readable module so
new contributors can understand *which* knobs exist without spelunking through
call sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


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
    desired_tags: Tuple[str, ...] = (
        "meso_mean",
        "mesomap",
        "timestamps",
        "dataqueue",
        # "psychopy",
        "treadmill",
        "wheel",
        "notes",
        "session_config",
        "meso_metadata",
        "pupil_metadata",
        "pupil_dlc",
        "suite2p",
    )
    logical_name_overrides: Dict[str, str] = field(default_factory=lambda: {
        "meso_metadata": "meso_meta",
        "pupil_metadata": "pupil_meta",
        "pupil_dlc": "pupil",
        "wheel": "encoder",
        "dataqueue": "time",
    })
    meta_columns: Tuple[str, ...] = ("Subject", "Session", "Task", "Source", "Key", "Value", "dtype")
    scope_key: str = "scope"
    session_scope: str = "session"
    experiment_scope: str = "experiment"
    time_basis_key: str = "time_basis"


@dataclass
class TimelineDefaults:
    """Options that describe how we discover and parse timeline CSV files."""

    dataqueue_glob: str = "*_dataqueue.csv"
    queue_column: str = "queue_elapsed"
    window_device_patterns: Tuple[str, ...] = ("dhyana", "mesoscope")


@dataclass
class DebugDefaults:
    """Hard-coded fixture shortcuts used across debugging notebooks/scripts."""

    sample_experiment_rel: Tuple[str, ...] = ("tests", "sample_experiment3")
    default_entry_index: int = 0


@dataclass
class Settings:
    """Container aggregating all configuration namespaces for callers."""

    sources: SourceMetaDefaults = field(default_factory=SourceMetaDefaults)
    dataset: DatasetLayout = field(default_factory=DatasetLayout)
    timeline: TimelineDefaults = field(default_factory=TimelineDefaults)
    debug: DebugDefaults = field(default_factory=DebugDefaults)


settings = Settings()


__all__ = ["settings"]
