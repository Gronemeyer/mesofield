"""Helpers that translate discovery manifests into tabular inventories.

These functions provide the "glue" between file-system discovery and the
MultiIndex DataFrame consumed by downstream loaders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from .datamodel import ManifestEntry
from .discover import discover_manifest
from .config import settings

IndexKey = tuple[str, str] | tuple[str, str, str | None]


def entries_to_inventory(
    manifest_entries: Iterable[ManifestEntry],
    *,
    root: Path,
    prefer_processed: bool = True,
    absolute_paths: bool = True,
    include_task_level: bool | None = None,
) -> pd.DataFrame:
    """Convert manifest entries into a wide DataFrame indexed by subject/session[/task]."""

    entries = list(manifest_entries)
    configured_index_names = list(settings.dataset.index_names)
    if len(configured_index_names) < 2:
        raise ValueError("dataset.index_names must include at least subject and session labels")
    subject_name, session_name = configured_index_names[:2]
    task_name = configured_index_names[2] if len(configured_index_names) > 2 else "Task"

    if not entries:
        index_names = [subject_name, session_name]
        if include_task_level:
            index_names.append(task_name)
        empty_index = pd.MultiIndex.from_tuples([], names=index_names)
        return pd.DataFrame(index=empty_index)

    if include_task_level is None:
        include_task = any(entry.task is not None for entry in entries)
    else:
        include_task = include_task_level

    root = Path(root)
    records: Dict[IndexKey, Dict[str, str]] = {}
    origins: Dict[IndexKey, Dict[str, str]] = {}

    for entry in entries:
        key: IndexKey
        if include_task:
            key = (entry.subject, entry.session, entry.task)
        else:
            key = (entry.subject, entry.session)

        if key not in records:
            records[key] = {}
            origins[key] = {}

        resolved_path = str((root / entry.path) if absolute_paths else entry.path)
        current_origin = origins[key].get(entry.tag)

        if prefer_processed:
            if current_origin == "processed" and entry.origin != "processed":
                continue
            if entry.origin == "processed":
                records[key][entry.tag] = resolved_path
                origins[key][entry.tag] = "processed"
                continue

        if current_origin is None or not prefer_processed or current_origin != "processed":
            records[key][entry.tag] = resolved_path
            origins[key][entry.tag] = entry.origin

    if not records:
        index_names = [subject_name, session_name]
        if include_task:
            index_names.append(task_name)
        empty_index = pd.MultiIndex.from_tuples([], names=index_names)
        return pd.DataFrame(index=empty_index)

    index_names = [subject_name, session_name] + ([task_name] if include_task else [])
    def sort_key(item: tuple[IndexKey, Dict[str, str]]) -> tuple[str, ...]:
        raw_key = item[0]
        return tuple("" if component is None else str(component) for component in raw_key)

    sorted_items = sorted(records.items(), key=sort_key)
    data = {key: value for key, value in sorted_items}
    data_frame = pd.DataFrame.from_dict(data, orient="index")
    data_frame.index = pd.MultiIndex.from_tuples(data_frame.index, names=index_names)
    data_frame.sort_index(inplace=True)
    return data_frame


def build_inventory(
    experiment_dir: Path,
    *,
    prefer_processed: bool = True,
    absolute_paths: bool = True,
    include_task_level: bool | None = None,
) -> pd.DataFrame:
    """Convenience wrapper combining discovery and inventory construction."""

    manifest = discover_manifest(experiment_dir)
    return entries_to_inventory(
        manifest.entries,
        root=manifest.root,
        prefer_processed=prefer_processed,
        absolute_paths=absolute_paths,
        include_task_level=include_task_level,
    )
