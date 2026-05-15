"""Reconstruct AcquisitionManifests for legacy mesofield sessions.

Mesofield only started writing a `mesokit_schema.AcquisitionManifest` once
the `Procedure._write_acquisition_manifest` hook landed. Sessions acquired
before that have no manifest, which means downstream tools (datakit,
databench) cannot ingest them through the contract path.

This module walks a BIDS-laid-out session directory (or an experiment
root containing many sessions) and synthesizes a manifest from what is
already on disk:

  - subject / session from the BIDS path (`sub-X/ses-Y/`)
  - task from the filename suffix (`..._task-Z_...`)
  - per-producer `output_path`, `bids_type`, `file_type` from the file tree
  - `started_at` / `ended_at` from the session's `*_timestamps.csv`
  - frame-metadata sidecars (`*_frame_metadata.json`) attached to their tiff

Fields that the legacy filesystem does not carry (hardware calibration,
mesofield_version, per-producer sampling_rate) are written as their
empty/default values. The manifest still gates ingest, but downstream
analyses that need calibration will need it filled in by hand.

Multi-task sessions get one manifest per task, written as
`manifest_task-<T>.json`; single-task sessions write `manifest.json`.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from mesokit_schema import (
    AcquisitionManifest,
    ProducerEntry,
    SessionIdentity,
    TimeBasis,
)


_BIDS_FILE_RE = re.compile(
    r"^(?P<timestamp>\d{8}_\d{6})"
    r"_sub-(?P<subject>[^_]+)"
    r"_ses-(?P<session>[^_]+)"
    r"_task-(?P<task>[^_]+)"
    r"_(?P<suffix>[^.]+)"
    r"\.(?P<extension>.+)$"
)

# Filenames whose BIDS suffix marks them as session metadata, not producer data.
_SESSION_LEVEL_SUFFIXES: frozenset[str] = frozenset({
    "configuration", "notes", "timestamps",
})

_SIDECAR_TAIL = "_frame_metadata.json"

# Default values for fields the legacy filesystem doesn't carry.
LEGACY_VERSION = "legacy"
LEGACY_CLOCK_SOURCE = "wall_unix_s"


def discover_sessions(root: Path) -> Iterator[Path]:
    """Yield session directories (`.../sub-X/ses-Y/`) under `root`.

    Accepts either a session dir, an experiment root containing `data/`,
    or any ancestor that contains BIDS-laid-out sessions.
    """
    root = Path(root).resolve()
    if _looks_like_session(root):
        yield root
        return

    # Common layouts: <exp>/data/sub-*/ses-*  or  <exp>/sub-*/ses-*
    for pattern in ("data/sub-*/ses-*", "sub-*/ses-*", "*/sub-*/ses-*"):
        matches = sorted(root.glob(pattern))
        if matches:
            for path in matches:
                if _looks_like_session(path):
                    yield path
            return


def _looks_like_session(path: Path) -> bool:
    return (
        path.is_dir()
        and path.name.startswith("ses-")
        and path.parent.name.startswith("sub-")
    )


def synthesize_manifests(session_dir: Path) -> dict[str, AcquisitionManifest]:
    """Build one manifest per task found under `session_dir`.

    Returns a {task: AcquisitionManifest} mapping. Single-task sessions
    return a one-entry dict; the caller decides the filename.
    """
    session_dir = Path(session_dir).resolve()
    subject = session_dir.parent.name.removeprefix("sub-")
    session = session_dir.name.removeprefix("ses-")

    files = [p for p in session_dir.rglob("*") if p.is_file()]
    sidecars = _index_sidecars(files)
    timestamps_path = _find_session_file(files, "timestamps")
    started_at, ended_at = _read_timestamps(timestamps_path) if timestamps_path else (None, None)
    config_path = _find_session_file(files, "configuration")
    config = _read_configuration(config_path) if config_path else {}

    # Partition producer files by task.
    by_task: dict[str, list[tuple[Path, re.Match[str]]]] = defaultdict(list)
    for path in files:
        m = _BIDS_FILE_RE.match(path.name)
        if not m:
            continue
        suffix = m.group("suffix")
        if suffix in _SESSION_LEVEL_SUFFIXES:
            continue
        if path.name.endswith(_SIDECAR_TAIL):
            continue
        by_task[m.group("task")].append((path, m))

    if not by_task:
        return {}

    fallback_started_at = started_at or _earliest_mtime(files)
    out: dict[str, AcquisitionManifest] = {}
    for task, entries in by_task.items():
        producers = [
            _producer_for(path, match, session_dir, sidecars)
            for path, match in sorted(entries, key=lambda pair: pair[0].name)
        ]
        out[task] = AcquisitionManifest(
            mesofield_version=LEGACY_VERSION,
            acquisition_complete=True,
            started_at=fallback_started_at,
            ended_at=ended_at,
            session=SessionIdentity(
                subject=subject,
                session=session,
                task=task,
                experimenter=config.get("experimenter"),
                protocol=config.get("protocol"),
            ),
            producers=producers,
        )
    return out


def manifest_filename(task: str, multi_task: bool) -> str:
    """Decide the on-disk filename for a synthesized manifest."""
    return f"manifest_task-{task}.json" if multi_task else "manifest.json"


# ---------------------------------------------------------------------------
# Helpers


def _index_sidecars(files: list[Path]) -> dict[Path, Path]:
    """Map each data file to its sibling `_frame_metadata.json` sidecar, if any."""
    index: dict[Path, Path] = {}
    for path in files:
        if not path.name.endswith(_SIDECAR_TAIL):
            continue
        base = path.name[: -len(_SIDECAR_TAIL)]
        for candidate in path.parent.iterdir():
            if (
                candidate.is_file()
                and candidate is not path
                and candidate.name.startswith(base + ".")
            ):
                index[candidate] = path
                break
    return index


def _find_session_file(files: list[Path], suffix: str) -> Optional[Path]:
    """Locate a session-level file by its BIDS suffix (e.g. 'timestamps')."""
    for path in files:
        m = _BIDS_FILE_RE.match(path.name)
        if m and m.group("suffix") == suffix:
            return path
    return None


def _read_timestamps(path: Path) -> tuple[Optional[datetime], Optional[datetime]]:
    """Return (start, stop) UTC-aware datetimes from a legacy timestamps.csv.

    The file is written by `DataSaver.save_timestamps` with columns
    ``device_id, started, stopped`` (older revisions used ``id, start_time,
    stop_time`` — both are accepted). The first row is the procedure-level
    timing; subsequent rows are per-device. Legacy values may be
    naive-local; we convert to UTC via `astimezone`.
    """
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            start = row.get("started") or row.get("start_time")
            stop = row.get("stopped") or row.get("stop_time")
            return _parse_dt(start), _parse_dt(stop)
    return None, None


def _read_configuration(path: Path) -> dict[str, str]:
    """Pull a flat `{Parameter: Value}` dict from a session configuration.csv."""
    out: dict[str, str] = {}
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (row.get("Parameter") or "").strip()
            val = (row.get("Value") or "").strip()
            if key:
                out[key] = val
    return out


def _parse_dt(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    raw = raw.strip()
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        # Treat naive as system-local and convert to UTC.
        dt = dt.astimezone(timezone.utc)
    return dt


def _earliest_mtime(files: list[Path]) -> datetime:
    """Fallback session start time when timestamps.csv is missing."""
    if not files:
        return datetime.now(timezone.utc)
    earliest = min(p.stat().st_mtime for p in files)
    return datetime.fromtimestamp(earliest, tz=timezone.utc)


def _producer_for(
    path: Path,
    match: re.Match[str],
    session_dir: Path,
    sidecars: dict[Path, Path],
) -> ProducerEntry:
    rel = path.resolve().relative_to(session_dir.resolve())
    bids_type = rel.parent.name if str(rel.parent) != "." else None
    sidecar_rel: Optional[str] = None
    if path in sidecars:
        sidecar_rel = str(sidecars[path].resolve().relative_to(session_dir.resolve()))
    suffix = match.group("suffix")
    return ProducerEntry(
        device_id=suffix,
        device_type=_guess_device_type(suffix, match.group("extension")),
        data_type=suffix,
        bids_type=bids_type,
        file_type=match.group("extension"),
        output_path=str(rel),
        metadata_path=sidecar_rel,
        sampling_rate_hz=None,
        time_basis=TimeBasis(clock_source=LEGACY_CLOCK_SOURCE),
        calibration={},
    )


def _guess_device_type(suffix: str, extension: str) -> str:
    if extension.endswith(("tiff", "tif", "mp4", "avi")):
        return "camera"
    if any(token in suffix for token in ("encoder", "wheel", "treadmill")):
        return "encoder"
    if "nidaq" in suffix:
        return "nidaq"
    return "device"
