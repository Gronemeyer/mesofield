"""File-system discovery helpers that produce machine-readable manifests."""

from __future__ import annotations

import re
from pathlib import Path

from ._utils._logger import get_logger
from .datamodel import Manifest, ManifestEntry
from .sources import SOURCE_REGISTRY

_BIDS_COMPONENT_PATTERN = re.compile(r"(sub|ses|task)-([A-Za-z0-9]+)", re.IGNORECASE)

logger = get_logger(__name__)


def discover_manifest(experiment_dir: Path | str) -> Manifest:
    """Return a :class:`Manifest` for ``experiment_dir`` by scanning registered sources."""

    root = Path(experiment_dir).resolve()
    entries = _discover_entries(root)
    if not entries:
        raise ValueError(f"No files discovered for {root}")
    entries.sort(key=lambda entry: (entry.tag, entry.path))
    return Manifest(root=root, entries=entries)


def _discover_entries(root: Path) -> list[ManifestEntry]:
    logger.info("Starting file discovery", extra={"phase": "discover", "experiment_dir": str(root)})

    entries: list[ManifestEntry] = []
    for tag in sorted(SOURCE_REGISTRY.keys()):
        logger.debug("Discovering files", extra={"phase": "discover", "tag": tag})
        source_class = SOURCE_REGISTRY[tag]

        for pattern in getattr(source_class, "patterns", ()):  # type: ignore[arg-type]
            for origin in ("processed", "data"):
                search_root = root / origin
                if not search_root.exists():
                    continue
                for file_path in search_root.glob(pattern):
                    rel_path = file_path.relative_to(root)
                    subject, session, task = _parse_components(rel_path, tag)
                    entries.append(
                        ManifestEntry(
                            tag=tag,
                            path=rel_path.as_posix(),
                            origin=origin,
                            subject=subject,
                            session=session,
                            task=task,
                        )
                    )
                    logger.debug(
                        "Found file",
                        extra={
                            "phase": "discover",
                            "tag": tag,
                            "path": rel_path.as_posix(),
                            "origin": origin,
                            "task": task,
                        },
                    )

    logger.info("Discovery complete", extra={"phase": "discover", "total_files": len(entries)})
    return entries


def _parse_components(relative_path: Path, tag: str) -> tuple[str, str, str | None]:
    captures: dict[str, str] = {}
    normalized_path = relative_path.as_posix()
    for label, value in _BIDS_COMPONENT_PATTERN.findall(normalized_path):
        captures.setdefault(label.lower(), value)

    parts = list(relative_path.parts)
    if parts and parts[0] in {"data", "processed"}:
        parts = parts[1:]

    subject = captures.get("sub")
    if subject is None and parts:
        candidate = parts[0]
        if candidate.startswith("sub-"):
            _, _, remainder = candidate.partition("-")
            subject = remainder or None
        else:
            subject = candidate

    session_value = captures.get("ses")
    session_token: str | None = None
    if session_value is not None:
        session_token = f"ses-{session_value}"
    else:
        for component in parts:
            if component.startswith("ses-"):
                session_token = component
                break

    task_value = captures.get("task")
    task_token: str | None = None
    if task_value is not None:
        task_token = f"task-{task_value}"
    else:
        match = re.search(r"task-([A-Za-z0-9]+)", normalized_path)
        if match:
            task_token = f"task-{match.group(1)}"
        else:
            task_token = "task-session"

    missing: list[str] = []
    if not subject:
        missing.append("subject")
    if not session_token:
        missing.append("session")
    if missing:
        raise ValueError(
            f"Missing {', '.join(missing)} token(s) in path '{normalized_path}' for tag '{tag}'"
        )

    assert subject is not None
    assert session_token is not None
    return subject, session_token, task_token


__all__ = ["discover_manifest"]