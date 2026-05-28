"""Version + provenance metadata for ``datakit``.

This module resolves the running package's version from git tags when the
package is being executed from a working tree, and falls back to installed
package metadata or the on-disk ``VERSION`` file otherwise. The
``build_meta()`` helper produces a small dictionary that is embedded into
materialized datasets so that pickled artefacts can be traced back to the
exact source revision that produced them.
"""

from __future__ import annotations

import datetime as _dt
import platform as _platform
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

_PACKAGE_NAME = "datakit"
_REPO_ROOT = Path(__file__).resolve().parent.parent
_VERSION_FILE = _REPO_ROOT / "VERSION"


def _run_git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0:
        return None
    value = out.stdout.strip()
    return value or None


def _installed_version() -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version as _v
    except ImportError:  # pragma: no cover - py<3.8
        return None
    try:
        return _v(_PACKAGE_NAME)
    except PackageNotFoundError:
        return None


def _file_version() -> Optional[str]:
    try:
        return _VERSION_FILE.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _normalise_remote_url(url: str) -> str:
    """Convert a git remote URL to an https:// browsable form when possible."""

    if url.startswith(("http://", "https://")):
        return url[:-4] if url.endswith(".git") else url
    # git@github.com:owner/repo.git -> https://github.com/owner/repo
    m = re.match(r"^[^@]+@([^:]+):(.+?)(\.git)?$", url)
    if m:
        host, path, _ = m.groups()
        return f"https://{host}/{path}"
    return url


@lru_cache(maxsize=1)
def _git_info() -> Dict[str, Any]:
    if not (_REPO_ROOT / ".git").exists():
        return {}
    commit = _run_git("rev-parse", "HEAD")
    if commit is None:
        return {}
    short = _run_git("rev-parse", "--short", "HEAD")
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    describe = _run_git("describe", "--tags", "--always", "--dirty")
    tag = _run_git("describe", "--tags", "--exact-match")
    status = _run_git("status", "--porcelain")
    remote = _run_git("config", "--get", "remote.origin.url")
    dirty = bool(status)

    info: Dict[str, Any] = {
        "git_commit": commit,
        "git_commit_short": short or (commit[:7] if commit else None),
        "git_branch": None if branch == "HEAD" else branch,
        "git_describe": describe,
        "git_tag": tag,
        "git_dirty": dirty,
    }
    if remote:
        url = _normalise_remote_url(remote)
        info["git_url"] = url
        info["git_link"] = f"{url}/commit/{commit}" if commit else url
    return info


@lru_cache(maxsize=1)
def get_version() -> str:
    """Best-effort version string for the running ``datakit`` package."""

    info = _git_info()
    describe = info.get("git_describe")
    if describe:
        suffix = "+dirty" if info.get("git_dirty") and not describe.endswith("-dirty") else ""
        return f"{describe}{suffix}"
    return _installed_version() or _file_version() or "0.0.0+unknown"


def build_meta() -> Dict[str, Any]:
    """Return a snapshot of provenance metadata for the running package.

    ``built_at`` is regenerated on every call so embedded copies record the
    moment a dataset was materialized rather than when the module was imported.
    """

    meta: Dict[str, Any] = {
        "package": _PACKAGE_NAME,
        "version": get_version(),
        "version_file": _file_version(),
        "version_installed": _installed_version(),
        "python": sys.version.split()[0],
        "python_implementation": _platform.python_implementation(),
        "platform": _platform.platform(),
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    meta.update(_git_info())
    return meta


__version__ = get_version()


__all__ = ["__version__", "get_version", "build_meta"]
