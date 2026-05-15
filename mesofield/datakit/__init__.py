"""Convenience bootstrap for the ``datakit`` package.

The module wires up a namespaced logger and re-exports high-level entry
points so exploratory notebooks can simply ``import datakit`` and start
interacting with :class:`datakit.experiment.ExperimentData` without extra
ceremony.
"""

from __future__ import annotations

import os

if os.getenv("DATAKIT_SAFE_MODE") == "1":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from ._utils._logger import get_logger

logger = get_logger("datakit")

from .experiment import ExperimentData, ExperimentMetadata  # noqa: E402  (import after logger to avoid circular import)
from .discover import discover_manifest  # noqa: E402  (import after logger to avoid circular import)
from .loader import ExperimentStore  # noqa: E402  (import after logger to avoid circular import)
from .config import settings  # noqa: E402  (import after logger to avoid circular import)


__all__ = [
    "logger",
    "get_logger",
    "ExperimentData",
    "ExperimentMetadata",
    "discover_manifest",
    "ExperimentStore",
    "settings",
]
