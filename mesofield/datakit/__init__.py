"""Datakit package entry point."""

from __future__ import annotations

import os

if os.getenv("DATAKIT_SAFE_MODE") == "1":
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")

from ._utils._logger import get_logger

logger = get_logger("datakit")

from ._version import __version__, build_meta, get_version  # noqa: E402
from .config import settings  # noqa: E402
from .core import Dataset, inspect_sources, load, load_dataset, load_path  # noqa: E402
from .datamodel import LoadedStream  # noqa: E402
from .explore import explore  # noqa: E402
from .profile import MaterializedMemoryReport, profile_materialized  # noqa: E402
from .shell import open_shell  # noqa: E402
from .sources.register import DataSource, LoadContext  # noqa: E402

__all__ = [
    "Dataset",
    "DataSource",
    "LoadContext",
    "LoadedStream",
    "MaterializedMemoryReport",
    "__version__",
    "build_meta",
    "explore",
    "get_logger",
    "get_version",
    "inspect_sources",
    "load",
    "load_dataset",
    "load_path",
    "logger",
    "open_shell",
    "profile_materialized",
    "settings",
]
