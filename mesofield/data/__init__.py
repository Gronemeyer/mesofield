"""Acquisition-time data management.

Centralises three responsibilities:

- :mod:`~mesofield.data.manager` orchestrates per-run data collection,
  notes, and timestamp writing.
- :mod:`~mesofield.data.writer` defines the OME-TIFF handler
  (:class:`OMEWriter`, backed by ome-writers) and the MP4 (:class:`CV2Writer`)
  frame handler.
- :mod:`~mesofield.data.batch` provides batch / post-hoc utilities used
  by analysis scripts.

The writer classes are re-exported from this package so that
``from mesofield.data import OMEWriter`` continues to work in existing
experiment scripts.
"""

try:
    from .writer import CV2Writer, OMEWriter, NullWriter
except ImportError:  # optional deps not installed (analysis-only env)
    CV2Writer = None  # type: ignore[assignment,misc]
    OMEWriter = None  # type: ignore[assignment,misc]
    NullWriter = None  # type: ignore[assignment,misc]

__all__ = ["CV2Writer", "OMEWriter", "NullWriter"]
