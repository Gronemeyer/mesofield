"""Acquisition-time data management.

Centralises three responsibilities:

- :mod:`~mesofield.data.manager` orchestrates per-run data collection,
  notes, and timestamp writing.
- :mod:`~mesofield.data.writer` defines the OME-TIFF (:class:`CustomWriter`)
  and MP4 (:class:`CV2Writer`) frame handlers.
- :mod:`~mesofield.data.batch` provides batch / post-hoc utilities used
  by analysis scripts.

``CustomWriter`` and ``CV2Writer`` are re-exported from this package so
that ``from mesofield.data import CustomWriter`` continues to work in
existing experiment scripts.
"""

try:
    from .writer import CustomWriter, CV2Writer
except ImportError:  # pymmcore-plus not installed (analysis-only env)
    CustomWriter = None  # type: ignore[assignment,misc]
    CV2Writer = None  # type: ignore[assignment,misc]

__all__ = ["CustomWriter", "CV2Writer"]
