"""Robust readers for mesofield OME-TIFF stacks.

Single-position acquisitions are written by :class:`mesofield.data.writer.OMEWriter`
as one uncompressed, contiguous (Big)TIFF, so they memory-map with zero copy. The
one failure mode is a *truncated* file: if an acquisition is interrupted (crash,
kill, power loss), ome-writers has already written an OME-XML header declaring the
full planned plane count, but fewer planes actually reached disk. tifffile then
sees an inconsistent OME series and ``memmap`` raises ``image data are not
memory-mappable`` (with an ``index ... out of range`` warning).

:func:`memmap_ome_stack` hides that: it memmaps normally when the file is complete
and transparently falls back to reading exactly the frames present when it isn't.
"""

from __future__ import annotations

from typing import Any


def memmap_ome_stack(path: str, mode: str = "r") -> Any:
    """Memory-map an OME-TIFF frame stack, tolerant of truncated acquisitions.

    Returns a read-only ``numpy.memmap`` (zero-copy). For a complete file this is
    the normal OME series memmap (preserving its N-D shape). For a file whose
    acquisition was interrupted -- OME-XML over-declares the plane count -- it
    retries with ``is_ome=False``, which memmaps the frames actually on disk as a
    plain contiguous ``(frames, H, W)`` stack. Page order equals acquisition
    order, so it is a drop-in for single-position stacks.
    """
    import tifffile

    try:
        return tifffile.memmap(path, mode=mode)
    except (ValueError, IndexError):
        # OME series inconsistent (truncated). Bypass OME interpretation and
        # memmap the IFDs that were actually flushed.
        return tifffile.memmap(path, mode=mode, is_ome=False)
