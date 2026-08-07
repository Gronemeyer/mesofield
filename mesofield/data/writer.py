"""OME-TIFF + MP4 writers for MDASequences.

Two MDA output handlers, both plain classes exposing the ``sequenceStarted`` /
``frameReady`` / ``sequenceFinished`` methods ``CMMCorePlus.run_mda`` connects by
name, and both emitting the ``<filename>_frame_metadata.json`` sidecar every
downstream mesofield parser (and the AcquisitionManifest ``metadata_path``) reads:

- :class:`OMEWriter` -- OME-TIFF, backed by the maintained ``ome-writers``
  library (incremental, flushed, no giant memmap pre-allocation). This replaced
  a memmap-based writer that accumulated ~9 GB of dirty pages in a 400 s
  dual-camera run and overflowed MMCore's circular buffer.
- :class:`CV2Writer` -- MP4/AVI via ``cv2.VideoWriter``.

Neither depends on ``pymmcore_plus.mda.handlers`` (deprecated upstream in favour
of ``ome-writers``). :class:`NullWriter` is a disk-free :class:`OMEWriter`
subclass used only for benchmarking.
"""

from typing import Any

from useq import MDAEvent

import numpy as np
from pathlib import Path
import json

FRAME_MD_FILENAME = "_frame_metadata.json"

# Codec selection lives in mesofield.data.codecs (the single source of truth,
# shared with the config wizard). Imported here for the CV2Writer below.
from mesofield.data.codecs import (  # noqa: E402
    configure_opencv_codec,
    default_fourcc,
    open_video_writer,
)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, object: Any) -> Any:
        if isinstance(object, MDAEvent):
            return None #ignore the MDAEvents for now
        return super().default(object)


_OME_SUFFIXES = (".ome.tiff", ".ome.tif", ".tiff", ".tif")


def _stringify_meta(meta: Any) -> dict[str, str]:
    """Flatten a per-frame metadata mapping to string-only values.

    ome-writers stores ``frame_metadata`` in an OME ``Map`` whose values must be
    strings. Scalars are stringified; nested/container values are JSON-encoded
    (via :class:`CustomJSONEncoder`, ``default=str`` for anything exotic).
    """
    out: dict[str, str] = {}
    for key, value in (meta or {}).items():
        if isinstance(value, str):
            out[str(key)] = value
        else:
            try:
                out[str(key)] = json.dumps(value, cls=CustomJSONEncoder, default=str)
            except Exception:
                out[str(key)] = str(value)
    return out


class OMEWriter:
    """OME-TIFF writer backed by the maintained ``ome-writers`` library.

    Drop-in replacement for :class:`CustomWriter` as an MDA output handler
    (``run_mda(output=...)``): it exposes the three signal handlers the runner
    connects by name (``sequenceStarted`` / ``frameReady`` / ``sequenceFinished``)
    and drives an :class:`ome_writers.OMEStream`.

    Why this exists (measured, not assumed): the memmap-based
    :class:`CustomWriter` pre-allocates the full multi-GB OME-TIFF and writes
    into it via ``numpy.memmap`` with no flush, so dirty pages accumulate in RAM
    (≈9 GB in a 400 s dual-camera run) until flush stalls back up MMCore's
    circular buffer and it overflows. ``ome-writers`` writes incrementally with
    real flushing and no giant up-front allocation, and is the path pymmcore-plus
    is migrating to (``pymmcore_plus.mda.handlers`` is deprecated).

    The mesofield contract is preserved: the ``<filename>_frame_metadata.json``
    sidecar every downstream parser (and the AcquisitionManifest ``metadata_path``)
    reads is still emitted from :meth:`finalize_metadata`, built from the same
    per-frame pymmcore-plus metadata, accumulated here exactly as the deprecated
    ``_5DWriterBase`` did.
    """

    def __init__(self, filename: Path | str) -> None:
        self._filename = str(filename)
        # Split off the OME/TIFF suffix so ome-writers reconstructs the exact
        # same output path (root_path + format.suffix == self._filename).
        self._root_path = self._filename
        self._suffix = ".ome.tiff"
        for suf in _OME_SUFFIXES:
            if self._filename.lower().endswith(suf):
                self._root_path = self._filename[: -len(suf)]
                self._suffix = suf
                break
        self._frame_metadata_filename = self._filename + FRAME_MD_FILENAME

        self._stream: Any = None
        self._sequence: Any = None
        # position key -> list of per-frame metadata (mirrors _5DWriterBase)
        from collections import defaultdict

        self.frame_metadatas: "defaultdict[str, list]" = defaultdict(list)
        self._position_key_map: dict[int, str] = {}

        # Off by default -- embedding per-frame metadata in the OME-XML does not
        # scale (see ``frameReady``). Opt in for small runs only.
        import os

        self._embed_frame_meta = bool(os.getenv("MESOFIELD_EMBED_FRAME_META"))

    # --- MDA signal handlers (connected by name by mda_listeners_connected) ---
    def sequenceStarted(self, seq: Any, meta: Any = None) -> None:
        self._sequence = seq
        self.frame_metadatas.clear()
        self._position_key_map.clear()

    def frameReady(self, frame: np.ndarray, event: MDAEvent, meta: Any) -> None:
        if self._stream is None:
            self._open_stream(frame)
        # By default do NOT embed per-frame metadata into the OME-XML. ome-writers
        # accumulates one OME ``Map`` per plane and serialises all of them at
        # ``close()``; for a long acquisition (e.g. 60k frames) that is a multi-GB,
        # multi-minute spike on the MDA worker thread at teardown that can OOM-kill
        # the process (silently, natively) -- especially across back-to-back runs.
        # The JSON sidecar written in ``finalize_metadata`` already holds the full,
        # unflattened per-frame metadata, so embedding is redundant. It can be
        # re-enabled for small runs via MESOFIELD_EMBED_FRAME_META=1.
        if self._embed_frame_meta:
            self._stream.append(frame, frame_metadata=_stringify_meta(meta))
        else:
            self._stream.append(frame)
        self.frame_metadatas[self._position_key(event)].append(meta or {})

    def sequenceFinished(self, seq: Any) -> None:
        # Write the sidecar FIRST: it is the source of truth for per-frame
        # metadata, and it must survive even if the ome-writers ``close()`` flush
        # fails or the process dies during it.
        self.finalize_metadata()
        if self._stream is not None:
            try:
                self._stream.close()
            finally:
                self._stream = None

    # --- helpers ------------------------------------------------------------
    def _open_stream(self, frame: np.ndarray) -> None:
        import ome_writers as ow

        h, w = int(frame.shape[-2]), int(frame.shape[-1])
        dims = ow.dims_from_useq(self._sequence, image_width=w, image_height=h)
        settings = ow.AcquisitionSettings(
            root_path=self._root_path,
            dimensions=dims,
            dtype=str(frame.dtype),
            format=ow.OmeTiffFormat(suffix=self._suffix),
            overwrite=True,
        )
        self._stream = ow.create_stream(settings)

    def _position_key(self, event: MDAEvent) -> str:
        pos_index = event.index.get("p", 0)
        if pos_index not in self._position_key_map:
            key = getattr(event, "pos_name", None) or f"p{pos_index}"
            self._position_key_map[pos_index] = key
        return self._position_key_map[pos_index]

    def finalize_metadata(self) -> None:
        """Write the per-frame metadata sidecar (mesofield's legacy contract)."""
        regular_dict = dict(self.frame_metadatas)
        json_str = json.dumps(regular_dict, indent=4, cls=CustomJSONEncoder)
        with open(self._frame_metadata_filename, "w") as fh:
            fh.write(json_str)


class NullWriter(OMEWriter):
    """Disk-free :class:`OMEWriter` used only for benchmarking.

    Runs the identical MDA drain path (``frameReady`` is still called per frame
    and metadata still accumulates) but never opens an ome-writers stream and
    never writes the sidecar, so no bytes hit disk and no dirty pages build up.
    Comparing a null-writer run against a real run isolates how much backlog/RAM
    the write path itself contributes.

    Enabled via ``MESOFIELD_NULL_WRITER=1`` (see ``BaseCamera._make_writer``);
    never selected in normal operation.
    """

    def frameReady(self, frame: np.ndarray, event: MDAEvent, meta: Any) -> None:
        # Accumulate metadata like the real path, but write nothing to disk.
        self.frame_metadatas[self._position_key(event)].append(meta or {})

    def sequenceFinished(self, seq: Any) -> None:
        # No stream to close, no sidecar to emit.
        return None


class CV2Writer:
    """Write frames to an mp4/avi video using OpenCV.

    Standalone MDA output handler -- it exposes the three signal methods the
    runner connects by name (``sequenceStarted`` / ``frameReady`` /
    ``sequenceFinished``, mirroring :class:`OMEWriter`) and drives
    ``cv2.VideoWriter`` directly, so it no longer depends on the deprecated
    ``pymmcore_plus.mda.handlers.OMETiffWriter``.

    Two usage modes share the same codec/fourcc/metadata logic:

    - **MDA-driven** (``sequenceStarted`` / ``frameReady`` / ``sequenceFinished``)
      when handed to ``CMMCorePlus.run_mda`` as an output handler.
    - **Direct** (``begin`` / ``add_frame`` / ``finish``) for cameras that run
      their own capture loop (e.g. :class:`OpenCVCamera`).

    Both modes emit the same ``<filename>_frame_metadata.json`` sidecar.
    """

    def __init__(self, filename: Path | str, fps: int = 30, fourcc: str | None = None) -> None:
        configure_opencv_codec()

        self._filename = str(filename)
        if not self._filename.endswith((".mp4", ".avi")):
            raise ValueError("filename must end with '.mp4' or '.avi'")
        self._fps = fps
        # ``None`` -> portable platform default (honours MESOFIELD_FOURCC).
        self._fourcc = fourcc if fourcc else default_fourcc(self._filename)
        # FFmpeg expects H.264-in-MP4 with the 'avc1' tag; OpenCV often gets
        # 'H264' from callers, which triggers a noisy fallback warning.
        if self._filename.endswith(".mp4") and self._fourcc.upper() == "H264":
            self._fourcc = "avc1"
        self._frame_metadata_filename = self._filename + FRAME_MD_FILENAME
        # Direct-use (non-MDA) capture-loop writer; opened by `begin`.
        self._direct_writer: Any = None

        # MDA-mode state (one cv2.VideoWriter per stage position).
        from collections import defaultdict

        self._sequence: Any = None
        self._writers: dict[str, Any] = {}
        self.frame_metadatas: "defaultdict[str, list]" = defaultdict(list)
        self._position_key_map: dict[int, str] = {}

    # --- MDA signal handlers (connected by name by mda_listeners_connected) ---
    def sequenceStarted(self, seq: Any, meta: Any = None) -> None:
        self._sequence = seq
        self.frame_metadatas.clear()
        self._position_key_map.clear()
        self._writers.clear()

    def frameReady(self, frame: np.ndarray, event: MDAEvent, meta: Any) -> None:
        import cv2

        key = self._position_key(event)
        writer = self._writers.get(key)
        if writer is None:
            is_color = frame.ndim == 3 and frame.shape[-1] in (3, 4)
            height, width = int(frame.shape[0]), int(frame.shape[1])
            writer, self._fourcc = open_video_writer(
                self._fname_for_position(key), self._fourcc, self._fps,
                (width, height), is_color,
            )
            self._writers[key] = writer

        if frame.dtype != np.uint8:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        writer.write(frame)
        self.frame_metadatas[key].append(meta or {})

    def sequenceFinished(self, seq: Any) -> None:
        for writer in self._writers.values():
            try:
                writer.release()
            except Exception:
                pass
        self._writers.clear()
        self.finalize_metadata()

    # --- helpers ------------------------------------------------------------
    def _position_key(self, event: MDAEvent) -> str:
        pos_index = event.index.get("p", 0)
        if pos_index not in self._position_key_map:
            key = getattr(event, "pos_name", None) or f"p{pos_index}"
            self._position_key_map[pos_index] = key
        return self._position_key_map[pos_index]

    def _fname_for_position(self, position_key: str) -> str:
        """Per-position filename; single-position runs keep the base name."""
        if (seq := self._sequence) and seq.sizes.get("p", 1) > 1:
            fname = self._filename.replace(".mp4", f"_{position_key}.mp4")
            return fname.replace(".avi", f"_{position_key}.avi")
        return self._filename

    def finalize_metadata(self) -> None:
        regular_dict = dict(self.frame_metadatas)
        json_str = json.dumps(regular_dict, indent=4, cls=CustomJSONEncoder)
        with open(self._frame_metadata_filename, "w") as file:
            file.write(json_str)

    # ----- direct (non-MDA) capture-loop interface ----------------------
    def begin(self, width: int, height: int, is_color: bool = True) -> None:
        """Open the underlying ``cv2.VideoWriter`` for a self-driven loop."""
        Path(self._filename).parent.mkdir(parents=True, exist_ok=True)
        self._direct_writer, self._fourcc = open_video_writer(
            self._filename, self._fourcc, self._fps, (width, height), is_color
        )

    def add_frame(self, frame: np.ndarray) -> None:
        """Write one frame to the direct-mode video (uint8 frames pass through)."""
        if self._direct_writer is None:
            raise RuntimeError("CV2Writer.add_frame called before begin()")
        if frame.dtype != np.uint8:
            import cv2

            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self._direct_writer.write(frame)

    def finish(self, extra_metadata: dict | None = None) -> None:
        """Release the direct-mode writer and write the metadata sidecar."""
        if self._direct_writer is not None:
            try:
                self._direct_writer.release()
            except Exception:
                pass
            self._direct_writer = None

        payload: dict[str, Any] = {"frame_metadatas": dict(self.frame_metadatas)}
        if extra_metadata:
            payload.update(extra_metadata)
        json_str = json.dumps(payload, indent=4, cls=CustomJSONEncoder)
        with open(self._frame_metadata_filename, "w") as file:
            file.write(json_str)
 