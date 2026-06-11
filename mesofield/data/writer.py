"""Custom OME.TIFF + MP4 writers for MDASequences.

`CustomWriter` extends pymmcore-plus's public :class:`OMETiffWriter` with two
additions:

1. ``bigtiff=True`` on the underlying tifffile call -- mesoscope acquisitions
   routinely exceed the classic-TIFF 4 GiB ceiling.
2. A per-frame ``<filename>_frame_metadata.json`` sidecar emitted from
   :meth:`finalize_metadata`, containing the same metadata pymmcore-plus
   accumulates internally. This is the legacy ``mesofield`` contract every
   downstream parser already reads.

The sidecar JSON is the current source of truth for per-frame metadata. The
broader goal (tracked separately) is to push the same fields into the OME-XML
embedded in the TIFF itself so the JSON becomes supplementary and redundant;
see the TODO in :meth:`OMETiffWriter._sequence_metadata` upstream.

`CV2Writer` subclasses the public :class:`OMETiffWriter` purely to reuse its
inherited ``frameReady`` plumbing (the machinery that turns MMCamera/MDA signals
into ``new_array`` / ``write_frame`` / ``store_frame_metadata`` calls and
accumulates pymmcore-plus metadata). It overrides every TIFF-specific method to
emit MP4/AVI instead -- there is no public MP4 handler in pymmcore-plus to
inherit from, and inheriting the public ``OMETiffWriter`` avoids depending on
pymmcore-plus's private ``_5d_writer_base`` module.
"""

from datetime import timedelta
from typing import TYPE_CHECKING, Any
import warnings

if TYPE_CHECKING:
    from pymmcore_plus.mda.metadata import SummaryMetaV1  # type: ignore

from pymmcore_plus.mda.handlers import OMETiffWriter
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

class CustomWriter(OMETiffWriter):
    """OME-TIFF writer extending pymmcore-plus's :class:`OMETiffWriter`.

    Two divergences from the public base:

    - Uses ``bigtiff=True`` so multi-GiB mesoscope acquisitions write cleanly.
    - Emits the per-frame metadata JSON sidecar mesofield's downstream parsers
      (and the AcquisitionManifest's ``metadata_path``) depend on.

    Everything else -- filename validation, frame writing, OME-XML sequence
    metadata, memmap handling -- is inherited from :class:`OMETiffWriter`.
    """

    def __init__(self, filename: Path | str) -> None:
        super().__init__(filename)
        self._frame_metadata_filename = self._filename + FRAME_MD_FILENAME

    def new_array(
        self, position_key: str, dtype: np.dtype, sizes: dict[str, int]
    ) -> np.memmap:
        """Mirror :meth:`OMETiffWriter.new_array` but with ``bigtiff=True``.

        Upstream's implementation hardcodes the ``imwrite`` call; we duplicate
        it here to flip the ``bigtiff`` flag. Keep the bodies in sync if a
        pymmcore-plus bump changes the upstream version.
        """
        from tifffile import imwrite, memmap

        dims, shape = zip(*sizes.items())

        metadata: dict[str, Any] = self._sequence_metadata()
        metadata["axes"] = "".join(dims).upper()

        if (seq := self.current_sequence) and seq.sizes.get("p", 1) > 1:
            ext = ".ome.tif" if self._is_ome else ".tif"
            fname = self._filename.replace(ext, f"_{position_key}{ext}")
        else:
            fname = self._filename

        imwrite(
            fname,
            shape=shape,
            bigtiff=True,
            dtype=dtype,
            metadata=metadata,
            imagej=not self._is_ome,
            ome=self._is_ome,
        )

        mmap = memmap(fname, dtype=dtype)
        mmap.shape = shape
        return mmap  # type: ignore

    def finalize_metadata(self) -> None:
        """Write the per-frame metadata sidecar.

        Called by ``OMETiffWriter.sequenceFinished`` after the last frame.
        Serialises ``self.frame_metadatas`` (the dict pymmcore-plus accumulates
        for us in ``frameReady``) to JSON at ``<filename>_frame_metadata.json``.
        """
        regular_dict = dict(self.frame_metadatas)
        json_str = json.dumps(regular_dict, indent=4, cls=CustomJSONEncoder)
        with open(self._frame_metadata_filename, "w") as fh:
            fh.write(json_str)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, object: Any) -> Any:
        if isinstance(object, MDAEvent):
            return None #ignore the MDAEvents for now
        return super().default(object)


class CV2Writer(OMETiffWriter):
    """Write frames to an mp4/avi video using OpenCV.

    Subclasses the public :class:`OMETiffWriter` only to reuse its inherited
    MDA-signal handling (``frameReady`` / ``sequenceStarted`` /
    ``sequenceFinished`` / ``store_frame_metadata`` and the
    ``frame_metadatas`` accumulation). Every TIFF-specific method
    (``__init__`` / ``new_array`` / ``write_frame`` / ``finalize_metadata``) is
    overridden below to emit video instead, so none of ``OMETiffWriter``'s
    tifffile machinery is ever reached.

    Two usage modes share the same codec/fourcc/metadata logic:

    - **MDA-driven** (``new_array`` / ``write_frame`` / ``finalize_metadata``)
      when handed to ``CMMCorePlus.run_mda`` as an output handler.
    - **Direct** (``begin`` / ``add_frame`` / ``finish``) for cameras that run
      their own capture loop (e.g. :class:`OpenCVCamera`).
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

        # `OMETiffWriter.sequenceStarted` only reorders position_sizes into
        # ImageJ axis order when `not self._is_ome`; setting it True makes that
        # override a pass-through to the base `sequenceStarted` (axis order is
        # irrelevant to video) and avoids an AttributeError.
        self._is_ome = True

        # Skip `OMETiffWriter.__init__` (it validates a .tif/.tiff filename and
        # imports tifffile); go straight to the base initializer to set up the
        # frameReady plumbing state (position_arrays, frame_metadatas, etc.).
        super(OMETiffWriter, self).__init__()

    def _codec_candidates(self) -> list[str]:
        """Return ordered codec candidates for the current output container."""
        primary = self._fourcc
        # MP4 fallback order favors compatibility when OpenH264 is unavailable.
        if self._filename.endswith(".mp4"):
            fallbacks = ["avc1", "H264", "mp4v"]
        else:
            fallbacks = ["XVID", "MJPG"]
        ordered = [primary] + [c for c in fallbacks if c.upper() != primary.upper()]
        return ordered

    def _open_writer(
        self, filename: str, fps: float, width: int, height: int, is_color: bool
    ) -> tuple[Any, str]:
        """Open cv2.VideoWriter using codec fallbacks; return (writer, used_fourcc)."""
        import cv2

        for code in self._codec_candidates():
            fourcc = cv2.VideoWriter.fourcc(*code)
            writer = cv2.VideoWriter(
                filename, fourcc, fps, (width, height), isColor=is_color
            )
            if writer.isOpened():
                if code.upper() != self._fourcc.upper():
                    warnings.warn(
                        (
                            f"VideoWriter fallback: requested fourcc={self._fourcc} "
                            f"but using {code} for '{filename}'."
                        ),
                        RuntimeWarning,
                        stacklevel=2,
                    )
                return writer, code
            try:
                writer.release()
            except Exception:
                pass
        raise RuntimeError(
            f"cv2.VideoWriter failed to open '{filename}' "
            f"(attempted fourcc={self._codec_candidates()}, fps={self._fps})"
        )

    def new_array(self, position_key: str, dtype: np.dtype, sizes: dict[str, int]):
        width = sizes["x"]
        height = sizes["y"]
        is_color = sizes.get("c", 1) > 1

        if (seq := self.current_sequence) and seq.sizes.get("p", 1) > 1:
            fname = self._filename.replace(".mp4", f"_{position_key}.mp4")
            fname = fname.replace(".avi", f"_{position_key}.avi")
        else:
            fname = self._filename

        writer, self._fourcc = open_video_writer(
            fname, self._fourcc, self._fps, (width, height), is_color
        )
        return writer

    def write_frame(self, ary: Any, index: tuple[int, ...], frame: np.ndarray) -> None:
        import cv2

        frame_8u = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ary.write(frame_8u)

    def finalize_metadata(self) -> None:
        for writer in self.position_arrays.values():
            try:
                writer.release()
            except Exception:
                pass

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
 