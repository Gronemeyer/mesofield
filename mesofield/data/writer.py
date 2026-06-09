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

if TYPE_CHECKING:
    from pymmcore_plus.mda.metadata import SummaryMetaV1  # type: ignore

from pymmcore_plus.mda.handlers import OMETiffWriter
from useq import MDAEvent

import numpy as np
from pathlib import Path
import json

FRAME_MD_FILENAME = "_frame_metadata.json"

# ─── H264 Video Codec ─────────────────────────────────────────────────────
# OpenH264 codec DLL for OpenCV video encoding (Windows only).
# The DLL lives at <repo-root>/external/video-codecs/.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CODEC_DIRECTORY = str(_REPO_ROOT / "external" / "video-codecs")
OPENH264_DLL_PATH = str(Path(CODEC_DIRECTORY) / "openh264-1.8.0-win64.dll")
# ─────────────────────────────────────────────────────────────────

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


def configure_opencv_codec() -> None:
    """Configure environment so OpenCV can locate the bundled OpenH264 DLL.

    Safe to call repeatedly. Silences OpenCV/FFMPEG logging and prepends the
    project's ``external/video-codecs`` directory to PATH / DLL search.
    """
    import os

    # Set environment variables to suppress OpenCV/FFMPEG output BEFORE importing cv2
    os.environ.setdefault('OPENCV_LOG_LEVEL', 'SILENT')
    os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'loglevel;quiet')
    os.environ.setdefault('OPENCV_VIDEOIO_DEBUG', '0')

    try:
        import cv2  # noqa: F401
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "opencv-python is required. Please `pip install opencv-python`."
        ) from e

    # OpenCV log silencing is version-dependent.
    try:
        if hasattr(cv2, 'setLogLevel'):
            cv2.setLogLevel(0)  # 0 = Silent
        elif (
            hasattr(cv2, 'utils')
            and hasattr(cv2.utils, 'logging')
            and hasattr(cv2.utils.logging, 'setLogLevel')
        ):
            cv2.utils.logging.setLogLevel(0)
    except Exception:
        pass

    os.environ['OPENH264_LIBRARY'] = OPENH264_DLL_PATH
    if CODEC_DIRECTORY not in os.environ.get('PATH', ''):
        os.environ['PATH'] = CODEC_DIRECTORY + os.pathsep + os.environ.get('PATH', '')
    if hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(CODEC_DIRECTORY)
        except (OSError, FileNotFoundError):
            pass


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

    def __init__(self, filename: Path | str, fps: int = 30, fourcc: str = "H264") -> None:
        configure_opencv_codec()

        self._filename = str(filename)
        if not self._filename.endswith((".mp4", ".avi")):
            raise ValueError("filename must end with '.mp4' or '.avi'")
        self._fps = fps
        self._fourcc = fourcc
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

    def new_array(self, position_key: str, dtype: np.dtype, sizes: dict[str, int]):
        import cv2

        width = sizes["x"]
        height = sizes["y"]
        is_color = sizes.get("c", 1) > 1

        if (seq := self.current_sequence) and seq.sizes.get("p", 1) > 1:
            fname = self._filename.replace(".mp4", f"_{position_key}.mp4")
            fname = fname.replace(".avi", f"_{position_key}.avi")
        else:
            fname = self._filename

        fourcc = cv2.VideoWriter.fourcc(*self._fourcc)
        writer = cv2.VideoWriter(fname, fourcc, self._fps, (width, height), isColor=is_color)
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
        import cv2

        Path(self._filename).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter.fourcc(*self._fourcc)
        writer = cv2.VideoWriter(
            self._filename, fourcc, float(self._fps), (width, height), isColor=is_color
        )
        if not writer.isOpened():
            raise RuntimeError(
                f"cv2.VideoWriter failed to open '{self._filename}' "
                f"(fourcc={self._fourcc}, fps={self._fps})"
            )
        self._direct_writer = writer

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
 