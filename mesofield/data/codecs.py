"""OpenCV video defaults — the single source of truth for fourcc and the
platform-dependent capture backend / pixel format.

Kept deliberately free of heavyweight imports (no ``pymmcore_plus``, ``cv2``
only imported lazily inside functions) so that both the acquisition writer
(:mod:`mesofield.data.writer`) / camera (:mod:`mesofield.devices.cameras`) and
the config wizard (:mod:`mesofield.gui.config_builder`) can import it without
dragging in the acquisition stack. Change a default here and every surface
follows.

OpenCV's FFMPEG ``VideoWriter`` needs a fourcc code. The portable choice — the
*only* codec bundled in every opencv-python wheel on Windows, Linux, and macOS
with no external library — is MPEG-4 (``mp4v`` in ``.mp4``, ``MJPG`` in
``.avi``). H.264 compresses better but needs extra pieces a plain
``pip install`` lacks (the Cisco OpenH264 DLL on Windows, libx264 on Linux), so
it is opt-in and falls back to the portable codec at runtime.

Selection precedence (see :func:`default_fourcc`): explicit caller/config value
-> ``MESOFIELD_FOURCC`` env var -> portable default for the container.
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Capture backend / pixel format (live view, not the saved file) ───────
# OpenCV capture backends offered in the wizard. Names map to ``cv2.CAP_<NAME>``.
CV_BACKENDS = ["ANY", "AVFOUNDATION", "MSMF", "DSHOW", "V4L2"]
# Capture pixel formats (CAP_PROP_FOURCC) offered in the wizard. "" = leave the
# camera default. This is the format the *camera delivers*, distinct from the
# writer codec below.
CAP_FOURCC_CHOICES = ["", "MJPG", "YUY2"]


def default_cv_backend() -> str:
    """Platform default capture backend.

    Windows uses DSHOW: it's the most reliable for USB webcams. MSMF (the other
    Windows option) frequently opens a camera that then delivers no frames.
    """
    return {"darwin": "AVFOUNDATION", "win32": "DSHOW"}.get(sys.platform, "V4L2")


def default_cap_fourcc() -> str:
    """Platform default capture pixel format.

    Windows USB webcams under DSHOW/MSMF typically deliver no frames in their
    default (raw/YUY2) mode and need MJPG forced; elsewhere the camera default
    is fine.
    """
    return "MJPG" if sys.platform == "win32" else ""


# ─── Writer codec (the saved file) ────────────────────────────────────────
_PORTABLE_FOURCC = {".mp4": "mp4v", ".avi": "MJPG"}
DEFAULT_FOURCC = "mp4v"

# Codecs offered in the config wizard's fourcc dropdown. 'mp4v' leads because
# it's the portable default; H264/avc1 are opt-in and fall back to mp4v.
FOURCC_CHOICES = ["mp4v", "MJPG", "H264", "avc1", "XVID"]

# OpenH264 DLL for H.264 encoding (Windows). Point ``MESOFIELD_OPENH264_LIBRARY``
# at a downloaded ``openh264-*.dll`` to enable H.264; otherwise we fall back to
# a repo-local copy if present (dev checkouts) and leave OpenCV to its own
# defaults when neither exists.
_REPO_CODEC_DIR = Path(__file__).resolve().parent.parent.parent / "external" / "video-codecs"
_REPO_OPENH264_DLL = _REPO_CODEC_DIR / "openh264-1.8.0-win64.dll"


def openh264_dll_path() -> Path | None:
    """Locate an OpenH264 DLL, or ``None`` if no usable copy is found."""
    env = os.environ.get("MESOFIELD_OPENH264_LIBRARY")
    if env and Path(env).is_file():
        return Path(env)
    if _REPO_OPENH264_DLL.is_file():
        return _REPO_OPENH264_DLL
    return None


def default_fourcc(filename: str) -> str:
    """Codec to use when a caller doesn't specify one.

    Honours ``MESOFIELD_FOURCC`` (a global override), else picks the portable
    codec matching the container extension.
    """
    env = os.environ.get("MESOFIELD_FOURCC")
    if env:
        return env
    ext = Path(filename).suffix.lower()
    return _PORTABLE_FOURCC.get(ext, DEFAULT_FOURCC)


def open_video_writer(
    filename: str,
    fourcc: str,
    fps: float,
    size: tuple[int, int],
    is_color: bool,
):
    """Open a ``cv2.VideoWriter``, falling back to a portable codec.

    Returns ``(writer, fourcc_used)``. If the requested fourcc can't be opened
    (e.g. H.264 on a box without the codec), retries once with the portable
    codec for the container and logs a warning naming the swap. Raises
    ``RuntimeError`` only if even the fallback fails to open.
    """
    import cv2

    def _try(code: str):
        w = cv2.VideoWriter(filename, cv2.VideoWriter.fourcc(*code), float(fps), size, isColor=is_color)
        return w if w.isOpened() else (w.release() or None)

    writer = _try(fourcc)
    if writer is not None:
        return writer, fourcc

    ext = Path(filename).suffix.lower()
    fallback = _PORTABLE_FOURCC.get(ext, DEFAULT_FOURCC)
    if fallback.upper() != fourcc.upper():
        writer = _try(fallback)
        if writer is not None:
            logger.warning(
                "cv2.VideoWriter could not open '%s' with fourcc '%s'; fell back "
                "to '%s'. Set the device 'fourcc' or MESOFIELD_FOURCC to silence "
                "this, or MESOFIELD_OPENH264_LIBRARY to enable H.264.",
                filename, fourcc, fallback,
            )
            return writer, fallback

    raise RuntimeError(
        f"cv2.VideoWriter failed to open '{filename}' with fourcc '{fourcc}' "
        f"and fallback '{fallback}' (fps={fps}, size={size})."
    )


def configure_opencv_codec() -> None:
    """Quiet OpenCV/FFMPEG logging and wire up the OpenH264 DLL if available.

    Safe to call repeatedly. Always silences OpenCV/FFMPEG logging. Only touches
    PATH / ``OPENH264_LIBRARY`` / the DLL search path when a usable OpenH264 DLL
    is actually found (via ``MESOFIELD_OPENH264_LIBRARY`` or a dev checkout);
    otherwise it leaves OpenCV's own codec resolution untouched.
    """
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

    dll = openh264_dll_path()
    if dll is None:
        return
    codec_dir = str(dll.parent)
    os.environ['OPENH264_LIBRARY'] = str(dll)
    if codec_dir not in os.environ.get('PATH', ''):
        os.environ['PATH'] = codec_dir + os.pathsep + os.environ.get('PATH', '')
    if hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(codec_dir)
        except (OSError, FileNotFoundError):
            pass
