import os
import concurrent.futures
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm
import tifffile

from scipy.ndimage import percentile_filter

# Defaults for sCMOS widefield with GCaMP8s — tune per setup.
DFF_DEFAULTS = dict(
    camera_offset=1602,    # validated from dark frames on 260422 in the dhyana-sensitivty experiment
    fs=49.9,
    percentile=8.0,
    window_seconds=60.0,    # 'collapse_first' only
    f0_floor=10.0,          # Rupprecht's "10 or 20 units" guard
    n_f0_samples=500,       # 'pixelwise' only — frames used for global F0
    chunk_frames=2000,      # 'pixelwise' only — temporal chunk for Pass 2
    fix_first_frame=True,  # replace frame 0 with a copy of frame 1 (hot-pixel/
                            # rolling-shutter artifact on the first acquired frame)
    max_frames=58227,       # optional cap for methods that should only process
                            # an initial segment of the stack
)


def _apply_first_frame_fix(arr, params):
    """If enabled and arr has >=2 frames along axis 0, overwrite arr[0] with arr[1]."""
    if params.get('fix_first_frame') and arr.shape[0] >= 2:
        arr[0] = arr[1]
    return arr

# Set environment variables to suppress OpenCV/FFMPEG output BEFORE importing cv2
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;quiet'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_FFMPEG'] = '0'

import cv2

# Set OpenCV logging to silent mode after import
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

# ─── H264 Video Codec ─────────────────────────────────────────────────────
# OpenH264 codec DLL for OpenCV video encoding (Windows only).
# The DLL lives at <repo-root>/external/video-codecs/.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CODEC_DIRECTORY = str(_REPO_ROOT / "external" / "video-codecs")
OPENH264_DLL_PATH = str(Path(CODEC_DIRECTORY) / "openh264-1.8.0-win64.dll")
# ─────────────────────────────────────────────────────────────────


# ─── Progress reporting ───────────────────────────────────────────────────
# Algorithm functions accept a `progress` callable and just call `progress(n)`
# after processing `n` frames. The default is a no-op, so algorithms stay
# unaware of whether progress is wired up. The orchestrator builds a single
# thread-safe reporter bound to a shared tqdm bar.

def _noop_progress(_n):
    pass


def chunked(total, chunk, progress=_noop_progress):
    """Yield (start, end) frame ranges and report progress after each chunk.

    Use in algorithms instead of a manual `range(0, T, chunk)` loop:

        for s, e in chunked(T, params['chunk_frames'], progress):
            ...work on frames[s:e]...
    """
    for s in range(0, total, chunk):
        e = min(s + chunk, total)
        yield s, e
        progress(e - s)


def _make_progress_reporter(total_work, desc, enabled=True):
    """Return (progress_fn, close_fn). progress_fn is thread-safe."""
    if not enabled:
        return _noop_progress, lambda: None
    pbar = tqdm(total=total_work, desc=desc, unit="frame",
                unit_scale=True, leave=False)
    lock = threading.Lock()

    def _update(n):
        with lock:
            pbar.update(n)

    return _update, pbar.close


# ─── ΔF/F algorithms ──────────────────────────────────────────────────────
# Each algorithm is a pure function (tiff_path, params, progress) -> np.ndarray.
# Register it in DFF_METHODS along with a `work(T, params)` estimator that
# returns the number of "frame units" the algorithm will process, so the
# orchestrator can size the progress bar without method-specific knowledge.

def _dff_collapse_first(tiff_path, params, progress=_noop_progress):
    """Method A: spatial mean → moving-percentile F0 → ΔF/F on the 1D trace.
    Cheap. Pixels weighted by photon yield."""
    tiff_array = tifffile.memmap(tiff_path)
    max_frames = params.get('max_frames')
    if max_frames is not None:
        tiff_array = tiff_array[:max_frames]
    T = tiff_array.shape[0]

    F = np.empty(T, dtype=np.float32)
    for s, e in chunked(T, params['chunk_frames'], progress):
        F[s:e] = np.mean(tiff_array[s:e], axis=(1, 2),
                         dtype=np.float64).astype(np.float32)

    # Suppress the first-frame acquisition artifact before any ΔF/F math,
    # otherwise the F0 estimator and the ratio at t=0 both get distorted.
    _apply_first_frame_fix(F, params)

    F -= params['camera_offset']
    w = max(3, int(round(params['window_seconds'] * params['fs'])))
    if w >= len(F):
        F0 = np.full_like(F, np.percentile(F, params['percentile']))
    else:
        F0 = percentile_filter(F, percentile=params['percentile'],
                               size=w, mode='nearest').astype(np.float32)
    F0 = np.maximum(F0, 0.0)
    return (F - F0) / (F0 + params['f0_floor'])


def _dff_pixelwise(tiff_path, params, progress=_noop_progress):
    """Method B: per-pixel global percentile F0 → ΔF/F → spatial mean.
    F0 from n_f0_samples temporally-distributed frames (cheap, accurate
    for slow drifts). No drift correction in F0 itself; detrend downstream
    if bleaching matters. Pixels weighted equally — amplifies low-SNR regions."""
    tiff_array = tifffile.memmap(tiff_path)
    T = tiff_array.shape[0]
    offset = params['camera_offset']

    # Pass 1: F0 from temporally subsampled stack (memory-bounded).
    # np.linspace(0, T-1, n) always includes index 0, so the first-frame
    # artifact would leak into the F0 percentile estimate — fix it here.
    n_samples = min(params['n_f0_samples'], T)
    sample_idx = np.linspace(0, T - 1, n_samples, dtype=int)
    sample_stack = np.asarray(tiff_array[sample_idx], dtype=np.float32)
    if sample_idx[0] == 0:
        _apply_first_frame_fix(sample_stack, params)
    sample_stack -= offset
    F0 = np.percentile(sample_stack, params['percentile'], axis=0)
    F0 = np.maximum(F0, 0.0).astype(np.float32)
    denom = F0 + params['f0_floor']
    del sample_stack
    progress(n_samples)

    # Pass 2: chunked ΔF/F → spatial mean per frame. Fix the first frame of
    # the very first block (before offset/ratio) so it propagates cleanly.
    out = np.empty(T, dtype=np.float32)
    for s, e in chunked(T, params['chunk_frames'], progress):
        block = np.asarray(tiff_array[s:e], dtype=np.float32)
        if s == 0:
            _apply_first_frame_fix(block, params)
        block -= offset
        out[s:e] = ((block - F0) / denom).mean(axis=(1, 2), dtype=np.float64)
    return out


# Registry: name -> (function, work-estimator). Add new algorithms here and
# they automatically participate in the shared progress bar.
DFF_METHODS = {
    'collapse_first': {
        'fn':   _dff_collapse_first,
        'work': lambda T, params: min(T, params['max_frames']) if params.get('max_frames') is not None else T,
    },
    'pixelwise': {
        'fn':   _dff_pixelwise,
        'work': lambda T, params: T + min(params['n_f0_samples'], T),
    },
}


def _tiff_frame_count(path):
    try:
        with tifffile.TiffFile(path) as tf:
            return len(tf.pages)
    except Exception:
        return int(tifffile.memmap(path).shape[0])


def dff_trace_from_tiff(tiff_paths, method='collapse_first',
                        show_progress=True, max_workers=None, **dff_params):
    """
    Compute ΔF/F traces concurrently across tiff files.

    method : key into DFF_METHODS (e.g. 'collapse_first', 'pixelwise').
    max_workers : cap this for 'pixelwise' if RAM-bound (each worker holds
                  ~ n_f0_samples * H * W * 4 bytes during Pass 1).
    **dff_params overrides DFF_DEFAULTS.

    Returns dict {tiff_path: np.ndarray (T,) float32}
    """
    params = {**DFF_DEFAULTS, **dff_params}
    spec = DFF_METHODS[method]
    worker, work_of = spec['fn'], spec['work']

    total_work = sum(work_of(_tiff_frame_count(p), params) for p in tiff_paths)
    progress, close = _make_progress_reporter(
        total_work, desc=f"ΔF/F ({method})", enabled=show_progress)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker, p, params, progress): p
                       for p in tiff_paths}
            results = {}
            for future in concurrent.futures.as_completed(futures):
                results[futures[future]] = future.result()
    finally:
        close()
    return results

def mean_trace_from_tiff(tiff_paths, show_progress=True, save=False):
    """
    Computes the mean traces for multiple TIFF files concurrently.
    Returns a dictionary mapping each file path to its mean trace.
    """
    import concurrent.futures
    
    def _compute_mean_trace(tiff_path):
        tiff_array = tifffile.memmap(tiff_path)
        return np.mean(tiff_array, axis=(1, 2))
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(_compute_mean_trace, path): path for path in tiff_paths}
        results = {}
        
        if show_progress:
            with tqdm(total=len(tiff_paths), desc="Computing mean traces", leave=False) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    path = futures[future]
                    results[path] = future.result()
                    pbar.update(1)
        else:
            for future in concurrent.futures.as_completed(futures):
                path = futures[future]
                results[path] = future.result()
    
    return results


def tiff_to_video(tiff_path: str,
                  output_path: str,
                  fps: int = 30,
                  output_format: str = "mp4",
                  use_color: bool = False,
                  show_progress: bool = True,
                  tqdm_position: int = 0):
    """
    Converts a multi-page TIFF stack to a video format.
    """

    tiff_array = tifffile.memmap(tiff_path)  # shape -> (num_frames, height, width) or (num_frames, height, width, channels)
    
    num_frames = tiff_array.shape[0]
    height = tiff_array.shape[1]
    width = tiff_array.shape[2] if not use_color else tiff_array.shape[2]
    
    if output_format.lower() == 'avi':
        fourcc = cv2.VideoWriter.fourcc(*'MJPG')
    elif output_format.lower() == 'mp4':
        fourcc = cv2.VideoWriter.fourcc(*'H264')
    else:
        raise ValueError(f"Unsupported output_format '{output_format}'. Use 'avi' or 'mp4'.")
    
    out = cv2.VideoWriter(
        filename=output_path,
        fourcc=fourcc,
        fps=fps,
        frameSize=(width, height),
        isColor=use_color
    )
    
    # Create a progress bar that updates and then clears itself when done.
    frame_iter = tqdm(
        range(num_frames),
        desc=f"Processing {os.path.basename(tiff_path)}",
        position=tqdm_position,
        leave=False,
        disable=not show_progress
    )
    
    for i in frame_iter:
        frame = tiff_array[i]
        if frame.dtype != np.uint8:
            # Normalize the frame to the range [0, 255] before converting to uint8
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            frame = cv2.convertScaleAbs(frame)
        out.write(frame)
    
    out.release()


def _convert_one(args):
    """
    Worker function to convert a single TIFF file to video.
    
    Parameters
    ----------
    args : tuple
        A tuple containing (file_path, processed_dir, output_format, fps, use_color, position)
    """
    file_path, processed_dir, output_format, fps, use_color, position = args
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(processed_dir, f"{base_filename}.{output_format}")
    
    tiff_to_video(
        tiff_path=file_path,
        output_path=output_path,
        fps=fps,
        output_format=output_format,
        use_color=use_color,
        show_progress=True,
        tqdm_position=position
    )
    # Optionally, you could write a message for each finished conversion:
    tqdm.write(f"Finished converting {os.path.basename(file_path)}")


def tiff_to_mp4(parent_directory, fps=30, output_format="mp4", use_color=False):
    """
    Parses the BIDS directory to find pupil.ome.tiff files and converts them to video.
    """
    found_files = []
    for root, dirs, files in os.walk(parent_directory):
        for file in files:
            if file.endswith("pupil.ome.tiff"):
                full_path = os.path.join(root, file)
                found_files.append(full_path)
    
    processed_dir = os.path.join(parent_directory, "data", "processed")
    print("Identified the following TIFF files:")
    for tiff_path in found_files:
        print(tiff_path)
    print(f"\nProcessed data will be saved to: {processed_dir}")
    
    user_input = input("\nContinue with conversion? (y/n): ")
    if user_input.lower().startswith('y'):
        os.makedirs(processed_dir, exist_ok=True)
        
        # Prepare arguments as a list of tuples, including unique progress bar positions
        args_list = [
            (file_path, processed_dir, output_format, fps, use_color, idx)
            for idx, file_path in enumerate(found_files)
        ]

        print("\nStarting conversion with multiprocessing...")
        futures = []
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Submit all tasks and store futures to ensure they all complete.
            for args in args_list:
                futures.append(executor.submit(_convert_one, args))
            # Optionally, wait for all futures to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()
                
    else:
        print("Conversion canceled.")

    print("\nConversion complete.")


def convert_video_to_h264(input_path: str, output_path: str):
    """Convert a single video file to H264 format."""
    import os
    import sys
    
    # Use global codec paths
    os.environ['OPENH264_LIBRARY'] = OPENH264_DLL_PATH
    
    # Add codec directory to PATH for conda environment
    if CODEC_DIRECTORY not in os.environ.get('PATH', ''):
        os.environ['PATH'] = CODEC_DIRECTORY + os.pathsep + os.environ.get('PATH', '')
    
    # Add to DLL search path on Windows (Python 3.8+)
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(CODEC_DIRECTORY)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter.fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    if not out.isOpened():
        raise ValueError(f"Could not initialize VideoWriter for {output_path} with H264 codec")
    
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(frame)
    
    cap.release()
    out.release()
    return output_path


def _convert_video_worker(args):
    """Worker function for video conversion."""
    input_path, output_dir, position = args
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_h264.mp4")
    
    try:
        convert_video_to_h264(input_path, output_path)
        tqdm.write(f"Converted {os.path.basename(input_path)} to H264")
    except Exception as e:
        tqdm.write(f"Error converting {os.path.basename(input_path)}: {str(e)}")
        raise


def batch_convert_to_h264(parent_directory: str, pattern: str = "*.mp4"):
    """
    Batch convert video files matching pattern to H264 format.
    
    Parameters
    ----------
    parent_directory : str
        Root directory to search for video files
    pattern : str
        Glob pattern to match files (e.g., "*.mp4", "*.avi", "pupil*.mp4")
    """
    import fnmatch
    
    # Validate input directory
    if not os.path.exists(parent_directory):
        print(f"Error: Directory does not exist: {parent_directory}")
        return
    
    # Find all matching video files
    found_files = []
    try:
        for root, dirs, files in os.walk(parent_directory):
            for file in files:
                if fnmatch.fnmatch(file, pattern):
                    full_path = os.path.join(root, file)
                    found_files.append(full_path)
    except Exception as e:
        print(f"Error scanning directory: {e}")
        return
    
    if not found_files:
        print(f"No video files found matching pattern: {pattern}")
        return
    
    output_dir = os.path.join(parent_directory, "data", "processed", "converted_mp4_codec")
    print(f"Found {len(found_files)} video files matching '{pattern}':")
    for video_path in found_files:
        print(video_path)
    print(f"\nConverted videos will be saved to: {output_dir}")
    
    user_input = input("\nContinue with H264 conversion? (y/n): ")
    if user_input.lower().startswith('y'):
        os.makedirs(output_dir, exist_ok=True)
        
        args_list = [
            (file_path, output_dir, idx)
            for idx, file_path in enumerate(found_files)
        ]

        print("\nStarting H264 conversion with multiprocessing...")
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(_convert_video_worker, args) for args in args_list]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    else:
        print("Conversion canceled.")

    print("\nH264 conversion complete.")


# ---------------------------------------------------------------------------
# Codec utility
# ---------------------------------------------------------------------------

def get_video_codec(video_path: str) -> str:
    """Return the FOURCC codec string for the given video file.

    Raises IOError if the file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path!r}")
        codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((codec_int >> 8 * i) & 0xFF) for i in range(4))
        return codec
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Crop + enhance (batch MP4 processing)
# ---------------------------------------------------------------------------

# Defaults for crop_enhance_mp4 — callers may override per-invocation.
_CROP_DEFAULTS = dict(
    frame_roi=0,
    frame_adjust=0,
    num_samples=3,
    roi_size=512,
    crop_size=256,
)


def make_square_roi(x, y, w, h, frame_shape, crop_size=256):
    """Convert a rectangular ROI to a square ROI by expanding to a bounding square."""
    frame_h, frame_w = frame_shape[:2]
    size = max(w, h, crop_size)
    center_x, center_y = x + w // 2, y + h // 2
    new_x = max(0, center_x - size // 2)
    new_y = max(0, center_y - size // 2)
    if new_x + size > frame_w:
        new_x = frame_w - size
    if new_y + size > frame_h:
        new_y = frame_h - size
    new_x = max(0, new_x)
    new_y = max(0, new_y)
    size = min(size, frame_w - new_x, frame_h - new_y)
    return int(new_x), int(new_y), int(size), int(size)


def select_rois(video_paths, cached_rois=None, frame_roi=0, crop_size=256):
    """Interactively select square ROIs for a list of videos (skipping cached)."""
    rois = dict(cached_rois or {})
    for path in video_paths:
        key = os.path.basename(path)
        if key in rois:
            continue
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_roi)
        _, frame = cap.read()
        cap.release()
        x, y, w, h = cv2.selectROI(f"Select ROI – {key} (will be made square)", frame, False, False)
        cv2.destroyAllWindows()
        x, y, w, h = make_square_roi(x, y, w, h, frame.shape, crop_size)
        rois[key] = [int(x), int(y), int(w), int(h)]
    return rois


def calibrate_adjust(samples, cached_adjust=None):
    """Interactive contrast / brightness / gamma calibration via OpenCV trackbars."""
    if cached_adjust:
        return cached_adjust["alpha"], cached_adjust["beta"], cached_adjust["gamma"]

    import numpy as _np

    def nothing(_):
        pass

    win = "Adjust – press s to save"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Contrast×100", win, 100, 300, nothing)
    cv2.createTrackbar("Brightness", win, 100, 200, nothing)
    cv2.createTrackbar("Gamma×100", win, 100, 300, nothing)

    while True:
        c = cv2.getTrackbarPos("Contrast×100", win) / 100.0
        b = cv2.getTrackbarPos("Brightness", win) - 100
        g = cv2.getTrackbarPos("Gamma×100", win) / 100.0
        invG = 1.0 / g if g > 0 else 1.0
        table = _np.array([((i / 255.0) ** invG) * 255 for i in range(256)], dtype=_np.uint8)
        adjusted = [cv2.LUT(cv2.convertScaleAbs(f, alpha=c, beta=b), table) for f in samples]
        combo = _np.hstack(adjusted)
        cv2.imshow(win, combo)
        if cv2.waitKey(1) & 0xFF == ord("s"):
            break

    cv2.destroyAllWindows()
    return c, b, g


def process_video_crop_enhance(path, roi, alpha, beta, gamma, output_dir, roi_size=512):
    """Crop, adjust, and upsample a single video file."""
    x, y, w, h = roi
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(
        os.path.join(output_dir, os.path.basename(path)), fourcc, fps, (roi_size, roi_size)
    )
    invG = 1.0 / gamma if gamma > 0 else 1.0
    table = np.array([((i / 255.0) ** invG) * 255 for i in range(256)], dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        crop = frame[y : y + h, x : x + w]
        adj = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
        adj = cv2.LUT(adj, table)
        upsampled = cv2.resize(adj, (roi_size, roi_size), interpolation=cv2.INTER_CUBIC)
        out.write(upsampled)

    cap.release()
    out.release()


if __name__ == "__main__":
    # Example usage
    parent_dir = r""  # Replace with your actual parent directory
    frames_per_second = 30

    tiff_to_mp4(
        parent_directory=parent_dir,
        fps=frames_per_second,
        output_format="mp4",
        use_color=False,
    )