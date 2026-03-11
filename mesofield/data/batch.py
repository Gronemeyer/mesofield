import os
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm
import tifffile

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
# OpenH264 codec paths for video conversion compatibility
# These paths point to external H.264 codec DLLs needed for OpenCV video encoding
# when the built-in codecs are not available or compatible with target software
# base project dir (mesofield/)
BASE_DIR = Path(__file__).resolve().parent.parent
# codec folder is under mesofield/external/video-codecs
CODEC_DIRECTORY = str(BASE_DIR / "external" / "video-codecs")
OPENH264_DLL_PATH = str(Path(CODEC_DIRECTORY) / "openh264-1.8.0-win64.dll")
# ─────────────────────────────────────────────────────────────────


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


if __name__ == "__main__":
    # Example usage
    parent_dir = r""  # Replace with your actual parent directory
    frames_per_second = 30
    
    tiff_to_mp4(
        parent_directory=parent_dir,
        fps=frames_per_second,
        output_format="mp4",
        use_color=False
    )