import cv2
import os
import multiprocessing as mp
import numpy as np
import json
from pathlib import Path

# Set environment variables to suppress OpenCV/FFMPEG output BEFORE importing cv2
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;quiet'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

if hasattr(cv2, 'setLogLevel'):
    cv2.setLogLevel(0)  # 0 = Silent

# ─── USER-ADJUSTABLE GLOBALS ─────────────────────────────────────────
INPUT_DIR      = r'D:\jgronemeyer\250627_HFSA\processed\pupil_mp4_links'
OUTPUT_DIR     = r'D:\jgronemeyer\250627_HFSA\processed\pupil_mp4_links\data\processed\cropped_enhanced'
CACHE_FILE     = r'D:\jgronemeyer\250627_HFSA\processed\pupil_mp4_links\data\processed\cropped_enhanced\crop_enhance_cache.json'
NUM_PROCESSES  = 10
FRAME_ROI      = 0    # index of frame for ROI selection
FRAME_ADJUST   = 0    # index of frame for contrast/brightness/gamma
NUM_SAMPLES    = 3    # how many ROI-cropped samples for adjustment
ROI_SIZE       = 512  # output size (square) after upsampling
CROP_SIZE      = 256  # minimum crop size for square selection
# OpenH264 codec paths for video conversion compatibility
# These paths point to external H.264 codec DLLs needed for OpenCV video encoding
# when the built-in codecs are not available or compatible with target software
# ─── H264 Video Codec ─────────────────────────────────────────────────────
# base project dir (mesofield/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# codec folder is under mesofield/external/video-codecs
CODEC_DIRECTORY = str(BASE_DIR / "external" / "video-codecs")
OPENH264_DLL_PATH = str(Path(CODEC_DIRECTORY) / "openh264-1.8.0-win64.dll")
# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────

# --- initialize H264 support before any VideoWriter calls ---
os.environ['OPENH264_LIBRARY'] = OPENH264_DLL_PATH
if CODEC_DIRECTORY not in os.environ.get('PATH', ''):
    os.environ['PATH'] = CODEC_DIRECTORY + os.pathsep + os.environ.get('PATH', '')
if hasattr(os, 'add_dll_directory') and os.path.exists(CODEC_DIRECTORY):
    os.add_dll_directory(CODEC_DIRECTORY)
elif not os.path.exists(CODEC_DIRECTORY):
    print(f"Warning: Codec directory not found: {CODEC_DIRECTORY}")
    print("H.264 encoding may not work properly without the codec files.")
    
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {'rois': {}, 'adjust': {}}


def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def make_square_roi(x, y, w, h, frame_shape):
    """Convert rectangular ROI to square ROI by expanding to minimum bounding square."""
    frame_h, frame_w = frame_shape[:2]
    
    # Use the larger dimension to make it square
    size = max(w, h)
    
    # Ensure minimum size
    size = max(size, CROP_SIZE)
    
    # Calculate center of original ROI
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Calculate new square coordinates
    new_x = max(0, center_x - size // 2)
    new_y = max(0, center_y - size // 2)
    
    # Ensure the square fits within frame bounds
    if new_x + size > frame_w:
        new_x = frame_w - size
    if new_y + size > frame_h:
        new_y = frame_h - size
    
    # Final bounds check
    new_x = max(0, new_x)
    new_y = max(0, new_y)
    size = min(size, frame_w - new_x, frame_h - new_y)
    
    return int(new_x), int(new_y), int(size), int(size)


def select_rois(video_paths, cached_rois):
    rois = cached_rois.copy()
    for path in video_paths:
        key = os.path.basename(path)
        if key in rois:
            continue
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_ROI)
        _, frame = cap.read()
        cap.release()
        
        # Get user selection
        x, y, w, h = cv2.selectROI(f'Select ROI – {key} (will be made square)', frame, False, False)
        cv2.destroyAllWindows()
        
        # Convert to square ROI
        x, y, w, h = make_square_roi(x, y, w, h, frame.shape)
        
        print(f"ROI for {key}: Square region ({w}x{h}) at ({x}, {y})")
        rois[key] = [int(x), int(y), int(w), int(h)]
    return rois


def calibrate_adjust(samples, cached_adjust):
    if cached_adjust:
        return cached_adjust['alpha'], cached_adjust['beta'], cached_adjust['gamma']

    # All samples are now the same size (ROI_SIZE x ROI_SIZE), so no padding needed
    def nothing(_): pass
    win = 'Adjust – press s to save'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Contrast×100', win, 100, 300, nothing)
    cv2.createTrackbar('Brightness',    win, 100, 200, nothing)
    cv2.createTrackbar('Gamma×100',     win, 100, 300, nothing)

    while True:
        c = cv2.getTrackbarPos('Contrast×100', win) / 100.0
        b = cv2.getTrackbarPos('Brightness',    win) - 100
        g = cv2.getTrackbarPos('Gamma×100',     win) / 100.0

        invG = 1.0 / g if g > 0 else 1.0
        table = np.array([((i / 255.0) ** invG) * 255 for i in range(256)], dtype=np.uint8)
        adjusted_list = [cv2.LUT(cv2.convertScaleAbs(f, alpha=c, beta=b), table) for f in samples]

        combo = np.hstack(adjusted_list)
        cv2.imshow(win, combo)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    cv2.destroyAllWindows()
    return c, b, g


def process_video(args):
    path, roi, alpha, beta, gamma = args
    x, y, w, h = roi
    cap = cv2.VideoCapture(path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    # Output video is always 512x512
    out    = cv2.VideoWriter(os.path.join(OUTPUT_DIR, os.path.basename(path)), fourcc, fps, (ROI_SIZE, ROI_SIZE))
    invG   = 1.0 / gamma if gamma > 0 else 1.0
    table  = np.array([((i / 255.0) ** invG) * 255 for i in range(256)], dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Crop the square region
        crop = frame[y:y+h, x:x+w]
        
        # Apply contrast/brightness/gamma adjustments
        adj  = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
        adj  = cv2.LUT(adj, table)
        
        # Upsample to 512x512 using cubic interpolation for better quality
        upsampled = cv2.resize(adj, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_CUBIC)
        
        out.write(upsampled)

    cap.release()
    out.release()


if __name__ == '__main__':
    vids = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.lower().endswith('.mp4')]
    if not vids:
        print(f'No .mp4 files found in {INPUT_DIR}')
        exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # load or initialize cache
    cache = load_cache()

    # ROI selection, respecting cache
    rois = select_rois(vids, cache.get('rois', {}))
    cache['rois'] = rois
    save_cache(cache)

    # prepare samples for adjustment
    samples = []
    for v in vids[:NUM_SAMPLES]:
        cap = cv2.VideoCapture(v)
        cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_ADJUST)
        _, frame = cap.read()
        cap.release()
        key = os.path.basename(v)
        x, y, w, h = rois[key]
        cropped = frame[y:y+h, x:x+w]
        # Upsample to ROI_SIZE for consistent preview
        upsampled = cv2.resize(cropped, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_CUBIC)
        samples.append(upsampled)

    # calibration, respecting cache
    alpha, beta, gamma = calibrate_adjust(samples, cache.get('adjust', {}))
    if not cache.get('adjust'):
        cache['adjust'] = {'alpha': alpha, 'beta': beta, 'gamma': gamma}
        save_cache(cache)

    # batch processing
    tasks = [(v, rois[os.path.basename(v)], alpha, beta, gamma) for v in vids]
    with mp.Pool(NUM_PROCESSES) as pool:
        pool.map(process_video, tasks)