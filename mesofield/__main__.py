import os
import logging

import click
from pathlib import Path

# Disable pymmcore-plus logger
package_logger = logging.getLogger('pymmcore-plus')
package_logger.setLevel(logging.CRITICAL)

# Disable debugger warning about the use of frozen modules
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

# Disable ipykernel logger
logging.getLogger("ipykernel.inprocess.ipkernel").setLevel(logging.WARNING)


'''
================================== Command Line Interface ======================================
Commands:
    launch: Launch the mesofield acquisition interface
        --params: Path to the config file
        
    batch_pupil: Convert the pupil videos to mp4 format
        --dir: Directory containing the BIDS formatted /data hierarchy
        
    convert_h264: Convert video files to H264 format for better compatibility
        --dir: Directory containing video files to convert
        --pattern: Glob pattern to match files (e.g., "*.mp4", "pupil*.mp4")
    
    install-drivers: Download Thorlabs Scientific Camera SDK and install native DLLs
        --mm-dir: Explicit Micro-Manager root directory (auto-detected when omitted)
        --keep-zip/--no-keep-zip: Keep the downloaded zip file after extraction
        
    plot_session: Plot the session data
        --dir: Path to experimental directory containing BIDS formatted /data hierarchy
        --sub: Subject ID (the name of the subject folder)
        --ses: Session ID (two digit number indicating the session)
        --save: Save the plot to the processing directory in the Experiment folder
        
    trace_meso: Get the mean trace of the mesoscopic data
        --dir: Path to experimental directory containing BIDS formatted /data hierarchy
        --sub: Subject ID (the name of the subject folder)
        
'''
@click.group()
def cli():
    """mesofields Command Line Interface"""


@cli.command()
@click.argument('config', type=click.Path(), required=False, default=None)
def launch(config):
    """Launch the Mesofield acquisition interface.

    CONFIG is an optional path to an experiment JSON config file.
    When omitted, Mesofield opens in a default state and the
    Configuration Wizard is shown for hot-loading configs.
    """
    import time
    
    from PyQt6.QtWidgets import QApplication, QSplashScreen
    from PyQt6.QtGui import QPixmap, QPainter, QFont
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor, QRadialGradient
    from PyQt6.QtGui import QIcon
    
    from mesofield.gui.maingui import MainWindow
    from mesofield.base import Procedure
    
    app = QApplication([])
    window_icon = QIcon(os.path.join(os.path.dirname(__file__), "gui", "Mesofield_icon.png"))
    app.setWindowIcon(window_icon)

    # PNG:
    # pixmap = QPixmap(r'mesofield\gui\Mesofield_icon.png')
    # pixmap = pixmap.scaled(500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    # splash = QSplashScreen(pixmap)
    #splash.setFixedSize(500, 500)

# ====================== Splash Screen with ASCII Art ========================= """

# Font: Sub-Zero; character width: Full, Character Height: Fitted
# https://patorjk.com/software/taag/#p=display&h=0&v=1&f=Sub-Zero&t=Mesofield
    ascii = r"""
 __    __     ______     ______     ______     ______   __      ____      __         _____
/\ "-./  \   /\  ___\   /\  ___\   /\  __ \   /\  ___\ /\ \   /\  ___\   /\ \       /\  __-.  
\ \ \-./\ \  \ \  __\   \ \___  \  \ \ \/\ \  \ \  __\ \ \ \  \ \  __\   \ \ \____  \ \ \/\ \ 
 \ \_\ \ \_\  \ \_____\  \/\_____\  \ \_____\  \ \_\    \ \_\  \ \_____\  \ \_____\  \ \____- 
  \/_/  \/_/   \/_____/   \/_____/   \/_____/   \/_/     \/_/   \/_____/   \/_____/   \/____/ 
                                                                                  
-------------------------  Mesofield Acquisition Interface  ---------------------------------
"""

    # Create a transparent pixmap
    pixmap = QPixmap(1100, 210)
    pixmap.fill(Qt.GlobalColor.transparent)

    # Build a radial gradient: dark center that fades out at the edges
    center = pixmap.rect().center()
    radius = max(pixmap.width(), pixmap.height()) / 2
    gradient = QRadialGradient(center.x(), center.y(), radius)
    gradient.setColorAt(0.0, QColor(1, 25, 5))  # solid dark center
    gradient.setColorAt(0.7, QColor(10, 15, 0, 200))  # keep dark until 80%
    gradient.setColorAt(1.0, QColor(0, 0, 0, 0))    # fully transparent edges

    painter = QPainter(pixmap)
    # Fill entire pixmap with the gradient block
    painter.fillRect(pixmap.rect(), gradient)

    # Draw the ASCII art on top
    painter.setPen(Qt.GlobalColor.green)
    painter.setFont(QFont("Courier", 12))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, ascii)
    painter.end()

    splash = QSplashScreen(pixmap)

    splash.show()
    app.processEvents()  # ensure the splash appears

    #TODO put this somewhere it belongs 
# ====================== End of Splash Screen logic ========================= '''
    time.sleep(0.5) #give the splash screen a moment to show :)
    procedure = Procedure(config)
    
    mesofield = MainWindow(procedure)
    mesofield.show()
    splash.finish(mesofield)
    app.exec()


@cli.command()
@click.argument('experiment_dir')
@click.option('--speed', default=1.0, show_default=True, help='Playback speed multiplier')
@click.option('--loop/--no-loop', default=False, show_default=True, help='Loop playback when finished')
def playback(experiment_dir: str, speed: float, loop: bool):
    """Launch Mesofield in playback mode for a recorded experiment."""

    from mesofield.playback import discover_playback_context, launch_playback_app

    context = discover_playback_context(Path(experiment_dir), speed=speed, loop=loop)
    launch_playback_app(context)


@cli.command()
@click.option('--path', help='Path to a tiff file')
@click.option('--dir',  help='Save the plot to the processing directory in the Experiment folder')
@click.option('--sub', help='Subject ID (the name of the subject folder)')
def trace_meso(path, dir, sub):
    import pandas as pd
    import mesofield.data.proc.load as load
    import mesofield.data.batch as batch
    
    if path:
        session_paths = [path]
        # find the parent experiment file assumn=ing path is under experimentdir/data/sub/ses/func/tiffile
        experiment_dir = Path(path).parents[4]
        print(f"DEBUG: Inferred experiment directory: {experiment_dir}")
        result = batch.mean_trace_from_tiff(session_paths)
        for path, trace in result.items():
            print(f"{path}: {trace[:10]}")
        # save to the processed dir of the experiment dir supporting Windows path
        outdir = Path(experiment_dir) / "processed"
        df = pd.DataFrame({"Slice": range(len(trace)), "Mean": trace})
        
        base_name = os.path.splitext(os.path.basename(path))[0]
        filename = f"{base_name}_meso-mean-trace.csv"
        df.to_csv(os.path.join(outdir, filename), index=False)
    else:
        datadict =  load.file_hierarchy(dir)

        # print(f"DEBUG: Available subject keys: {list(datadict.keys())}")
        # print(f"DEBUG: Available session keys for subject '{sub}': {list(datadict[sub].keys())}")

        session_paths = []
        for key in sorted(datadict[sub].keys()):
            if key.isdigit():
                session = datadict[sub][key]
                if 'widefield' in session:
                    print(f"DEBUG: Session {key} widefield keys: {list(session['widefield'].keys())}")
                    if 'meso_tiff' in session['widefield']:
                        path = session['widefield']['meso_tiff']
                        print(f"DEBUG: Found session {key}, meso_tiff path: {path}")
                        session_paths.append(path)
                    else:
                        print(f"WARNING: Session {key} 'widefield' keys: {list(session['widefield'].keys())} (missing 'meso_tiff')")
                else:
                    print(f"WARNING: Session {key} missing 'widefield' key. Available keys: {list(session.keys())}")
        print(f"DEBUG: Collected session_paths: {session_paths}")
        
        results = batch.mean_trace_from_tiff(session_paths)
        for path, trace in results.items():
            print(f"{path}: {trace[:10]}") 
            
        outdir = os.path.join(dir, "processed", sub)
        os.makedirs(outdir, exist_ok=True)

        for path, trace in results.items():
            df = pd.DataFrame({"Slice": range(len(trace)), "Mean": trace})
            base_name = os.path.splitext(os.path.basename(path))[0]
            filename = f"{base_name}_meso-mean-trace.csv"
            df.to_csv(os.path.join(outdir, filename), index=False)


@cli.command()
@click.option('--dir', required=True, help='Experiment directory containing BIDS formatted /data hierarchy')
@click.option('--sub', default=None, help='Single subject ID to process (default: all subjects)')
@click.option('--frame', default=1, show_default=True, help='0-based frame index to extract from each tiff')
@click.option('--rotate', default=0, show_default=True, type=int, help='Rotate each frame by N degrees (positive=clockwise, negative=counter-clockwise)')
@click.option('--filter', 'ses_filter', default=None, help='Session range START:END (1-based, exclusive end). E.g. 1:10 keeps sessions 1-9.')
def montage_meso(dir, sub, frame, rotate, ses_filter):
    """Extract a single frame from each session's widefield tiff and save a per-subject montage.

    Each task within a session becomes its own row.  Columns are sessions,
    so the output image is a grid of (tasks x sessions) panels.
    """
    import numpy as np
    import tifffile
    import mesofield.data.proc.load as load
    from PIL import Image, ImageDraw, ImageFont

    datadict = load.file_hierarchy(dir)
    subjects = [sub] if sub else sorted(datadict.keys())

    outdir = os.path.join(dir, "processed")
    os.makedirs(outdir, exist_ok=True)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    label_height = 30

    for subject in subjects:
        sessions = datadict[subject]
        ses_keys = sorted(k for k in sessions.keys() if k.isdigit())

        # Apply session range filter
        if ses_filter:
            parts = ses_filter.split(':')
            start = int(parts[0]) if parts[0] else 1
            end = int(parts[1]) if len(parts) > 1 and parts[1] else None
            ses_keys = [k for k in ses_keys if int(k) >= start and (end is None or int(k) < end)]

        # Collect the superset of task names across all sessions (preserving order)
        all_tasks = []
        for sk in ses_keys:
            for tk in sessions[sk]:
                if tk not in all_tasks:
                    all_tasks.append(tk)

        # grid[task][ses_key] = normalised 8-bit numpy image
        grid: dict[str, dict[str, np.ndarray]] = {t: {} for t in all_tasks}

        for ses_key in ses_keys:
            session = sessions[ses_key]
            for task in all_tasks:
                if task not in session or 'meso_tiff' not in session[task]:
                    print(f"WARNING: sub-{subject} ses-{ses_key} task-{task} missing meso_tiff, skipping")
                    continue

                tiff_path = session[task]['meso_tiff']
                try:
                    tiff_array = tifffile.memmap(tiff_path)
                    if tiff_array.shape[0] <= frame:
                        print(f"WARNING: {tiff_path} has only {tiff_array.shape[0]} frames, skipping")
                        continue
                    img = np.array(tiff_array[frame])
                except Exception as e:
                    print(f"ERROR reading {tiff_path}: {e}")
                    continue

                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img_norm = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img_norm = np.zeros_like(img, dtype=np.uint8)

                if rotate:
                    # PIL rotates counter-clockwise, so negate for clockwise convention
                    img_norm = np.array(Image.fromarray(img_norm).rotate(-rotate, expand=True))

                grid[task][ses_key] = img_norm

        # Skip subject if nothing was loaded
        if not any(grid[t] for t in all_tasks):
            print(f"WARNING: No frames found for sub-{subject}, skipping")
            continue

        # Determine a uniform cell size across the whole grid
        all_imgs = [img for task_imgs in grid.values() for img in task_imgs.values()]
        cell_h = max(img.shape[0] for img in all_imgs)
        cell_w = max(img.shape[1] for img in all_imgs)

        def _resize(img):
            if img.shape[0] == cell_h and img.shape[1] == cell_w:
                return img
            return np.array(Image.fromarray(img).resize((cell_w, cell_h), Image.LANCZOS))

        def _blank():
            return np.zeros((cell_h, cell_w), dtype=np.uint8)

        def _label_header(text, width):
            header = Image.new('L', (width, label_height), color=0)
            draw = ImageDraw.Draw(header)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            draw.text(((width - text_w) // 2, 4), text, fill=255, font=font)
            return np.array(header)

        # Helper: render vertical text as a strip image
        def _vertical_text(text, height, strip_w=40):
            """Render *text* rotated 90° so it reads bottom-to-top (left side)
            or top-to-bottom (right side).  Returns an (height, strip_w) uint8 array."""
            # draw text horizontally first, then rotate
            tmp = Image.new('L', (height, strip_w), color=0)
            draw = ImageDraw.Draw(tmp)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(((height - tw) // 2, (strip_w - th) // 2), text, fill=255, font=font)
            return tmp  # will be rotated by caller

        side_strip_w = 40
        grid_height = label_height + cell_h * len(all_tasks)
        grid_width = cell_w * len(ses_keys)

        # --- Left column: vertical subject name (reads bottom-to-top) ---
        subject_label = f"sub-{subject}"
        left_pil = _vertical_text(subject_label, grid_height, side_strip_w)
        left_strip = np.array(left_pil.rotate(90, expand=True))  # CCW 90°

        # --- Right column: vertical task names (one per row, reads top-to-bottom) ---
        right_pieces = []
        # blank for column-header row
        right_pieces.append(np.zeros((label_height, side_strip_w), dtype=np.uint8))
        for task in all_tasks:
            task_label = task if task else "default"
            tmp = _vertical_text(task_label, cell_h, side_strip_w)
            right_pieces.append(np.array(tmp.rotate(-90, expand=True)))  # CW 90°
        right_strip = np.vstack(right_pieces)

        # --- Build the data columns (one per session) ---
        columns = []
        for ses_key in ses_keys:
            col_header = _label_header(f"ses-{ses_key}", cell_w)
            panels = [col_header]
            for task in all_tasks:
                img = grid[task].get(ses_key)
                panels.append(_resize(img) if img is not None else _blank())
            columns.append(np.vstack(panels))

        montage = np.hstack([left_strip] + columns + [right_strip])
        montage_img = Image.fromarray(montage)

        filename = f"sub-{subject}_frame-{frame}_montage.png"
        save_path = os.path.join(outdir, filename)
        montage_img.save(save_path)
        n_cells = sum(len(v) for v in grid.values())
        print(f"Saved: {save_path}  ({len(all_tasks)} tasks x {len(ses_keys)} sessions, {n_cells} images)")

    print("Done.")


@cli.command()
@click.option('--dir', help='Directory containing the BIDS formatted /data hierarchy')
def batch_pupil(dir):
    """Convert the pupil videos to mp4 format."""
    from mesofield.data.batch import tiff_to_mp4
        
    tiff_to_mp4(
        parent_directory=dir,
        fps=30,
        output_format="mp4",
        use_color=False
    )


@cli.command()
def psychopy():
    import sys
    from PyQt6.QtWidgets import QApplication
    import tests.test_psychopy as test_psychopy
    
    app = QApplication(sys.argv)
    gui = test_psychopy.DillPsychopyGui()
    gui.show()
    sys.exit(app.exec())


@cli.command()
@click.option('--params', default='hardware.yaml', help='Path to the config file')
def get_fps(params):
    import json
    from tqdm import tqdm   
    import numpy as np
    import datetime
    from useq import MDAEvent, MDASequence
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.metadata import FrameMetaV1
    from mesofield.config import ExperimentConfig
    
    frame_metadata: FrameMetaV1 = None
    
    config = ExperimentConfig(params)
    config.hardware.initialize(config)

    # measure over a fixed number of frames to get fps
    num_frames = 300
    mmc: CMMCorePlus = config.hardware.ThorCam.core
    #mmc.setExposure(50)
    sequence = MDASequence(time_plan={"interval": 0, "loops": num_frames})

    # ask user for desired duration (in seconds)
    duration = float(input("Enter duration in seconds for file‐size estimate: "))
    num_animals = int(input("Enter number of animals: "))
    num_sessions = int(input("Enter number of sessions: "))

    times = []
    pbar = tqdm(total=num_frames, desc="Acquiring frames")
    img_size = 0

    @mmc.mda.events.frameReady.connect
    def new_frame(img: np.ndarray, event: MDAEvent, metadata: dict):
        
        nonlocal img_size
        nonlocal frame_metadata
        # frame timestamps
        frame_time_str = metadata['camera_metadata']['TimeReceivedByCore']
        times.append(datetime.datetime.fromisoformat(frame_time_str))

        # single instance of frame metadata for printing:
        if frame_metadata is None:
            frame_metadata = metadata
            
        # record single image size once
        if img_size == 0:
            img_size = img.nbytes
        pbar.update(1)

    # @mmc.mda.events.sequenceStarted.connect
    # def on_start(sequence: MDASequence, metadata: dict):
    #     print("Measuring framerate...")

    # run acquisition
    mmc.run_mda(sequence, block=True)
    pbar.close()

    # compute fps
    deltas = [(t2 - t1).total_seconds() for t1, t2 in zip(times[:-1], times[1:])]
    fps = 1 / np.mean(deltas)

    # estimate file size for the user‐specified duration
    estimated_frames = int(fps * duration)
    estimated_bytes = img_size * estimated_frames
    estimated_mb = estimated_bytes / (1024**2)
    estimated_gb = estimated_bytes / (1024**3)
    total_gbs = estimated_gb * num_animals * num_sessions
    summary = {
        "Camera Device": mmc.getCameraDevice(),
        "Exposure (ms)": mmc.getExposure(),
        "Camera Metadata": frame_metadata["camera_metadata"],
        "Measured FPS": round(fps, 2),
        "Duration (s)": duration,
        "Frames": estimated_frames,
        "Individual TIFF Stack Size (MB)": round(estimated_gb * 1024, 2),
        "Animals": num_animals,
        "Sessions": num_sessions,
        "Total Estimated Size (GB)": round(total_gbs, 2)
    }

    print(json.dumps(summary, indent=4))

@cli.command()
@click.option('--yaml_path', default='tests/dev.yaml', help='Path to the YAML config file')
@click.option('--json_path', default='tests/devsub.json', help='Path to the JSON config file')
def ipython(yaml_path, json_path):
    """Load iPython terminal with ExperimentConfig in a dev configuration."""
    from mesofield.config import ExperimentConfig
    from IPython import embed
        
    config = ExperimentConfig(yaml_path)
    config.load_parameters(json_path)
    embed(header='Mesofield ExperimentConfig Terminal. Type `config.` + TAB ', local={'config': config})


@cli.command()
@click.option('--dir', 'experiment_dir', required=True, help='Directory containing BIDS data')
@click.option('--db', 'db_path', required=True, help='Path to the HDF5 database')
def refresh_db(experiment_dir, db_path):
    """Rebuild the database from files on disk."""
    from mesofield.io.h5db import H5Database

    db = H5Database(db_path)
    db.refresh(experiment_dir)
    click.echo(f"Database refreshed from {experiment_dir}")

@cli.command()
@click.option('--dir', required=True, help='Directory containing video files to convert')
@click.option('--pattern', default='*.mp4', help='Glob pattern to match files (e.g., "*.mp4", "pupil*.mp4")')
def convert_h264(dir, pattern):
    """Convert video files to H264 format for better compatibility."""
    from mesofield.data.batch import batch_convert_to_h264
    
    batch_convert_to_h264(
        parent_directory=dir,
        pattern=pattern
    )
    
@cli.command('install-drivers')
@click.option('--mm-dir', 'mm_dir', default=None,
              help='Explicit path to the Micro-Manager root directory. '
                   'Auto-detected from pymmcore-plus when omitted.')
@click.option('--keep-zip/--no-keep-zip', default=False, show_default=True,
              help='Keep the downloaded zip file after extraction.')
def install_drivers(mm_dir, keep_zip):
    """Download Thorlabs Scientific Camera SDK and install native DLLs into Micro-Manager.

    This command performs the following steps:

    \b
      1. Locate (or install) the Micro-Manager device adapters via pymmcore-plus.
      2. Download the Thorlabs Scientific Camera Interfaces SDK.
      3. Extract the SDK into mesofield/external/drivers/.
      4. Copy the 64-bit native DLLs into the Micro-Manager root directory.
    """
    import shutil
    import subprocess
    import sys
    import tempfile
    import urllib.request
    import zipfile

    THORLABS_SDK_URL = (
        "https://media.thorlabs.com/contentassets/"
        "039fcbaaafa0457eb2901466cf0b9489/"
        "scientific_camera_interfaces_windows-2.1.zip"
        "?v=1116040458"
    )
    # Relative path inside the extracted zip that contains the 64-bit DLLs
    DLL_SUBPATH = Path(
        "Scientific Camera Interfaces",
        "SDK",
        "Native Toolkit",
        "dlls",
        "Native_64_lib",
    )

    EXTERNAL_DRIVERS_DIR = Path(__file__).resolve().parent / "external" / "drivers"

    # ---- Step 1: Resolve the Micro-Manager root directory ----
    if mm_dir is not None:
        mm_root = Path(mm_dir)
    else:
        mm_root = _resolve_micromanager_root()

    if not mm_root.is_dir():
        click.secho(f"ERROR: Micro-Manager directory does not exist: {mm_root}", fg="red")
        raise SystemExit(1)

    click.echo(f"Micro-Manager root: {mm_root}")

    # ---- Step 2: Download the Thorlabs SDK zip ----
    EXTERNAL_DRIVERS_DIR.mkdir(parents=True, exist_ok=True)
    zip_dest = EXTERNAL_DRIVERS_DIR / "scientific_camera_interfaces_windows-2.1.zip"

    if zip_dest.exists():
        click.echo(f"Zip already present at {zip_dest}, skipping download.")
    else:
        click.echo("Downloading Thorlabs Scientific Camera Interfaces SDK …")
        try:
            _download_with_progress(THORLABS_SDK_URL, zip_dest)
        except Exception as exc:
            click.secho(f"Download failed: {exc}", fg="red")
            raise SystemExit(1)

    # ---- Step 3: Extract into external/drivers/ ----
    extract_dir = EXTERNAL_DRIVERS_DIR / "scientific_camera_interfaces"
    if extract_dir.exists():
        click.echo(f"Extraction folder already exists at {extract_dir}, skipping extraction.")
    else:
        click.echo(f"Extracting SDK to {extract_dir} …")
        try:
            with zipfile.ZipFile(zip_dest, "r") as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile as exc:
            click.secho(f"Bad zip file: {exc}", fg="red")
            raise SystemExit(1)

    # ---- Step 4: Copy 64-bit native DLLs into Micro-Manager root ----
    dll_source = extract_dir / DLL_SUBPATH
    if not dll_source.is_dir():
        # The zip might have a single top-level folder; search for a match
        candidates = list(extract_dir.rglob("Native_64_lib"))
        if candidates:
            dll_source = candidates[0]
        else:
            click.secho(
                f"ERROR: Could not locate Native_64_lib inside the extracted archive.\n"
                f"Expected at: {dll_source}",
                fg="red",
            )
            raise SystemExit(1)

    dll_files = list(dll_source.glob("*.dll"))
    if not dll_files:
        click.secho(f"WARNING: No .dll files found in {dll_source}", fg="yellow")
        raise SystemExit(1)

    click.echo(f"Copying {len(dll_files)} DLL(s) from {dll_source} → {mm_root}")
    for dll in dll_files:
        dest = mm_root / dll.name
        shutil.copy2(dll, dest)
        click.echo(f"  ✓ {dll.name}")

    # ---- Cleanup ----
    if not keep_zip and zip_dest.exists():
        zip_dest.unlink()
        click.echo("Removed downloaded zip file.")

    click.secho("\nThorlabs Scientific Camera DLLs installed successfully.", fg="green")


def _resolve_micromanager_root() -> Path:
    """Locate the Micro-Manager installation via pymmcore-plus.

    Falls back to running ``mmcore install`` when no installation is found.
    """
    import subprocess
    import sys

    try:
        from pymmcore_plus import find_micromanager
        mm_path = find_micromanager()
        if mm_path:
            return Path(mm_path)
    except ImportError:
        click.secho(
            "pymmcore-plus is not installed.  Install it first:\n"
            "  pip install pymmcore-plus",
            fg="red",
        )
        raise SystemExit(1)
    except Exception:
        pass  # fall through to mmcore install

    # No existing installation – offer to install device adapters
    click.echo("No Micro-Manager installation detected.")
    if click.confirm("Run 'mmcore install' to install Micro-Manager device adapters?", default=True):
        subprocess.check_call([sys.executable, "-m", "pymmcore_plus", "install"])
        # Re-resolve after install
        try:
            from pymmcore_plus import find_micromanager
            mm_path = find_micromanager()
            if mm_path:
                return Path(mm_path)
        except Exception:
            pass

    click.secho("ERROR: Could not locate a Micro-Manager installation.", fg="red")
    raise SystemExit(1)


def _download_with_progress(url: str, dest: Path, chunk_size: int = 1024 * 64):
    """Download *url* to *dest* with a simple progress indicator."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "mesofield-installer/1.0"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest, "wb") as fh:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    click.echo(f"\r  {pct:3d}% ({downloaded // 1024:,} KB)", nl=False)
        if total:
            click.echo()  # newline after progress


@cli.command()
@click.option('--dir', required=True, help='Experiment directiory with pupil files')
@click.option('--sub', required=True, help='Subject ID (the name of the subject folder)')
@click.option('--ses', required=True, help='Session ID (the name of the session folder)')
def plot_session(dir, sub, ses):
    """Plot the pupil data from this session."""
    import pandas as pd
    import matplotlib.pyplot as plt
    import mesofield.data.proc.load as load
    import mesofield.data.batch as batch
    import mesofield.data.proc as proc
    
    datadict =  load.file_hierarchy(dir)

    print(f"DEBUG: Available subject keys: {list(datadict.keys())}")
    print(f"DEBUG: Available session keys for subject '{sub}': {list(datadict[sub].keys())}")
    print(f"DEBUG: Keys within session '{ses}': {list(datadict[sub][ses].keys())}")
    
    data = pd.DataFrame(pd.read_pickle(r"D:\jgronemeyer\240324_HFSA\processed\dlc_output\20250408_174515_sub-STREHAB05_ses-10_task-widefield_pupil.omeDLC_Resnet50_DLC-HFSAApr20shuffle2_snapshot_010_full.pickle")).head()
    proc_data = proc.process_deeplabcut_pupil_data(data)
    print(proc_data.head())

if __name__ == "__main__":
    cli()