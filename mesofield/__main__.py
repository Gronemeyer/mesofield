import os

import click
from pathlib import Path

# Disable debugger warning about the use of frozen modules
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"


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
    from mesofield.base import Procedure, load_procedure_from_config
    
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
    procedure = load_procedure_from_config(config) if config else Procedure(config)
    
    mesofield = MainWindow(procedure)
    mesofield.show()
    splash.finish(mesofield)
    app.exec()


# ---------------------------------------------------------------------------
# Explore pickle file and launch IPython
# ---------------------------------------------------------------------------

@cli.command('explore-pickle')
@click.argument('pickle_path', type=click.Path(exists=True, dir_okay=False))
def explore_pickle(pickle_path):
    """Explore a pickle file using datakit.explore and launch an IPython terminal with the report."""
    import sys
    import pickle
    from mesofield.datakit import explore
    try:
        with open(pickle_path, 'rb') as f:
            dataset = pickle.load(f)
    except Exception as exc:
        click.secho(f"Failed to load pickle file: {exc}", fg="red")
        sys.exit(1)
    report = explore(dataset, print_output=True)
    try:
        from IPython import embed
        click.secho("Launching IPython shell. The 'report' variable contains the exploration result.", fg="green")
        embed(header=f"Exploration report for {pickle_path}\nType 'report' to see the summary.")
        embed(dataset=dataset, report=report)
    except ImportError:
        click.secho("IPython is not installed. Please install it to use this feature.", fg="red")
        sys.exit(1)



@cli.command()
@click.argument('config', type=click.Path(exists=True, dir_okay=False), required=False, default=None)
def viewer(config):
    """Launch the standalone TIFF ROI viewer.

    CONFIG is an optional path to an ``experiment.json``. When provided, the
    viewer's "Open TIFF…" dialog opens in that experiment's data directory
    (``<experiment>/data`` if it exists, otherwise the JSON's parent dir).
    Hardware is NOT initialized — this is a read-only inspection tool.
    """
    import json
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QIcon
    from mesofield.data.proc.analysis import TiffViewer

    initial_dir = ""
    if config:
        cfg_path = Path(config).resolve()
        try:
            with open(cfg_path) as f:
                data = json.load(f)
        except Exception:
            data = {}
        # Prefer an explicit save_dir from the config; fall back to
        # <experiment>/data, then the JSON's parent directory.
        save_dir = data.get('save_dir') if isinstance(data, dict) else None
        candidates = []
        if save_dir:
            candidates.append(Path(save_dir))
            candidates.append(Path(save_dir) / 'data')
        candidates.append(cfg_path.parent / 'data')
        candidates.append(cfg_path.parent)
        for c in candidates:
            if c and c.exists():
                initial_dir = str(c)
                break

    app = QApplication([])
    icon_path = os.path.join(os.path.dirname(__file__), "gui", "Mesofield_icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    win = TiffViewer(initial_dir=initial_dir or None)
    win.resize(1100, 800)
    win.show()
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


# ---------------------------------------------------------------------------
# Datakit-based meso tiff discovery helpers
# ---------------------------------------------------------------------------

def _discover_meso_entries(experiment_dir, subject_filter=None):
    """Return ManifestEntry objects for mesoscope tiff files.

    Uses datakit's ``discover_manifest`` to scan the BIDS hierarchy,
    filters for ``meso_metadata`` entries (the JSON sidecars), then
    rewrites each entry's path to point at the tiff itself.
    """
    from mesofield.datakit.discover import discover_manifest
    from mesofield.datakit.datamodel import ManifestEntry

    manifest = discover_manifest(Path(experiment_dir))
    tiff_entries = []
    for entry in manifest.entries:
        if entry.tag != "meso_metadata":
            continue
        if subject_filter and entry.subject != subject_filter:
            continue
        # Derive the tiff path from the metadata sidecar path:
        #   ...mesoscope.ome.tiff_frame_metadata.json  →  ...mesoscope.ome.tiff
        tiff_path = entry.path.replace("_frame_metadata.json", "")
        tiff_entries.append(ManifestEntry(
            tag="meso_tiff",
            path=tiff_path,
            origin=entry.origin,
            subject=entry.subject,
            session=entry.session,
            task=entry.task,
        ))
    return tiff_entries


def _discover_meso_tiffs(experiment_dir, subject_filter=None):
    """Return a list of absolute tiff paths and the experiment root."""
    root = Path(experiment_dir).resolve()
    entries = _discover_meso_entries(experiment_dir, subject_filter)
    paths = [str(root / e.path) for e in entries]
    return paths, root


@cli.command()
@click.option('--path', help='Path to a single tiff file (bypass discovery)')
@click.option('--dir',  help='Experiment directory to discover mesoscope tiffs')
@click.option('--sub', default=None, help='Subject ID to filter (default: all)')
def trace_meso(path, dir, sub):
    """Compute mean fluorescence traces from mesoscope OME-TIFF stacks.

    Uses datakit discovery to find mesoscope tiffs in a BIDS hierarchy,
    or accepts a single --path for quick one-off analysis.
    """
    import pandas as pd
    import mesofield.data.batch as batch

    if path:
        tiff_paths = [path]
        experiment_dir = Path(path).parents[4]
        outdir = Path(experiment_dir) / "processed"
    elif dir:
        tiff_paths, experiment_dir = _discover_meso_tiffs(dir, sub)
        outdir = Path(dir) / "processed" / sub if sub else Path(dir) / "processed"
    else:
        raise click.UsageError("Provide --path or --dir")

    if not tiff_paths:
        click.secho("No mesoscope tiffs found.", fg="yellow")
        return

    os.makedirs(outdir, exist_ok=True)
    click.echo(f"Computing mean traces for {len(tiff_paths)} tiff(s)...")
    results = batch.mean_trace_from_tiff(tiff_paths)

    for tiff_path, trace in results.items():
        df = pd.DataFrame({"Slice": range(len(trace)), "Mean": trace})
        base_name = os.path.splitext(os.path.basename(tiff_path))[0]
        filename = f"{base_name}_meso-mean-trace.csv"
        save_path = os.path.join(outdir, filename)
        df.to_csv(save_path, index=False)
        click.echo(f"  {filename}  ({len(trace)} frames)")

    click.secho(f"Saved {len(results)} trace(s) to {outdir}", fg="green")


@cli.command()
@click.option('--dir', required=True, help='Experiment directory containing BIDS formatted /data hierarchy')
@click.option('--sub', default=None, help='Single subject ID to process (default: all subjects)')
@click.option('--frame', default=1, show_default=True, help='0-based frame index to extract from each tiff')
@click.option('--rotate', default=0, show_default=True, type=int, help='Rotate each frame by N degrees (positive=clockwise, negative=counter-clockwise)')
@click.option('--filter', 'ses_filter', default=None, help='Session range START:END (1-based, exclusive end). E.g. 1:10 keeps sessions 1-9.')
def montage_meso(dir, sub, frame, rotate, ses_filter):
    """Extract a single frame from each session's widefield tiff and save a per-subject montage.

    Uses datakit to discover mesoscope tiffs in the BIDS hierarchy.
    Each task within a session becomes its own row.  Columns are sessions,
    so the output image is a grid of (tasks x sessions) panels.
    """
    import numpy as np
    import tifffile
    from PIL import Image, ImageDraw, ImageFont

    # --- Discover tiffs via datakit ---
    entries = _discover_meso_entries(dir, sub)
    if not entries:
        click.secho("No mesoscope tiffs found.", fg="yellow")
        return

    # Build a nested dict: subject -> session -> task -> tiff_path
    tree: dict[str, dict[str, dict[str, str]]] = {}
    root = Path(dir).resolve()
    for entry in entries:
        s, ses, task = entry.subject, entry.session, entry.task or "default"
        tree.setdefault(s, {}).setdefault(ses, {})[task] = str(root / entry.path)

    subjects = [sub] if sub else sorted(tree.keys())

    outdir = os.path.join(dir, "processed")
    os.makedirs(outdir, exist_ok=True)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    label_height = 30

    for subject in subjects:
        if subject not in tree:
            click.echo(f"WARNING: sub-{subject} not found, skipping")
            continue
        sessions = tree[subject]
        ses_keys = sorted(sessions.keys())

        # Apply session range filter
        if ses_filter:
            parts = ses_filter.split(':')
            start = int(parts[0]) if parts[0] else 1
            end = int(parts[1]) if len(parts) > 1 and parts[1] else None
            ses_keys = [k for k in ses_keys if int(k) >= start and (end is None or int(k) < end)]

        # Collect the superset of task names across all sessions (preserving order)
        all_tasks: list[str] = []
        for sk in ses_keys:
            for tk in sessions[sk]:
                if tk not in all_tasks:
                    all_tasks.append(tk)

        # grid[task][ses_key] = normalised 8-bit numpy image
        grid: dict[str, dict[str, np.ndarray]] = {t: {} for t in all_tasks}

        for ses_key in ses_keys:
            session = sessions[ses_key]
            for task in all_tasks:
                if task not in session:
                    click.echo(f"WARNING: sub-{subject} ses-{ses_key} task-{task} missing, skipping")
                    continue

                tiff_path = session[task]
                try:
                    tiff_array = tifffile.memmap(tiff_path)
                    if tiff_array.shape[0] <= frame:
                        click.echo(f"WARNING: {tiff_path} has only {tiff_array.shape[0]} frames, skipping")
                        continue
                    img = np.array(tiff_array[frame])
                except Exception as e:
                    click.echo(f"ERROR reading {tiff_path}: {e}")
                    continue

                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img_norm = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img_norm = np.zeros_like(img, dtype=np.uint8)

                if rotate:
                    img_norm = np.array(Image.fromarray(img_norm).rotate(-rotate, expand=True))

                grid[task][ses_key] = img_norm

        if not any(grid[t] for t in all_tasks):
            click.echo(f"WARNING: No frames found for sub-{subject}, skipping")
            continue

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

        def _vertical_text(text, height, strip_w=40):
            tmp = Image.new('L', (height, strip_w), color=0)
            draw = ImageDraw.Draw(tmp)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(((height - tw) // 2, (strip_w - th) // 2), text, fill=255, font=font)
            return tmp

        side_strip_w = 40
        grid_height = label_height + cell_h * len(all_tasks)

        subject_label = f"sub-{subject}"
        left_pil = _vertical_text(subject_label, grid_height, side_strip_w)
        left_strip = np.array(left_pil.rotate(90, expand=True))

        right_pieces = [np.zeros((label_height, side_strip_w), dtype=np.uint8)]
        for task in all_tasks:
            task_label = task if task else "default"
            tmp = _vertical_text(task_label, cell_h, side_strip_w)
            right_pieces.append(np.array(tmp.rotate(-90, expand=True)))
        right_strip = np.vstack(right_pieces)

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
        click.echo(f"Saved: {save_path}  ({len(all_tasks)} tasks x {len(ses_keys)} sessions, {n_cells} images)")

    click.secho("Done.", fg="green")


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


def _resolve_init_hardware(rig, hardware):
    """Resolve the ``hardware`` argument for :func:`scaffold_experiment`.

    Returns a ``Path`` (copy a canonical rig file) or ``"dev"`` / ``"blank"``.
    ``--hardware`` wins over ``--rig``; with neither, an interactive picker
    over the rig store plus the ``dev`` / ``blank`` built-ins is shown.
    """
    from mesofield.scaffold import rigs

    if hardware:
        return Path(hardware)
    if rig:
        if rig in ("dev", "blank"):
            return rig
        try:
            return rigs._resolve_existing(rig)
        except FileNotFoundError as exc:
            click.secho(str(exc), fg="red")
            raise SystemExit(1)

    choices = rigs.list_rigs() + ["dev", "blank"]
    click.echo("Select a hardware configuration for this experiment:")
    for name in rigs.list_rigs():
        click.echo(f"  {name}    (canonical rig)")
    click.echo("  dev      (mock devices -- runs without hardware)")
    click.echo("  blank    (fill-out template)")
    picked = click.prompt(
        "Rig", type=click.Choice(choices), default="blank", show_choices=False
    )
    if picked in ("dev", "blank"):
        return picked
    return rigs.rig_path(picked)


@cli.command('init')
@click.argument('directory', type=click.Path())
@click.option('--name', default=None,
              help='Experiment protocol name (default: directory basename uppercased).')
@click.option('--force', is_flag=True,
              help='Overwrite an existing non-empty directory.')
@click.option('--rig', default=None,
              help="Canonical rig to copy hardware.yaml from "
                   "(or 'dev'/'blank'). Skips the interactive picker.")
@click.option('--hardware', default=None, type=click.Path(exists=True, dir_okay=False),
              help='Explicit hardware.yaml file to copy in (overrides --rig).')
def init(directory, name, force, rig, hardware):
    """Scaffold a new mesofield experiment in DIRECTORY.

    Generates `experiment.json`, `hardware.yaml`, `procedure.py`, and a
    `devices/` subdirectory with an annotated thermal-sensor example.

    The `hardware.yaml` is chosen interactively: a canonical rig from this
    machine's rig store (see `mesofield rig`), `dev` (mock devices, runs
    without hardware), or `blank` (a fill-out template). Use --rig/--hardware
    to skip the prompt.
    """
    from mesofield.scaffold import scaffold_experiment

    hardware_choice = _resolve_init_hardware(rig, hardware)
    try:
        out = scaffold_experiment(
            Path(directory), name=name, force=force, hardware=hardware_choice,
        )
    except FileExistsError as exc:
        click.secho(str(exc), fg="red")
        raise SystemExit(1)
    click.secho(f"Scaffolded experiment at {out}", fg="green")
    click.echo("Next steps:")
    click.echo(f"  1. cd {out}")
    if hardware_choice == "dev":
        click.echo("  2. python procedure.py    # runs the mock acquisition")
        click.echo(f"  3. open data/sub-SUBJ01/ses-01/manifest.json")
    else:
        click.echo("  2. review hardware.yaml   # confirm it matches this rig")
        click.echo("  3. python procedure.py    # runs the acquisition")
    click.echo("Read the generated README.md for customization tips.")


@cli.group('rig')
def rig():
    """Manage this machine's canonical hardware.yaml configurations.

    A rig is a named hardware.yaml stored in this computer's OS config
    directory. `mesofield init` copies a rig into each new experiment so
    experiment folders stay self-contained.
    """


@rig.command('list')
def rig_list():
    """List the canonical rigs registered on this machine."""
    from mesofield.scaffold import rigs

    names = rigs.list_rigs()
    if not names:
        click.echo(f"No rigs registered. Store: {rigs.rigs_dir()}")
        click.echo("Add one with 'mesofield rig add' or 'mesofield rig new'.")
        return
    click.echo(f"Rigs in {rigs.rigs_dir()}:")
    for name in names:
        click.secho(f"  {name}", fg="cyan")
        try:
            devices = rigs.rig_devices(name)
        except Exception as exc:
            click.secho(f"      (could not read devices: {exc})", fg="red")
            continue
        if not devices:
            click.echo("      (no devices declared)")
        for dev_name, dev_type in devices:
            click.echo(f"      - {dev_name}  (type: {dev_type})")


@rig.command('add')
@click.argument('name')
@click.argument('path', type=click.Path(exists=True, dir_okay=False))
@click.option('--force', is_flag=True, help='Overwrite an existing rig.')
def rig_add(name, path, force):
    """Copy an existing hardware.yaml at PATH into the store as NAME."""
    from mesofield.scaffold import rigs

    try:
        dst = rigs.add_rig(name, Path(path), force=force)
    except FileExistsError as exc:
        click.secho(str(exc), fg="red")
        raise SystemExit(1)
    except Exception as exc:
        click.secho(f"Failed to add rig: {exc}", fg="red")
        raise SystemExit(1)
    click.secho(f"Registered rig {name!r} at {dst}", fg="green")


@rig.command('new')
@click.argument('name')
@click.option('--force', is_flag=True, help='Overwrite an existing rig.')
def rig_new(name, force):
    """Scaffold a blank fill-out hardware template in the store as NAME."""
    from mesofield.scaffold import rigs

    try:
        dst = rigs.new_rig(name, force=force)
    except FileExistsError as exc:
        click.secho(str(exc), fg="red")
        raise SystemExit(1)
    click.secho(f"Created rig template at {dst}", fg="green")
    click.echo("Edit it to declare this machine's real devices, then use it")
    click.echo(f"with 'mesofield init <dir> --rig {name}'.")


@rig.command('show')
@click.argument('name')
def rig_show(name):
    """Print the path and contents of rig NAME."""
    from mesofield.scaffold import rigs

    try:
        path = rigs._resolve_existing(name)
    except FileNotFoundError as exc:
        click.secho(str(exc), fg="red")
        raise SystemExit(1)
    click.echo(f"# {path}")
    click.echo(path.read_text(encoding="utf-8"))


@rig.command('remove')
@click.argument('name')
def rig_remove(name):
    """Delete rig NAME from the store."""
    from mesofield.scaffold import rigs

    try:
        rigs.remove_rig(name)
    except FileNotFoundError as exc:
        click.secho(str(exc), fg="red")
        raise SystemExit(1)
    click.secho(f"Removed rig {name!r}", fg="green")


@cli.command('retrofit-manifest')
@click.argument('path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--force', is_flag=True,
              help='Overwrite existing manifest.json files (default: skip).')
@click.option('--dry-run', is_flag=True,
              help='Print what would be written without touching disk.')
def retrofit_manifest(path, force, dry_run):
    """Reconstruct mesokit-schema AcquisitionManifests for legacy sessions.

    PATH is either a single session directory (``data/sub-X/ses-Y/``) or an
    experiment root that contains many sessions under ``data/``. Hardware
    calibration is not present in legacy acquisitions; producers get an
    empty ``calibration`` dict. Frame-metadata sidecars (``*_frame_metadata.json``)
    are attached to their tiffs. Multi-task sessions write one manifest
    per task as ``manifest_task-<T>.json``.
    """
    from mesofield.utils.retrofit import (
        discover_sessions,
        manifest_filename,
        synthesize_manifests,
    )

    sessions = list(discover_sessions(Path(path)))
    if not sessions:
        click.secho(f"No BIDS sessions found under {path}", fg="yellow")
        raise SystemExit(1)

    written = skipped = empty = 0
    for session in sessions:
        by_task = synthesize_manifests(session)
        if not by_task:
            click.echo(f"empty {session}  (no producer files)")
            empty += 1
            continue
        multi_task = len(by_task) > 1
        for task, manifest in by_task.items():
            out_path = session / manifest_filename(task, multi_task)
            if out_path.exists() and not force:
                click.echo(f"skip  {out_path}  (exists; use --force to overwrite)")
                skipped += 1
                continue
            verb = "would write" if dry_run else "wrote"
            click.echo(
                f"{verb}  {out_path}  ({len(manifest.producers)} producers, task={task})"
            )
            if not dry_run:
                manifest.write(out_path)
                written += 1

    summary = (
        f"\n{'(dry-run) ' if dry_run else ''}"
        f"sessions={len(sessions)} written={written} skipped={skipped} empty={empty}"
    )
    click.echo(summary)

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


@cli.command('build-dataset')
@click.argument('input_path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--output', '-o', 'output_path', type=click.Path(), default=None,
              help='Output file path (default: <experiment>/processed/YYMMDD_dataset_mvp.<fmt>)')
@click.option('--tags', '-t', multiple=True, default=None,
              help='Source tags to include (repeatable; default: all configured tags)')
@click.option('--format', '-f', 'fmt', type=click.Choice(['h5', 'parquet', 'csv', 'pickle']),
              default='h5', show_default=True, help='Output format')
@click.option('--progress', is_flag=True, help='Show a progress bar during materialization')
@click.option('--shell', is_flag=True, help='Drop into an IPython session after building')
def build_dataset(input_path, output_path, tags, fmt, progress, shell):
    """Build a materialized dataset from an experiment directory.

    Discovers the BIDS hierarchy under INPUT_PATH, loads all registered
    data sources, and writes a single dataset file.
    """
    from mesofield.datakit.core import Dataset

    ds = Dataset.from_directory(
        Path(input_path),
        sources=list(tags) if tags else None,
    )
    if output_path is None:
        from datetime import datetime
        stem = datetime.now().strftime("%y%m%d") + "_dataset_mvp"
        ext = {"h5": ".h5", "parquet": ".parquet", "csv": ".csv", "pickle": ".pkl"}[fmt]
        output_path = Path(input_path) / "processed" / (stem + ext)
    result_path = ds.save(
        Path(output_path),
        format={"h5": "hdf5", "parquet": "parquet", "csv": "csv", "pickle": "pickle"}[fmt],
        strict=True,
        progress=progress,
    )
    click.secho(f"Dataset saved to {result_path}", fg="green")
    if shell:
        try:
            from IPython import embed
            import pandas as pd
            df = pd.read_pickle(result_path) if str(result_path).endswith('.pkl') else pd.read_hdf(result_path)
            click.echo(f"Loaded dataset as 'df' ({df.shape[0]} rows × {df.shape[1]} cols)")
            embed(colors="neutral")
        except ImportError:
            click.secho("IPython not installed; skipping shell.", fg="yellow")


@cli.command('export-hardware')
@click.argument('procedure_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--output', '-o', default=None, type=click.Path(),
              help='Output path for the hardware.yaml '
                   '(default: hardware.yaml beside the procedure file).')
def export_hardware(procedure_path, output):
    """Export a scripted procedure's hardware to a reusable hardware.yaml rig file.

    PROCEDURE_PATH is a procedure.py whose `define_hardware` builds devices in
    Python. The devices are instantiated and serialized into a `type:`-tagged
    hardware.yaml that can later be loaded the normal file-based way.
    """
    from mesofield.base import load_procedure_from_config

    out = output or os.path.join(
        os.path.dirname(os.path.abspath(procedure_path)), "hardware.yaml"
    )
    procedure = load_procedure_from_config(procedure_path)
    procedure.hardware.to_yaml(out)
    click.secho(f"Exported hardware configuration to {out}", fg="green")


@cli.command()
@click.option('--dir', required=True, help='Experiment directiory with pupil files')
@click.option('--sub', required=True, help='Subject ID (the name of the subject folder)')
@click.option('--ses', required=True, help='Session ID (the name of the session folder)')
def plot_session(dir, sub, ses):
    """Plot the pupil data from this session."""
    import pandas as pd
    import matplotlib.pyplot as plt
    click.secho(
        "plot_session is deprecated. Use `mesofield build-dataset` + datakit instead.",
        fg="yellow",
    )
    raise SystemExit(1)

if __name__ == "__main__":
    cli()