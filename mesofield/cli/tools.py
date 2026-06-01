"""``mesofield tools`` — setup, export, and diagnostic utilities.

Lower-frequency helpers: native-driver installation, hardware export,
framerate benchmarking, the PsychoPy test harness, and a dev config shell.
"""

from __future__ import annotations

import os
from pathlib import Path

import click

from ._richhelp import RichGroup


@click.group('tools', cls=RichGroup)
def tools():
    """Setup, export, and diagnostic utilities."""


# ---------------------------------------------------------------------------
# install-drivers
# ---------------------------------------------------------------------------


@tools.command('install-drivers')
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

    EXTERNAL_DRIVERS_DIR = (
        Path(__file__).resolve().parent.parent / "external" / "drivers"
    )

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


# ---------------------------------------------------------------------------
# export-hardware
# ---------------------------------------------------------------------------


@tools.command('export-hardware')
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


# ---------------------------------------------------------------------------
# fps
# ---------------------------------------------------------------------------


@tools.command('fps')
@click.option('--params', default='hardware.yaml', help='Path to the config file')
def fps(params):
    """Measure camera framerate and estimate dataset file sizes."""
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

    # run acquisition
    mmc.run_mda(sequence, block=True)
    pbar.close()

    # compute fps
    deltas = [(t2 - t1).total_seconds() for t1, t2 in zip(times[:-1], times[1:])]
    fps_value = 1 / np.mean(deltas)

    # estimate file size for the user‐specified duration
    estimated_frames = int(fps_value * duration)
    estimated_bytes = img_size * estimated_frames
    estimated_gb = estimated_bytes / (1024**3)
    total_gbs = estimated_gb * num_animals * num_sessions
    summary = {
        "Camera Device": mmc.getCameraDevice(),
        "Exposure (ms)": mmc.getExposure(),
        "Camera Metadata": frame_metadata["camera_metadata"],
        "Measured FPS": round(fps_value, 2),
        "Duration (s)": duration,
        "Frames": estimated_frames,
        "Individual TIFF Stack Size (MB)": round(estimated_gb * 1024, 2),
        "Animals": num_animals,
        "Sessions": num_sessions,
        "Total Estimated Size (GB)": round(total_gbs, 2)
    }

    print(json.dumps(summary, indent=4))


# ---------------------------------------------------------------------------
# psychopy
# ---------------------------------------------------------------------------


@tools.command('psychopy')
def psychopy():
    """Launch the PsychoPy test harness GUI (development tool)."""
    import sys
    from PyQt6.QtWidgets import QApplication
    import tests.test_psychopy as test_psychopy
    from mesofield.gui import theme

    app = QApplication(sys.argv)
    theme.apply_theme(app)
    gui = test_psychopy.DillPsychopyGui()
    gui.show()
    sys.exit(app.exec())


# ---------------------------------------------------------------------------
# config-shell
# ---------------------------------------------------------------------------


@tools.command('config-shell')
@click.option('--yaml_path', default='tests/dev.yaml', help='Path to the YAML config file')
@click.option('--json_path', default='tests/devsub.json', help='Path to the JSON config file')
def config_shell(yaml_path, json_path):
    """Load an IPython terminal with an ExperimentConfig in a dev configuration."""
    from mesofield.config import ExperimentConfig
    from IPython import embed

    config = ExperimentConfig(yaml_path)
    config.load_json(json_path)
    embed(header='Mesofield ExperimentConfig Terminal. Type `config.` + TAB ', local={'config': config})
