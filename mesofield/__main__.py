import os
import logging

import click

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
@click.option('--config', required=True, help='Path to experiment configuration (JSON or YAML)')
def launch(config):
    import time
    
    from PyQt6.QtWidgets import QApplication, QSplashScreen
    from PyQt6.QtGui import QPixmap, QPainter, QFont
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor, QRadialGradient
    
    from mesofield.gui.maingui import MainWindow
    from mesofield.base import Procedure, create_procedure
    from mesofield.config import ExperimentConfig
    
    app = QApplication([])


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
    if not os.path.exists(config):
        raise click.ClickException(f"Configuration file not found: {config}")

    try:
        experiment_config = ExperimentConfig.from_file(config)
    except Exception as exc:  # pragma: no cover - CLI failure path
        raise click.ClickException(f"Failed to load configuration: {exc}") from exc

    display_keys = getattr(experiment_config, "display_keys", None)

    time.sleep(2) #give the splash screen a moment to show :)
    procedure = create_procedure(
        Procedure,
        config=experiment_config,
    )
    
    mesofield = MainWindow(procedure, display_keys=display_keys)
    mesofield.show()
    splash.finish(mesofield)
    app.exec()


@cli.command()
@click.option('--dir',  help='Save the plot to the processing directory in the Experiment folder')
@click.option('--sub', help='Subject ID (the name of the subject folder)')
def trace_meso(dir, sub):
    import pandas as pd
    import mesofield.data.proc.load as load
    import mesofield.data.batch as batch
    
    datadict =  load.file_hierarchy(dir)

    session_paths = []
    for key in sorted(datadict[sub].keys()):
        if key.isdigit():
            session_paths.append(datadict[sub][key]['widefield']['meso_tiff'])
    
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

if __name__ == "__main__":
    cli()