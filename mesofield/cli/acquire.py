"""Acquisition-workflow commands: launch, init, playback, viewer.

These are the top-level day-to-day entry points and are attached directly to
the root ``mesofield`` group (no subgroup prefix).
"""

from __future__ import annotations

import os
from pathlib import Path

import click


# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------


@click.command()
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
    from mesofield.gui import theme
    from mesofield.base import Procedure, load_procedure_from_config

    app = QApplication([])
    theme.apply_theme(app)
    window_icon = QIcon(os.path.join(os.path.dirname(os.path.dirname(__file__)), "gui", "Mesofield_icon.png"))
    app.setWindowIcon(window_icon)

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

    time.sleep(0.5)  # give the splash screen a moment to show :)
    procedure = load_procedure_from_config(config) if config else Procedure(config)

    mesofield = MainWindow(procedure)
    mesofield.show()
    splash.finish(mesofield)
    app.exec()


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


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


@click.command('init')
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


# ---------------------------------------------------------------------------
# playback
# ---------------------------------------------------------------------------


@click.command()
@click.argument('experiment_dir')
@click.option('--speed', default=1.0, show_default=True, help='Initial playback speed multiplier')
@click.option('--loop/--no-loop', default=False, show_default=True, help='Loop playback when finished')
def playback(experiment_dir: str, speed: float, loop: bool):
    """Launch the Mesofield playback viewer for a recorded experiment.

    EXPERIMENT_DIR is an experiment root (containing ``data/sub-*/ses-*``) or a
    session directory. The viewer is read-only: pick a subject / session / task,
    then play, pause and scrub the recorded camera streams. Treadmill data, when
    present in the session's ``*_dataqueue.csv``, is plotted beneath the cameras.
    """
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QIcon

    from mesofield.gui.playback_window import PlaybackWindow
    from mesofield.gui import theme

    app = QApplication.instance() or QApplication([])
    theme.apply_theme(app)
    icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gui", "Mesofield_icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    window = PlaybackWindow(experiment_dir=experiment_dir, speed=speed, loop=loop)
    window.resize(1200, 900)
    window.show()
    app.exec()


# ---------------------------------------------------------------------------
# viewer
# ---------------------------------------------------------------------------


@click.command()
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
    from mesofield.gui import theme
    theme.apply_theme(app)
    icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gui", "Mesofield_icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    win = TiffViewer(initial_dir=initial_dir or None)
    win.resize(1100, 800)
    win.show()
    app.exec()
