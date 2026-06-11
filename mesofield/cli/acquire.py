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


def _resolve_launch_target(arg):
    """Resolve a ``launch`` argument to a filesystem path (or ``None``).

    Resolution order, so the common cases just work and a typo can't silently
    boot the wrong thing:

    1. an existing path (``hardware.yaml`` / ``experiment.json`` / ``procedure.py``
       / a directory) is used verbatim;
    2. a canonical **rig name** from this machine's store
       (``mesofield rig list``) resolves to its ``hardware.yaml``;
    3. the literal ``dev`` boots a throwaway mock rig (runs without hardware);
    4. anything else prints the known rigs and returns ``None`` so Mesofield
       opens in its default state with the Configuration Wizard.
    """
    if not arg:
        return None
    if os.path.exists(arg):
        return arg

    from mesofield.scaffold import rigs

    try:
        return str(rigs._resolve_existing(arg))
    except FileNotFoundError:
        pass

    if arg == "dev":
        import tempfile
        from mesofield.scaffold.experiment import _hardware_yaml_mock

        fd, tmp = tempfile.mkstemp(prefix="mesofield_dev_", suffix=".yaml")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(_hardware_yaml_mock())
        return tmp

    click.secho(
        f"No path or rig named {arg!r}. Known rigs: "
        f"{', '.join(rigs.list_rigs()) or '(none)'}.\n"
        "Opening in default state -- pick a rig in the Configuration Wizard.",
        fg="yellow",
    )
    return None


@click.command()
@click.argument('config', type=click.Path(), required=False, default=None)
def launch(config):
    """Launch the Mesofield acquisition interface.

    CONFIG is optional and may be a canonical **rig name** (see
    ``mesofield rig list``), the literal ``dev`` (a mock rig that runs without
    hardware), or a path to a ``hardware.yaml`` (the rig to bring up), an
    ``experiment.json``, a scripted ``procedure.py``, or an experiment directory
    containing them. The rig is the only thing needed to launch; experiment
    parameters can be sideloaded or generated from the Configuration Wizard.
    When omitted, Mesofield opens in a default state and the wizard is shown.
    """
    from mesofield.gui.maingui import run_gui
    from mesofield.base import load_procedure_from_config

    procedure = load_procedure_from_config(_resolve_launch_target(config))
    run_gui(procedure)


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
@click.option('--speed', default=1.0, show_default=True, help='Playback speed multiplier')
@click.option('--loop/--no-loop', default=False, show_default=True, help='Loop playback when finished')
def playback(experiment_dir: str, speed: float, loop: bool):
    """Launch Mesofield in playback mode for a recorded experiment."""

    from mesofield.playback import (
        discover_playback_context,
        discover_playback_sessions,
        launch_playback_app,
    )

    sessions = discover_playback_sessions(Path(experiment_dir))
    context = discover_playback_context(Path(experiment_dir), speed=speed, loop=loop)
    launch_playback_app(context, browser_sessions=sessions)


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
