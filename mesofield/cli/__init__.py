"""Mesofield command-line interface.

The CLI is organized into a small set of day-to-day acquisition commands at
the top level, plus four command groups:

\b
    mesofield launch | init | playback | viewer   acquisition workflow
    mesofield rig ...        manage canonical hardware.yaml rigs
    mesofield datakit ...    build / explore / profile / inspect datasets
    mesofield process ...    batch-process & convert recorded data
    mesofield tools ...      setup, export, and diagnostic utilities

Run ``mesofield <cmd> --help`` (or ``mesofield <group> --help``) for details.
"""

from __future__ import annotations

import os

import click

# Disable debugger warning about the use of frozen modules
os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")

from ._richhelp import RichGroup
from .acquire import init, launch, playback, viewer
from .datakit import datakit
from .process import process
from .rig import rig
from .tools import tools


@click.group(cls=RichGroup)
def cli():
    """Mesofield Command Line Interface."""


# Set the free-standing acquisition commands (launch/init/playback/viewer)
# apart from the command groups in the help tree.
cli.loose_command_heading = "acquisition workflow"


# --- Top-level acquisition commands ---
cli.add_command(launch)
cli.add_command(init)
cli.add_command(playback)
cli.add_command(viewer)

# --- Command groups ---
cli.add_command(rig)
cli.add_command(datakit)
cli.add_command(process)
cli.add_command(tools)


if __name__ == "__main__":
    cli()
