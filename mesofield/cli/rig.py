"""``mesofield rig`` — manage this machine's canonical hardware.yaml configs."""

from __future__ import annotations

from pathlib import Path

import click

from ._richhelp import RichGroup


@click.group('rig', cls=RichGroup)
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
