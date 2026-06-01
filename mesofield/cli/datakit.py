"""``mesofield datakit`` — build, explore, profile, and inspect datasets.

A thin Click surface over :mod:`mesofield.datakit`. Each command maps to a
public datakit API:

\b
    build     -> Dataset.from_directory(...).save(...)
    explore   -> datakit.explore(target)
    profile   -> datakit.profile_materialized(pkl)
    inspect   -> datakit.inspect_sources(Dataset)
    shell     -> datakit.open_shell(target)
"""

from __future__ import annotations

from pathlib import Path

import click

from ._richhelp import RichGroup


@click.group('datakit', cls=RichGroup)
def datakit():
    """Build, explore, profile, and inspect materialized datasets."""


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


@datakit.command('build')
@click.argument('input_path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--output', '-o', 'output_path', type=click.Path(), default=None,
              help='Output file path (default: <experiment>/processed/YYMMDD_dataset_mvp.<fmt>)')
@click.option('--tags', '-t', multiple=True, default=None,
              help='Source tags to include (repeatable; default: all configured tags)')
@click.option('--format', '-f', 'fmt', type=click.Choice(['h5', 'parquet', 'csv', 'pickle']),
              default='h5', show_default=True, help='Output format')
@click.option('--progress', is_flag=True, help='Show a progress bar during materialization')
@click.option('--shell', 'open_shell_after', is_flag=True,
              help='Drop into an IPython session after building')
def build(input_path, output_path, tags, fmt, progress, open_shell_after):
    """Build a materialized dataset from an experiment directory.

    Discovers the BIDS hierarchy under INPUT_PATH, loads all registered
    data sources, and writes a single dataset file.
    """
    from mesofield.datakit import Dataset

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
    if open_shell_after:
        from mesofield.datakit import open_shell
        open_shell(result_path)


# ---------------------------------------------------------------------------
# explore
# ---------------------------------------------------------------------------


@datakit.command('explore')
@click.argument('target', type=click.Path(exists=True))
@click.option('--hdf-key', default='dataset', show_default=True,
              help='HDF5 key to read when TARGET is an .h5/.hdf5 file.')
def explore(target, hdf_key):
    """Print a structural report for a dataset.

    TARGET may be an experiment directory (pre-load inventory report), or a
    materialized ``.pkl`` / ``.h5`` file (post-load DataFrame report).
    """
    from mesofield.datakit import explore as explore_fn

    explore_fn(Path(target), print_output=True, hdf_key=hdf_key)


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


@datakit.command('profile')
@click.argument('pickle_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--json', 'json_path', type=click.Path(), default=None,
              help='Write a JSON report to this path.')
@click.option('--verbose', is_flag=True,
              help='Print the detailed per-column breakdown instead of just the summary.')
@click.option('--top-cells', default=20, show_default=True,
              help='Number of largest individual cells to include.')
def profile(pickle_path, json_path, verbose, top_cells):
    """Profile the memory / storage footprint of a materialized dataset.

    PICKLE_PATH is a materialized ``.pkl`` file produced by ``datakit build``.
    """
    from mesofield.datakit import profile_materialized

    report = profile_materialized(Path(pickle_path), top_n_cells=top_cells)
    click.echo(report.verbose(top_n_cells=top_cells) if verbose else report.summary())
    if json_path:
        out = report.to_json(json_path)
        click.secho(f"\nWrote JSON report to: {out}", fg="green")


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


@datakit.command('inspect')
@click.argument('input_path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--tags', '-t', multiple=True, default=None,
              help='Source tags to report (repeatable; default: all discovered tags)')
def inspect(input_path, tags):
    """Print per-source coverage for an experiment directory.

    Discovers the BIDS hierarchy under INPUT_PATH and reports how many rows
    carry each registered source (present / total / missing / coverage),
    without loading any file payloads.
    """
    from mesofield.datakit import Dataset, inspect_sources

    ds = Dataset.from_directory(Path(input_path))
    summary = inspect_sources(ds, sources=list(tags) if tags else None)
    if summary.empty:
        click.secho("No registered sources found in the inventory.", fg="yellow")
        return
    click.echo(summary.to_string())


# ---------------------------------------------------------------------------
# shell
# ---------------------------------------------------------------------------


@datakit.command('shell')
@click.argument('target', type=click.Path(exists=True), required=False, default=None)
@click.option('--hdf-key', default='dataset', show_default=True,
              help='HDF5 key to read when TARGET is an .h5/.hdf5 file.')
def shell(target, hdf_key):
    """Open an IPython shell pre-loaded with a datakit object.

    TARGET may be an experiment directory (seeds ``dataset`` + ``inventory``)
    or a materialized ``.pkl`` / ``.h5`` file (seeds ``df``). Omit TARGET for
    a bare datakit shell. Each session also exposes ``datakit``, ``explore``,
    and a pre-computed ``report``.
    """
    from mesofield.datakit import open_shell

    open_shell(Path(target) if target else None, hdf_key=hdf_key)
