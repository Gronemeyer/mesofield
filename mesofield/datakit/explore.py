"""Agnostic dataset and experiment exploration.

Lightweight introspection for both pre-load discovery results
(:class:`datakit.Dataset`) and post-load materialized outputs
(``pandas.DataFrame``).  Output uses ``rich`` when available and falls
back to plain indented text otherwise.

Quick start::

    from mesofield.datakit import explore, Dataset

    # Pre-load: inspect a discovered Dataset (or directory path)
    explore("path/to/experiment")
    explore(Dataset.from_directory(root))

    # Post-load: inspect a materialized DataFrame or pickle / HDF5
    explore("path/to/dataset.pkl")
    explore(materialized_df)

    # Programmatic access (no printing)
    report = explore(dataset, print_output=False)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TagSummary:
    """Per-source statistics for a discovered Dataset."""

    tag: str
    file_count: int
    coverage_pct: float
    extensions: tuple[str, ...]


@dataclass(frozen=True)
class DatasetInventoryReport:
    """Pre-load overview of a :class:`datakit.Dataset` inventory."""

    roots: tuple[str, ...]
    n_subjects: int
    n_sessions: int
    n_tasks: int
    has_task_level: bool
    subjects: tuple[str, ...]
    sessions: tuple[str, ...]
    tasks: tuple[str, ...]
    tags: tuple[TagSummary, ...]
    n_total_files: int
    coverage_matrix: dict[str, dict[str, float]]  # tag -> {subject: pct}


@dataclass(frozen=True)
class ColumnInfo:
    """Type and structure information for a single materialized column."""

    source: str
    feature: str
    dtype: str
    detail: str


@dataclass(frozen=True)
class MaterializedReport:
    """Post-load overview of a materialized dataset DataFrame."""

    shape: tuple[int, int]
    index_names: tuple[str, ...]
    index_counts: dict[str, int]
    n_sources: int
    n_features: int
    memory_mb: float
    sources: tuple[str, ...]
    source_features: dict[str, tuple[str, ...]]
    columns: tuple[ColumnInfo, ...]
    coverage: dict[str, float]
    hierarchy: dict[str, Any]


# ---------------------------------------------------------------------------
# Value inspection helpers
# ---------------------------------------------------------------------------


def _describe_value(val: object) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    if isinstance(val, np.ndarray):
        return f"ndarray {val.dtype} {val.shape}"
    if isinstance(val, pd.DataFrame):
        return f"DataFrame {val.shape[0]}x{val.shape[1]}"
    if isinstance(val, pd.Series):
        return f"Series len={len(val)}"
    if isinstance(val, dict):
        n = len(val)
        keys_preview = ", ".join(list(val.keys())[:3])
        if n > 3:
            keys_preview += ", ..."
        return f"dict({n}) [{keys_preview}]"
    if isinstance(val, (list, tuple)):
        return f"{type(val).__name__} len={len(val)}"
    return type(val).__name__


def _inspect_column(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "all null"
    sample = non_null.iloc[0]
    desc = _describe_value(sample)
    if len(non_null) > 1:
        other_desc = _describe_value(non_null.iloc[-1])
        if other_desc != desc:
            desc += " (varies)"
    return desc


# ---------------------------------------------------------------------------
# Inventory exploration (pre-load)
# ---------------------------------------------------------------------------


def explore_inventory(dataset: Any) -> DatasetInventoryReport:
    """Analyse a :class:`datakit.Dataset` without loading any files."""
    from .core import Dataset

    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected datakit.Dataset, got {type(dataset).__name__}")

    inv = dataset.inventory
    idx = inv.index
    subjects = tuple(dataset.subjects)
    sessions = tuple(dataset.sessions)
    has_task = dataset.has_task_level
    tasks: tuple[str, ...] = ()
    if has_task and idx.nlevels >= 3:
        tasks = tuple(sorted(idx.get_level_values(2).dropna().unique().tolist()))

    # Per-tag summaries
    tag_summaries: list[TagSummary] = []
    coverage_matrix: dict[str, dict[str, float]] = {}
    total_files = 0

    for tag in sorted(inv.columns):
        col = inv[tag]
        count = int(col.notna().sum())
        total_files += count
        coverage_pct = round(float(col.notna().mean()) * 100.0, 1) if len(col) else 0.0

        exts = sorted({Path(str(v)).suffix.lower() for v in col.dropna().tolist() if str(v)})
        exts_tuple = tuple(e for e in exts if e)

        tag_summaries.append(
            TagSummary(
                tag=tag,
                file_count=count,
                coverage_pct=coverage_pct,
                extensions=exts_tuple,
            )
        )

        if subjects:
            subj_cov: dict[str, float] = {}
            for subj in subjects:
                try:
                    sub_slice = inv.xs(subj, level=0)[tag]
                    subj_cov[subj] = round(float(sub_slice.notna().mean()) * 100.0, 1)
                except KeyError:
                    subj_cov[subj] = 0.0
            coverage_matrix[tag] = subj_cov

    roots = tuple(str(r) for r in dataset.roots) or ("<in-memory>",)

    return DatasetInventoryReport(
        roots=roots,
        n_subjects=len(subjects),
        n_sessions=len(sessions),
        n_tasks=len(tasks),
        has_task_level=has_task,
        subjects=subjects,
        sessions=sessions,
        tasks=tasks,
        tags=tuple(tag_summaries),
        n_total_files=total_files,
        coverage_matrix=coverage_matrix,
    )


# ---------------------------------------------------------------------------
# Materialized exploration (post-load)
# ---------------------------------------------------------------------------


def explore_materialized(dataset: pd.DataFrame) -> MaterializedReport:
    """Analyse a materialized dataset DataFrame."""
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(dataset).__name__}")

    shape = dataset.shape
    idx = dataset.index
    raw_names = tuple(idx.names) if hasattr(idx, "names") else ()
    index_names: tuple[str, ...] = tuple(
        str(n) if n is not None else f"level_{i}" for i, n in enumerate(raw_names)
    )
    index_counts: dict[str, int] = {}
    for i, label in enumerate(index_names):
        index_counts[str(label)] = int(idx.get_level_values(i).nunique())

    cols = dataset.columns
    if isinstance(cols, pd.MultiIndex) and cols.nlevels >= 2:
        sources = tuple(sorted(cols.get_level_values(0).unique().tolist()))
        src_features: dict[str, tuple[str, ...]] = {}
        for src in sources:
            feats = sorted(
                cols.get_level_values(1)[cols.get_level_values(0) == src].unique().tolist()
            )
            src_features[src] = tuple(feats)
        n_features = int(cols.get_level_values(1).nunique())
    else:
        sources = ()
        src_features = {}
        n_features = len(cols)

    memory_mb = round(int(dataset.memory_usage(deep=True).sum()) / (1024 * 1024), 2)

    col_infos: list[ColumnInfo] = []
    for col_key in cols:
        if isinstance(col_key, tuple):
            src, feat = str(col_key[0]), str(col_key[1])
        else:
            src, feat = "", str(col_key)
        series = dataset[col_key]
        dt = str(series.dtype)
        detail = _inspect_column(series) if dt == "object" else dt
        col_infos.append(ColumnInfo(source=src, feature=feat, dtype=dt, detail=detail))

    coverage: dict[str, float] = {}
    if isinstance(cols, pd.MultiIndex) and cols.nlevels >= 2:
        for src in sources:
            src_cols = dataset.xs(src, axis=1, level=0, drop_level=True)
            non_null = src_cols.notna().any(axis=1).mean()  # type: ignore[call-overload]
            coverage[src] = round(float(non_null) * 100.0, 1)
    else:
        for c in cols:
            coverage[str(c)] = round(float(dataset[c].notna().mean()) * 100.0, 1)

    hierarchy: dict[str, Any] = {}
    for row_key in idx:
        parts = row_key if isinstance(row_key, tuple) else (row_key,)
        node = hierarchy
        for part in parts:
            key = str(part)
            node = node.setdefault(key, {})

    return MaterializedReport(
        shape=shape,
        index_names=index_names,
        index_counts=index_counts,
        n_sources=len(sources),
        n_features=n_features,
        memory_mb=memory_mb,
        sources=sources,
        source_features=src_features,
        columns=tuple(col_infos),
        coverage=coverage,
        hierarchy=hierarchy,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_HAS_RICH: bool | None = None


def _rich_available() -> bool:
    global _HAS_RICH
    if _HAS_RICH is None:
        try:
            import rich  # noqa: F401

            _HAS_RICH = True
        except ImportError:
            _HAS_RICH = False
    return _HAS_RICH


def _render_inventory_rich(report: DatasetInventoryReport) -> str:
    from io import StringIO

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree

    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120)

    overview = (
        f"[bold]Root(s):[/bold] {', '.join(report.roots)}\n"
        f"[bold]Subjects:[/bold] {report.n_subjects}  "
        f"[bold]Sessions:[/bold] {report.n_sessions}  "
        f"[bold]Tasks:[/bold] {report.n_tasks}  "
        f"[bold]Files:[/bold] {report.n_total_files}\n"
        f"[bold]Task-level index:[/bold] {report.has_task_level}"
    )
    console.print(Panel(overview, title="Dataset Inventory", border_style="blue"))

    tree = Tree("[bold]Subjects[/bold]")
    for subj in report.subjects:
        subj_node = tree.add(f"[cyan]{subj}[/cyan]")
        for ses in report.sessions:
            ses_node = subj_node.add(f"[green]{ses}[/green]")
            for task in report.tasks:
                ses_node.add(f"[dim]{task}[/dim]")
    console.print(tree)

    table = Table(title="Source Tags", show_lines=False)
    table.add_column("Tag", style="bold")
    table.add_column("Files", justify="right")
    table.add_column("Coverage %", justify="right")
    table.add_column("Extensions")
    for ts in report.tags:
        cov_style = (
            "green" if ts.coverage_pct >= 80 else ("yellow" if ts.coverage_pct >= 50 else "red")
        )
        table.add_row(
            ts.tag,
            str(ts.file_count),
            f"[{cov_style}]{ts.coverage_pct:.0f}%[/{cov_style}]",
            ", ".join(ts.extensions) if ts.extensions else "-",
        )
    console.print(table)

    if len(report.subjects) > 1 and report.coverage_matrix:
        cov_table = Table(title="Coverage by Subject", show_lines=True)
        cov_table.add_column("Subject", style="bold")
        for ts in report.tags:
            cov_table.add_column(ts.tag, justify="center", max_width=8)
        for subj in report.subjects:
            cells: list[str] = []
            for ts in report.tags:
                pct = report.coverage_matrix.get(ts.tag, {}).get(subj, 0)
                style = "green" if pct >= 80 else ("yellow" if pct >= 50 else "red")
                cells.append(f"[{style}]{pct:.0f}%[/{style}]")
            cov_table.add_row(subj, *cells)
        console.print(cov_table)

    return buf.getvalue()


def _render_inventory_plain(report: DatasetInventoryReport) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("DATASET INVENTORY")
    lines.append("=" * 60)
    lines.append(f"  Root(s):    {', '.join(report.roots)}")
    lines.append(f"  Subjects:   {report.n_subjects}")
    lines.append(f"  Sessions:   {report.n_sessions}")
    lines.append(f"  Tasks:      {report.n_tasks}")
    lines.append(f"  Files:      {report.n_total_files}")
    lines.append(f"  Task-level: {report.has_task_level}")
    lines.append("")
    lines.append("STRUCTURE")
    lines.append("-" * 40)
    for subj in report.subjects:
        lines.append(f"  {subj}")
        for ses in report.sessions:
            lines.append(f"    {ses}")
            for task in report.tasks:
                lines.append(f"      {task}")
    lines.append("")
    lines.append("SOURCE TAGS")
    lines.append("-" * 40)
    lines.append(f"  {'Tag':<25} {'Files':>5}  {'Coverage':>8}  Extensions")
    for ts in report.tags:
        ext_str = ", ".join(ts.extensions) if ts.extensions else "-"
        lines.append(
            f"  {ts.tag:<25} {ts.file_count:>5}  {ts.coverage_pct:>7.0f}%  {ext_str}"
        )
    return "\n".join(lines)


def _build_hierarchy_tree(
    node: Any, tree: dict[str, Any], depth: int, level_names: tuple[str, ...]
) -> None:
    styles = ["cyan", "green", "dim"]
    style = styles[depth] if depth < len(styles) else ""
    for key, children in sorted(tree.items()):
        child_node = node.add(f"[{style}]{key}[/{style}]" if style else key)
        if isinstance(children, dict) and children:
            _build_hierarchy_tree(child_node, children, depth + 1, level_names)


def _render_materialized_rich(report: MaterializedReport) -> str:
    from io import StringIO

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree

    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120)

    idx_parts = ", ".join(f"{k}={v}" for k, v in report.index_counts.items())
    overview = (
        f"[bold]Shape:[/bold] {report.shape[0]} rows x {report.shape[1]} columns\n"
        f"[bold]Index:[/bold] ({idx_parts})\n"
        f"[bold]Sources:[/bold] {report.n_sources}  "
        f"[bold]Features:[/bold] {report.n_features}  "
        f"[bold]Memory:[/bold] {report.memory_mb:.1f} MB"
    )
    console.print(Panel(overview, title="Materialized Dataset", border_style="blue"))

    tree = Tree("[bold]Sources[/bold]")
    for src in report.sources:
        src_node = tree.add(f"[cyan]{src}[/cyan]")
        for feat in report.source_features.get(src, ()):
            src_node.add(f"[dim]{feat}[/dim]")
    console.print(tree)

    type_table = Table(title="Column Types", show_lines=False)
    type_table.add_column("Source", style="bold")
    type_table.add_column("Feature")
    type_table.add_column("dtype")
    type_table.add_column("Detail")
    for ci in report.columns:
        type_table.add_row(ci.source, ci.feature, ci.dtype, ci.detail)
    console.print(type_table)

    cov_table = Table(title="Source Coverage", show_lines=False)
    cov_table.add_column("Source", style="bold")
    cov_table.add_column("Available %", justify="right")
    for src in report.sources:
        pct = report.coverage.get(src, 0)
        style = "green" if pct >= 80 else ("yellow" if pct >= 50 else "red")
        cov_table.add_row(src, f"[{style}]{pct:.0f}%[/{style}]")
    console.print(cov_table)

    if report.hierarchy:
        idx_tree = Tree("[bold]Index Hierarchy[/bold]")
        _build_hierarchy_tree(idx_tree, report.hierarchy, 0, report.index_names)
        console.print(idx_tree)

    return buf.getvalue()


def _render_materialized_plain(report: MaterializedReport) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("MATERIALIZED DATASET")
    lines.append("=" * 60)
    idx_parts = ", ".join(f"{k}={v}" for k, v in report.index_counts.items())
    lines.append(f"  Shape:    {report.shape[0]} rows x {report.shape[1]} columns")
    lines.append(f"  Index:    ({idx_parts})")
    lines.append(f"  Sources:  {report.n_sources}")
    lines.append(f"  Features: {report.n_features}")
    lines.append(f"  Memory:   {report.memory_mb:.1f} MB")
    lines.append("")
    lines.append("STRUCTURE")
    lines.append("-" * 40)
    for src in report.sources:
        feats = report.source_features.get(src, ())
        lines.append(f"  {src}")
        for f in feats:
            lines.append(f"    {f}")
    lines.append("")
    lines.append("COLUMN TYPES")
    lines.append("-" * 40)
    lines.append(f"  {'Source':<20} {'Feature':<20} {'dtype':<12} Detail")
    for ci in report.columns:
        lines.append(
            f"  {ci.source:<20} {ci.feature:<20} {ci.dtype:<12} {ci.detail}"
        )
    lines.append("")
    lines.append("SOURCE COVERAGE")
    lines.append("-" * 40)
    for src in report.sources:
        pct = report.coverage.get(src, 0)
        bar_n = int(pct / 5)
        bar = "#" * bar_n + "." * (20 - bar_n)
        lines.append(f"  {src:<20} [{bar}] {pct:.0f}%")
    return "\n".join(lines)


def _render(report: DatasetInventoryReport | MaterializedReport) -> str:
    if isinstance(report, DatasetInventoryReport):
        return (
            _render_inventory_rich(report)
            if _rich_available()
            else _render_inventory_plain(report)
        )
    return (
        _render_materialized_rich(report)
        if _rich_available()
        else _render_materialized_plain(report)
    )


# ---------------------------------------------------------------------------
# Public API — single discoverable namespace
# ---------------------------------------------------------------------------


class _Explore:
    """Callable namespace for dataset and experiment exploration.

    Use ``explore(target)`` for auto-dispatch, or pick a specific entry
    point via dot notation:

    - ``explore(target)`` — auto-dispatch on a Dataset, DataFrame, or path
    - ``explore.inventory(dataset)`` — pre-load inventory report
    - ``explore.materialized(df)`` — post-load DataFrame report
    - ``explore.path(p)`` — load and report from a directory / .pkl / .h5
    - ``explore.render(report)`` — format a report as a string

    Report dataclasses are also exposed for type hints:
    ``explore.DatasetInventoryReport``, ``explore.MaterializedReport``,
    ``explore.ColumnInfo``, ``explore.TagSummary``.
    """

    # Report dataclasses (exposed for type hints + isinstance checks)
    DatasetInventoryReport = DatasetInventoryReport
    MaterializedReport = MaterializedReport
    ColumnInfo = ColumnInfo
    TagSummary = TagSummary

    def __call__(
        self,
        target: Any,
        *,
        print_output: bool = True,
        hdf_key: str = "dataset",
    ) -> DatasetInventoryReport | MaterializedReport:
        """Explore the structure of a Dataset or materialized DataFrame.

        Parameters
        ----------
        target
            One of:
            - :class:`datakit.Dataset`
            - ``pandas.DataFrame`` (materialized result)
            - ``Path`` or ``str`` to a directory (experiment root),
              ``.pkl``, or ``.h5`` / ``.hdf5`` file
        print_output
            If ``True`` (default), print a formatted summary to stdout.
        hdf_key
            HDF5 key used when ``target`` is an ``.h5`` file.
        """
        from .core import Dataset

        if isinstance(target, Dataset):
            report: DatasetInventoryReport | MaterializedReport = explore_inventory(target)
        elif isinstance(target, pd.DataFrame):
            report = explore_materialized(target)
        elif isinstance(target, (str, Path)):
            report = self.path(Path(target), hdf_key=hdf_key, print_output=False)
        else:
            raise TypeError(f"Unsupported target type: {type(target).__name__}")

        if print_output:
            print(_render(report))
        return report

    def inventory(
        self, dataset: Any, *, print_output: bool = False
    ) -> DatasetInventoryReport:
        """Build a pre-load :class:`DatasetInventoryReport` from a Dataset."""
        report = explore_inventory(dataset)
        if print_output:
            print(_render(report))
        return report

    def materialized(
        self, dataset: pd.DataFrame, *, print_output: bool = False
    ) -> MaterializedReport:
        """Build a post-load :class:`MaterializedReport` from a DataFrame."""
        report = explore_materialized(dataset)
        if print_output:
            print(_render(report))
        return report

    def path(
        self,
        path: Union[str, Path],
        *,
        hdf_key: str = "dataset",
        print_output: bool = False,
    ) -> DatasetInventoryReport | MaterializedReport:
        """Build a report from a directory, ``.pkl``, or ``.h5`` / ``.hdf5`` file."""
        from .core import Dataset

        p = Path(path).expanduser().resolve()

        if p.is_dir():
            report: DatasetInventoryReport | MaterializedReport = explore_inventory(
                Dataset.from_directory(p)
            )
        else:
            if not p.is_file():
                raise FileNotFoundError(f"Path does not exist: {p}")
            suffix = p.suffix.lower()
            if suffix == ".pkl":
                df = pd.read_pickle(p)
            elif suffix in (".h5", ".hdf5"):
                df = pd.read_hdf(p, key=hdf_key)
            else:
                raise ValueError(
                    f"Unsupported file type: {suffix} "
                    "(expected directory, .pkl, .h5, or .hdf5)"
                )
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
            report = explore_materialized(df)

        if print_output:
            print(_render(report))
        return report

    @staticmethod
    def render(report: DatasetInventoryReport | MaterializedReport) -> str:
        """Render a report as a formatted string (rich if available, else plain)."""
        return _render(report)

    def __repr__(self) -> str:
        return (
            "<datakit.explore — call as explore(target); see "
            "explore.inventory, explore.materialized, explore.path, explore.render>"
        )


explore = _Explore()


__all__ = ["explore"]

