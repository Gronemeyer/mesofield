"""Memory / storage profiler for materialized datakit datasets.

Profiles a materialized ``pandas.DataFrame`` (or a saved ``.pkl`` / ``.h5``
file containing one), with first-class support for the nested object
payloads typical of datakit outputs:

- Object-dtype columns containing embedded ``pandas.DataFrame`` instances
  (recursed into, per inner column).
- Object-dtype columns containing ``numpy.ndarray`` payloads.
- ``meta`` columns (or any object cell) holding nested ``dict`` payloads —
  recursively profiled key by key.
- Arbitrary Python objects, sized via ``sys.getsizeof`` with cycle-safe
  recursion into containers (``list``/``tuple``/``set``/``dict``).

The profiler leans on the pandas ``memory_usage(deep=True)`` API for
contiguous-dtype columns and the index, and only falls back to recursive
sizing for ``object`` payloads where pandas reports only the pointer cost.

Output
------
:func:`profile_materialized` returns a :class:`MaterializedMemoryReport`
that can render:

- ``summary()``    — concise human-readable summary string
- ``verbose()``    — detailed human-readable breakdown
- ``to_dict()``    — JSON-serialisable nested dict
- ``to_json(path)``— write JSON file to disk

CLI usage::

    mesofield datakit profile path/to/materialized.pkl --verbose
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Sizing helpers
# ---------------------------------------------------------------------------

# Cap on recursion depth for nested container sizing to avoid pathological
# graphs.  Most datakit payloads are shallow (df-of-df, dict-of-arrays).
_MAX_DEPTH = 32


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except Exception:
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


def _deep_size(value: Any, seen: set[int] | None = None, depth: int = 0) -> int:
    """Recursive, cycle-safe size estimate in bytes for arbitrary objects.

    Uses pandas / numpy native sizing where available, falls back to
    ``sys.getsizeof`` + container recursion otherwise.
    """
    if seen is None:
        seen = set()
    if depth > _MAX_DEPTH:
        return int(sys.getsizeof(value))
    obj_id = id(value)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if value is None:
        return 0
    if isinstance(value, np.ndarray):
        return int(value.nbytes + sys.getsizeof(value))
    if isinstance(value, pd.DataFrame):
        return int(value.memory_usage(index=True, deep=True).sum())
    if isinstance(value, pd.Series):
        return int(value.memory_usage(index=True, deep=True))
    if isinstance(value, pd.Index):
        return int(value.memory_usage(deep=True))
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        return int(sys.getsizeof(value))
    if isinstance(value, dict):
        size = sys.getsizeof(value)
        for k, v in value.items():
            size += _deep_size(k, seen, depth + 1)
            size += _deep_size(v, seen, depth + 1)
        return int(size)
    if isinstance(value, (list, tuple, set, frozenset)):
        size = sys.getsizeof(value)
        for item in value:
            size += _deep_size(item, seen, depth + 1)
        return int(size)
    if hasattr(value, "__dict__"):
        return int(sys.getsizeof(value) + _deep_size(vars(value), seen, depth + 1))
    return int(sys.getsizeof(value))


def _flatten_dict_sizes(
    d: dict, prefix: str = "", out: dict[str, int] | None = None, depth: int = 0
) -> dict[str, int]:
    """Flatten a nested dict into ``{dotted.key: bytes}`` for profiling."""
    if out is None:
        out = {}
    if depth > _MAX_DEPTH:
        out[prefix or "<root>"] = _deep_size(d)
        return out
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            _flatten_dict_sizes(v, key, out, depth + 1)
        else:
            out[key] = _deep_size(v)
    return out


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ColumnMemory:
    column: str
    source: str
    feature: str
    dtype: str
    n_total: int
    n_non_null: int
    n_null: int
    pandas_deep_bytes: int
    pointer_array_bytes: int
    object_payload_bytes: int
    estimated_total_bytes: int
    avg_non_null_cell_bytes: int
    max_non_null_cell_bytes: int
    value_type_counts: dict[str, int] = field(default_factory=dict)
    nested_dataframe_inner_bytes: dict[str, int] = field(default_factory=dict)
    nested_dict_key_bytes: dict[str, int] = field(default_factory=dict)


@dataclass
class CellMemory:
    row: str
    column: str
    source: str
    feature: str
    value_type: str
    estimated_bytes: int


@dataclass
class MaterializedMemoryReport:
    source_path: str | None
    shape: tuple[int, int]
    index_names: tuple[str, ...]
    column_levels: tuple[str, ...]
    index_bytes: int
    columns_index_bytes: int
    pandas_deep_total_bytes: int
    estimated_total_bytes: int
    columns: list[ColumnMemory]
    by_source_bytes: dict[str, int]
    by_source_feature_bytes: dict[str, int]  # "source.feature" -> bytes
    largest_cells: list[CellMemory]

    # --- Rendering ---------------------------------------------------------

    @staticmethod
    def _fmt_bytes(n: int) -> str:
        n = int(n)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if abs(n) < 1024.0 or unit == "TB":
                return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,} B"
            n /= 1024.0
        return f"{n:.1f} TB"

    def summary(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("MATERIALIZED DATASET MEMORY PROFILE")
        lines.append("=" * 60)
        if self.source_path:
            lines.append(f"  Source:        {self.source_path}")
        lines.append(f"  Shape:         {self.shape[0]} rows x {self.shape[1]} columns")
        lines.append(f"  Index levels:  {', '.join(self.index_names) or '-'}")
        lines.append(f"  Column levels: {', '.join(self.column_levels) or '-'}")
        lines.append(f"  Index bytes:   {self._fmt_bytes(self.index_bytes)}")
        lines.append(f"  Columns bytes: {self._fmt_bytes(self.columns_index_bytes)}")
        lines.append(
            f"  Pandas deep:   {self._fmt_bytes(self.pandas_deep_total_bytes)} "
            "(pointer-only for object dtypes)"
        )
        lines.append(
            f"  Estimated:     {self._fmt_bytes(self.estimated_total_bytes)} "
            "(recursive object payload sizing)"
        )
        lines.append("")
        lines.append("TOP SOURCES BY ESTIMATED SIZE")
        lines.append("-" * 40)
        top_sources = sorted(self.by_source_bytes.items(), key=lambda kv: kv[1], reverse=True)[:10]
        for src, b in top_sources:
            lines.append(f"  {src:<30} {self._fmt_bytes(b):>12}")
        return "\n".join(lines)

    def verbose(self, *, top_n_cells: int = 20) -> str:
        lines = [self.summary(), ""]

        lines.append("PER-COLUMN BREAKDOWN")
        lines.append("-" * 40)
        header = (
            f"  {'source.feature':<40} {'dtype':<12} {'non_null':>8} "
            f"{'pandas':>12} {'estimated':>12}"
        )
        lines.append(header)
        sorted_cols = sorted(self.columns, key=lambda c: c.estimated_total_bytes, reverse=True)
        for c in sorted_cols:
            lines.append(
                f"  {c.column:<40} {c.dtype:<12} {c.n_non_null:>8} "
                f"{self._fmt_bytes(c.pandas_deep_bytes):>12} "
                f"{self._fmt_bytes(c.estimated_total_bytes):>12}"
            )
            if c.value_type_counts:
                types = ", ".join(f"{t}={n}" for t, n in c.value_type_counts.items())
                lines.append(f"      types: {types}")
            if c.nested_dataframe_inner_bytes:
                lines.append("      nested DataFrame inner columns:")
                inner_sorted = sorted(
                    c.nested_dataframe_inner_bytes.items(), key=lambda kv: kv[1], reverse=True
                )
                for name, b in inner_sorted:
                    lines.append(f"        {name:<32} {self._fmt_bytes(b):>12}")
            if c.nested_dict_key_bytes:
                lines.append("      nested dict keys (aggregated across rows):")
                key_sorted = sorted(
                    c.nested_dict_key_bytes.items(), key=lambda kv: kv[1], reverse=True
                )
                for name, b in key_sorted[:20]:
                    lines.append(f"        {name:<32} {self._fmt_bytes(b):>12}")
                if len(key_sorted) > 20:
                    lines.append(f"        ... and {len(key_sorted) - 20} more keys")

        if self.largest_cells:
            lines.append("")
            lines.append(f"TOP {min(top_n_cells, len(self.largest_cells))} LARGEST CELLS")
            lines.append("-" * 40)
            lines.append(
                f"  {'row':<40} {'source.feature':<30} {'type':<14} {'size':>12}"
            )
            for cm in self.largest_cells[:top_n_cells]:
                lines.append(
                    f"  {cm.row:<40} {cm.column:<30} {cm.value_type:<14} "
                    f"{self._fmt_bytes(cm.estimated_bytes):>12}"
                )

        return "\n".join(lines)

    # --- Serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "shape": list(self.shape),
            "index_names": list(self.index_names),
            "column_levels": list(self.column_levels),
            "index_bytes": int(self.index_bytes),
            "columns_index_bytes": int(self.columns_index_bytes),
            "pandas_deep_total_bytes": int(self.pandas_deep_total_bytes),
            "estimated_total_bytes": int(self.estimated_total_bytes),
            "by_source_bytes": {k: int(v) for k, v in self.by_source_bytes.items()},
            "by_source_feature_bytes": {
                k: int(v) for k, v in self.by_source_feature_bytes.items()
            },
            "columns": [
                {
                    "column": c.column,
                    "source": c.source,
                    "feature": c.feature,
                    "dtype": c.dtype,
                    "n_total": c.n_total,
                    "n_non_null": c.n_non_null,
                    "n_null": c.n_null,
                    "pandas_deep_bytes": c.pandas_deep_bytes,
                    "pointer_array_bytes": c.pointer_array_bytes,
                    "object_payload_bytes": c.object_payload_bytes,
                    "estimated_total_bytes": c.estimated_total_bytes,
                    "avg_non_null_cell_bytes": c.avg_non_null_cell_bytes,
                    "max_non_null_cell_bytes": c.max_non_null_cell_bytes,
                    "value_type_counts": c.value_type_counts,
                    "nested_dataframe_inner_bytes": c.nested_dataframe_inner_bytes,
                    "nested_dict_key_bytes": c.nested_dict_key_bytes,
                }
                for c in self.columns
            ],
            "largest_cells": [
                {
                    "row": cm.row,
                    "column": cm.column,
                    "source": cm.source,
                    "feature": cm.feature,
                    "value_type": cm.value_type,
                    "estimated_bytes": cm.estimated_bytes,
                }
                for cm in self.largest_cells
            ],
        }

    def to_json(self, path: Union[str, Path], *, indent: int = 2) -> Path:
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent, default=str)
        return out


# ---------------------------------------------------------------------------
# Core profiler
# ---------------------------------------------------------------------------


def _col_path(col_key: Any) -> str:
    if isinstance(col_key, tuple):
        return ".".join(str(x) for x in col_key)
    return str(col_key)


def _source_feature(col_key: Any) -> tuple[str, str]:
    if isinstance(col_key, tuple):
        src = str(col_key[0]) if len(col_key) > 0 else ""
        feat = str(col_key[1]) if len(col_key) > 1 else ""
        return src, feat
    return str(col_key), ""


def _row_path(row_key: Any) -> str:
    if isinstance(row_key, tuple):
        return ".".join(str(x) for x in row_key)
    return str(row_key)


def profile_materialized(
    target: Union[pd.DataFrame, str, Path],
    *,
    top_n_cells: int = 20,
    source_path: str | None = None,
) -> MaterializedMemoryReport:
    """Build a :class:`MaterializedMemoryReport` from a DataFrame or saved file.

    Parameters
    ----------
    target
        A materialized ``pandas.DataFrame`` or a path-like pointing to a
        ``.pkl`` / ``.pickle`` file produced by datakit.
    top_n_cells
        How many of the largest individual object cells to keep in the
        report.
    source_path
        Optional override for the path recorded in the report (useful when
        passing an already-loaded DataFrame).
    """
    if isinstance(target, (str, Path)):
        p = Path(target).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Pickle file not found: {p}")
        df = pd.read_pickle(p)
        resolved_source = str(p)
    elif isinstance(target, pd.DataFrame):
        df = target
        resolved_source = source_path
    else:
        raise TypeError(
            f"Expected pandas DataFrame or path to pickle, got {type(target).__name__}"
        )

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Loaded object is not a DataFrame, got {type(df).__name__}"
        )

    index_bytes = int(df.index.memory_usage(deep=True))
    columns_index_bytes = int(df.columns.memory_usage(deep=True))
    pandas_deep_total_bytes = int(df.memory_usage(index=True, deep=True).sum())

    column_reports: list[ColumnMemory] = []
    by_source_bytes: dict[str, int] = defaultdict(int)
    by_source_feature_bytes: dict[str, int] = defaultdict(int)
    all_cells: list[CellMemory] = []

    n_rows = int(len(df))

    for col_key in df.columns:
        series = df[col_key]
        path = _col_path(col_key)
        source, feature = _source_feature(col_key)
        non_null = int(series.notna().sum())
        null_count = int(n_rows - non_null)
        dtype = str(series.dtype)
        pandas_deep_bytes = int(series.memory_usage(index=False, deep=True))

        nested_df_inner: dict[str, int] = defaultdict(int)
        nested_dict_keys: dict[str, int] = defaultdict(int)
        type_counter: Counter[str] = Counter()

        if dtype != "object":
            pointer_array_bytes = pandas_deep_bytes
            object_payload_bytes = 0
            estimated_total_bytes = pandas_deep_bytes
            avg_cell = int(pandas_deep_bytes / non_null) if non_null else 0
            max_cell = int(series.dtype.itemsize) if hasattr(series.dtype, "itemsize") else avg_cell
            type_counter[dtype] = non_null
        else:
            pointer_array_bytes = int(series.memory_usage(index=False, deep=False))
            object_payload_bytes = 0
            max_cell = 0

            for row_key, value in series.items():
                if _is_missing(value):
                    continue

                type_name = type(value).__name__
                type_counter[type_name] += 1

                if isinstance(value, pd.DataFrame):
                    cell_bytes = int(value.memory_usage(index=True, deep=True).sum())
                    nested_df_inner["<index>"] += int(value.index.memory_usage(deep=True))
                    for inner_col in value.columns:
                        inner_series = value[inner_col]
                        nested_df_inner[str(inner_col)] += int(
                            inner_series.memory_usage(index=False, deep=True)
                        )
                elif isinstance(value, pd.Series):
                    cell_bytes = int(value.memory_usage(index=True, deep=True))
                elif isinstance(value, np.ndarray):
                    cell_bytes = int(value.nbytes + sys.getsizeof(value))
                elif isinstance(value, dict):
                    cell_bytes = _deep_size(value)
                    for k, b in _flatten_dict_sizes(value).items():
                        nested_dict_keys[k] += int(b)
                else:
                    cell_bytes = _deep_size(value)

                object_payload_bytes += cell_bytes
                if cell_bytes > max_cell:
                    max_cell = cell_bytes

                all_cells.append(
                    CellMemory(
                        row=_row_path(row_key),
                        column=path,
                        source=source,
                        feature=feature,
                        value_type=type_name,
                        estimated_bytes=int(cell_bytes),
                    )
                )

            estimated_total_bytes = int(pointer_array_bytes + object_payload_bytes)
            avg_cell = int(object_payload_bytes / non_null) if non_null else 0

        by_source_bytes[source] += int(estimated_total_bytes)
        by_source_feature_bytes[path] += int(estimated_total_bytes)

        column_reports.append(
            ColumnMemory(
                column=path,
                source=source,
                feature=feature,
                dtype=dtype,
                n_total=n_rows,
                n_non_null=non_null,
                n_null=null_count,
                pandas_deep_bytes=pandas_deep_bytes,
                pointer_array_bytes=int(pointer_array_bytes),
                object_payload_bytes=int(object_payload_bytes),
                estimated_total_bytes=int(estimated_total_bytes),
                avg_non_null_cell_bytes=int(avg_cell),
                max_non_null_cell_bytes=int(max_cell),
                value_type_counts=dict(type_counter),
                nested_dataframe_inner_bytes=dict(nested_df_inner),
                nested_dict_key_bytes=dict(nested_dict_keys),
            )
        )

    estimated_values_total = sum(c.estimated_total_bytes for c in column_reports)
    estimated_total_bytes = int(index_bytes + columns_index_bytes + estimated_values_total)

    all_cells.sort(key=lambda cm: cm.estimated_bytes, reverse=True)
    largest_cells = all_cells[:top_n_cells]

    raw_index_names = tuple(df.index.names) if hasattr(df.index, "names") else ()
    index_names = tuple(
        str(n) if n is not None else f"level_{i}" for i, n in enumerate(raw_index_names)
    )
    if isinstance(df.columns, pd.MultiIndex):
        column_levels = tuple(
            str(n) if n is not None else f"level_{i}"
            for i, n in enumerate(df.columns.names)
        )
    else:
        column_levels = ("columns",)

    return MaterializedMemoryReport(
        source_path=resolved_source,
        shape=tuple(df.shape),
        index_names=index_names,
        column_levels=column_levels,
        index_bytes=index_bytes,
        columns_index_bytes=columns_index_bytes,
        pandas_deep_total_bytes=pandas_deep_total_bytes,
        estimated_total_bytes=estimated_total_bytes,
        columns=column_reports,
        by_source_bytes=dict(by_source_bytes),
        by_source_feature_bytes=dict(by_source_feature_bytes),
        largest_cells=largest_cells,
    )


__all__ = [
    "CellMemory",
    "ColumnMemory",
    "MaterializedMemoryReport",
    "profile_materialized",
]
