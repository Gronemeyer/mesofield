"""Core orchestration for datakit.

Provides :class:`Dataset` — the user-facing entry point that wraps a
discovered file inventory and materializes it via the source registry.

Failure policy:
- ``materialize(strict=True)`` (default) raises on the first error with
  ``(subject, session, task, source, path)`` context.
- ``materialize(strict=False)`` continues past errors and (with
  ``return_errors=True``) returns a long-format error DataFrame.
- :meth:`validate` runs every cell, never raises, and returns the same
  long-format DataFrame.
"""

from __future__ import annotations

import traceback as _tb
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._utils._logger import get_logger
from ._version import build_meta as _build_meta
from .config import settings
from .datamodel import LoadedStream, Manifest, ManifestEntry
from .discover import discover_manifest
from .sources import SOURCE_REGISTRY
from .sources.register import LoadContext

logger = get_logger(__name__)

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Inventory construction
# ---------------------------------------------------------------------------


def _entries_to_inventory(
    entries: Iterable[ManifestEntry],
    *,
    root: Path,
    prefer_processed: bool = True,
    include_task_level: bool | None = None,
) -> pd.DataFrame:
    """Flatten manifest entries into a wide DataFrame indexed by subject/session[/task]."""
    entries = list(entries)
    subject_name, session_name, task_name = settings.dataset.index_names

    if include_task_level is None:
        include_task = any(e.task is not None for e in entries)
    else:
        include_task = bool(include_task_level)

    index_names = [subject_name, session_name] + ([task_name] if include_task else [])

    if not entries:
        return pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=index_names))

    records: dict[tuple, dict[str, str]] = {}
    origins: dict[tuple, dict[str, str]] = {}
    for entry in entries:
        key = (entry.subject, entry.session, entry.task) if include_task else (entry.subject, entry.session)
        records.setdefault(key, {})
        origins.setdefault(key, {})
        resolved = str(root / entry.path)
        current_origin = origins[key].get(entry.tag)
        if prefer_processed and current_origin == "processed" and entry.origin != "processed":
            continue
        records[key][entry.tag] = resolved
        origins[key][entry.tag] = entry.origin

    keys = sorted(records.keys(), key=lambda k: tuple("" if v is None else str(v) for v in k))
    index = pd.MultiIndex.from_tuples(keys, names=index_names)
    return pd.DataFrame([records[k] for k in keys], index=index)


# ---------------------------------------------------------------------------
# Topological sort + experiment-window resolution
# ---------------------------------------------------------------------------


def _toposort(active: Sequence[str]) -> list[str]:
    """Order ``active`` so each tag follows its declared dependencies."""
    active_set = set(active)
    state: dict[str, int] = {}  # 0=visiting, 1=done
    order: list[str] = []

    def visit(tag: str, stack: tuple[str, ...]) -> None:
        if state.get(tag) == 1:
            return
        if state.get(tag) == 0:
            raise RuntimeError(f"Cycle in source 'requires' graph: {' -> '.join(stack + (tag,))}")
        state[tag] = 0
        cls_ = SOURCE_REGISTRY.get(tag)
        for dep in (getattr(cls_, "requires", ()) or ()):
            if dep in active_set:
                visit(dep, stack + (tag,))
        state[tag] = 1
        order.append(tag)

    for tag in active:
        visit(tag, ())
    return order


def _experiment_window(frame: Optional[pd.DataFrame]) -> Optional[tuple[float, float]]:
    if frame is None or "device_id" not in frame.columns:
        return None
    time_col = settings.timeline.queue_column
    if time_col not in frame.columns:
        return None
    devices = frame["device_id"].astype(str)
    mask = pd.Series(False, index=frame.index)
    for pat in settings.timeline.window_device_patterns:
        mask |= devices.str.contains(pat, case=False, na=False, regex=False)
    if not mask.any():
        return None
    times = pd.to_numeric(frame.loc[mask, time_col], errors="coerce").dropna()
    if times.empty:
        return None
    return float(times.iloc[0]), float(times.iloc[-1])


def _make_context(
    idx: tuple,
    inventory_row: Mapping[str, Any],
    deps: Mapping[str, LoadedStream | None],
) -> LoadContext:
    dq = deps.get("dataqueue")
    dq_frame = dq.value if dq is not None and isinstance(dq.value, pd.DataFrame) else None
    return LoadContext(
        subject=str(idx[0]),
        session=str(idx[1]) if len(idx) > 1 else "unknown",
        task=str(idx[2]) if len(idx) > 2 else None,
        inventory_row=dict(inventory_row),
        dependencies=dict(deps),
        master_timeline=np.asarray(dq.t, dtype=np.float64) if dq is not None else None,
        experiment_window=_experiment_window(dq_frame),
        dataqueue_frame=dq_frame,
        dataqueue_meta=dq.meta if dq is not None else None,
    )


# ---------------------------------------------------------------------------
# Payload flattening
# ---------------------------------------------------------------------------


def _series_to_cell(s: pd.Series) -> Any:
    if s.empty:
        return np.nan
    if len(s) == 1:
        return s.iloc[0]
    try:
        if int(s.nunique(dropna=False)) <= 1:
            return s.iloc[0]
    except TypeError:
        pass
    return s.to_numpy()


def _flatten(stream: LoadedStream) -> dict[tuple[str, str], Any]:
    """Flatten a stream payload + per-cell meta into ``{(tag, feature): value}``."""
    tag = stream.tag
    payload = stream.value
    cells: dict[tuple[str, str], Any]
    if isinstance(payload, pd.DataFrame):
        cells = {(tag, str(c)): _series_to_cell(payload[c]) for c in payload.columns}
    elif isinstance(payload, pd.Series):
        cells = {(tag, "values"): _series_to_cell(payload)}
    elif isinstance(payload, (np.ndarray, list, tuple)):
        cells = {(tag, "values"): np.asarray(payload)}
    elif isinstance(payload, dict):
        cells = {(tag, str(k)): v for k, v in payload.items()}
    else:
        cells = {(tag, "value"): payload}

    meta = stream.meta or {}
    scope = meta.get(settings.dataset.scope_key)
    if scope in (settings.dataset.session_scope, settings.dataset.experiment_scope):
        return cells
    skip = {
        settings.dataset.scope_key,
        settings.sources.meta_camera_key,
        settings.sources.meta_timeseries_key,
        settings.sources.meta_source_key,
        settings.sources.meta_interval_key,
    }
    meta_dict = {k: v for k, v in meta.items() if k not in skip}
    if meta_dict:
        cells[(tag, "meta")] = meta_dict
    return cells


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


_RECORD_COLS = ["status", "path", "error_type", "message", "traceback"]


def _norm(value: Optional[Union[str, Sequence[str]]]) -> Optional[set[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    if isinstance(value, SequenceABC):
        return {str(v) for v in value}
    raise TypeError(f"Expected str or sequence, got {type(value).__name__}")


class Dataset:
    """Discovery + materialization for a BIDS-style experiment hierarchy."""

    def __init__(
        self,
        inventory: pd.DataFrame,
        *,
        sources: Optional[Iterable[str]] = None,
        roots: Sequence[PathLike] = (),
    ) -> None:
        if not isinstance(inventory, pd.DataFrame):
            raise TypeError("inventory must be a pandas DataFrame")
        if not isinstance(inventory.index, pd.MultiIndex):
            raise ValueError("inventory must have a MultiIndex (Subject, Session[, Task])")
        n = inventory.index.nlevels
        if n < 2 or n > 3:
            raise ValueError(f"inventory must have 2 or 3 index levels; got {n}")

        inv = inventory.copy()
        inv.index.set_names(list(settings.dataset.index_names[:n]), inplace=True)
        self._inventory = inv

        if sources is None:
            tags = tuple(t for t in inv.columns if t in SOURCE_REGISTRY)
        else:
            tags = tuple(sources)
            unknown = [t for t in tags if t not in SOURCE_REGISTRY]
            if unknown:
                raise KeyError(f"Unknown source tag(s): {sorted(unknown)}")
        self._sources = tags
        self._roots = tuple(Path(r) for r in roots)
        self._notes: list[str] = []

    # ---- Construction -------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        root: Union[PathLike, Sequence[PathLike]],
        *,
        sources: Optional[Iterable[str]] = None,
        prefer_processed: bool = True,
        include_task_level: bool | None = True,
    ) -> "Dataset":
        """Discover one or more experiment roots and build a Dataset.

        Pass a single path or a sequence of paths; multiple roots are
        concatenated row-wise.
        """
        roots = [root] if isinstance(root, (str, Path)) else list(root)
        if not roots:
            raise ValueError("at least one root is required")
        frames: list[pd.DataFrame] = []
        resolved: list[Path] = []
        for r in roots:
            manifest = discover_manifest(Path(r).expanduser())
            frames.append(_entries_to_inventory(
                manifest.entries,
                root=manifest.root,
                prefer_processed=prefer_processed,
                include_task_level=include_task_level,
            ))
            resolved.append(manifest.root)
        combined = pd.concat(frames, sort=False).sort_index() if len(frames) > 1 else frames[0]
        return cls(combined, sources=sources, roots=resolved)

    # ---- Properties ---------------------------------------------------

    @property
    def inventory(self) -> pd.DataFrame:
        return self._inventory.copy()

    @property
    def sources(self) -> Tuple[str, ...]:
        return self._sources

    @property
    def roots(self) -> Tuple[Path, ...]:
        return self._roots

    @property
    def subjects(self) -> list[str]:
        return sorted(self._inventory.index.get_level_values(0).unique().tolist())

    @property
    def sessions(self) -> list[str]:
        return sorted(self._inventory.index.get_level_values(1).unique().tolist())

    @property
    def has_task_level(self) -> bool:
        return self._inventory.index.nlevels >= 3

    @property
    def notes(self) -> list[str]:
        """Free-form notes attached to this dataset."""
        return list(self._notes)

    def add_note(self, note: str) -> "Dataset":
        """Append a free-form note; persisted via ``df.attrs['datakit_notes']``."""
        if not isinstance(note, str):
            raise TypeError("note must be a string")
        self._notes.append(note)
        return self

    @property
    def meta(self) -> dict:
        """Provenance metadata for the running datakit package.

        The same dictionary is attached to the materialized DataFrame as
        ``df.attrs["datakit"]`` so it persists through pickle round-trips.
        """
        return _build_meta()

    def __repr__(self) -> str:
        return (
            f"Dataset(rows={len(self._inventory)}, sources={len(self._sources)}, "
            f"task_level={self.has_task_level})"
        )

    # ---- Filtering ----------------------------------------------------

    def _row_mask(
        self,
        subject: Optional[set[str]],
        session: Optional[set[str]],
        task: Optional[set[str]],
    ) -> pd.Series:
        inv = self._inventory
        mask = pd.Series(True, index=inv.index)
        for level, want in enumerate((subject, session, task)):
            if want is None or level >= inv.index.nlevels:
                continue
            vals = inv.index.get_level_values(level).astype(str).isin({str(v) for v in want})
            mask = mask & pd.Series(vals, index=inv.index)
        return mask

    def include(
        self,
        *,
        subject: Optional[Union[str, Sequence[str]]] = None,
        session: Optional[Union[str, Sequence[str]]] = None,
        task: Optional[Union[str, Sequence[str]]] = None,
        source: Optional[Union[str, Sequence[str]]] = None,
    ) -> "Dataset":
        """Keep only rows/sources matching the given filters (AND-combined).

        Each keyword accepts either a single string or a sequence of strings;
        ``None`` (the default) means "no constraint on this axis". All
        provided filters are combined with logical AND, so adding a keyword
        narrows the result. Returns a new ``Dataset`` — the original is
        unchanged, so calls chain naturally.

        Examples
        --------
        >>> ds.include(subject="STREHAB07")              # one subject
        >>> ds.include(subject=["STREHAB07", "STREHAB08"])  # multiple subjects
        >>> ds.include(session="ses-05", task="task-widefield")
        >>> ds.include(source=["dataqueue", "treadmill"])    # drop other sources
        >>> ds.include(subject="STREHAB07").include(task="task-movies")  # chain
        """
        mask = self._row_mask(_norm(subject), _norm(session), _norm(task))
        srcs = _norm(source)
        new_sources = tuple(t for t in self._sources if t in srcs) if srcs is not None else self._sources
        return Dataset(self._inventory.loc[mask].copy(), sources=new_sources, roots=self._roots)

    def exclude(
        self,
        *,
        subject: Optional[Union[str, Sequence[str]]] = None,
        session: Optional[Union[str, Sequence[str]]] = None,
        task: Optional[Union[str, Sequence[str]]] = None,
        source: Optional[Union[str, Sequence[str]]] = None,
    ) -> "Dataset":
        """Drop rows/sources matching the given filters.

        Like :meth:`include`, every keyword accepts a string or a sequence
        of strings, and combining keywords narrows what gets removed.
        Behavior depends on which axes are provided:

        - **source only** — drop those source columns globally.
        - **row axes only** (subject/session/task) — drop matching rows.
        - **both** — NaN out only the listed source columns within matching
          rows; rows and other sources are preserved.

        Examples
        --------
        >>> ds.exclude(subject="STREHAB07")                   # drop a subject
        >>> ds.exclude(source="psychopy")                     # drop a source globally
        >>> ds.exclude(session=["ses-01", "ses-02"])          # drop multiple sessions
        >>> ds.exclude(subject="STREHAB07", source="pupil_dlc")
        ... # blank pupil_dlc only for STREHAB07; other rows/sources untouched
        """
        subj, ses, tsk, srcs = _norm(subject), _norm(session), _norm(task), _norm(source)
        has_row = any(s is not None for s in (subj, ses, tsk))
        mask = self._row_mask(subj, ses, tsk)
        new_inv = self._inventory.copy()
        new_sources = self._sources
        if srcs is None:
            new_inv = new_inv.loc[~mask].copy()
        elif not has_row:
            new_sources = tuple(t for t in self._sources if t not in srcs)
        else:
            cols = [c for c in new_inv.columns if c in srcs]
            if cols:
                new_inv.loc[mask, cols] = np.nan
        return Dataset(new_inv, sources=new_sources, roots=self._roots)

    def head(self, n: int = 3) -> "Dataset":
        """Return a new ``Dataset`` containing only the first ``n`` rows.

        Convenience for quick tests; equivalent to slicing the inventory
        with ``.iloc[:n]`` while preserving sources and roots.
        """
        if not isinstance(n, int) or n < 0:
            raise ValueError(f"n must be a non-negative int; got {n!r}")
        return Dataset(self._inventory.iloc[:n].copy(), sources=self._sources, roots=self._roots)

    def select(
        self,
        subject: str,
        session: str,
        task: Optional[str] = None,
    ) -> "Dataset":
        """Return a new ``Dataset`` containing exactly one inventory row.

        Positional shorthand for ``include(subject=..., session=..., task=...)``
        intended for the common "give me this one cell" use case. Unlike
        :meth:`include`, all arguments must be single strings — for
        multi-value or partial filtering use :meth:`include` directly:

        >>> ds.select("STREHAB07", "ses-05", "task-widefield")  # one cell
        >>> ds.include(subject="STREHAB07", session="ses-05")   # all tasks for that session

        Raises ``KeyError`` if no row matches and ``ValueError`` if more
        than one row matches (e.g. when ``task`` is omitted on a
        task-level inventory and multiple tasks exist for the session).
        """
        for name, value in (("subject", subject), ("session", session), ("task", task)):
            if value is not None and not isinstance(value, str):
                raise TypeError(
                    f"select(): {name} must be a string; got {type(value).__name__}. "
                    "Use Dataset.include() for multi-value filtering."
                )
        ds = self.include(subject=subject, session=session, task=task)
        n = len(ds._inventory)
        if n == 0:
            key = (subject, session) if task is None else (subject, session, task)
            raise KeyError(f"No inventory entry for {key}")
        if n > 1:
            key = (subject, session) if task is None else (subject, session, task)
            raise ValueError(
                f"select() matched {n} rows for {key}; "
                "narrow with task=... or use Dataset.include() instead."
            )
        return ds

    # ---- Materialize / validate --------------------------------------

    def _iter_cells(self, *, strict: bool, progress: bool) -> tuple[
        list[dict[tuple[str, str], Any]], list[dict[str, Any]]
    ]:
        order = _toposort(self._sources)
        n_levels = self._inventory.index.nlevels
        index_names = list(settings.dataset.index_names)[:n_levels]

        rows: list[dict[tuple[str, str], Any]] = []
        records: list[dict[str, Any]] = []

        bar = None
        if progress:
            try:
                from tqdm import tqdm
                bar = tqdm(total=len(self._inventory) * len(order), unit="load")
            except ImportError:
                pass

        for idx, row in self._inventory.iterrows():
            if not isinstance(idx, tuple):
                idx = (idx,)
            inv_row = row.to_dict()
            row_cells: dict[tuple[str, str], Any] = {}
            cell_streams: dict[str, LoadedStream | None] = {}

            base_record: dict[str, Any] = {name: idx[i] if i < len(idx) else None
                                           for i, name in enumerate(index_names)}

            for tag in order:
                cls_ = SOURCE_REGISTRY[tag]
                rec = {**base_record, "Source": tag, "path": None,
                       "error_type": None, "message": None, "traceback": None}

                path_value = inv_row.get(tag)
                if path_value is None or (isinstance(path_value, float) and pd.isna(path_value)):
                    rec["status"] = "missing"
                    records.append(rec)
                    cell_streams[tag] = None
                    if bar: bar.update(1)
                    continue

                deps = {req: cell_streams.get(req)
                        for req in (getattr(cls_, "requires", ()) or ())}
                ctx = _make_context(idx, inv_row, deps)
                path_str = str(path_value)
                rec["path"] = path_str

                try:
                    stream = cls_().load(Path(path_str), context=ctx)
                except Exception as exc:  # noqa: BLE001
                    if strict:
                        if bar: bar.close()
                        raise RuntimeError(
                            f"Error loading source '{tag}' for {idx} at '{path_str}': "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    rec["status"] = "error"
                    rec["error_type"] = type(exc).__name__
                    rec["message"] = str(exc)
                    rec["traceback"] = _tb.format_exc()
                    records.append(rec)
                    cell_streams[tag] = None
                    if bar: bar.update(1)
                    continue

                cell_streams[tag] = stream
                row_cells.update(_flatten(stream))
                rec["status"] = "ok"
                records.append(rec)
                if bar: bar.update(1)

            rows.append(row_cells)

        if bar: bar.close()
        return rows, records

    def validate(self, *, progress: bool = False) -> pd.DataFrame:
        """Run every (cell, source); report status without raising."""
        _, records = self._iter_cells(strict=False, progress=progress)
        n_levels = self._inventory.index.nlevels
        cols = list(settings.dataset.index_names)[:n_levels] + ["Source"] + _RECORD_COLS
        return pd.DataFrame(records, columns=cols)

    def materialize(
        self,
        *,
        strict: bool = True,
        return_errors: bool = False,
        progress: bool = False,
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
        """Build the materialized DataFrame.

        With ``strict=True`` (default) the first error is raised with full
        ``(subject, session, task, source, path)`` context. With
        ``strict=False`` failed cells are blanked; pass ``return_errors=True``
        to also receive the long-format error frame produced by
        :meth:`validate`.
        """
        rows, records = self._iter_cells(strict=strict, progress=progress)
        idx = self._inventory.index

        all_cols: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for r in rows:
            for c in r.keys():
                if c not in seen:
                    seen.add(c)
                    all_cols.append(c)

        if not all_cols:
            df = pd.DataFrame(index=idx,
                              columns=pd.MultiIndex.from_tuples([], names=["Source", "Feature"]))
        else:
            data = {c: [r.get(c, np.nan) for r in rows] for c in all_cols}
            df = pd.DataFrame(data, index=idx)
            df.columns = pd.MultiIndex.from_tuples(all_cols, names=["Source", "Feature"])

        # Embed provenance metadata so pickled artefacts can be traced back to
        # the exact datakit revision that produced them. ``DataFrame.attrs``
        # round-trips through ``to_pickle``/``read_pickle``.
        df.attrs["datakit"] = _build_meta()
        df.attrs["datakit_sources"] = list(self._sources)
        df.attrs["datakit_roots"] = [str(r) for r in self._roots]
        df.attrs["datakit_notes"] = list(self._notes)

        if return_errors:
            n_levels = idx.nlevels
            cols = list(settings.dataset.index_names)[:n_levels] + ["Source"] + _RECORD_COLS
            return df, pd.DataFrame(records, columns=cols)
        return df

    # ---- Persistence --------------------------------------------------

    def save(
        self,
        path: PathLike,
        *,
        format: Optional[str] = None,
        strict: bool = True,
        progress: bool = False,
        hdf_key: str = "dataset",
    ) -> Path:
        """Materialize and write to disk. Pickle by default; HDF5 via
        ``format="hdf5"`` or a ``.h5``/``.hdf5`` suffix."""
        out = Path(path).expanduser()
        suffix = out.suffix.lower()
        fmt = (format or ("hdf5" if suffix in {".h5", ".hdf5"} else "pickle")).lower()
        df = self.materialize(strict=strict, progress=progress)
        out.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "pickle":
            df.to_pickle(out)
        elif fmt == "hdf5":
            df.to_hdf(out, key=hdf_key, mode="w", format="fixed")
        else:
            raise ValueError(f"Unsupported format: {fmt!r}")
        return out


# ---------------------------------------------------------------------------
# Top-level convenience API
# ---------------------------------------------------------------------------


def load(
    root: Union[PathLike, Sequence[PathLike]],
    *,
    sources: Optional[Iterable[str]] = None,
    prefer_processed: bool = True,
    include_task_level: bool | None = True,
    progress: bool = True,
    strict: bool = True,
    return_errors: bool = False,
) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
    """One-shot discovery + materialization.

    Equivalent to ``Dataset.from_directory(root, ...).materialize(...)``.
    Use :meth:`Dataset.from_directory` directly when you need to filter
    (``.include`` / ``.exclude``) before materializing.
    """
    ds = Dataset.from_directory(
        root,
        sources=sources,
        prefer_processed=prefer_processed,
        include_task_level=include_task_level,
    )
    return ds.materialize(strict=strict, progress=progress, return_errors=return_errors)


def load_path(tag: str, path: PathLike) -> LoadedStream:
    """Ad-hoc single-file load via the registered source for ``tag``.

    Builds a minimal :class:`LoadContext` so sources without ``requires``
    or sibling-path lookups can be exercised directly. Sources declaring
    dependencies will receive ``None`` for them in ``context.dependencies``
    and must either degrade gracefully or raise.
    """
    if tag not in SOURCE_REGISTRY:
        raise KeyError(f"Unknown source tag: {tag!r}")
    cls_ = SOURCE_REGISTRY[tag]
    p = Path(path)
    ctx = LoadContext(
        subject="unknown",
        session="unknown",
        task=None,
        inventory_row={tag: str(p)},
        dependencies={},
    )
    return cls_().load(p, context=ctx)


def inspect_sources(
    inventory_or_dataset: Union["Dataset", pd.DataFrame],
    sources: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return a per-source coverage summary for an inventory.

    The returned DataFrame is indexed by source tag with columns
    ``present``, ``total``, ``missing``, and ``coverage`` (fraction of rows
    with a non-null path). Accepts either a :class:`Dataset` or a raw
    inventory DataFrame.

    When ``sources`` is omitted, every registered tag found in the
    inventory's columns is reported.
    """
    if isinstance(inventory_or_dataset, Dataset):
        inv = inventory_or_dataset.inventory
    elif isinstance(inventory_or_dataset, pd.DataFrame):
        inv = inventory_or_dataset
    else:
        raise TypeError(
            "inspect_sources expects a Dataset or DataFrame; got "
            f"{type(inventory_or_dataset).__name__}"
        )

    if sources is None:
        tags = [c for c in inv.columns if c in SOURCE_REGISTRY]
    else:
        tags = list(sources)

    total = len(inv)
    rows: list[dict[str, Any]] = []
    for tag in tags:
        if tag not in inv.columns:
            rows.append({
                "source": tag,
                "present": 0,
                "total": total,
                "missing": total,
                "coverage": 0.0,
            })
            continue
        present = int(inv[tag].notna().sum())
        rows.append({
            "source": tag,
            "present": present,
            "total": total,
            "missing": total - present,
            "coverage": (present / total) if total else 0.0,
        })
    return pd.DataFrame(rows).set_index("source")


__all__ = [
    "Dataset",
    "LoadContext",
    "LoadedStream",
    "load",
    "load_path",
    "inspect_sources",
]
