"""Data-ingestion workflow.

The module keeps the happy path front-and-centre:

1. Discover the experiment hierarchy (subjects → sessions → tasks) and express it
   as a MultiIndex inventory frame.
2. Register logical data sources as ``pd.Series`` of file paths coupled with loader
   callables.
3. Materialise each source into predictable column blocks (scalars, numpy arrays,
   or preserved raw tables for structured outputs).
4. Persist the resulting dataset out to disk with standard pandas HDF5 helpers.

"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ._utils._logger import get_logger
from mesofield.datakit.config import settings
from mesofield.datakit.datamodel import LoadedStream
from mesofield.datakit.inventory import build_inventory
from mesofield.datakit.sources.behavior.dataqueue import DataqueueSource
from mesofield.datakit.sources import get_source_class
from mesofield.datakit.sources.register import SourceContext

logger = get_logger(__name__)


LoaderFn = Callable[[str, SourceContext], object]


def _ensure_datasource_registry() -> None:
    """Import datakit sources so the registry module is initialized."""
    from mesofield.datakit import sources as _  # noqa: F401


def _source_class(tag: str):
    return get_source_class(tag)


def _make_loader(tag: str) -> LoaderFn:
    def load(path_str: str, context: SourceContext) -> object:
        source = get_source_class(tag)()
        return source.load(Path(path_str), context=context)

    return load


def _structured_default(tag: str) -> bool:
    cls = _source_class(tag)
    return not getattr(cls, "flatten_payload", True)


@dataclass
class SeriesSpec:
    """Declarative description of a file-backed source registered with the store."""

    name: str
    files: pd.Series  # MultiIndex aligned to (Subject, Session, Task)
    loader: LoaderFn
    structured: bool = False

    def aligned_to(self, index: pd.Index) -> SeriesSpec:
        target_index = pd.MultiIndex.from_tuples(index.tolist(), names=index.names)
        if not self.files.index.equals(target_index):
            reindexed = self.files.reindex(target_index)
            return SeriesSpec(self.name, reindexed, self.loader, self.structured)
        return self


class ExperimentStore:
    """Builder that turns an inventory DataFrame into a rich dataset.

    Register one or more :class:`SeriesSpec` instances (each describing a source
    tag, file paths, and loader callable) then call :meth:`materialize` to fetch
    the data and flatten it into a single MultiIndex DataFrame plus normalized
    metadata.
    """

    def __init__(self, source: str | Path | pd.DataFrame):
        if isinstance(source, (str, Path)):
            src_path = Path(source)
            if src_path.is_dir():
                self.inventory = self._build_inventory(src_path)
            elif src_path.is_file() and src_path.suffix.lower() in {".h5", ".hdf5"}:
                self.inventory = pd.read_hdf(src_path)
            else:
                raise ValueError(f"Unsupported source: {source}")
        elif isinstance(source, pd.DataFrame):
            self.inventory = source.copy()
        else:
            raise TypeError("source must be a path or pre-built DataFrame")

        if not isinstance(self.inventory.index, pd.MultiIndex):
            raise ValueError("Inventory requires a MultiIndex on (Subject, Session, Task)")

        index_names = settings.dataset.index_names
        expected_levels = len(index_names)
        if self.inventory.index.nlevels != expected_levels:
            raise ValueError(
                f"Inventory must have {expected_levels} index levels {index_names};"
                f" got {self.inventory.index.nlevels}"
            )

        self.inventory.index.set_names(index_names, inplace=True)
        self._specs = []
        self._materialized = None
        self._master_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.meta_frame = pd.DataFrame(columns=self._meta_columns())
        self.session_attrs = {}
        self.experiment_attrs = {}
        self.time_basis = {}
        self._meta_rows = []

    @staticmethod
    def _build_inventory(root: Path) -> pd.DataFrame:
        inventory = build_inventory(root, include_task_level=True)
        if not isinstance(inventory.index, pd.MultiIndex):
            raise ValueError("Discovery inventory must provide a MultiIndex")
        index_names = settings.dataset.index_names
        expected_levels = len(index_names)
        if inventory.index.nlevels != expected_levels:
            raise ValueError(
                f"Discovery inventory is expected to have {expected_levels} levels, got {inventory.index.nlevels}"
            )
        inventory.index.set_names(index_names, inplace=True)
        return inventory

    def register_series(self, name: str, files: pd.Series, loader: LoaderFn, *, structured: bool = False) -> None:
        logical_overrides = settings.dataset.logical_name_overrides
        resolved_name = logical_overrides.get(name, name)
        spec = SeriesSpec(name=resolved_name, files=files, loader=loader, structured=structured)
        self._specs.append(spec.aligned_to(self.inventory.index))

    def register_source(
        self,
        tag: str,
        files: pd.Series,
        *,
        logical_name: str | None = None,
        structured: bool | None = None,
    ) -> None:
        if structured is None:
            structured = _structured_default(tag)
        loader = _make_loader(tag)
        logical_overrides = settings.dataset.logical_name_overrides
        resolved_name = logical_name or logical_overrides.get(tag, tag)
        self.register_series(resolved_name, files, loader, structured=structured)

    def register_sources(
        self,
        tags: Iterable[str],
        *,
        logical_name_overrides: Dict[str, str] | None = None,
    ) -> list[str]:
        name_overrides = logical_name_overrides or settings.dataset.logical_name_overrides
        missing: list[str] = []
        for tag in tags:
            if tag not in self.inventory.columns:
                missing.append(tag)
                continue
            logical_name = name_overrides.get(tag, tag)
            self.register_source(
                tag,
                self.inventory[tag],
                logical_name=logical_name,
            )
        return missing

    # ------------------------------------------------------------------
    # Materialization helpers
    # ------------------------------------------------------------------
    def _flatten_frame(self, df: pd.DataFrame, prefix: str, structured: bool) -> dict:
        if structured:
            return {(prefix, "raw_output"): df}
        payload: dict[tuple[str, str], object] = {}
        for col in df.columns:
            col_series = df[col]
            payload[(prefix, str(col))] = self._series_to_cell(col_series)
        return payload

    @staticmethod
    def _series_to_cell(series: pd.Series) -> object:
        if series.empty:
            return np.nan
        if len(series) == 1:
            return series.iloc[0]
        if ExperimentStore._series_is_constant(series):
            return series.iloc[0]
        return series.to_numpy()

    @staticmethod
    def _series_is_constant(series: pd.Series) -> bool:
        if series.empty:
            return True
        try:
            unique = series.nunique(dropna=False)
        except TypeError:
            first = series.iloc[0]
            try:
                return bool(series.apply(lambda value: (value == first) or (pd.isna(value) and pd.isna(first))).all())
            except Exception:
                return False
        return unique <= 1

    def _load_file_result(
        self,
        path: str,
        spec: SeriesSpec,
        idx: Tuple[str, str, str],
    ) -> tuple[dict, Dict[str, Any] | None]:
        if pd.isna(path):
            return {}, None
        subject, session, task = idx
        inventory_row = self.inventory.loc[idx].to_dict()
        master = self._ensure_master_timeline(idx, inventory_row)
        context = SourceContext(
            subject=subject,
            session=session,
            task=task,
            inventory_row=inventory_row,
            master_timeline=master.get("master_timeline"),
            experiment_window=master.get("experiment_window"),
            dataqueue_frame=master.get("dataqueue_frame"),
            dataqueue_meta=master.get("dataqueue_meta"),
        )
        try:
            data = spec.loader(path, context)
        except Exception as exc:
            context = (
                f"Error loading source '{spec.name}' for "
                f"({subject}, {session}, {task}) at '{path}'. "
                f"Cause: {exc!r}"
            )
            raise RuntimeError(context) from exc
        if isinstance(data, LoadedStream):
            payload = data.value
            if isinstance(payload, pd.DataFrame):
                return self._flatten_frame(payload, spec.name, spec.structured), data.meta
            if isinstance(payload, pd.Series):
                values = payload.to_numpy() if len(payload) > 1 else payload.iloc[0]
                return {(spec.name, "values"): values}, data.meta
            if isinstance(payload, (np.ndarray, list)):
                return {(spec.name, "values"): np.asarray(payload)}, data.meta
            return {(spec.name, "value"): payload}, data.meta
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        if isinstance(data, pd.DataFrame):
            return self._flatten_frame(data, spec.name, spec.structured), None
        if isinstance(data, pd.Series):
            if spec.structured:
                return {(spec.name, "raw_output"): data.to_frame().T}, None
            values = self._series_to_cell(data)
            return {(spec.name, "values"): values}, None
        if isinstance(data, (np.ndarray, list)):
            return {(spec.name, "values"): np.asarray(data)}, None
        return {(spec.name, "value"): data}, None

    def _ensure_master_timeline(
        self,
        idx: Tuple[str, str, str],
        inventory_row: Dict[str, Any],
    ) -> Dict[str, Any]:
        if idx in self._master_cache:
            return self._master_cache[idx]

        dataqueue_path = inventory_row.get("dataqueue")
        if dataqueue_path is None or (isinstance(dataqueue_path, float) and pd.isna(dataqueue_path)):
            cached = {
                "master_timeline": None,
                "experiment_window": None,
                "dataqueue_frame": None,
                "dataqueue_meta": None,
            }
            self._master_cache[idx] = cached
            return cached

        dq_path = Path(str(dataqueue_path))
        dq_source = DataqueueSource()
        master_timeline, frame, meta = dq_source.build_timeseries(dq_path)
        experiment_window = self._resolve_experiment_window(frame)
        cached = {
            "master_timeline": master_timeline,
            "experiment_window": experiment_window,
            "dataqueue_frame": frame,
            "dataqueue_meta": meta,
        }
        self._master_cache[idx] = cached
        return cached

    def _resolve_experiment_window(self, frame: pd.DataFrame) -> tuple[float, float] | None:
        device_col = "device_id"
        time_col = settings.timeline.queue_column
        if device_col not in frame.columns or time_col not in frame.columns:
            return None

        patterns = settings.timeline.window_device_patterns
        device_series = frame[device_col].astype(str)
        mask = pd.Series(False, index=frame.index)
        for pattern in patterns:
            mask |= device_series.str.contains(pattern, case=False, na=False, regex=False)

        if not mask.any():
            return None

        times = pd.to_numeric(frame.loc[mask, time_col], errors="coerce").dropna()
        if times.empty:
            return None
        return float(times.iloc[0]), float(times.iloc[-1])

    def _load_file(self, path: str, spec: SeriesSpec, idx: Tuple[str, str, str]) -> dict:
        data, meta = self._load_file_result(path, spec, idx)
        if meta:
            self._record_meta(idx, spec.name, meta)
        return data

    def materialize(self, *, progress: bool = False) -> pd.DataFrame:
        if not self._specs:
            raise RuntimeError("No sources registered. Call register_series first.")
        self._reset_meta_state()
        rows_by_idx = {idx: {} for idx in self.inventory.index}
        total_loads = len(self.inventory.index) * len(self._specs)
        bar = None
        if progress:
            from tqdm import tqdm

            bar = tqdm(
                total=total_loads,
                desc="materialize",
                unit="load",
                miniters=1,
            )
        for spec in self._specs:
            if bar is not None:
                bar.set_description(f"materialize:{spec.name}")
            for idx in self.inventory.index:
                path = spec.files.loc[idx]
                rows_by_idx[idx].update(self._load_file(path, spec, idx))
                if bar is not None:
                    bar.update(1)
        if bar is not None:
            bar.close()
        rows = [rows_by_idx[idx] for idx in self.inventory.index]
        df = pd.DataFrame(rows, index=self.inventory.index)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns.set_names(["Source", "Feature"], inplace=True)
        else:
            column_values = list(df.columns)
            tuple_columns = [col for col in column_values if isinstance(col, tuple) and len(col) == 2]
            if len(tuple_columns) == len(column_values):
                df.columns = pd.MultiIndex.from_tuples(tuple_columns, names=["Source", "Feature"])
            else:
                df.columns = pd.MultiIndex.from_product([["default"], df.columns], names=["Source", "Feature"])
        self._finalize_meta_frame()
        meta_table = self._meta_table_to_wide()
        if not meta_table.empty:
            df = pd.concat([df, meta_table], axis=1)
            df.columns.set_names(["Source", "Feature"], inplace=True)
        self._materialized = df
        # df.attrs["meta_frame"] = self.meta_frame.to_dict(orient="records")
        # df.attrs["meta_frame_columns"] = list(self.meta_frame.columns)
        # df.attrs["session_attrs"] = dict(self.session_attrs)
        # df.attrs["experiment_attrs"] = dict(self.experiment_attrs)
        # df.attrs["time_basis"] = dict(self.time_basis)
        # df.attrs["meta_columns"] = list(self._meta_columns())
        return df

    def release(self) -> None:
        """Release materialized data and metadata to free memory."""

        self._materialized = None
        self._reset_meta_state()

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _meta_columns() -> List[str]:
        return list(settings.dataset.meta_columns)

    def _reset_meta_state(self) -> None:
        self._meta_rows.clear()
        self.session_attrs.clear()
        self.experiment_attrs.clear()
        self.time_basis.clear()
        self.meta_frame = pd.DataFrame(columns=self._meta_columns())

    def _record_meta(self, idx: Tuple[str, str, str], source: str, meta: Dict[str, Any]) -> None:
        if not meta:
            return
        subject, session, task = idx
        scope_key = settings.dataset.scope_key
        scope = meta.get(scope_key, "stream")
        values = {k: v for k, v in meta.items() if k != scope_key}
        if scope == settings.dataset.session_scope:
            target = self.session_attrs.setdefault((subject, session), {})
            target.update(values)
            return
        if scope == settings.dataset.experiment_scope:
            self.experiment_attrs.update(values)
            return
        meta_columns = list(settings.dataset.meta_columns)
        subject_col, session_col, task_col = settings.dataset.index_names
        source_col = meta_columns[3] if len(meta_columns) > 3 else "Source"
        key_col = meta_columns[4] if len(meta_columns) > 4 else "Key"
        value_col = meta_columns[5] if len(meta_columns) > 5 else "Value"
        dtype_col = meta_columns[6] if len(meta_columns) > 6 else "dtype"
        for key, value in values.items():
            self._meta_rows.append(
                {
                    subject_col: subject,
                    session_col: session,
                    task_col: task,
                    source_col: source,
                    key_col: key,
                    value_col: value,
                    dtype_col: type(value).__name__,
                }
            )
            if key == settings.dataset.time_basis_key and isinstance(value, str):
                self.time_basis[source] = value

    def _finalize_meta_frame(self) -> None:
        if self._meta_rows:
            self.meta_frame = pd.DataFrame(self._meta_rows, columns=self._meta_columns())
        else:
            self.meta_frame = pd.DataFrame(columns=self._meta_columns())

    def _meta_table_to_wide(self) -> pd.DataFrame:
        if self.meta_frame.empty:
            return pd.DataFrame(index=self.inventory.index)
        meta_columns = list(settings.dataset.meta_columns)
        subject_col, session_col, task_col = settings.dataset.index_names
        source_col = meta_columns[3] if len(meta_columns) > 3 else "Source"
        key_col = meta_columns[4] if len(meta_columns) > 4 else "Key"
        value_col = meta_columns[5] if len(meta_columns) > 5 else "Value"
        index_cols = [subject_col, session_col, task_col]
        meta = self.meta_frame.copy()
        missing = [col for col in index_cols + [source_col, key_col, value_col] if col not in meta.columns]
        if missing:
            return pd.DataFrame(index=self.inventory.index)
        pivot = meta.pivot_table(
            index=index_cols,
            columns=[source_col, key_col],
            values=value_col,
            aggfunc="first",
        )
        pivot = pivot.reindex(self.inventory.index)
        if not pivot.empty:
            pivot.columns = pd.MultiIndex.from_tuples(
                [(str(source), f"meta.{key}") for source, key in pivot.columns],
                names=["Source", "Feature"],
            )
        grouped = (
            meta.groupby(index_cols + [source_col], dropna=False)[[key_col, value_col]]
            .apply(lambda group: {row[key_col]: row[value_col] for _, row in group.iterrows()})
        )
        meta_dict = grouped.unstack(source_col)
        meta_dict = meta_dict.reindex(self.inventory.index)
        if not meta_dict.empty:
            meta_dict.columns = pd.MultiIndex.from_tuples(
                [(str(source), "meta") for source in meta_dict.columns],
                names=["Source", "Feature"],
            )
        if pivot.empty and meta_dict.empty:
            return pd.DataFrame(index=self.inventory.index)
        return pd.concat([frame for frame in (pivot, meta_dict) if not frame.empty], axis=1)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def to_hdf(self, path: str | Path, key: str = "hfsa_mvp", *, mode: Literal["a", "w", "r+"] = "w") -> None:
        if self._materialized is None:
            raise RuntimeError("Call materialize() before saving")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._materialized.to_hdf(path, key=key, mode=mode, format="fixed")

    # Convenience ------------------------------------------------------
    def describe(self) -> pd.DataFrame:
        """Return a tidy summary of the registered sources and their coverage."""
        summary_rows = []
        for spec in self._specs:
            coverage = spec.files.notna().mean()
            summary_rows.append({
                "source": spec.name,
                "structured": spec.structured,
                "coverage_pct": round(float(coverage * 100), 2),
            })
        return pd.DataFrame(summary_rows)


__all__ = ["ExperimentStore", "SeriesSpec", "launch_dataset_shell"]


# ---------------------------------------------------------------------------
# Default pipeline wiring (parity with the removed datakit.dataset helpers)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DefaultSource:
    """Metadata describing how to pull a :class:`DataSource` into the store."""

    tag: str
    logical_name: str
    loader: LoaderFn
    structured: bool = False


def _build_default_sources() -> tuple[DefaultSource, ...]:
    _ensure_datasource_registry()
    desired_tags = settings.dataset.desired_tags
    sources: list[DefaultSource] = []
    logical_name_overrides = settings.dataset.logical_name_overrides

    for tag in desired_tags:
        try:
            loader = _make_loader(tag)
            structured = _structured_default(tag)
        except (ValueError, KeyError) as exc:
            logger.warning(
                "Skipping default source; no loader registered",
                extra={"phase": "mvp_default_sources", "tag": tag, "error": str(exc)},
            )
            continue

        logical_name = logical_name_overrides.get(tag, tag)
        sources.append(DefaultSource(tag=tag, logical_name=logical_name, loader=loader, structured=structured))
    return tuple(sources)


# Deferred to a lazy attribute (via __getattr__ below) so importing this
# module does not force the source registry to populate -- the producer-paired
# parsers live in mesofield.io.devices, and importing them eagerly creates a
# circular import while datakit is still bootstrapping.
_DEFAULT_SOURCES: tuple[DefaultSource, ...] | None = None


def _get_default_sources() -> tuple[DefaultSource, ...]:
    global _DEFAULT_SOURCES
    if _DEFAULT_SOURCES is None:
        _DEFAULT_SOURCES = _build_default_sources()
    return _DEFAULT_SOURCES


def __getattr__(name: str):
    if name == "DEFAULT_SOURCES":
        return _get_default_sources()
    raise AttributeError(name)


def build_default_dataset(
    input_path: Path,
    *,
    output_path: Optional[Path] = None,
    sources: Iterable[DefaultSource] | None = None,
    tags: Iterable[str] | None = None,
) -> Path:
    """Replicate the classic experiment build using the minimalist loader."""

    inventory_frame = build_inventory(input_path, include_task_level=True)
    store = ExperimentStore(inventory_frame)
    base_inventory = store.inventory
    missing_columns: list[str] = []

    if sources is None:
        sources = _get_default_sources()

    if tags is not None:
        missing_columns = store.register_sources(tags)
    elif sources is not None:
        for src in sources:
            if src.tag in base_inventory.columns:
                series = base_inventory[src.tag]
            else:
                missing_columns.append(src.tag)
                continue

            store.register_series(src.logical_name, series, loader=src.loader, structured=src.structured)
    else:
        missing_columns = store.register_sources(settings.dataset.desired_tags)

    if missing_columns:
        logger.warning(
            "Skipping sources missing from inventory",
            extra={
                "phase": "mvp_default_dataset",
                "missing_tags": sorted(missing_columns),
            },
        )

    store.materialize()
    timestamp = datetime.datetime.now().strftime("%y%m%d")
    default_output = (input_path / "processed" / f"{timestamp}_dataset_mvp.h5").resolve()
    resolved_output = Path(output_path).resolve() if output_path is not None else default_output
    store.to_hdf(resolved_output)
    return resolved_output


def launch_dataset_shell(source: str | Path | pd.DataFrame, *, key: str = "hfsa_mvp") -> None:
    """Open an embedded IPython REPL with a dataset object preloaded.

    ``source`` can be an in-memory :class:`pandas.DataFrame` or a path to an
    HDF5 file produced by :func:`build_default_dataset`. The interactive session
    exposes two convenience names: ``dataset`` (the DataFrame) and
    ``dataset_path`` (when available).
    """

    namespace: dict[str, object] = {}

    if isinstance(source, pd.DataFrame):
        dataset = source
        dataset_path: Path | None = None
    else:
        dataset_path = Path(source).resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        if dataset_path.is_dir():
            raise ValueError("Provide a dataset HDF5 file, not a directory")
        dataset = pd.read_hdf(dataset_path, key=key)
        namespace["dataset_path"] = dataset_path

    namespace["dataset"] = dataset

    header = (
        "Embedded datakit MVP shell.\n"
        "Variables available: dataset (DataFrame)"
        + ("; dataset_path" if "dataset_path" in namespace else "")
    )

    try:
        from IPython import embed
    except ImportError:  # pragma: no cover - optional dependency
        from code import interact

        interact(header, local=namespace)
        return

    embed(header=header, user_ns=namespace)


def _cli():
    import typer

    def main(
        input_path: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True, help="HFSA data root"),
        output_path: Optional[Path] = typer.Option(None, help="Optional output HDF5 path"),
    ) -> None:
        path = build_default_dataset(input_path, output_path=output_path)
        launch_dataset_shell(path)
        typer.echo(f"Saved MVP dataset to {path}")

    typer.run(main)


if __name__ == "__main__":
    _cli()
