"""Build a pandas pickle from this experiment's session outputs.

Reads each `manifest.json` produced by `procedure.py`, ingests each
declared producer (`wheel` CSV, `meso` / `pupil` OME-TIFF + frame-metadata
sidecars), and writes a single `processed/dataset.pkl` you can load with
``pd.read_pickle(...)``.

The dataset has the shape databench consumes:

  Subject  Session  Task     | Source   Signal
  -----------------------------------------------------
  DEMO     01       freeview | wheel    timestamp, payload
                             | meso     frame_index, TimeReceivedByCore
                             | pupil    frame_index, TimeReceivedByCore

The mock producers don't write columns that match datakit's real
SOURCE_REGISTRY parsers (the wheel parser expects ``Clicks, Time,
Speed``; the mesoscope parser globs ``*_mesoscope.ome.tiff…``). We
therefore use the AcquisitionManifest as the source of truth and do a
manifest-driven ingest -- which is the pattern lab programmers will
write against their own producers.

The script does, however, import freely from ``mesofield.datakit`` and
``mesokit_schema``: the manifest models, the loader's
``ExperimentStore`` (only for the final pickle write convention),
``hash_file`` for provenance, etc. The point is the datakit module is
the toolkit; the parsers shipped with it are one option among many.

Run::

    python load_dataset.py             # uses ./data
    python load_dataset.py --root ./   # explicit
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# mesokit-schema: the typed manifest contract.
from mesokit_schema import (
    AcquisitionManifest,
    DatasetManifest,
    ProducerEntry,
    SourceVersion,
    TimeBasis,
)
from mesokit_schema.dataset import hash_file

# mesofield.datakit: the ingest toolkit. We don't use SOURCE_REGISTRY's
# real-hardware parsers here (mock filenames don't match their globs),
# but we lean on the datakit module's package surface for everything else.
import mesofield.datakit  # noqa: F401 — ensures datakit's logger is wired


def discover_sessions(root: Path) -> list[Path]:
    """Find every `data/sub-*/ses-*/manifest.json` under `root`."""
    data_root = root / "data"
    if not data_root.exists():
        return []
    return sorted(data_root.glob("sub-*/ses-*/manifest.json"))


def load_producer_table(session_dir: Path, producer: ProducerEntry) -> pd.DataFrame:
    """Read one producer's output (or its sidecar) into a DataFrame."""
    if producer.file_type.endswith("csv"):
        return pd.read_csv(session_dir / producer.output_path)

    if producer.file_type == "ome.tiff":
        # The pixel data lives in the TIFF (inspect with ImageJ); the
        # time-axis metadata we merge into the dataset comes from the
        # frame-metadata sidecar.
        if not producer.metadata_path:
            raise RuntimeError(
                f"Camera producer {producer.device_id!r} has no metadata_path"
            )
        sidecar = json.loads((session_dir / producer.metadata_path).read_text())
        return pd.DataFrame(sidecar.get("p0", []))

    raise RuntimeError(
        f"load_dataset.py: unknown file_type {producer.file_type!r} for "
        f"producer {producer.device_id!r}; extend this loader for new types."
    )


def build_session_table(session_dir: Path, manifest: AcquisitionManifest) -> pd.DataFrame:
    """Multi-source DataFrame for a single session.

    Anchored to the mesoscope (primary) frame count; shorter sources
    are NaN-padded, longer ones truncated. Trades temporal precision
    for an inspectable, regular table.
    """
    primary = next(
        (p for p in manifest.producers if p.data_type == "frames" and p.bids_type == "func"),
        manifest.producers[0],
    )
    primary_df = load_producer_table(session_dir, primary)
    anchor_rows = len(primary_df)

    columns: dict[tuple[str, str], list] = {}

    def _add(source: str, df: pd.DataFrame) -> None:
        for col in df.columns:
            values = df[col].iloc[:anchor_rows].tolist()
            if len(values) < anchor_rows:
                values = values + [None] * (anchor_rows - len(values))
            columns[(source, col)] = values

    for producer in manifest.producers:
        df = load_producer_table(session_dir, producer)
        # Use device_id as the source key so two cameras (meso / pupil)
        # appear as distinct sources even though both share data_type=frames.
        _add(producer.device_id, df)

    table = pd.DataFrame(columns)
    table.columns = pd.MultiIndex.from_tuples(table.columns, names=["Source", "Signal"])

    session = manifest.session
    task = session.task or "default"
    table.index = pd.MultiIndex.from_tuples(
        [(session.subject, session.session, task)] * len(table),
        names=["Subject", "Session", "Task"],
    )
    return table


def build_dataset(root: Path) -> tuple[Path, Path, pd.DataFrame]:
    """Ingest every session under `root/data/`. Returns (pickle, manifest, df)."""
    manifest_paths = discover_sessions(root)
    if not manifest_paths:
        raise FileNotFoundError(
            f"No AcquisitionManifests found under {root / 'data'}. "
            f"Run `python procedure.py` first."
        )

    tables: list[pd.DataFrame] = []
    acq_hashes: list[str] = []
    for mpath in manifest_paths:
        acq = AcquisitionManifest.read(mpath)
        if not acq.acquisition_complete:
            print(f"[skip] {mpath} reports acquisition_complete=False")
            continue
        tables.append(build_session_table(mpath.parent, acq))
        acq_hashes.append(acq.content_hash())
        print(f"[ok]   {acq.session.subject}/{acq.session.session}: "
              f"{len(tables[-1])} rows × {len(tables[-1].columns)} cols")

    if not tables:
        raise RuntimeError("No complete sessions to ingest.")

    dataset = pd.concat(tables, axis=0).sort_index()
    out_dir = root / "processed"
    out_dir.mkdir(exist_ok=True)
    pkl_path = out_dir / "dataset.pkl"
    dataset.to_pickle(pkl_path)

    ds_manifest = DatasetManifest(
        datakit_version="manifest-driven-ingest",
        built_at=datetime.now(timezone.utc),
        upstream_acquisition_hash=acq_hashes[0] if len(acq_hashes) == 1 else None,
        data_file="dataset.pkl",
        data_content_hash=hash_file(pkl_path),
        time_basis=TimeBasis(
            clock_source="derived",
            description="Anchored on mesoscope frame count per session.",
        ),
        source_versions=[
            SourceVersion(
                tag="multi-source",
                version="0.1.0",
                parser_class="load_dataset.build_session_table",
            )
        ],
        columns=[(s, c) for (s, c) in dataset.columns.tolist()],
    )
    ds_manifest_path = out_dir / "dataset_manifest.json"
    ds_manifest.write(ds_manifest_path)

    return pkl_path, ds_manifest_path, dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent,
        help="Experiment root containing data/ (default: this file's dir)",
    )
    args = parser.parse_args()

    pkl, ds_manifest_path, df = build_dataset(args.root.resolve())
    print()
    print(f"Wrote {pkl}")
    print(f"Wrote {ds_manifest_path}")
    print()
    print("--- first 3 rows ---")
    print(df.head(3))
    print()
    print(f"--- columns ({len(df.columns)} total) ---")
    for col in df.columns:
        print(f"  {col}")
    print()
    print("Load in Python: pd.read_pickle({})".format(repr(str(pkl))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
