"""End-to-end smoke test of the mesofield acquire -> ingest -> consume
pipeline, glued together by the shared mesokit-schema contract.

What this script does:

    1. mesofield (acquisition)
       Runs ``experiments/pipeline_demo`` headlessly with two synthetic
       producers (mock wheel encoder + mock camera). Real CSV, real
       OME-TIFF, real per-frame metadata sidecar land in a BIDS
       layout, and an ``AcquisitionManifest`` is written at the
       session root.

    2. mesofield.processing (intermediate)
       Runs a small ``MockRegionalMeans(ProcessorRunner)`` on the
       TIFF. The runner hashes the input, computes 4-quadrant means
       per frame, writes ``processed/regional_means.csv``, and emits
       a ``mock_regional_means.process.json`` ProcessingManifest
       beside it -- closing the provenance chain past acquisition.

    3. mesofield.datakit (ingest)
       The inline ingester reads BOTH manifests, parses each declared
       file (CSV, frame-metadata JSON, processed CSV), builds a
       (Subject, Session, Task) MultiIndex DataFrame with
       (Source, Signal) columns, and emits a DatasetManifest plus
       ``data.pkl``. ``upstream_acquisition_hash`` chains the
       dataset back to the acquisition.

    4. mesofield.datakit (consumer)
       Loads the pickle back via ``mesofield.datakit.load_dataset`` and
       sanity-checks the dataset shape. This is the role the separate
       ``databench`` package used to fill; the loader now lives in
       ``mesofield.datakit`` so no separate install is needed.

The script then runs an explicit **checks suite**: positive assertions
across the contract (manifest round-trips, content-hash chain through
acquisition AND processing, dataset shape, time-basis sanity,
calibration preservation) and **negative tests** that prove each
fence in the schema and ingester actually catches misbehavior
(tampered pickle, mutated manifest, forged schema_version, premature
acquisition, unknown fields, missing producer files, tampered
processing input).

Run with the ``mesofield`` conda env so all three repos plus pydantic
are available::

    PYTHONPATH=. /Users/anaconda3/envs/mesofield/bin/python test_pipeline.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

# --- mesokit-schema (the shared contract) ----------------------------------
from mesokit_schema import (
    SCHEMA_VERSION,
    AcquisitionManifest,
    DatasetManifest,
    InputRef,
    ProcessingManifest,
    ProducerEntry,
    SourceVersion,
    TimeBasis,
)
from mesokit_schema.dataset import hash_file

# --- mesofield (producer + processing) -------------------------------------
from mesofield.base import load_procedure
from mesofield.processing import ProcessorRunner

# --- mesofield.datakit (ingest + consumer) ---------------------------------
# `datakit` lives inside mesofield now. The inline ingester below is
# manifest-driven; `load_dataset` is the consumer-side loader (the role the
# separate `databench` package used to fill -- no separate install needed).
import mesofield.datakit  # noqa: F401
from mesofield.datakit import load_dataset
try:
    from mesofield import __version__ as _DATAKIT_VERSION  # type: ignore[attr-defined]
except Exception:
    _DATAKIT_VERSION = "0.0.0+unknown"


REPO_ROOT = Path(__file__).resolve().parent
# Resolve the shipped demo relative to the repo (tests/ -> repo root) so the
# suite is portable across machines instead of hardcoding one developer's path.
MESOFIELD_DEMO = REPO_ROOT.parent / "experiments" / "pipeline_demo"


# ---------------------------------------------------------------------------
# Expected values — derived from the demo experiment's own config files so
# changes to duration / subject / calibration don't silently desync the test.
# Hardcoded constants below belong to the device *class contract*, not the
# experiment config:
#   - EXPECTED_WHEEL_SOURCE_TAG: MockEncoderDevice.data_type ("samples")
#   - EXPECTED_WHEEL_SIGNAL_COLS: BaseDataProducer.save_data() columns
#   - EXPECTED_CAMERA_SOURCE_TAG: MockFrameProducer.data_type ("frames")
#   - EXPECTED_PROCESSOR_TAG: the MockRegionalMeans tool_name
EXPECTED_WHEEL_SOURCE_TAG = "samples"
EXPECTED_WHEEL_SIGNAL_COLS = {"timestamp", "payload"}
EXPECTED_CAMERA_SOURCE_TAG = "frames"
EXPECTED_CAMERA_SIGNAL_COLS = {"frame_index", "TimeReceivedByCore"}
EXPECTED_PROCESSOR_TAG = "mock_regional_means"
EXPECTED_PROCESSOR_REGIONS = 4


class _Expectations:
    """Snapshot of the demo experiment's declared inputs."""

    __slots__ = (
        "subject", "session", "task", "protocol", "experimenter",
        "duration_s", "wheel_device_id", "wheel_calibration",
        "camera_device_id", "camera_calibration",
    )

    def __init__(self, demo_dir: Path) -> None:
        import yaml

        experiment = json.loads((demo_dir / "experiment.json").read_text())
        cfg = experiment["Configuration"]
        subjects = experiment["Subjects"]
        subject_id, subject_block = next(iter(subjects.items()))

        hardware = yaml.safe_load((demo_dir / "hardware.yaml").read_text())
        device_stanzas = {
            key: val for key, val in hardware.items()
            if isinstance(val, dict) and "type" in val
        }

        self.subject = subject_id
        self.session = subject_block["session"]
        self.task = subject_block["task"]
        self.protocol = cfg["protocol"]
        self.experimenter = cfg["experimenter"]
        self.duration_s = float(cfg["duration"])

        wheel = device_stanzas["wheel"]
        self.wheel_device_id = "wheel"
        self.wheel_calibration = {
            k: wheel[k] for k in ("cpr", "diameter_mm", "sample_interval_ms")
            if k in wheel
        }

        camera = device_stanzas["camera"]
        self.camera_device_id = "camera"
        self.camera_calibration = {
            "width": camera["width"],
            "height": camera["height"],
            "frame_interval_ms": camera["frame_interval_ms"],
        }


EXP = _Expectations(MESOFIELD_DEMO)


# ---------------------------------------------------------------------------
# Check harness
# ---------------------------------------------------------------------------
class CheckTracker:
    """Accumulates pass/fail across check steps so one bad assert doesn't hide the rest."""

    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def section(self, title: str) -> None:
        print(f"\n[checks] {title}")

    @contextmanager
    def step(self, label: str) -> Iterator[None]:
        prefix = f"  - {label}"
        try:
            yield
        except Exception as exc:
            tb = traceback.format_exc(limit=2)
            print(f"{prefix} ... FAIL ({exc.__class__.__name__}: {exc})")
            self.failed.append((label, tb))
            return
        print(f"{prefix} ... ok")
        self.passed.append(label)

    def summary(self) -> int:
        total = len(self.passed) + len(self.failed)
        print(f"\n=== {len(self.passed)}/{total} checks passed ===")
        if self.failed:
            print("Failures:")
            for label, _tb in self.failed:
                print(f"  - {label}")
            return 1
        return 0


# ---------------------------------------------------------------------------
# Stage 1 — mesofield acquisition
# ---------------------------------------------------------------------------
def run_mesofield(workdir: Path) -> tuple[Path, dict]:
    """Run the demo procedure headlessly.

    Returns ``(session_dir, runtime)`` where ``runtime`` is a snapshot of the
    live Procedure state captured *before the proc object is discarded* --
    used by run_checks to lock the core invariants:

      - every hardware DataProducer registered to the DataManager queue
      - ExperimentConfig / HardwareManager device state is well-formed
        (exactly one primary, cameras tracked)
    """
    print(f"[mesofield] staging demo experiment in {workdir}")
    shutil.copytree(MESOFIELD_DEMO, workdir, dirs_exist_ok=True)

    config_path = workdir / "experiment.json"
    proc = load_procedure(str(config_path))

    print("[mesofield] running headlessly (duration cap will stop it)")
    duration = float(proc.config.get("duration", 2))
    finished = proc.run_until_finished(timeout=duration + 5.0)
    if not finished:
        raise RuntimeError("mesofield procedure did not finish within timeout")

    session_dir = (
        Path(proc.data_dir)
        / f"sub-{proc.config.subject}"
        / f"ses-{proc.config.session}"
    )
    manifest_path = session_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"AcquisitionManifest missing: {manifest_path}. "
            "Did Procedure._write_acquisition_manifest run?"
        )

    hw = proc.config.hardware
    runtime = {
        # device_ids registered to the DataManager (signals.data -> queue)
        "data_device_ids": sorted(
            getattr(d, "device_id", getattr(d, "id", "?")) for d in proc.data.devices
        ),
        # device_ids the HardwareManager built from the YAML
        "hardware_device_ids": sorted(hw.devices.keys()),
        # cameras the HardwareManager tracked (drives the MDA viewer)
        "camera_ids": sorted(
            getattr(c, "device_id", getattr(c, "id", "?")) for c in hw.cameras
        ),
        # devices flagged primary -- must be exactly one
        "primary_ids": sorted(
            getattr(d, "device_id", getattr(d, "id", "?"))
            for d in hw.devices.values()
            if getattr(d, "is_primary", False)
        ),
    }

    print(f"[mesofield] session dir: {session_dir}")
    print(f"[mesofield] manifest:    {manifest_path}")
    print(f"[mesofield] data.devices: {runtime['data_device_ids']}")
    return session_dir, runtime


# ---------------------------------------------------------------------------
# Stage 2 — Intermediate processor
# ---------------------------------------------------------------------------
class MockRegionalMeans(ProcessorRunner):
    """Toy mesomap-shaped processor: reads a TIFF, writes 4-quadrant means CSV.

    Defined inline so the test exercises the ProcessorRunner harness without
    pulling a real processor package into the dependency graph.
    """

    tool_name = "mock_regional_means"
    tool_version = "0.1.0"

    def run(self, inputs, *, n_regions: int = 4):
        if n_regions != 4:
            raise ValueError("This toy processor only knows the 4-quadrant case.")
        import tifffile

        tiff_path = Path(inputs[0])
        stack = tifffile.imread(tiff_path)
        if stack.ndim != 3:
            raise ValueError(f"Expected 3-D stack, got shape {stack.shape}")
        n_frames, h, w = stack.shape

        rows = np.empty((n_frames, n_regions), dtype=np.float64)
        rows[:, 0] = stack[:, : h // 2, : w // 2].mean(axis=(1, 2))
        rows[:, 1] = stack[:, : h // 2, w // 2:].mean(axis=(1, 2))
        rows[:, 2] = stack[:, h // 2:, : w // 2].mean(axis=(1, 2))
        rows[:, 3] = stack[:, h // 2:, w // 2:].mean(axis=(1, 2))

        df = pd.DataFrame(rows, columns=[f"region_{i}" for i in range(n_regions)])
        # tiff_path is at session/func/<tiff>; write to session/processed/.
        out_dir = tiff_path.parent.parent / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "regional_means.csv"
        df.to_csv(out_path, index=False)
        return [out_path]


def run_processor(session_dir: Path, acq: AcquisitionManifest) -> tuple[Path, ProcessingManifest]:
    """Run MockRegionalMeans on the camera TIFF and return its sidecar path."""
    camera_entry = _find_producer(acq, "frames")
    tiff_path = session_dir / camera_entry.output_path
    runner = MockRegionalMeans()
    outputs, manifest = runner(
        [tiff_path],
        upstream=acq,
        session_root=session_dir,
        n_regions=EXPECTED_PROCESSOR_REGIONS,
    )
    sidecar = runner.manifest_path(outputs)
    print(f"[processor] wrote {outputs[0]}")
    print(f"[processor] wrote {sidecar}")
    return sidecar, manifest


def _find_producer(acq: AcquisitionManifest, data_type: str) -> ProducerEntry:
    matches = [p for p in acq.producers if p.data_type == data_type]
    if not matches:
        raise AssertionError(f"No producer with data_type={data_type!r}")
    if len(matches) > 1:
        raise AssertionError(f"Ambiguous data_type={data_type!r}: {len(matches)} matches")
    return matches[0]


# ---------------------------------------------------------------------------
# Stage 3 — datakit (manifest-driven ingest)
# ---------------------------------------------------------------------------
class IngestError(RuntimeError):
    """Raised when the ingester refuses to process a manifest."""


def _require_compatible_schema(version: str) -> None:
    ours_major = SCHEMA_VERSION.split(".")[0]
    theirs_major = version.split(".")[0]
    if ours_major != theirs_major:
        raise IngestError(
            f"Incompatible schema_version {version!r}; ingester supports {ours_major}.x.y "
            f"(current: {SCHEMA_VERSION})"
        )


def _load_producer_table(session_root: Path, producer: ProducerEntry) -> pd.DataFrame:
    """Read one producer's output into a DataFrame. Knows about CSV and frame-metadata JSON."""
    file_type = producer.file_type
    csv_path = session_root / producer.output_path

    if file_type.endswith("csv"):
        return pd.read_csv(csv_path)

    if file_type == "ome.tiff":
        # The frame data itself isn't merged into the dataset; the
        # per-frame metadata sidecar is. Read from metadata_path.
        if not producer.metadata_path:
            raise IngestError(
                f"Camera producer {producer.device_id!r} has no metadata_path; "
                f"can't ingest a TIFF without its frame metadata sidecar."
            )
        sidecar_path = session_root / producer.metadata_path
        with open(sidecar_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        records = payload.get("p0", [])
        return pd.DataFrame(records)

    raise IngestError(
        f"Unknown file_type {file_type!r} for producer {producer.device_id!r}"
    )


def ingest_with_datakit(
    session_root: Path,
) -> tuple[Path, Path, AcquisitionManifest, list[ProcessingManifest], DatasetManifest]:
    """Read both manifests, merge declared outputs into a MultiIndex DataFrame."""
    acq_path = session_root / "manifest.json"
    acq = AcquisitionManifest.read(acq_path)
    _require_compatible_schema(acq.schema_version)
    if not acq.acquisition_complete:
        raise IngestError("Refusing to ingest: AcquisitionManifest reports acquisition_complete=false")
    print(f"[datakit ] read AcquisitionManifest (schema {acq.schema_version}, "
          f"{len(acq.producers)} producers)")

    processing_manifests: list[ProcessingManifest] = []
    processed_dir = session_root / "processed"
    if processed_dir.exists():
        for path in sorted(processed_dir.glob("*.process.json")):
            pm = ProcessingManifest.read(path)
            _require_compatible_schema(pm.schema_version)
            if pm.upstream_acquisition_hash and pm.upstream_acquisition_hash != acq.content_hash():
                raise IngestError(
                    f"ProcessingManifest {path.name} references a different upstream "
                    f"acquisition hash; refuses to ingest mismatched provenance"
                )
            processing_manifests.append(pm)
        print(f"[datakit ] read {len(processing_manifests)} ProcessingManifest(s)")

    subject = acq.session.subject
    session = acq.session.session
    task = acq.session.task or "default"

    # Anchor everyone to the camera's frame count (the canonical clock for this
    # demo); shorter sources are padded with NaN, longer ones are truncated.
    raw_frames: list[tuple[ProducerEntry, pd.DataFrame]] = []
    n_rows = 0
    for producer in acq.producers:
        df = _load_producer_table(session_root, producer)
        if df.empty:
            raise IngestError(f"Producer {producer.device_id!r} output is empty")
        raw_frames.append((producer, df))
        if producer.data_type == EXPECTED_CAMERA_SOURCE_TAG:
            n_rows = len(df)
    if n_rows == 0:
        n_rows = max(len(df) for _, df in raw_frames)

    processed_frames: list[tuple[ProducerEntry, pd.DataFrame, ProcessingManifest]] = []
    for pm in processing_manifests:
        for output in pm.outputs:
            path = session_root / output.output_path
            if not path.exists():
                raise IngestError(
                    f"ProcessingManifest declared {output.output_path!r} but file is missing"
                )
            df = pd.read_csv(path)
            processed_frames.append((output, df, pm))

    columns: dict[tuple[str, str], list] = {}

    def _add(source: str, df: pd.DataFrame) -> None:
        for col in df.columns:
            values = df[col].iloc[:n_rows].tolist()
            if len(values) < n_rows:
                values = values + [None] * (n_rows - len(values))
            columns[(source, col)] = values

    for producer, df in raw_frames:
        _add(producer.data_type, df)
    for output, df, _pm in processed_frames:
        _add(output.data_type, df)

    if not columns:
        raise IngestError("No data ingested from any producer.")

    table = pd.DataFrame(columns)
    table.columns = pd.MultiIndex.from_tuples(table.columns, names=["Source", "Signal"])
    table.index = pd.MultiIndex.from_tuples(
        [(subject, session, task)] * len(table),
        names=["Subject", "Session", "Task"],
    )

    out_dir = session_root / "processed"
    out_dir.mkdir(exist_ok=True)
    pkl_path = out_dir / "data.pkl"
    table.to_pickle(pkl_path)

    source_versions = [
        SourceVersion(
            tag=p.data_type,
            version="0.1.0",
            parser_class="test_pipeline.ingest_with_datakit",
        )
        for p in acq.producers
    ]
    for pm in processing_manifests:
        for output in pm.outputs:
            source_versions.append(
                SourceVersion(
                    tag=output.data_type,
                    version=pm.tool_version,
                    parser_class=f"{pm.tool_name}.declare_outputs",
                )
            )

    ds_manifest = DatasetManifest(
        datakit_version=str(_DATAKIT_VERSION),
        built_at=datetime.now(timezone.utc),
        upstream_acquisition_hash=acq.content_hash(),
        data_file="data.pkl",
        data_content_hash=hash_file(pkl_path),
        time_basis=TimeBasis(
            clock_source="derived",
            description="Anchored on camera frame count; encoder truncated.",
        ),
        source_versions=source_versions,
        columns=[(s, c) for (s, c) in table.columns.tolist()],
    )
    ds_manifest_path = out_dir / "manifest.json"
    ds_manifest.write(ds_manifest_path)

    print(f"[datakit ] wrote {pkl_path}")
    print(f"[datakit ] wrote {ds_manifest_path}")
    return pkl_path, ds_manifest_path, acq, processing_manifests, ds_manifest


# ---------------------------------------------------------------------------
# Stage 4 — consumer (mesofield.datakit.load_dataset)
# ---------------------------------------------------------------------------
def load_with_consumer(pkl_path: Path) -> pd.DataFrame:
    table = load_dataset(pkl_path)
    print(f"[consumer] loaded shape={table.shape} "
          f"index_levels={list(table.index.names)} "
          f"col_levels={list(table.columns.names)}")
    return table


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def run_checks(
    session_root: Path,
    acq: AcquisitionManifest,
    proc_manifests: list[ProcessingManifest],
    ds: DatasetManifest,
    pkl_path: Path,
    ds_manifest_path: Path,
    proc_sidecar_path: Path,
    table: pd.DataFrame,
    runtime: dict,
) -> CheckTracker:
    tracker = CheckTracker()
    acq_path = session_root / "manifest.json"

    # -- Core invariants: producers -> DataQueue, hardware/config state -----
    # These lock the non-negotiable runtime behaviour: every DataProducer
    # the HardwareManager built must register to the DataManager queue and
    # land rows in dataqueue.csv, while ExperimentConfig / HardwareManager
    # device state stays well-formed (exactly one primary, cameras tracked).
    tracker.section("Core invariants (DataProducer registration + state)")

    with tracker.step("HardwareManager built the expected devices"):
        assert runtime["hardware_device_ids"], "no hardware devices built"
        # Every producer named in the manifest came from a hardware device.
        manifest_ids = {p.device_id for p in acq.producers}
        assert manifest_ids.issubset(set(runtime["hardware_device_ids"])), (
            manifest_ids, runtime["hardware_device_ids"],
        )

    with tracker.step("exactly one primary device (HardwareManager._validate_primary)"):
        assert len(runtime["primary_ids"]) == 1, runtime["primary_ids"]

    with tracker.step("camera-type producers are tracked in hardware.cameras"):
        # The demo's `camera` producer (data_type=frames) must reach the
        # MDA gui's viewer iteration via HardwareManager.cameras.
        camera_producers = {p.device_id for p in acq.producers if p.data_type == "frames"}
        assert camera_producers.issubset(set(runtime["camera_ids"])), (
            camera_producers, runtime["camera_ids"],
        )

    with tracker.step("every hardware DataProducer registered to the DataManager"):
        # Procedure.initialize_hardware registers eagerly; the manifest's
        # producers must all be in proc.data.devices.
        assert manifest_ids.issubset(set(runtime["data_device_ids"])), (
            manifest_ids, runtime["data_device_ids"],
        )

    dataqueue_files = list(session_root.rglob("*_dataqueue.csv"))
    with tracker.step("exactly one dataqueue.csv written"):
        assert len(dataqueue_files) == 1, dataqueue_files

    dq = pd.read_csv(dataqueue_files[0]) if dataqueue_files else pd.DataFrame()
    with tracker.step("dataqueue.csv has a device_id column and rows"):
        assert "device_id" in dq.columns, list(dq.columns)
        assert len(dq) > 0, "dataqueue.csv is empty"

    with tracker.step("every manifest producer emitted rows into the dataqueue"):
        dq_devices = set(dq["device_id"].astype(str).unique())
        missing = {p.device_id for p in acq.producers} - dq_devices
        assert not missing, (
            f"producers absent from dataqueue: {sorted(missing)}; "
            f"queue carried: {sorted(dq_devices)}"
        )

    with tracker.step("each producer also wrote its own individual output file"):
        # The "individual files AND dataqueue" invariant: both must hold.
        for p in acq.producers:
            assert (session_root / p.output_path).exists(), p.output_path

    # -- Acquisition manifest -----------------------------------------------
    tracker.section("AcquisitionManifest (multi-producer)")

    with tracker.step("schema_version matches package SCHEMA_VERSION"):
        assert acq.schema_version == SCHEMA_VERSION

    with tracker.step("acquisition_complete is True"):
        assert acq.acquisition_complete is True

    with tracker.step("round-trips to equal object"):
        assert AcquisitionManifest.read(acq_path) == acq

    with tracker.step("session identity matches experiment.json"):
        assert acq.session.subject == EXP.subject
        assert acq.session.session == EXP.session
        assert acq.session.task == EXP.task
        assert acq.session.protocol == EXP.protocol
        assert acq.session.experimenter == EXP.experimenter

    with tracker.step("exactly 2 producers (wheel + camera)"):
        assert len(acq.producers) == 2

    wheel_p = _find_producer(acq, EXPECTED_WHEEL_SOURCE_TAG)
    camera_p = _find_producer(acq, EXPECTED_CAMERA_SOURCE_TAG)

    with tracker.step("wheel producer has expected device_id + bids_type=beh + no metadata_path"):
        assert wheel_p.device_id == EXP.wheel_device_id
        assert wheel_p.bids_type == "beh"
        assert wheel_p.metadata_path is None
        for k, v in EXP.wheel_calibration.items():
            assert wheel_p.calibration.get(k) == v, (k, wheel_p.calibration.get(k), v)

    with tracker.step("wheel producer declares its dataqueue payload schema"):
        assert wheel_p.dataqueue_schema is not None
        assert wheel_p.dataqueue_schema.device_id == EXP.wheel_device_id
        assert wheel_p.dataqueue_schema.payload_format == "scalar"

    with tracker.step("camera producer has expected device_id + bids_type=func + metadata_path"):
        assert camera_p.device_id == EXP.camera_device_id
        assert camera_p.bids_type == "func"
        assert camera_p.metadata_path is not None
        for k, v in EXP.camera_calibration.items():
            assert camera_p.calibration.get(k) == v, (k, camera_p.calibration.get(k), v)

    with tracker.step("camera producer has no dataqueue schema (it doesn't push to queue)"):
        assert camera_p.dataqueue_schema is None

    with tracker.step("every producer's output file exists on disk"):
        for p in acq.producers:
            assert (session_root / p.output_path).exists(), p.output_path

    with tracker.step("camera frame-metadata sidecar exists on disk"):
        assert (session_root / camera_p.metadata_path).exists()

    with tracker.step("camera TIFF has the declared frame count (≥ 1 frame)"):
        import tifffile
        stack = tifffile.imread(session_root / camera_p.output_path)
        assert stack.ndim == 3 and stack.shape[0] >= 1
        assert stack.shape[1] == EXP.camera_calibration["height"]
        assert stack.shape[2] == EXP.camera_calibration["width"]

    with tracker.step("camera sidecar JSON has one record per frame"):
        with open(session_root / camera_p.metadata_path) as fh:
            sidecar = json.load(fh)
        records = sidecar.get("p0", [])
        import tifffile
        stack = tifffile.imread(session_root / camera_p.output_path)
        assert len(records) == stack.shape[0]
        for rec in records:
            assert "frame_index" in rec and "TimeReceivedByCore" in rec

    with tracker.step("acquisition wall time fits inside duration + slack"):
        assert acq.ended_at is not None and acq.started_at is not None
        elapsed = (acq.ended_at - acq.started_at).total_seconds()
        assert 0 < elapsed < EXP.duration_s + 5.0, elapsed

    # -- BaseCamera live-view contract --------------------------------------
    # Independent of the headless acquisition: instantiate a fresh
    # MockFrameProducer and exercise the BaseCamera.snap / start_live /
    # stop_live methods. These are the same methods the MDA gui's snap
    # and live buttons will call on every camera (MMCamera, OpenCVCamera,
    # MockFrameProducer) once the gui's mmcore-only branch is replaced.
    tracker.section("BaseCamera live-view contract (snap / start_live / stop_live)")

    from mesofield.devices.mocks import MockFrameProducer
    from mesofield.devices.base_camera import BaseCamera

    with tracker.step("MockFrameProducer subclasses BaseCamera"):
        assert issubclass(MockFrameProducer, BaseCamera)

    cam_cfg = {"id": "snap_test", "width": 16, "height": 16, "frame_interval_ms": 20}
    cam = MockFrameProducer(cam_cfg)
    try:
        with tracker.step("snap() returns an ndarray with the configured shape"):
            frame = cam.snap()
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (16, 16)

        with tracker.step("start_live() runs without recording (no output_path)"):
            assert cam.output_path is None
            cam.start_live()
            # Give the run loop a moment to produce a couple of frames.
            import time
            time.sleep(0.1)
            assert cam.is_active
    finally:
        with tracker.step("stop_live() halts the capture thread cleanly"):
            cam.stop_live()
            assert not cam.is_active
            assert cam._thread is None

    # -- BaseCamera writer selection ----------------------------------------
    tracker.section("BaseCamera._make_writer dispatch")

    with tracker.step("file_type='ome.tiff' -> CustomWriter"):
        from mesofield.data.writer import CustomWriter
        cam.file_type = "ome.tiff"
        writer_path = str(session_root / "snap_test.ome.tiff")
        w = cam._make_writer(writer_path)
        assert isinstance(w, CustomWriter)

    with tracker.step("file_type='mp4' -> CV2Writer"):
        from mesofield.data.writer import CV2Writer
        cam.file_type = "mp4"
        w = cam._make_writer(str(session_root / "snap_test.mp4"))
        assert isinstance(w, CV2Writer)

    with tracker.step("unknown file_type raises ValueError"):
        cam.file_type = "raw"
        try:
            cam._make_writer(str(session_root / "snap_test.raw"))
        except ValueError:
            pass
        else:
            raise AssertionError("_make_writer accepted unknown file_type")

    # -- ProcessingManifest -------------------------------------------------
    tracker.section("ProcessingManifest (mock_regional_means)")

    with tracker.step("exactly 1 ProcessingManifest written"):
        assert len(proc_manifests) == 1
    pm = proc_manifests[0]

    with tracker.step("tool_name + tool_version match the runner"):
        assert pm.tool_name == EXPECTED_PROCESSOR_TAG
        assert pm.tool_version == "0.1.0"

    with tracker.step("upstream_acquisition_hash chains to the acquisition"):
        assert pm.upstream_acquisition_hash == acq.content_hash()

    with tracker.step("inputs[0].content_hash matches the camera TIFF on disk"):
        assert len(pm.inputs) == 1
        actual_hash = hash_file(session_root / camera_p.output_path)
        assert pm.inputs[0].content_hash == actual_hash

    with tracker.step("parameters round-trip the runtime kwargs"):
        assert pm.parameters.get("n_regions") == EXPECTED_PROCESSOR_REGIONS

    with tracker.step("outputs declare regional_means.csv as a ProducerEntry"):
        assert len(pm.outputs) == 1
        out = pm.outputs[0]
        assert out.data_type == EXPECTED_PROCESSOR_TAG
        assert out.file_type == "csv"
        assert (session_root / out.output_path).exists()

    with tracker.step("manifest round-trips to equal object"):
        assert ProcessingManifest.read(proc_sidecar_path) == pm

    # -- DatasetManifest (ingest -> consumer) -------------------------------
    tracker.section("DatasetManifest")

    with tracker.step("DatasetManifest.upstream_acquisition_hash == AcquisitionManifest.content_hash"):
        assert ds.upstream_acquisition_hash == acq.content_hash()

    with tracker.step("DatasetManifest.data_content_hash matches recomputed hash"):
        assert ds.data_content_hash == hash_file(pkl_path)

    with tracker.step("DatasetManifest round-trips to equal object"):
        assert DatasetManifest.read(ds_manifest_path) == ds

    with tracker.step("source_versions cover raw + processed sources"):
        tags = {sv.tag for sv in ds.source_versions}
        assert EXPECTED_WHEEL_SOURCE_TAG in tags
        assert EXPECTED_CAMERA_SOURCE_TAG in tags
        assert EXPECTED_PROCESSOR_TAG in tags

    with tracker.step("DatasetManifest.columns matches the table's columns"):
        declared = {tuple(c) for c in ds.columns}
        actual = set(table.columns.tolist())
        assert declared == actual, (declared ^ actual)

    # -- Dataset shape ------------------------------------------------------
    tracker.section("Dataset shape (consumer view)")

    with tracker.step("index is MultiIndex [Subject, Session, Task]"):
        assert isinstance(table.index, pd.MultiIndex)
        assert list(table.index.names) == ["Subject", "Session", "Task"]

    with tracker.step("index has one unique key matching the acquisition"):
        unique = set(table.index.unique().tolist())
        assert unique == {(EXP.subject, EXP.session, EXP.task)}, unique

    with tracker.step("columns are MultiIndex [Source, Signal]"):
        assert isinstance(table.columns, pd.MultiIndex)
        assert list(table.columns.names) == ["Source", "Signal"]

    with tracker.step(f"wheel source {EXPECTED_WHEEL_SOURCE_TAG!r} carries timestamp + payload"):
        signals = {sig for src, sig in table.columns if src == EXPECTED_WHEEL_SOURCE_TAG}
        assert EXPECTED_WHEEL_SIGNAL_COLS.issubset(signals), signals

    with tracker.step(f"camera source {EXPECTED_CAMERA_SOURCE_TAG!r} carries frame metadata"):
        signals = {sig for src, sig in table.columns if src == EXPECTED_CAMERA_SOURCE_TAG}
        assert EXPECTED_CAMERA_SIGNAL_COLS.issubset(signals), signals

    with tracker.step(f"processed source {EXPECTED_PROCESSOR_TAG!r} carries {EXPECTED_PROCESSOR_REGIONS} region columns"):
        signals = {sig for src, sig in table.columns if src == EXPECTED_PROCESSOR_TAG}
        expected = {f"region_{i}" for i in range(EXPECTED_PROCESSOR_REGIONS)}
        assert expected.issubset(signals), signals

    with tracker.step("row count == camera frame count == processor output rows"):
        import tifffile
        n_frames = tifffile.imread(session_root / camera_p.output_path).shape[0]
        means_rows = len(pd.read_csv(session_root / pm.outputs[0].output_path))
        assert len(table) == n_frames == means_rows

    with tracker.step("camera columns have no NaN (anchored to camera length)"):
        cam_cols = [c for c in table.columns if c[0] == EXPECTED_CAMERA_SOURCE_TAG]
        nan_count = int(table[cam_cols].isna().sum().sum())
        assert nan_count == 0, f"{nan_count} NaN in camera columns"

    # -- Time basis sanity --------------------------------------------------
    tracker.section("Time basis sanity")

    with tracker.step("wheel timestamps (first n_frames) monotonically non-decreasing"):
        ts = table[(EXPECTED_WHEEL_SOURCE_TAG, "timestamp")].astype(float).to_numpy()
        ts = ts[~np.isnan(ts)]
        assert (ts[1:] >= ts[:-1]).all()

    with tracker.step("wheel timestamps inside [started_at, ended_at + 1s]"):
        ts = table[(EXPECTED_WHEEL_SOURCE_TAG, "timestamp")].astype(float).to_numpy()
        ts = ts[~np.isnan(ts)]
        lo = acq.started_at.timestamp()
        hi = acq.ended_at.timestamp() + 1.0
        assert (ts >= lo).all() and (ts <= hi).all()

    with tracker.step("camera frame_index is sequential 0..N-1"):
        idx = table[(EXPECTED_CAMERA_SOURCE_TAG, "frame_index")].astype(int).to_numpy()
        assert (idx == np.arange(len(idx))).all()

    with tracker.step("regional means differ across quadrants (structured signal)"):
        means = table[
            [(EXPECTED_PROCESSOR_TAG, f"region_{i}") for i in range(EXPECTED_PROCESSOR_REGIONS)]
        ].to_numpy()
        # baselines (1000, 2000, 3000, 4000) per quadrant; means should differ by > 500.
        per_region = means.mean(axis=0)
        assert per_region[1] - per_region[0] > 500
        assert per_region[3] - per_region[2] > 500

    # -- Negative tests -----------------------------------------------------
    tracker.section("Negative tests (each must raise)")

    with tracker.step("tampered pickle bytes -> recomputed hash differs"):
        original = pkl_path.read_bytes()
        try:
            pkl_path.write_bytes(original + b"\x00")
            assert hash_file(pkl_path) != ds.data_content_hash
        finally:
            pkl_path.write_bytes(original)

    with tracker.step("tampered TIFF bytes -> recomputed hash differs from processing input hash"):
        # Tamper a *copy* rather than the original: the OME-TIFF can still be
        # memory-mapped by the writer, and reopening it for writing fails on
        # Windows ([Errno 22]). Hashing a mutated copy proves the same thing
        # without depending on the original being rewritable.
        tiff_path = session_root / camera_p.output_path
        tampered = tiff_path.parent / (tiff_path.name + ".tampered")
        shutil.copyfile(tiff_path, tampered)
        try:
            with open(tampered, "ab") as fh:
                fh.write(b"\x00")
            assert hash_file(tampered) != pm.inputs[0].content_hash
        finally:
            tampered.unlink(missing_ok=True)

    with tracker.step("mutated AcquisitionManifest -> hash no longer matches upstream"):
        mutated = acq.model_copy(update={"notes": "tampered"})
        assert mutated.content_hash() != ds.upstream_acquisition_hash

    with tracker.step("AcquisitionManifest with unknown field -> ValidationError"):
        forged = json.loads(acq_path.read_text())
        forged["unexpected_field"] = "boom"
        try:
            AcquisitionManifest.model_validate(forged)
        except ValidationError:
            pass
        else:
            raise AssertionError("pydantic accepted an unexpected field")

    with tracker.step("forged schema_version='9.9.9' -> IngestError"):
        forged_path = session_root / "manifest.forged_schema.json"
        forged = json.loads(acq_path.read_text())
        forged["schema_version"] = "9.9.9"
        forged_path.write_text(json.dumps(forged))
        try:
            _ingest_forged_manifest(session_root, forged_path)
        except IngestError:
            pass
        else:
            raise AssertionError("ingester accepted a future major schema_version")
        finally:
            forged_path.unlink(missing_ok=True)

    with tracker.step("forged acquisition_complete=False -> IngestError"):
        forged_path = session_root / "manifest.forged_complete.json"
        forged = json.loads(acq_path.read_text())
        forged["acquisition_complete"] = False
        forged_path.write_text(json.dumps(forged))
        try:
            _ingest_forged_manifest(session_root, forged_path)
        except IngestError:
            pass
        else:
            raise AssertionError("ingester accepted an incomplete acquisition")
        finally:
            forged_path.unlink(missing_ok=True)

    with tracker.step("forged producer output path -> IngestError"):
        forged_path = session_root / "manifest.forged_path.json"
        forged = json.loads(acq_path.read_text())
        forged["producers"][0]["output_path"] = "does/not/exist.csv"
        forged_path.write_text(json.dumps(forged))
        try:
            _ingest_forged_manifest(session_root, forged_path)
        except IngestError:
            pass
        else:
            raise AssertionError("ingester accepted a missing producer file")
        finally:
            forged_path.unlink(missing_ok=True)

    with tracker.step("ProcessingManifest with bad upstream_acquisition_hash -> ingester refuses"):
        forged_pm = pm.model_copy(update={"upstream_acquisition_hash": "0" * 64})
        forged_path = (session_root / "processed" / "mock_regional_means.forged.process.json")
        forged_path.write_text(forged_pm.to_json())
        # Move the real sidecar out of the way so the ingester sees the forged one.
        real_dest = proc_sidecar_path.with_suffix(".bak")
        proc_sidecar_path.rename(real_dest)
        try:
            ingest_with_datakit(session_root)
        except IngestError:
            pass
        else:
            raise AssertionError("ingester accepted a forged processing-upstream hash")
        finally:
            forged_path.unlink(missing_ok=True)
            real_dest.rename(proc_sidecar_path)

    with tracker.step("load_dataset rejects unsupported file extension"):
        bogus = session_root / "data.xyz"
        bogus.write_text("not a dataset")
        try:
            load_dataset(bogus)
        except ValueError:
            pass
        else:
            raise AssertionError("load_dataset accepted .xyz")
        finally:
            bogus.unlink(missing_ok=True)

    return tracker


def _ingest_forged_manifest(session_root: Path, manifest_path: Path) -> None:
    """Run the same validation gates the real ingester uses, against a forged manifest."""
    acq = AcquisitionManifest.read(manifest_path)
    _require_compatible_schema(acq.schema_version)
    if not acq.acquisition_complete:
        raise IngestError("Refusing to ingest: acquisition_complete=false")
    for producer in acq.producers:
        if not (session_root / producer.output_path).exists():
            raise IngestError(
                f"Producer {producer.device_id!r} output path missing: {producer.output_path}"
            )


# ---------------------------------------------------------------------------
# pytest entry point
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_end_to_end(tmp_path):
    """Acquire -> process -> ingest -> consume on the demo rig; all checks pass.

    The same flow as :func:`main` but rooted in pytest's ``tmp_path`` and
    asserting the CheckTracker recorded zero failures -- so a single bad check
    surfaces with its label + traceback instead of a bare exit code.
    """
    session_root, runtime = run_mesofield(tmp_path / "experiment")
    acq = AcquisitionManifest.read(session_root / "manifest.json")
    proc_sidecar_path, _ = run_processor(session_root, acq)
    pkl_path, ds_manifest_path, acq, proc_manifests, ds = ingest_with_datakit(session_root)
    table = load_with_consumer(pkl_path)

    tracker = run_checks(
        session_root, acq, proc_manifests, ds, pkl_path, ds_manifest_path,
        proc_sidecar_path, table, runtime,
    )
    assert not tracker.failed, "pipeline checks failed:\n" + "\n".join(
        f"  - {label}\n{tb}" for label, tb in tracker.failed
    )


# ---------------------------------------------------------------------------
def main() -> int:
    print(f"=== Pipeline smoke test (mesokit-schema {SCHEMA_VERSION}) ===")
    tmp = Path(tempfile.mkdtemp(prefix="pipeline_demo_"))
    try:
        session_root, runtime = run_mesofield(tmp / "experiment")
        acq = AcquisitionManifest.read(session_root / "manifest.json")
        proc_sidecar_path, _ = run_processor(session_root, acq)
        pkl_path, ds_manifest_path, acq, proc_manifests, ds = ingest_with_datakit(session_root)
        table = load_with_consumer(pkl_path)

        print("\n--- first 3 rows ---")
        print(table.head(3))

        tracker = run_checks(
            session_root, acq, proc_manifests, ds, pkl_path, ds_manifest_path,
            proc_sidecar_path, table, runtime,
        )
        rc = tracker.summary()
        print(f"\nArtifacts kept at: {tmp}")
        return rc
    except Exception as exc:
        print(f"\nFAILED before checks ran: {exc.__class__.__name__}: {exc}")
        raise


if __name__ == "__main__":
    sys.exit(main())
