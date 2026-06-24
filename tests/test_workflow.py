"""End-to-end Procedure orchestration + data-outcome test on the mock rig.

Drives the *real* ``Procedure.run`` lifecycle (construct -> initialize hardware
-> arm -> start -> duration-timer cleanup -> save) with the shipped mock devices
from ``mesofield/devices/mocks.py``, then asserts the on-disk acquisition
outcomes and the emitted manifest.

This replaces the previous hand-rolled ``DummyCamera``/``DummyEncoder``/
``DummyWriter`` fakes (and the obsolete ``DataSaver.writer_for`` monkeypatch)
with production code paths, so the test cannot silently drift from real device
behavior. Covers areas: Procedure orchestration, hardware lifecycle, and data
management / acquisition outcomes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

# Importing the module registers the ``mock_wheel`` / ``mock_camera`` device
# types used by the fixture's hardware.yaml.
import mesofield.devices.mocks  # noqa: F401
from mesofield.base import Procedure
from mesofield.data.manager import DataManager


def test_procedure_workflow(tmp_path, hardware_yaml, experiment_json):
    hw = hardware_yaml(camera=True)          # primary mock wheel + mock camera
    cfg_json = experiment_json(duration=1)   # 1s wall-clock duration cap (int)
    out_dir = tmp_path / "out"

    proc = Procedure(
        hardware=str(hw),
        config=str(cfg_json),
        experiment_directory=str(out_dir),
    )

    # Hardware brought up and a DataManager created at construction.
    assert isinstance(proc.data, DataManager)
    assert len(proc.hardware.devices) == 2
    assert proc.hardware.primary is not None

    proc.add_note("integration note")

    # Real run: arms a 0.3s duration timer that drives cleanup + save.
    assert proc.run_until_finished(timeout=15) is True

    paths = proc.data.save.paths

    # Always-produced session artifacts.
    assert Path(paths.configuration).exists()
    assert Path(paths.timestamps).exists()
    assert Path(paths.notes).exists(), "a note was added but notes.txt is missing"
    assert Path(paths.queue).exists()

    # Every non-camera device wrote its own data file via DataSaver.all_hardware.
    for dev_id, dev in proc.hardware.devices.items():
        if getattr(dev, "device_type", None) != "camera":
            assert Path(paths.hardware[dev_id]).exists(), f"missing output for {dev_id}"

    # Data actually flowed from the devices through the queue logger.
    with open(paths.queue, newline="") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) > 1, "dataqueue has only a header -- no device data captured"

    # The acquisition manifest landed at the session root and lists every
    # device that produced an output (mesokit-schema contract).
    manifest_files = list(Path(out_dir).rglob("manifest.json"))
    assert manifest_files, "no AcquisitionManifest written"
    manifest = json.loads(manifest_files[0].read_text())
    producer_ids = {p["device_id"] for p in manifest.get("producers", [])}
    assert producer_ids == set(proc.hardware.devices.keys())
