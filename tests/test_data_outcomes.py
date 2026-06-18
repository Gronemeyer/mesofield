"""Data-management units: DataPaths layout + DataSaver outputs.

The integrated outcomes (per-device files, manifest, writer dispatch) are
covered by ``test_workflow`` and ``test_pipeline``. This module unit-tests the
pieces those don't isolate: the BIDS path layout ``DataPaths.build`` produces,
and ``DataSaver.save_queue``'s dict-payload column fan-out.
"""

from __future__ import annotations

import csv
from datetime import datetime

import pytest

# Register mock device types for the hardware_yaml fixture.
import mesofield.devices.mocks  # noqa: F401
from mesofield.config import ExperimentConfig
from mesofield.data.manager import DataPaths, DataSaver


# --------------------------------------------------------------------------- #
# DataPaths.build -- BIDS layout per device
# --------------------------------------------------------------------------- #
def test_datapaths_build_bids_layout(experiment_config, tmp_path):
    cfg = experiment_config(camera=True)
    cfg.experiment_dir = str(tmp_path)
    cfg.hardware.initialize(cfg)  # materialize the mock devices

    paths = DataPaths.build(cfg)

    assert set(paths.hardware) == {"wheel", "camera"}
    wheel = paths.hardware["wheel"].replace("\\", "/")
    camera = paths.hardware["camera"].replace("\\", "/")
    assert "/beh/" in wheel and wheel.endswith("_wheel.csv")
    assert "/func/" in camera and camera.endswith("_meso.ome.tiff")

    assert paths.configuration.endswith("_configuration.csv")
    assert paths.timestamps.endswith("_timestamps.csv")
    assert paths.queue.replace("\\", "/").endswith("_dataqueue.csv")


# --------------------------------------------------------------------------- #
# DataSaver.save_queue -- dict payload fan-out
# --------------------------------------------------------------------------- #
@pytest.fixture
def saver(tmp_path):
    cfg = ExperimentConfig()
    cfg.experiment_dir = str(tmp_path)
    return DataSaver(cfg)


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.reader(fh))


def test_save_queue_fans_dict_payloads_to_columns(saver, tmp_path):
    ts = datetime.now().isoformat()
    rows = [
        [0.0, ts, None, "wheel", 5],                              # scalar payload
        [0.1, ts, 1234, "treadmill", {"distance": 1.5, "speed": 2.0}],  # dict
    ]
    out = tmp_path / "queue.csv"
    saver.save_queue(rows, str(out))

    table = _read(out)
    header = table[0]
    assert header == [
        "queue_elapsed", "packet_ts", "device_ts", "device_id",
        "payload", "distance", "speed",
    ]
    by_device = {r[3]: r for r in table[1:]}
    # Scalar payload sits in the payload column; dict columns blank.
    assert by_device["wheel"][4] == "5"
    assert by_device["wheel"][5:] == ["", ""]
    # Dict payload fans out; the payload column is blank.
    assert by_device["treadmill"][4] == ""
    assert by_device["treadmill"][5:] == ["1.5", "2.0"]


def test_save_queue_scalar_only_has_single_payload_column(saver, tmp_path):
    rows = [[0.0, "t", None, "wheel", 3], [0.1, "t", None, "wheel", 4]]
    out = tmp_path / "queue.csv"
    saver.save_queue(rows, str(out))

    table = _read(out)
    assert table[0] == ["queue_elapsed", "packet_ts", "device_ts", "device_id", "payload"]
    assert [r[4] for r in table[1:]] == ["3", "4"]


# --------------------------------------------------------------------------- #
# DataSaver.configuration + save_timestamps
# --------------------------------------------------------------------------- #
def test_configuration_writes_parameter_value_csv(saver):
    saver.cfg.set("subject", "M99")
    saver.configuration()

    table = _read(saver.paths.configuration)
    assert table[0] == ["Parameter", "Value"]
    rows = {r[0]: r[1] for r in table[1:]}
    assert rows.get("subject") == "M99"


def test_save_timestamps_writes_header_and_row(saver):
    saver.save_timestamps("PROTO", "2026-01-01T00:00:00", "2026-01-01T00:00:05")

    table = _read(saver.paths.timestamps)
    assert table[0] == ["device_id", "started", "stopped"]
    assert table[1][:3] == ["PROTO", "2026-01-01T00:00:00", "2026-01-01T00:00:05"]
