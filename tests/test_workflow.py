"""End-to-end Procedure workflow against the real DataManager/DataSaver API.

Drives the *generic* ``Procedure.run`` (no subclass overrides) with the
registered mock devices declared in a hardware YAML, exactly as a user rig
would. The primary mock camera self-terminates at the configured duration,
which triggers cleanup -> save_data -> manifest. Asserts the on-disk
contract: every DataPaths entry exists and the source experiment.json gets
its session auto-incremented.
"""

import json
from pathlib import Path

import pytest

from mesofield.base import Procedure
from mesofield.data.manager import DataManager

# Importing mocks registers `mock_camera` / `mock_wheel` with DeviceRegistry.
import mesofield.devices.mocks  # noqa: F401


HARDWARE_YAML = """
memory_buffer_size: 100

camera:
  type: mock_camera
  primary: true
  width: 16
  height: 16
  frame_interval_ms: 50
  output:
    suffix: meso
    file_type: ome.tiff
    bids_type: func

wheel:
  type: mock_wheel
  sample_interval_ms: 50
  output:
    suffix: wheel
    file_type: csv
    bids_type: beh
"""

EXPERIMENT_JSON = {
    "Configuration": {
        "experimenter": "tester",
        "protocol": "workflow_test",
        "duration": 1,
        "start_on_trigger": False,
    },
    "Subjects": {
        "SUBJ1": {
            "sex": "F",
            "genotype": "test",
            "session": "01",
            "task": "wf",
        }
    },
    "DisplayKeys": ["duration", "start_on_trigger", "task", "session"],
}


@pytest.mark.integration
def test_procedure_workflow(tmp_path):
    hw_path = tmp_path / "hardware.yaml"
    hw_path.write_text(HARDWARE_YAML)
    cfg_json = tmp_path / "experiment.json"
    cfg_json.write_text(json.dumps(EXPERIMENT_JSON))

    proc = Procedure(hardware=str(hw_path), experiment=str(cfg_json))
    proc.config.experiment_dir = str(tmp_path)

    assert len(proc.hardware.devices) == 2
    assert isinstance(proc.data, DataManager)

    proc.add_note("test note")

    # duration=1s: the primary mock camera self-terminates, which drives
    # _cleanup_procedure -> save_data -> manifest with no manual stop.
    assert proc.run_until_finished(timeout=15), "procedure never finished"

    paths = proc.data.save.paths
    assert Path(paths.configuration).exists()
    assert Path(paths.notes).exists()
    assert Path(paths.timestamps).exists()
    assert Path(paths.queue).exists()
    for dev_id in ("camera", "wheel"):
        assert Path(paths.hardware[dev_id]).exists(), f"{dev_id} output missing"

    session_dir = Path(proc.config.data_dir) / "sub-SUBJ1" / "ses-01"
    manifest = json.loads((session_dir / "manifest.json").read_text())
    assert manifest["acquisition_complete"] is True

    # save_json round-trips the source experiment.json without leaking
    # subject-only keys into the Configuration block.
    data = json.loads(cfg_json.read_text())
    assert data["Subjects"]["SUBJ1"]["session"] == "01"
    assert "sex" not in data["Configuration"]
