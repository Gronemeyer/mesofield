import json
import textwrap
import time
from pathlib import Path
from datetime import datetime
import types

import pandas as pd
import pytest

from mesofield.base import Procedure, ProcedureConfig
from mesofield.config import ExperimentConfig
from mesofield.data.manager import DataManager, DataSaver
from mesofield import DeviceRegistry


class DummyEvent:
    def __init__(self):
        self._callbacks = []

    def connect(self, cb):
        self._callbacks.append(cb)

    def emit(self, val):
        for cb in self._callbacks:
            cb(val)


class DummyWriter:
    def __init__(self, path: str):
        self._filename = path
        self._frame_metadata_filename = path + "_meta.json"
        Path(self._frame_metadata_filename).write_text("{}")

    def write_frame(self, *a, **k):
        pass


class DummyCamera:
    device_type = "camera"
    file_type = "ome.tiff"
    bids_type = "func"
    sampling_rate = 1.0

    def __init__(self, cfg: dict | None = None):
        self.device_id = cfg.get("id", "cam") if cfg else "cam"
        self.id = self.device_id
        self.name = cfg.get("name", self.device_id) if cfg else "cam"
        self.data_event = DummyEvent()

        class _Core:
            def __init__(self):
                self.mda = types.SimpleNamespace(events=DummyEvent())

        self.core = _Core()
        self.output_path = ""
        self.metadata_path = None
        self.started = False

    def start(self):
        self.started = True
        return True

    def stop(self):
        self.started = False
        return True

    def save_data(self, path=None):
        assert path is not None
        Path(path).write_text("camera")

    def get_data(self):
        return None


class DummyEncoder:
    device_type = "encoder"
    device_id = "encoder"
    file_type = "csv"
    bids_type = "beh"

    def __init__(self, *a, **k):
        self.output_path = ""
        self.started = False

    def start_recording(self):
        self.started = True

    def start(self):
        self.start_recording()
        return True

    def stop(self):
        self.started = False
        return True

    def save_data(self, path=None):
        assert path is not None
        Path(path).write_text("encoder")

    def get_data(self):
        return []


# patch DataSaver.writer_for to avoid heavy dependencies
def _dummy_writer_for(self: DataSaver, camera: DummyCamera):
    path = self.cfg.make_path(camera.name, "dat", camera.bids_type)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    camera.output_path = path
    self.paths.writers[camera.name] = path
    Path(path).write_text("data")
    return DummyWriter(path)


class DummyProcedure(Procedure):
    def run(self):
        self.data.start_queue_logger()
        for cam in self.hardware.cameras:
            self.data.save.writer_for(cam)
            cam.start()
        self.hardware.encoder.start()
        self.start_time = datetime.now()
        self.data.queue.push("cam1", "frame")
        self.data.queue.push("encoder", 1)
        time.sleep(0.1)
        self.hardware.encoder.stop()
        for cam in self.hardware.cameras:
            cam.stop()
        self.stopped_time = datetime.now()
        self.data.stop_queue_logger()
        self.save_data()
        self.data.update_database()


def test_procedure_workflow(tmp_path, monkeypatch):
    # register dummy devices
    DeviceRegistry._registry["camera"] = DummyCamera
    DeviceRegistry._registry["encoder"] = DummyEncoder
    monkeypatch.setattr(DataSaver, "writer_for", _dummy_writer_for)
    monkeypatch.setattr("mesofield.hardware.SerialWorker", DummyEncoder)
    monkeypatch.setattr("mesofield.hardware.EncoderSerialInterface", DummyEncoder)

    hw_path = tmp_path / "hardware.yaml"
    hw_path.write_text(
        textwrap.dedent(
            """
            memory_buffer_size: 1
            encoder:
              type: wheel
              port: COM1
            cameras:
              - id: cam1
                name: cam1
                backend: dummy
            """
        )
    )

    cfg_json = tmp_path / "config.json"
    json.dump({
        "Configuration": {
            "experimenter": "tester",
            "protocol": "exp1",
            "experiment_directory": str(tmp_path),
            "hardware_config_file": str(hw_path),
            "database_path": str(tmp_path / "db.h5"),
            "duration": 1,
            "start_on_trigger": False,
            "led_pattern": ["4", "4"]
        },
        "Subjects": {
            "SUBJ1": {
                "sex": "F",
                "genotype": "test",
                "DOB": "2024-01-01",
                "DOS": "2024-01-02",
                "session": "01",
                "task": "wf"
            }
        },
        "DisplayKeys": ["duration", "start_on_trigger", "task", "session"]
    }, cfg_json.open("w"))

    pcfg = ProcedureConfig(
        experiment_id="exp1",
        experimentor="tester",
        hardware_yaml=str(hw_path),
        data_dir=str(tmp_path),
        json_config=str(cfg_json),
    )

    proc = DummyProcedure(pcfg)

    # configuration loaded
    assert len(proc.hardware.devices) == 2
    assert isinstance(proc.data, DataManager)

    proc.add_note("test note")
    proc.run()

    paths = proc.data.save.paths
    assert Path(paths.configuration).exists()
    assert Path(paths.notes).exists()
    assert Path(paths.timestamps).exists()
    assert Path(paths.hardware["encoder"]).exists()
    assert Path(paths.writers["cam1"]).exists()
    assert Path(paths.queue).exists()

    db = proc.data.base
    assert db and db.path.exists()
    df = db.read("datapaths")
    assert isinstance(df, pd.DataFrame) and not df.empty

    # configuration JSON updated with incremented session
    with open(cfg_json) as f:
        data = json.load(f)
    assert data["Subjects"]["SUBJ1"]["session"] == "02"
    # subject-only keys should not be written to the Configuration block
    assert "sex" not in data["Configuration"]
