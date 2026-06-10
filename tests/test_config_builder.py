"""Tests for the guided config builder + rig-store serialization.

Covers the "hard to fail" guarantee: a builder-assembled mapping always
round-trips to a parseable rig that HardwareManager accepts, and a mock rig
instantiates real device objects without any hardware or custom procedure.py.
"""

from __future__ import annotations

import pytest

from mesofield.scaffold import rigs


def _use_temp_store(monkeypatch, tmp_path):
    """Point the rig store at an isolated temp dir."""
    monkeypatch.setattr(rigs, "rigs_dir", lambda: tmp_path)


def _mock_rig_doc() -> dict:
    return {
        "memory_buffer_size": 1000,
        "camera": {
            "type": "mock_camera", "width": 64, "height": 64,
            "frame_interval_ms": 50, "primary": True,
            "output": {"suffix": "meso", "file_type": "ome.tiff", "bids_type": "func"},
        },
        "wheel": {
            "type": "mock_wheel", "sample_interval_ms": 50, "cpr": 2400,
            "diameter_mm": 80,
            "output": {"suffix": "wheel", "file_type": "csv", "bids_type": "beh"},
        },
    }


def test_save_rig_roundtrips_and_instantiates_devices(monkeypatch, tmp_path):
    _use_temp_store(monkeypatch, tmp_path)

    path = rigs.save_rig("unit_rig", _mock_rig_doc())
    assert path.is_file()
    assert "unit_rig" in rigs.list_rigs()

    from mesofield.config import ExperimentConfig

    cfg = ExperimentConfig(str(path))
    assert cfg.hardware.is_configured
    cfg.hardware.initialize(cfg)
    assert set(cfg.hardware.devices.keys()) == {"camera", "wheel"}


def test_save_rig_refuses_overwrite_without_force(monkeypatch, tmp_path):
    _use_temp_store(monkeypatch, tmp_path)

    rigs.save_rig("dup", {"memory_buffer_size": 1})
    with pytest.raises(FileExistsError):
        rigs.save_rig("dup", {"memory_buffer_size": 2})
    # force=True overwrites cleanly
    assert rigs.save_rig("dup", {"memory_buffer_size": 2}, force=True).is_file()


def test_mock_device_types_registered_by_default():
    """The builder's mock catalog must resolve without a custom procedure.py."""
    from mesofield import DeviceRegistry
    import mesofield.hardware  # noqa: F401  (triggers device registration)

    assert DeviceRegistry.get_class("mock_wheel") is not None
    assert DeviceRegistry.get_class("mock_camera") is not None


def test_build_experiment_doc_single_subject():
    from mesofield.gui.config_builder import build_experiment_doc

    doc = build_experiment_doc(
        configuration={"experimenter": "jake", "protocol": "DEMO",
                       "duration": 7, "start_on_trigger": True},
        tasks=["run"],
        variables=[],
        subjects=[{"subject": "M1", "session": "03", "task": "run"}],
    )
    assert doc["Configuration"]["duration"] == 7
    assert doc["Configuration"]["start_on_trigger"] is True
    assert doc["Configuration"]["task"] == "run"  # single task -> scalar
    assert doc["Subjects"]["M1"] == {"session": "03", "task": "run"}
    assert "subject" in doc["DisplayKeys"]


def test_build_experiment_doc_multi_subject_variables_and_blanks():
    from mesofield.gui.config_builder import build_experiment_doc

    doc = build_experiment_doc(
        configuration={"experimenter": "j", "protocol": "P", "duration": 5,
                       "start_on_trigger": False},
        tasks=["baseline", "stim"],         # multiple -> list (runtime dropdown)
        variables=[("genotype", str, True), ("weight", int, False)],  # weight hidden
        subjects=[
            {"subject": "M1", "session": "01", "task": "baseline",
             "genotype": "wt", "weight": "22"},
            {"subject": "M2", "session": "01", "task": "stim",
             "genotype": "", "weight": ""},   # blanks -> omitted ("or not")
        ],
    )
    assert doc["Configuration"]["task"] == ["baseline", "stim"]
    assert doc["Subjects"]["M1"] == {
        "session": "01", "task": "baseline", "genotype": "wt", "weight": 22,
    }
    assert doc["Subjects"]["M2"] == {"session": "01", "task": "stim"}  # no blanks
    # weight is recorded per-subject but hidden from the app -> not in DisplayKeys
    assert doc["DisplayKeys"] == [
        "subject", "session", "task", "genotype",
        "experimenter", "protocol", "duration",
    ]


def test_device_specs_catalog_keys():
    from mesofield.gui.config_builder import DEVICE_SPECS

    assert {"opencv_camera", "camera", "wheel", "mock_wheel", "mock_camera",
            "psychopy", "mouseportal"} <= set(DEVICE_SPECS)


def test_stimulus_specs_have_no_output_and_launch_fields():
    """Stimulus apps are subprocess plumbing: no output stream, never primary."""
    from mesofield.gui.config_builder import DEVICE_SPECS

    for type_key in ("psychopy", "mouseportal"):
        spec = DEVICE_SPECS[type_key]
        assert spec.stimulus is True
        assert spec.category == "Stimulus"
        assert spec.output == []  # not a DataProducer
        keys = {f.key for f in spec.fields}
        assert {"app_dir", "python_exe", "ready_timeout"} <= keys  # transitional plumbing

    mp = DEVICE_SPECS["mouseportal"]
    mp_keys = {f.key for f in mp.fields}
    assert {"udp_port", "treadmill_id", "tail_seconds"} <= mp_keys
