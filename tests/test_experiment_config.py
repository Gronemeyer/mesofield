"""ExperimentConfig / ConfigRegister state, validation, and BIDS paths.

Complements ``test_config_simplification`` (LED normalization + dropdown
choices) with coverage of registered defaults, type validation/coercion,
session zero-padding, ``make_path`` layout + uniqueness, value-change
callbacks, and JSON/dict loading.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mesofield.config import ExperimentConfig, parse_task_from_filename


@pytest.fixture
def cfg():
    # No hardware path -> empty HardwareManager; fast and headless.
    return ExperimentConfig()


def test_defaults_are_registered(cfg):
    assert cfg.get("subject") == "sub"
    assert cfg.get("duration") == 60
    assert cfg.get("start_on_trigger") is False
    assert cfg.led_pattern == ["4", "4"]
    assert cfg.experiment_dir_is_set is False
    assert cfg.notes == []


def test_set_coerces_to_registered_type(cfg):
    cfg.set("duration", "5")  # str -> int (registered type)
    assert cfg.get("duration") == 5
    assert isinstance(cfg.get("duration"), int)


def test_set_invalid_type_raises(cfg):
    with pytest.raises(TypeError):
        cfg.set("duration", "not-an-int")


def test_session_is_zero_padded(cfg):
    cfg.set("session", "1")
    assert cfg.session == "01"
    cfg.set("session", 7)
    assert cfg.session == "07"


def test_register_callback_fires_on_set(cfg):
    seen = []
    cfg.register_callback("subject", lambda k, v: seen.append((k, v)))
    cfg.set("subject", "M123")
    assert seen == [("subject", "M123")]


def test_make_path_bids_layout(tmp_path, cfg):
    cfg.experiment_dir = str(tmp_path)
    cfg.set("subject", "001")
    cfg.set("session", "1")
    cfg.set("task", "rest")

    p = Path(cfg.make_path("meso", "ome.tiff", "func", create_dir=True))

    assert p.parent.as_posix().endswith("data/sub-001/ses-01/func")
    assert p.name.endswith("sub-001_ses-01_task-rest_meso.ome.tiff")


def test_make_path_returns_unique_paths(tmp_path, cfg):
    cfg.experiment_dir = str(tmp_path)
    p1 = cfg.make_path("x", "csv", "beh", create_dir=True)
    Path(p1).write_text("")
    p2 = cfg.make_path("x", "csv", "beh", create_dir=True)
    assert p2 != p1
    assert not Path(p2).exists()  # never collides with an existing file


def test_load_json_applies_values_and_directory(tmp_path, cfg):
    j = tmp_path / "experiment.json"
    j.write_text(
        json.dumps(
            {
                "Configuration": {
                    "experimenter": "tester",
                    "duration": 3,
                    "protocol": ["A", "B"],  # list -> dropdown choices
                },
                "Subjects": {"S1": {"session": "01", "task": "go"}},
                "DisplayKeys": ["duration"],
            }
        )
    )
    cfg.load_json(str(j))

    assert cfg.get("experimenter") == "tester"
    assert cfg.get("duration") == 3
    assert cfg.get("protocol") == "A"           # first choice selected
    assert cfg.get_choices("protocol") == ["A", "B"]
    assert "S1" in cfg.subjects
    assert cfg.experiment_dir == str(tmp_path)   # follows the JSON's dir
    assert cfg.experiment_dir_is_set is True


def test_load_dict_flat_mapping(cfg):
    cfg.load_dict({"duration": 9, "task": "abc"})
    assert cfg.get("duration") == 9
    assert cfg.get("task") == "abc"


# --- PsychoPy task->script mapping -------------------------------------------


def test_parse_task_from_filename():
    assert parse_task_from_filename("Gratings_task-grat_v0.9.py") == "grat"
    assert parse_task_from_filename("/abs/path/task-mov_natural.py") == "mov"
    assert parse_task_from_filename("no_token_here.py") is None


def test_psychopy_block_drives_task_choices_and_path(tmp_path, cfg):
    j = tmp_path / "experiment.json"
    j.write_text(
        json.dumps(
            {
                "Configuration": {"duration": 5},
                "Subjects": {"S1": {"session": "01"}},
                "PsychoPy": {
                    "grat": "Gratings_task-grat.py",
                    "mov": "natural_task-mov.py",
                },
                "DisplayKeys": ["task"],
            }
        )
    )
    cfg.load_json(str(j))

    # Task choices are derived from the map keys (sorted), defaulting to the first.
    assert cfg.get_choices("task") == ["grat", "mov"]
    assert cfg.get("task") == "grat"
    assert cfg.psychopy_path == str(tmp_path / "Gratings_task-grat.py")

    # Switching the task switches the resolved script.
    cfg.set("task", "mov")
    assert cfg.psychopy_path == str(tmp_path / "natural_task-mov.py")


def test_psychopy_entry_with_trial_duration(tmp_path, cfg):
    cfg.experiment_dir = str(tmp_path)
    cfg.set("duration", 100)
    cfg.update_psychopy(
        {
            "mov": {"file": "task-mov.py", "trial_duration": 20},
            "grat": "task-grat.py",  # plain string -> no per-task trial duration
        }
    )

    # Dict entry: script resolves, trial_duration drives num_trials (100 // 20).
    cfg.set("task", "mov")
    assert cfg.psychopy_path == str(tmp_path / "task-mov.py")
    assert cfg.trial_duration == 20
    assert cfg.num_trials == 5
    assert cfg.psychopy_parameters["num_trials"] == 5
    assert cfg.psychopy_parameters["trial_duration"] == 20

    # Plain-string entry: falls back to the stored num_trials default.
    cfg.set("task", "grat")
    assert cfg.psychopy_path == str(tmp_path / "task-grat.py")
    assert cfg.num_trials == 20

    # The dict entry round-trips to disk as a dict.
    j = tmp_path / "experiment.json"
    j.write_text(json.dumps({"Configuration": {}, "Subjects": {}}))
    cfg._json_file_path = str(j)
    cfg.update_psychopy({"mov": {"file": "task-mov.py", "trial_duration": 20}})
    written = json.loads(j.read_text())
    assert written["PsychoPy"] == {"mov": {"file": "task-mov.py", "trial_duration": 20}}


def test_psychopy_path_legacy_fallback(tmp_path, cfg):
    cfg.experiment_dir = str(tmp_path)
    cfg.set("psychopy_filename", "single_script.py")
    # No PsychoPy map -> legacy single-script behavior.
    assert cfg.psychopy_path == str(tmp_path / "single_script.py")


def test_psychopy_path_absolute_entry(tmp_path, cfg):
    cfg.experiment_dir = str(tmp_path)
    abs_script = tmp_path / "elsewhere" / "task-spont.py"
    cfg.set("task", "spont")
    cfg.set("psychopy", {"spont": str(abs_script)})
    assert cfg.psychopy_path == str(abs_script)


def test_update_psychopy_persists_block(tmp_path, cfg):
    j = tmp_path / "experiment.json"
    j.write_text(
        json.dumps(
            {
                "Configuration": {"duration": 5},
                "Subjects": {"S1": {"session": "01"}},
                "DisplayKeys": ["task"],
            }
        )
    )
    cfg.load_json(str(j))

    cfg.update_psychopy({"a": "task-a.py", "b": "task-b.py"})

    # In-memory choices updated.
    assert cfg.get_choices("task") == ["a", "b"]
    assert cfg.get("task") == "a"

    # Top-level PsychoPy block written back to disk.
    written = json.loads(j.read_text())
    assert written["PsychoPy"] == {"a": "task-a.py", "b": "task-b.py"}


# --- Task-gated stimulus selection (PsychoPy + MousePortal on one rig) --------


def test_stimulus_task_choices_are_unioned(tmp_path, cfg):
    j = tmp_path / "experiment.json"
    j.write_text(
        json.dumps(
            {
                "Configuration": {"duration": 5, "task": ["baseline"]},
                "Subjects": {"S1": {"session": "01"}},
                "PsychoPy": {"grat": "task-grat.py"},
                "MousePortal": {"task": "corridor", "experiment": {"num_blocks": 1}},
                "DisplayKeys": ["task"],
            }
        )
    )
    cfg.load_json(str(j))

    # The dropdown is the union of the plain Configuration task, the PsychoPy
    # map keys, and MousePortal's bound task.
    assert cfg.get_choices("task") == ["baseline", "corridor", "grat"]
    # A still-valid current task (from Configuration) is preserved.
    assert cfg.get("task") == "baseline"


def test_psychopy_device_serves_only_its_tasks(cfg):
    from mesofield.devices.psychopy_device import PsychoPyDevice

    dev = PsychoPyDevice({})
    cfg.set("psychopy", {"grat": "task-grat.py"})
    assert dev.serves_task("grat", cfg) is True
    assert dev.serves_task("mov", cfg) is False
    # No map -> legacy single-script behavior: serves every task.
    cfg.set("psychopy", {})
    assert dev.serves_task("anything", cfg) is True


def test_mouseportal_device_serves_only_its_task(cfg):
    from mesofield.devices.mouseportal_device import MousePortalDevice

    dev = MousePortalDevice({})
    cfg.set("mouseportal", {"task": "corridor"})
    assert dev.serves_task("corridor", cfg) is True
    assert dev.serves_task("grat", cfg) is False
    # No bound task -> serves every task (single-stimulus rig).
    cfg.set("mouseportal", {})
    assert dev.serves_task("anything", cfg) is True
