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

from mesofield.config import ExperimentConfig


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
