import json

from mesofield.config import ExperimentConfig


class DummyHardwareManager:
    def __init__(self, *args, **kwargs):
        self.cameras = ()


def test_led_pattern_is_normalized_in_set(monkeypatch, tmp_path):
    monkeypatch.setattr("mesofield.config.HardwareManager", DummyHardwareManager)

    cfg = ExperimentConfig(path=str(tmp_path))
    cfg.set("led_pattern", "422222442")

    assert cfg.get("led_pattern") == list("422222442")
    assert cfg.led_pattern == list("422222442")


def test_load_json_keeps_led_text_logic_and_non_led_dropdown_choices(monkeypatch, tmp_path):
    monkeypatch.setattr("mesofield.config.HardwareManager", DummyHardwareManager)

    cfg_json = tmp_path / "config.json"
    payload = {
        "Configuration": {
            "duration": 10,
            "task": ["widefield", "blue"],
            "led_pattern": "422",
        },
        "Subjects": {
            "S1": {
                "session": "01",
                "task": "widefield",
            }
        },
        "DisplayKeys": ["duration", "task", "led_pattern"],
    }
    cfg_json.write_text(json.dumps(payload), encoding="utf-8")

    cfg = ExperimentConfig(path=str(tmp_path))
    cfg.load_json(str(cfg_json))

    assert cfg.get_choices("task") == ["widefield", "blue"]
    assert cfg.get("task") == "widefield"

    assert cfg.get_choices("led_pattern") is None
    assert cfg.led_pattern == ["4", "2", "2"]
