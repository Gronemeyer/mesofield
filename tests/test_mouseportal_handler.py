from __future__ import annotations

import json
from pathlib import Path
from pathlib import Path
import sys
import types
from typing import Any, cast

import pytest

from mesofield.config import ExperimentConfig
from mesofield.subprocesses.mouseportal import MousePortal


class DummyHardwareManager:
    def __init__(self, path: str) -> None:  # pragma: no cover - simple stub
        self.devices: dict[str, Any] = {}
        self.cameras = ()
        self.yaml = {}


def _make_dummy_config(tmp_path: Path) -> types.SimpleNamespace:
    plugin_cfg = {
        "env_path": sys.executable,
        "script_path": str(tmp_path / "runportal.py"),
        "socket_host": "127.0.0.1",
        "socket_port": 9999,
        "output_suffix": "mouseportal",
        "output_extension": "jsonl",
        "bids_type": "beh",
    }
    cfg = types.SimpleNamespace()
    cfg.plugins = {"mouseportal": {"config": plugin_cfg}}
    cfg.save_dir = str(tmp_path)
    cfg.hardware = types.SimpleNamespace(devices={})
    return cfg


def test_mouseportal_emits_messages_and_registers(tmp_path: Path) -> None:
    cfg = cast(ExperimentConfig, _make_dummy_config(tmp_path))
    portal = MousePortal(cfg, launch_process=False)

    captured: list[tuple[dict[str, Any], Any]] = []
    portal.data_event.connect(lambda payload, device_ts: captured.append((payload, device_ts)))

    message = {"type": "status", "time": 1.23, "position": 2.0, "velocity": 3.5}
    portal._handle_socket_payload(message)

    assert captured, "expected data_event to emit a payload"
    assert captured[0][0]["type"] == "status"
    assert captured[0][1] == pytest.approx(1.23)

    drained = portal.drain_messages()
    assert drained and drained[0]["message"]["type"] == "status"

    assert cfg.hardware.devices[portal.device_id] is portal


def test_mouseportal_save_data(tmp_path: Path) -> None:
    cfg = cast(ExperimentConfig, _make_dummy_config(tmp_path))
    portal = MousePortal(cfg, launch_process=False)

    portal._handle_socket_payload({"type": "status", "time": 1.0})
    portal._handle_socket_payload({"type": "event", "time_received": 2.0, "name": "test"})

    output_path = tmp_path / "mouseportal.jsonl"
    portal.save_data(str(output_path))

    assert output_path.exists()
    with output_path.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 2
    assert lines[0]["message"]["type"] == "status"


def test_experiment_config_registers_plugins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "hardware.yaml"
    config_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr("mesofield.config.HardwareManager", DummyHardwareManager)

    cfg = ExperimentConfig(str(config_path))
    cfg.load_json(str(Path(__file__).with_name("devcfg.json")))

    assert cfg.get("plugins") == cfg.plugins.settings
    assert "mouseportal" in cfg.plugins.settings
    assert cfg.plugins.is_enabled("mouseportal")


def test_mouseportal_expands_asset_dir_paths(tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    cfg_ns = _make_dummy_config(tmp_path)
    plugins = cfg_ns.plugins
    if hasattr(plugins, "get_settings"):
        plugin_settings = plugins.get_settings("mouseportal") or {}
        plugin_cfg = plugin_settings.setdefault("config", {})
    else:
        plugin_cfg = plugins["mouseportal"]["config"]
    plugin_cfg.update(
        {
            "asset_dir": str(assets_dir),
            "left_wall_texture": "left.png",
            "right_wall_texture": "right.png",
            "ceiling_texture": "ceiling.png",
            "floor_texture": "floor.png",
        }
    )

    portal = MousePortal(cast(ExperimentConfig, cfg_ns), launch_process=False)

    expected_asset_dir = Path(assets_dir).resolve()
    assert portal.cfg["asset_dir"] == expected_asset_dir.as_posix()
    assert portal.cfg["left_wall_texture"] == (expected_asset_dir / "left.png").as_posix()
    assert portal.cfg["right_wall_texture"] == (expected_asset_dir / "right.png").as_posix()
    assert portal.cfg["ceiling_texture"] == (expected_asset_dir / "ceiling.png").as_posix()
    assert portal.cfg["floor_texture"] == (expected_asset_dir / "floor.png").as_posix()