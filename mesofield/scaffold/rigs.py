"""Machine-level store of canonical ``hardware.yaml`` rig configurations.

A ``hardware.yaml`` is rig-specific -- it pins COM ports, camera ids, device
indices, and Micro-Manager ``.cfg`` paths to one physical computer. Rather
than hand-copying the right file into every new experiment, each machine keeps
a small store of named canonical configs in its OS-native config directory
(via :mod:`platformdirs`). ``mesofield init`` copies a chosen rig into the new
experiment so the experiment folder stays self-contained.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import platformdirs
import yaml

from mesofield.scaffold.experiment import hardware_yaml_template


def rigs_dir() -> Path:
    """Return (creating if needed) the directory holding canonical rig files."""
    path = Path(platformdirs.user_config_dir("mesofield", appauthor=False)) / "rigs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_rigs() -> list[str]:
    """Return the sorted names of every rig in the store."""
    return sorted(
        p.stem for p in rigs_dir().iterdir()
        if p.suffix in (".yaml", ".yml") and p.is_file()
    )


def rig_path(name: str) -> Path:
    """Return the path a rig named ``name`` resolves to (``.yaml``)."""
    return rigs_dir() / f"{name}.yaml"


def _resolve_existing(name: str) -> Path:
    """Return an existing rig's path, accepting either ``.yaml`` or ``.yml``."""
    yaml_path = rig_path(name)
    if yaml_path.is_file():
        return yaml_path
    yml_path = rigs_dir() / f"{name}.yml"
    if yml_path.is_file():
        return yml_path
    raise FileNotFoundError(
        f"No rig named {name!r}. Known rigs: {', '.join(list_rigs()) or '(none)'}"
    )


def add_rig(name: str, source: Path, *, force: bool = False) -> Path:
    """Copy an existing ``hardware.yaml`` into the store under ``name``.

    The source is parsed with :func:`yaml.safe_load` first so a malformed
    file is rejected before it lands in the store.
    """
    source = Path(source)
    with open(source, "r", encoding="utf-8") as fh:
        yaml.safe_load(fh)  # validate it parses; result intentionally unused

    dst = rig_path(name)
    if dst.exists() and not force:
        raise FileExistsError(
            f"Rig {name!r} already exists at {dst}. Pass force=True to overwrite."
        )
    shutil.copyfile(source, dst)
    return dst


def new_rig(name: str, *, force: bool = False) -> Path:
    """Write a blank fill-out hardware template into the store under ``name``."""
    dst = rig_path(name)
    if dst.exists() and not force:
        raise FileExistsError(
            f"Rig {name!r} already exists at {dst}. Pass force=True to overwrite."
        )
    dst.write_text(hardware_yaml_template(), encoding="utf-8")
    return dst


def remove_rig(name: str) -> None:
    """Delete a rig from the store."""
    _resolve_existing(name).unlink()


# Top-level hardware.yaml keys that configure the rig but are not devices.
_NON_DEVICE_KEYS = frozenset({
    "memory_buffer_size", "viewer_type", "widgets",
    "blue_led_power_mw", "violet_led_power_mw",
})


def rig_devices(name: str) -> list[tuple[str, str]]:
    """Return ``(device_name, device_type)`` pairs declared by rig ``name``.

    Mirrors how :class:`~mesofield.hardware.HardwareManager` reads the YAML:
    every top-level mapping (other than scalar config keys) is a device, and
    a ``cameras:`` list expands to one entry per camera.
    """
    with open(_resolve_existing(name), "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}

    devices: list[tuple[str, str]] = []
    for key, value in doc.items():
        if key in _NON_DEVICE_KEYS:
            continue
        if key == "cameras" and isinstance(value, list):
            for cam in value:
                if not isinstance(cam, dict):
                    continue
                cam_name = cam.get("id") or cam.get("name") or "camera"
                cam_type = cam.get("type") or cam.get("backend") or "camera"
                devices.append((str(cam_name), str(cam_type)))
            continue
        if isinstance(value, dict):
            devices.append((key, str(value.get("type", key))))
    return devices
