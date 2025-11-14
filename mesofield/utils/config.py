"""Simplified configuration helpers for Mesofield."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml

try:  # Python 3.11+
    import tomllib  # type: ignore
except ImportError:  # pragma: no cover - fallback for older Python
    tomllib = None

from mesofield.utils._logger import get_logger

logger = get_logger(__name__)

SUPPORTED_FORMATS = {".json", ".yaml", ".yml"}
if tomllib:
    SUPPORTED_FORMATS.add(".toml")

Schema = Dict[str, Dict[str, Any]]


# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

MOUSEPORTAL_SCHEMA: Schema = {
    "window_width": {"type": int, "default": 800, "min": 100, "max": 4096},
    "window_height": {"type": int, "default": 600, "min": 100, "max": 4096},
    "fullscreen": {"type": bool, "default": False},
    "segment_length": {"type": float, "default": 10.0, "min": 0.1},
    "corridor_width": {"type": float, "default": 4.0, "min": 0.1},
    "wall_height": {"type": float, "default": 3.0, "min": 0.1},
    "num_segments": {"type": int, "default": 10, "min": 2, "max": 50},
    "camera_height": {"type": float, "default": 2.0, "min": 0.1},
    "fog_density": {"type": float, "default": 0.06, "min": 0.0, "max": 1.0},
    "left_wall_texture": {"type": str, "default": ""},
    "right_wall_texture": {"type": str, "default": ""},
    "ceiling_texture": {"type": str, "default": ""},
    "floor_texture": {"type": str, "default": ""},
    "serial_port": {"type": str, "default": "dev"},
    "gain": {"type": float, "default": 1.0, "min": 0.0},
    "input_reversed": {"type": bool, "default": False},
    "speed_scaling": {"type": float, "default": 5.0, "min": 0.1},
    "data_logging_file": {"type": str, "default": "movement.csv"},
    "event_log_file": {"type": str, "default": "events.json"},
}

HARDWARE_SCHEMA: Schema = {
    "backend": {"type": str, "default": "micromanager", "choices": ["micromanager", "opencv"]},
    "viewer_type": {"type": str, "default": "static"},
    "widgets": {"type": list, "default": []},
    "cameras": {"type": list, "default": []},
}

EXPERIMENT_SCHEMA: Schema = {
    "subject": {"type": str, "default": "default", "required": True},
    "session": {"type": str, "default": "001", "required": True},
    "task": {"type": str, "default": "default", "required": True},
    "duration": {"type": int, "default": 60, "min": 1},
    "trial_duration": {"type": int, "default": None, "min": 1},
    "start_on_trigger": {"type": bool, "default": False},
    "psychopy_filename": {"type": str, "default": "experiment.py"},
}

SCHEMAS = {
    "mouseportal": MOUSEPORTAL_SCHEMA,
    "hardware": HARDWARE_SCHEMA,
    "experiment": EXPERIMENT_SCHEMA,
}


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def load_config_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a configuration file, inferring the format from its extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if suffix == ".toml" and tomllib:
        with path.open("rb") as handle:
            return tomllib.load(handle)

    raise ValueError(f"Unsupported config format: {suffix}. Supported: {sorted(SUPPORTED_FORMATS)}")


def save_config_file(config: Dict[str, Any], path: Union[str, Path], format_override: Optional[str] = None) -> None:
    """Persist a configuration dictionary to disk."""
    path = Path(path)
    suffix = (format_override or path.suffix or ".yaml").lower()
    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".json":
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return
    if suffix in {".yaml", ".yml"}:
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, default_flow_style=False, indent=2)
        return
    raise ValueError(f"Unsupported save format: {suffix}")


# ---------------------------------------------------------------------------
# Schema utilities
# ---------------------------------------------------------------------------


def _coerce_value(value: Any, expected_type: type) -> Any:
    if expected_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return bool(value)
    try:
        return expected_type(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - error branch
        raise ValueError(f"Expected {expected_type.__name__}, got {value!r}") from exc


def _apply_schema(raw: Dict[str, Any], schema: Schema, name: str) -> Dict[str, Any]:
    normalised: Dict[str, Any] = {}

    for key, meta in schema.items():
        expected = meta.get("type", str)
        default = meta.get("default")
        required = meta.get("required", False)
        value = raw.get(key, default)

        if value is None:
            if required and default is None:
                raise ValueError(f"Missing required parameter '{key}' in {name} config")
            normalised[key] = value
            continue

        value = _coerce_value(value, expected)

        min_val = meta.get("min")
        max_val = meta.get("max")
        if min_val is not None and value < min_val:
            raise ValueError(f"Parameter '{key}' must be >= {min_val}, got {value}")
        if max_val is not None and value > max_val:
            raise ValueError(f"Parameter '{key}' must be <= {max_val}, got {value}")

        choices = meta.get("choices")
        if choices and value not in choices:
            raise ValueError(f"Parameter '{key}' must be one of {choices}, got {value}")

        normalised[key] = value

    for key, value in raw.items():
        if key not in normalised:
            logger.debug("Unrecognised %s config parameter '%s'", name, key)
            normalised[key] = value

    return normalised


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_mouseportal_config(path: Union[str, Path]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load a MousePortal configuration and split out the optional engine block."""
    raw = load_config_file(path)
    engine = raw.pop("engine", {}) if isinstance(raw, dict) else {}
    runtime = _apply_schema(raw, MOUSEPORTAL_SCHEMA, "mouseportal")
    return runtime, engine


def load_hardware_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load hardware configuration with defaults and light validation."""
    raw = load_config_file(path)
    try:
        return _apply_schema(raw, HARDWARE_SCHEMA, "hardware")
    except ValueError as exc:
        logger.warning("Hardware config validation failed (%s); returning raw values", exc)
        return raw


def load_experiment_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load experiment configuration without imposing Mesofield-specific schema.

    ExperimentConfig is responsible for interpreting the returned mapping.  Here
    we only ensure that the file exists and contains a dictionary so that all
    structural decisions happen in one place.
    """
    raw = load_config_file(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Experiment configuration must be a mapping, got {type(raw).__name__}")
    return dict(raw)


def validate_config_file(path: Union[str, Path], config_type: str = "mouseportal") -> bool:
    """Validate a configuration file and return True on success."""
    schema = SCHEMAS.get(config_type)
    if schema is None:
        raise ValueError(f"Unknown config type: {config_type}")

    try:
        raw = load_config_file(path)
        _apply_schema(raw, schema, config_type)
        return True
    except Exception as exc:  # pragma: no cover - failure path
        logger.error("Validation failed for %s: %s", path, exc)
        return False


def create_config_template(config_type: str, output_path: Union[str, Path]) -> None:
    """Write a template file containing all defaults for the specified type."""
    schema = SCHEMAS.get(config_type)
    if schema is None:
        raise ValueError(f"Unknown config type: {config_type}")

    template = {key: meta.get("default") for key, meta in schema.items()}
    save_config_file(template, output_path)
    logger.info("Generated %s template: %s", config_type, output_path)


# ---------------------------------------------------------------------------
# Lightweight wrappers
# ---------------------------------------------------------------------------


@dataclass
class MousePortalConfig:
    data: Dict[str, Any]

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - simple delegation
        try:
            return self.data[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.data)


@dataclass
class HardwareConfig:
    data: Dict[str, Any]

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - simple delegation
        try:
            return self.data[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.data)
