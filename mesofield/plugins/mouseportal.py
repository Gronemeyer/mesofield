"""MousePortal plugin that launches the virtual corridor and captures events."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import zmq
import yaml

from mesofield.plugins import Plugin, register_plugin
from mesofield.protocols import ProcedurePlugin
from mesofield.subprocesses.mouseportal import MousePortal
from mesofield.utils._logger import get_logger

logger = get_logger("MousePortalPlugin")


@register_plugin("mouse_portal")
class MousePortalPlugin(Plugin, ProcedurePlugin):
    """Manage the MousePortal subprocess and pipe events into Mesofield."""

    def __init__(self, *, name: str, config: Optional[dict[str, Any]] = None, metadata: Optional[dict[str, Any]] = None) -> None:
        super().__init__(name=name, config=config, metadata=metadata)
        self._ctx = zmq.Context.instance()
        self._controller: Optional[MousePortal] = None
        self._data_manager = None
        self._experiment_config = None
        self._device_id = self.config.get("device_id", "mouse_portal")
        self._event_topic = self.config.get("topic", "exp-event")
        self._dev_mode = bool(self.config.get("dev_mode", False))
        self._ready_attempts = int(self.config.get("ready_attempts", 12))
        self._ping_timeout = float(self.config.get("ping_timeout_s", 1.0))
        self._normalized_spec: Optional[Dict[str, Any]] = None
        self._runtime_spec_path: Optional[Path] = None
        self._movement_path: Optional[str] = None
        self._event_path: Optional[str] = None
        self._running = False
        self._pending_payload: Optional[Dict[str, Any]] = None

    # lifecycle -----------------------------------------------------
    def attach(self, *, experiment_config: Any, data_manager: Any) -> None:
        super().attach(experiment_config=experiment_config, data_manager=data_manager)
        self._experiment_config = experiment_config
        self._data_manager = data_manager

        unified_config_path = self.config.get("unified_config_path")
        preloaded_unified: Optional[Dict[str, Any]] = None
        if unified_config_path:
            source = Path(unified_config_path)
            if source.exists():
                preloaded_unified = self._load_mapping(source)
                self._apply_plugin_settings(dict(preloaded_unified.get("plugin") or {}))
            else:
                logger.error("MousePortal unified config not found: %s", source)

        cfg_override = self.config.get("config_path") if not unified_config_path else None
        env_path = self.config.get("python_executable") or self.config.get("env_path")
        script_path = self.config.get("script_path")

        self._controller = MousePortal(
            experiment_config,
            cfg_path=cfg_override,
            python_executable=env_path,
            script_path=script_path,
            plugin_options=self.config,
            context=self._ctx,
        )

        payload = self._prepare_payload(unified_config_path, preloaded=preloaded_unified)
        self._pending_payload = payload
        logger.info(
            "MousePortal configured (dev_mode=%s, unified=%s, legacy_cfg=%s)",
            self._dev_mode,
            bool(unified_config_path),
            bool(cfg_override),
        )

    def start(self) -> None:
        super().start()

    def stop(self) -> None:
        try:
            self.end_experiment()
        finally:
            self._cleanup_artifacts()
            super().stop()

    # public API ----------------------------------------------------
    def begin_experiment(self, *, dev_mode: Optional[bool] = None) -> None:
        controller = self._require_controller()
        if self._running:
            logger.debug("MousePortal experiment already running")
            return

        extra_args: list[str] = []
        mode = self._dev_mode if dev_mode is None else bool(dev_mode)
        if mode:
            extra_args.append("--dev")

        payload = self._pending_payload.copy() if self._pending_payload else {}

        manager = self._data_manager
        if manager is None:
            raise RuntimeError("DataManager not available for mouse portal plugin")

        self._movement_path = manager.allocate_output_path(
            "mouse_portal_movement",
            suffix="mouseportal_movement",
            extension="csv",
            bids_type="beh",
        )
        self._event_path = manager.allocate_output_path(
            "mouse_portal_events",
            suffix="mouseportal_events",
            extension="json",
            bids_type="beh",
        )
        payload["data_logging_file"] = self._movement_path
        payload["event_log_file"] = self._event_path
        self._apply_duration_hint()
        controller.set_cfg(payload)

        controller.launch(extra_args=extra_args or None)
        try:
            controller.start_event_stream(self._handle_event)
            self._wait_for_ready()
            controller.start()
        except Exception:
            controller.stop_event_stream()
            controller.terminate()
            raise
        self._running = True
        logger.info("MousePortal experiment started")

    def end_experiment(self) -> None:
        controller = self._controller
        if controller is None:
            return
        if self._running:
            try:
                controller.stop(timeout=2.0)
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("MousePortal stop command failed: %s", exc)
        controller.stop_event_stream()
        controller.terminate()
        self._running = False

    # helpers -------------------------------------------------------
    def _require_controller(self) -> MousePortal:
        if self._controller is None:
            raise RuntimeError("MousePortal controller not initialised")
        return self._controller

    def _apply_plugin_settings(self, overrides: Optional[Dict[str, Any]]) -> None:
        if not overrides:
            return
        sanitized = {k: v for k, v in overrides.items() if v is not None}
        if not sanitized:
            return
        self.config.update(sanitized)
        if "device_id" in sanitized:
            self._device_id = sanitized["device_id"]
        if "topic" in sanitized:
            self._event_topic = sanitized["topic"]
        if "ready_attempts" in sanitized:
            self._ready_attempts = int(sanitized["ready_attempts"])
        if "ping_timeout_s" in sanitized:
            self._ping_timeout = float(sanitized["ping_timeout_s"])
        if "dev_mode" in sanitized:
            self._dev_mode = bool(sanitized["dev_mode"])

    def _prepare_payload(
        self, unified_config_path: Optional[str], *, preloaded: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if unified_config_path:
            return self._load_unified_config(unified_config_path, preloaded=preloaded)

        payload = self._load_controller_config()
        if not payload:
            payload = {}
        self._materialize_spec(payload)
        return payload

    def _load_controller_config(self) -> Dict[str, Any]:
        controller = self._require_controller()
        if controller.cfg_path is None:
            logger.warning("MousePortal config_path not provided; using defaults")
            return {}
        try:
            config = controller.load_cfg()
            # Auto-detect dev mode from serial_port setting
            if config.get("serial_port") == "dev":
                self._dev_mode = True
                logger.info("Auto-enabled dev mode based on serial_port='dev'")
            return config
        except FileNotFoundError:
            logger.warning("MousePortal configuration file not found; starting from defaults")
            return {}

    def _load_unified_config(
        self, config_path: str, *, preloaded: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Load unified configuration that contains both rendering config and experiment sequence."""
        source = Path(config_path)
        if preloaded is None and not source.exists():
            logger.error("MousePortal unified config not found: %s", source)
            return {}

        # Load the unified YAML file
        unified_config = dict(preloaded) if preloaded is not None else self._load_mapping(source)

        # Remove helper sections that should not be forwarded to the controller
        plugin_overrides = unified_config.pop("plugin", None)
        metadata = unified_config.pop("metadata", None)
        unified_config.pop("visual_diagram", None)
        if plugin_overrides and preloaded is None:
            self._apply_plugin_settings(dict(plugin_overrides))
        if metadata:
            logger.debug("Loaded MousePortal metadata: %s", metadata)

        # Extract engine configuration for experiment sequence
        engine_config = unified_config.get("engine", {})
        if engine_config:
            # Normalize and materialize the experiment spec
            normalized = self._normalize_spec(engine_config)
            self._normalized_spec = normalized
            
            # Create runtime spec file
            controller = self._require_controller()
            runtime_dir = controller.runtime_path.parent
            runtime_dir.mkdir(parents=True, exist_ok=True)
            runtime_spec = runtime_dir / f"{controller.runtime_path.stem}-spec.json"
            with runtime_spec.open("w", encoding="utf-8") as fh:
                json.dump(normalized, fh, indent=2)
            self._runtime_spec_path = runtime_spec
            
            # Update engine config in payload
            unified_config["engine"] = {"spec_path": str(runtime_spec)}
            controller._sequence_path = runtime_spec
        
        # Auto-detect dev mode from serial_port setting
        if unified_config.get("serial_port") == "dev":
            self._dev_mode = True
            logger.info("Auto-enabled dev mode based on serial_port='dev'")
            
        return unified_config

    def _materialize_spec(self, payload: Dict[str, Any]) -> None:
        sequence_path = self.config.get("sequence_yaml") or self.config.get("sequence_path")
        if not sequence_path:
            return
        source = Path(sequence_path)
        if not source.exists():
            logger.error("MousePortal sequence spec not found: %s", source)
            return
        spec_dict = self._load_mapping(source)
        normalized = self._normalize_spec(spec_dict)
        self._normalized_spec = normalized

        controller = self._require_controller()
        runtime_dir = controller.runtime_path.parent
        runtime_dir.mkdir(parents=True, exist_ok=True)
        runtime_spec = runtime_dir / f"{controller.runtime_path.stem}-spec.json"
        with runtime_spec.open("w", encoding="utf-8") as fh:
            json.dump(normalized, fh, indent=2)
        self._runtime_spec_path = runtime_spec

        engine_cfg = payload.setdefault("engine", {})
        engine_cfg["spec_path"] = str(runtime_spec)
        controller._sequence_path = runtime_spec  # keep runtime override persistent

    def _load_mapping(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as fh:
            if path.suffix.lower() in {".yaml", ".yml"}:
                data = yaml.safe_load(fh) or {}
            else:
                data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("MousePortal experiment spec must be a mapping")
        return data

    def _normalize_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        blocks: list[Dict[str, Any]] = []
        for block in spec.get("blocks", []):
            trials: list[Dict[str, Any]] = []
            for trial in block.get("trials", []):
                routine = trial.get("routine")
                cfg = dict(trial.get("config", {}))
                if isinstance(routine, dict):
                    cfg = {**cfg, **{k: v for k, v in routine.items() if k not in {"type", "name"}}}
                    routine_name = routine.get("type") or routine.get("name")
                else:
                    routine_name = routine
                if routine_name is None:
                    raise ValueError("Each trial in MousePortal spec must declare a routine")
                trials.append(
                    {
                        "name": trial.get("name", routine_name),
                        "routine": routine_name,
                        "config": cfg,
                    }
                )
            blocks.append(
                {
                    "name": block.get("name", "block"),
                    "policy": block.get("policy", "sequential"),
                    "repeats": int(block.get("repeats", 1)),
                    "trials": trials,
                }
            )
        return {"rng_seed": int(spec.get("rng_seed", 0)), "blocks": blocks}

    def _estimate_total_duration(self) -> float:
        if not self._normalized_spec:
            return 0.0
        total = 0.0
        for block in self._normalized_spec.get("blocks", []):
            block_total = 0.0
            for trial in block.get("trials", []):
                cfg = trial.get("config", {}) or {}
                duration = self._resolve_duration(cfg)
                block_total += duration
            total += block_total * int(block.get("repeats", 1))
        return total

    def _resolve_duration(self, cfg: Dict[str, Any]) -> float:
        for key in ("duration_max_s", "duration_s", "duration", "duration_max"):
            if key in cfg:
                value = cfg[key]
                maybe = self._coerce_duration(value)
                if maybe is not None:
                    return maybe
        if "duration_min_s" in cfg:
            maybe = self._coerce_duration(cfg["duration_min_s"])
            if maybe is not None:
                return maybe
        return 0.0

    @staticmethod
    def _coerce_duration(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    def _apply_duration_hint(self) -> None:
        if self._experiment_config is None:
            return
        total = self._estimate_total_duration()
        if total <= 0:
            return
        try:
            self._experiment_config.set("mouse_portal_estimated_duration_s", total)
        except Exception:  # pragma: no cover - registry might reject new key types
            pass
        try:
            current = self._experiment_config.get("duration", 0) or 0
            current_val = float(current)
        except (TypeError, ValueError):
            current_val = 0.0
        if total > current_val:
            self._experiment_config.set("duration", int(math.ceil(total)))

    def _wait_for_ready(self) -> None:
        controller = self._require_controller()
        for attempt in range(self._ready_attempts):
            try:
                reply = controller.send_command("ping", timeout=self._ping_timeout)
            except TimeoutError:
                time.sleep(0.5)
                continue
            if isinstance(reply, dict):
                return
        raise RuntimeError("MousePortal command server did not respond in time")

    def _handle_event(self, payload: Dict[str, Any]) -> None:
        if self._data_manager is None:
            return
        device_ts = payload.get("time_abs")
        self._data_manager.queue.push(
            self._device_id,
            payload,
            device_ts=device_ts,
            topic=self._event_topic,
        )

    def _cleanup_artifacts(self) -> None:
        if self._runtime_spec_path and self._runtime_spec_path.exists():
            try:
                self._runtime_spec_path.unlink()
            except OSError:  # pragma: no cover - best effort cleanup
                pass
        self._runtime_spec_path = None


__all__ = ["MousePortalPlugin"]
