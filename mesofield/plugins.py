from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from mesofield.config import ExperimentConfig

from mesofield.subprocesses.mouseportal import (
    MousePortal,
    drive_mouseportal_trials,
    ensure_mouseportal,
    shutdown_mouseportal,
)


@dataclass
class PluginHook:
    sync: Callable[["ExperimentConfig", Optional[Any], Optional[Any]], Optional[Any]]
    start: Optional[Callable[["ExperimentConfig", Any], None]] = None
    drive: Optional[Callable[["ExperimentConfig", Any, Iterable[Any]], None]] = None
    shutdown: Optional[Callable[["ExperimentConfig", Any], None]] = None


class PluginManager:
    """Manage lifecycle hooks for optional experiment plugins."""

    def __init__(self, config: "ExperimentConfig") -> None:
        self._config = config
        self._hooks: Dict[str, PluginHook] = {}
        self._controllers: Dict[str, Any] = {}
        self._context: Dict[str, Any] = {}
        self._settings: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Plugin configuration helpers
    # ------------------------------------------------------------------
    @property
    def settings(self) -> Dict[str, Dict[str, Any]]:
        return self._settings

    def clear(self) -> None:
        self._settings.clear()
        self.shutdown()

    def load_settings(self, raw: Any) -> Dict[str, Dict[str, Any]]:
        self._settings = self.normalize_plugins(raw)
        return self._settings

    def normalize_plugins(self, raw: Any) -> Dict[str, Dict[str, Any]]:
        if not isinstance(raw, dict):
            return {}
        normalized: Dict[str, Dict[str, Any]] = {}
        for name, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            cfg = entry.get("config") if isinstance(entry.get("config"), dict) else {}
            normalized[name] = {
                "enabled": bool(entry.get("enabled")),
                "config": cfg,
            }
        return normalized

    def enabled_plugins(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: entry
            for name, entry in self._settings.items()
            if isinstance(entry, dict) and entry.get("enabled")
        }

    def is_enabled(self, name: str) -> bool:
        entry = self._settings.get(name)
        return bool(isinstance(entry, dict) and entry.get("enabled"))

    def get_settings(self, name: str) -> Optional[Dict[str, Any]]:
        entry = self._settings.get(name)
        return entry if isinstance(entry, dict) else None

    def iter_enabled_plugin_configs(
        self,
        plan_payload: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        for name, entry in self.enabled_plugins().items():
            yield name, self.build_plugin_configuration(entry, plan_payload)

    def build_plugin_configuration(
        self,
        plugin_entry: Dict[str, Any],
        plan_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        config_payload = copy.deepcopy(plugin_entry.get("config", {}))
        if plan_payload:
            config_payload.setdefault("experiment", plan_payload.get("definition"))
            config_payload["compiled_plan"] = copy.deepcopy(plan_payload)
        return config_payload

    def update_plugin_entry(self, name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self.normalize_plugins({name: payload})
        entry = normalized.get(name, {"enabled": False, "config": {}})
        self._settings[name] = entry
        return entry

    # ------------------------------------------------------------------
    # Built-in plugin hooks
    # ------------------------------------------------------------------

    def register_mouseportal_hooks(self) -> None:
        self.register(
            "mouseportal",
            PluginHook(
                sync=self._mouseportal_sync,
                start=self._mouseportal_start,
                drive=self._mouseportal_drive,
                shutdown=self._mouseportal_shutdown,
            ),
        )

    def get_plugin_definition(self) -> Optional[Dict[str, Any]]:
        for entry in self.enabled_plugins().values():
            config_payload = entry.get("config")
            if isinstance(config_payload, dict):
                definition = config_payload.get("experiment")
                if isinstance(definition, dict):
                    return copy.deepcopy(definition)
        return None

    def get_plugin_payload(
        self,
        name: str,
        plan_payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        entry = self.enabled_plugins().get(name)
        if not entry:
            return None
        return self.build_plugin_configuration(entry, plan_payload or None)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, name: str, hook: PluginHook) -> None:
        self._hooks[name] = hook

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def refresh(self, *, data_manager: Optional[Any] = None) -> Dict[str, Any]:
        if data_manager is not None:
            self._context["data_manager"] = data_manager
        cached_manager = self._context.get("data_manager")

        previous = dict(self._controllers)
        controllers: Dict[str, Any] = {}
        for name, hook in self._hooks.items():
            if not self.is_enabled(name):
                continue
            controller = None
            existing = previous.get(name)
            if hook.sync is not None:
                controller = hook.sync(self._config, existing, cached_manager)
            if controller is not None:
                controllers[name] = controller

        stale_names = set(previous) - set(controllers)
        for name in stale_names:
            self._shutdown_controller(name, previous[name])

        self._controllers = controllers
        return controllers

    def start(self) -> None:
        for name, controller in self._controllers.items():
            hook = self._hooks.get(name)
            if not hook or hook.start is None:
                continue
            try:
                hook.start(self._config, controller)
            except Exception as exc:  # pragma: no cover - defensive logging
                self._log_warning("%s plugin failed to start: %s", name, exc)

    def drive(self, trials: Optional[Iterable[Any]] = None) -> None:
        trial_sequence = trials if trials is not None else getattr(
            self._config, "experiment_trials", []
        )
        for name, controller in self._controllers.items():
            hook = self._hooks.get(name)
            if not hook or hook.drive is None:
                continue
            try:
                hook.drive(self._config, controller, trial_sequence)
            except Exception as exc:  # pragma: no cover - defensive logging
                self._log_warning("%s plugin run step failed: %s", name, exc)

    def shutdown(self) -> None:
        for name, controller in list(self._controllers.items()):
            self._shutdown_controller(name, controller)
        self._controllers.clear()

    def ensure(self, name: str, *, data_manager: Optional[Any] = None) -> Optional[Any]:
        controllers = self.refresh(data_manager=data_manager)
        return controllers.get(name)

    def get(self, name: str) -> Optional[Any]:
        return self._controllers.get(name)

    def shutdown_one(self, name: str) -> None:
        controller = self._controllers.pop(name, None)
        if controller is not None:
            self._shutdown_controller(name, controller)

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _shutdown_controller(self, name: str, controller: Any) -> None:
        if controller is None:
            return
        hook = self._hooks.get(name)
        if hook and hook.shutdown is not None:
            try:
                hook.shutdown(self._config, controller)
                return
            except Exception as exc:  # pragma: no cover - defensive logging
                self._log_warning("%s plugin shutdown failed: %s", name, exc)
                return
        shutdown_fn = getattr(controller, "shutdown", None)
        if callable(shutdown_fn):
            try:
                shutdown_fn()
            except Exception as exc:  # pragma: no cover - defensive logging
                self._log_warning("%s plugin shutdown failed: %s", name, exc)

    def _log_warning(self, message: str, *args: Any) -> None:
        logger = getattr(self._config, "logger", None)
        if logger is not None:
            logger.warning(message, *args)

    # ------------------------------------------------------------------
    # MousePortal helpers
    # ------------------------------------------------------------------
    def _mouseportal_sync(
        self,
        config: "ExperimentConfig",
        existing: Optional[Any],
        data_manager: Optional[Any],
    ) -> Optional[MousePortal]:
        logger = getattr(config, "logger", None)
        portal = ensure_mouseportal(
            config,
            data_manager=data_manager,
            portal=existing if isinstance(existing, MousePortal) else None,
            logger=logger,
        )
        return portal

    def _mouseportal_start(
        self,
        config: "ExperimentConfig",
        controller: Any,
    ) -> None:
        if isinstance(controller, MousePortal):
            controller.start()

    def _mouseportal_drive(
        self,
        config: "ExperimentConfig",
        controller: Any,
        trials: Iterable[Any],
    ) -> None:
        if isinstance(controller, MousePortal):
            logger = getattr(config, "logger", None)
            plan_payload = getattr(config, "experiment_plan_payload", None)

            definition = None
            plan_id = None
            if isinstance(plan_payload, dict):
                definition = plan_payload.get("definition")
                if plan_payload.get("plan_id") is not None:
                    plan_id = str(plan_payload["plan_id"])

            if isinstance(definition, dict):
                plan_definition = copy.deepcopy(definition)

                def _extract_bool(source: Mapping[str, Any], key: str) -> Optional[bool]:
                    value = source.get(key)
                    return bool(value) if isinstance(value, bool) else None

                def _extract_float(source: Mapping[str, Any], key: str) -> Optional[float]:
                    value = source.get(key)
                    if isinstance(value, (int, float)):
                        return float(value)
                    return None

                default_mode = plan_definition.get("default_mode")
                if isinstance(default_mode, str) and default_mode.strip():
                    default_mode = default_mode.strip()
                else:
                    default_mode = None

                auto_start = _extract_bool(plan_definition, "auto_start")
                auto_advance = _extract_bool(plan_definition, "auto_advance")
                inter_trial_interval = _extract_float(plan_definition, "inter_trial_interval")

                try:
                    controller.load_experiment_plan(
                        plan_definition,
                        plan_id=plan_id,
                        default_mode=default_mode,
                        auto_start=auto_start,
                        auto_advance=auto_advance,
                        inter_trial_interval=inter_trial_interval,
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    if logger is not None:
                        logger.warning("MousePortal failed to load experiment plan: %s", exc)
                else:
                    try:
                        controller.run_experiment(plan_id=plan_id, restart=True)
                        if logger is not None:
                            logger.info(
                                "MousePortal experiment plan started%s",
                                f" (plan_id={plan_id})" if plan_id else "",
                            )
                    except Exception as exc:  # pragma: no cover - defensive logging
                        if logger is not None:
                            logger.warning("MousePortal failed to start experiment plan: %s", exc)
                return

            # Fallback to legacy per-trial driving when no plan definition is available
            drive_mouseportal_trials(controller, trials, logger=logger)

    def _mouseportal_shutdown(
        self,
        config: "ExperimentConfig",
        controller: Any,
    ) -> None:
        if isinstance(controller, MousePortal):
            logger = getattr(config, "logger", None)
            shutdown_mouseportal(controller, logger=logger)
