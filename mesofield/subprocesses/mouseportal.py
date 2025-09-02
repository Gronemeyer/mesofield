from __future__ import annotations

"""MousePortal subprocess controller.

This module provides a small helper class that launches the external
``runportal.py`` script in a separate Python environment using
``QProcess``.  Configuration parameters are pulled from the
``ExperimentConfig`` plugin entry ``plugins['mouseportal']['config']``.

The external script is expected to accept a ``--cfg`` argument pointing
to a JSON file containing the runtime configuration.  Any stdout
produced by the subprocess (as well as commands sent to it) are pushed
into the :class:`~mesofield.data.manager.DataManager` queue so they can
be logged alongside other experiment data.

The interface mirrors the behaviour of the ``parent_test.py`` example
provided by the user: configuration management, start/stop helpers and a
few convenience methods for common commands.
"""

import json
import os
import sys
import time
from typing import Callable, Optional, TYPE_CHECKING

from PyQt6.QtCore import QProcess

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from mesofield.config import ExperimentConfig
    from mesofield.data.manager import DataManager


class MousePortal:
    """Controller for the MousePortal ``runportal.py`` subprocess."""

    def __init__(
        self,
        config: ExperimentConfig,
        data_manager: DataManager | None = None,
        parent=None,
    ) -> None:
        self._config = config
        self._data = data_manager
        self._parent = parent

        # ------------------------------------------------------------------
        # Extract plugin configuration
        # ------------------------------------------------------------------
        plugin_cfg = (
            getattr(config, "plugins", {}).get("mouseportal", {}).get("config", {})
        )

        # Paths for the python interpreter and the external script
        self.python_executable = plugin_cfg.get("env_path", sys.executable)
        self.script = plugin_cfg.get("script_path", "runportal.py")

        # Remaining parameters are written to a runtime JSON file
        self.cfg = {
            k: v for k, v in plugin_cfg.items() if k not in {"env_path", "script_path"}
        }

        # Location of runtime configuration JSON
        base_dir = getattr(config, "save_dir", os.getcwd())
        self.runtime_path = os.path.join(base_dir, "mouseportal_runtime.json")

        # ------------------------------------------------------------------
        # QProcess setup
        # ------------------------------------------------------------------
        self.process = QProcess(self._parent)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_ready_read)
        self.process.finished.connect(self._on_finished)

        # Optional callbacks for GUI use
        self.output_callback: Optional[Callable[[str], None]] = None
        self.finished_callback: Optional[Callable[[int, QProcess.ExitStatus], None]] = None

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def set_cfg(self, cfg: dict) -> None:
        """Replace the current runtime configuration dictionary."""
        self.cfg = cfg

    def save_runtime(self, cfg: Optional[dict] = None) -> None:
        """Write configuration to ``self.runtime_path``."""
        cfg_to_write = cfg if cfg is not None else self.cfg
        os.makedirs(os.path.dirname(self.runtime_path), exist_ok=True)
        with open(self.runtime_path, "w", encoding="utf-8") as f:
            json.dump(cfg_to_write, f, indent=2)

    def remove_runtime(self) -> None:
        try:
            if os.path.isfile(self.runtime_path):
                os.remove(self.runtime_path)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Process control
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Launch the external MousePortal process."""
        if self.is_running:
            return
        self.save_runtime()
        args = [self.script, "--cfg", self.runtime_path]
        self.process.start(self.python_executable, args)

    def end(self, wait_ms: int = 3000) -> None:
        """Politely ask the subprocess to exit and wait for completion."""
        if not self.is_running:
            self.remove_runtime()
            return
        self.send_cmd("end")
        self.process.waitForFinished(wait_ms)
        self.remove_runtime()

    def terminate(self, wait_ms: int = 3000) -> None:
        """Forcefully terminate the subprocess."""
        if self.is_running:
            self.process.terminate()
            self.process.waitForFinished(wait_ms)
        self.remove_runtime()

    @property
    def is_running(self) -> bool:
        return self.process.state() == QProcess.ProcessState.Running

    # ------------------------------------------------------------------
    # Command helpers
    # ------------------------------------------------------------------
    def send_cmd(self, cmd: str) -> None:
        """Send a command string to the subprocess via ``stdin``."""
        if not self.is_running:
            return
        ts = time.time()
        payload = f"{cmd} {ts}\n".encode()
        self.process.write(payload)
        if self._data is not None:
            # Log command to DataManager queue
            self._data.queue.push("mouseportal", {"cmd": cmd}, device_ts=ts)

    def start_trial(self) -> None:
        self.send_cmd("start_trial")

    def stop_trial(self) -> None:
        self.send_cmd("stop_trial")

    def mark_event(self, label: str = "button") -> None:
        self.send_cmd(f"mark_event {label}")

    # ------------------------------------------------------------------
    # I/O and callbacks
    # ------------------------------------------------------------------
    def set_output_callback(self, cb: Callable[[str], None]) -> None:
        self.output_callback = cb

    def set_finished_callback(
        self, cb: Callable[[int, QProcess.ExitStatus], None]
    ) -> None:
        self.finished_callback = cb

    def _on_ready_read(self) -> None:
        raw = self.process.readAllStandardOutput().data().decode(errors="replace")
        if not raw:
            return
        for line in raw.splitlines():
            if self.output_callback:
                self.output_callback(line)
            if self._data is not None:
                self._data.queue.push("mouseportal", line)

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        if self.finished_callback:
            self.finished_callback(exit_code, exit_status)
        self.remove_runtime()
        if self._data is not None:
            self._data.queue.push(
                "mouseportal", {"finished": exit_code, "status": exit_status.value}
            )


__all__ = ["MousePortal"]

