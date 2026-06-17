"""Base class for stimulus devices that launch an external app subprocess.

Captures the lifecycle shared by every "launch an external stimulus program,
wait for a stdout readiness handshake, terminate on stop" device (MousePortal
today; PsychoPy is a candidate to adopt this later). Subclasses implement a few
hooks; the base wires the mesofield device contract
(:class:`~mesofield.protocols.StimulusDevice`): ``DeviceSignals``,
``initialize/arm/start/stop/shutdown``, ``status``/``metadata``, and a
GUI-safe readiness wait.

A stimulus device is *not* a :class:`~mesofield.protocols.DataProducer`: it
never emits on ``signals.data``. ``signals.started`` fires on the readiness
handshake; ``signals.finished`` fires on ``stop`` or subprocess exit.

Required subclass surface
-------------------------
- ``ready_token`` (class attr) -- stdout substring the child prints when ready.
- :meth:`build_command` -- the argv to launch.

Optional hooks
--------------
- ``launch_phase`` (class attr, ``"arm"`` or ``"start"``) -- when to launch.
  ``"arm"`` guarantees the stimulus is up *before* recording starts, since a
  Procedure runs all ``arm_all`` before ``start_all``.
- :meth:`prepare` -- per-run prep (generate config, open side channels) before launch.
- :meth:`preflight` -- return an error string to abort launch with a clear message.
- :meth:`launch_cwd` / :meth:`launch_env` -- subprocess working dir / environment.
- :meth:`on_stop` -- teardown of any side channels opened in :meth:`prepare`.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional

from mesofield.signals import DeviceSignals
from mesofield.utils._logger import get_logger
from mesofield.devices.subprocesses.base import SubprocessSupervisor


class SubprocessStimulusDevice:
    """Lifecycle skeleton for subprocess-backed stimulus devices."""

    device_type: ClassVar[str] = "stimulus"
    file_type: str = ""
    bids_type: Optional[str] = None

    # Subclass contract / tunables.
    ready_token: ClassVar[str] = "READY"
    launch_phase: ClassVar[str] = "start"   # "arm" | "start"
    default_device_id: ClassVar[str] = "stimulus"

    def __init__(self, cfg: Dict[str, Any]):
        from psygnal import SignalInstance

        self.signals = DeviceSignals()
        self.cfg = dict(cfg)
        self.device_id: str = cfg.get("id", self.default_device_id)
        self.is_primary: bool = bool(cfg.get("primary", False))
        self.logger = get_logger(
            f"{type(self).__module__}.{type(self).__name__}[{self.device_id}]"
        )

        # Common launch parameters.
        self.app_dir: Optional[str] = cfg.get("app_dir")
        self.python_exe: Optional[str] = cfg.get("python_exe")
        self.ready_timeout: float = float(cfg.get("ready_timeout", 30.0))

        # GUI-facing lifecycle status: "loaded" -> "launching" -> "ready"
        # (or "running"/"failed") -> "stopped". ``status_changed`` lets a GUI
        # panel reflect the subprocess state instead of failing silently.
        self._gui_status: str = "loaded"
        self.status_changed: SignalInstance = SignalInstance((str,))

        # Per-run state.
        self._config: Any = None
        self._process: Optional[SubprocessSupervisor] = None
        self._started: Optional[datetime] = None
        self._stopped: Optional[datetime] = None

    # -- GUI status -----------------------------------------------------
    @property
    def gui_status(self) -> str:
        return self._gui_status

    def _set_status(self, status: str) -> None:
        self._gui_status = status
        try:
            self.status_changed.emit(status)
        except Exception as exc:
            self.logger.debug(f"status_changed emit failed: {exc}")

    # -- subclass hooks (override as needed) ----------------------------
    def prepare(self, config: Any) -> None:
        """Per-run prep before launch (config files, side channels). No-op."""
        return None

    def preflight(self) -> Optional[str]:
        """Return an actionable error string to abort launch, or None to proceed."""
        return None

    def build_command(self) -> List[str]:
        """Return the full argv used to launch the stimulus subprocess."""
        raise NotImplementedError

    def launch_cwd(self) -> Optional[str]:
        return self.app_dir

    def launch_env(self) -> Optional[Dict[str, str]]:
        return None

    def on_stop(self) -> None:
        """Teardown for anything opened in :meth:`prepare`. No-op by default."""
        return None

    # -- lifecycle ------------------------------------------------------
    def initialize(self) -> bool:
        return True

    def arm(self, config: Any) -> None:
        self._config = config
        self.prepare(config)
        if self.launch_phase == "arm":
            self._launch_and_wait()

    def start(self) -> bool:
        if self.launch_phase == "arm":
            # Launched during arm; just confirm it is alive.
            if self._process is None or not self._process.is_running():
                self.logger.warning(f"{self.device_id} is not running at start().")
                return False
            self._started = self._started or datetime.now()
            return True
        return self._launch_and_wait()

    def stop(self) -> bool:
        self._stopped = datetime.now()
        try:
            self.on_stop()
        except Exception as exc:
            self.logger.debug(f"on_stop failed: {exc}")
        if self._process is not None:
            self._process.terminate()
        self._set_status("stopped")
        self.signals.finished.emit()
        return True

    def shutdown(self) -> None:
        self.stop()
        self._process = None

    # -- launch helpers -------------------------------------------------
    def _launch_and_wait(self) -> bool:
        problem = self.preflight()
        if problem is not None:
            self.logger.error(problem)
            self._set_status("failed")
            return False

        self._process = SubprocessSupervisor(
            self.build_command(),
            ready_token=self.ready_token,
            cwd=self.launch_cwd(),
            env=self.launch_env(),
            on_ready=lambda: self.signals.started.emit(),
            on_finished=self._on_finished,
            name=self.device_id,
        )
        self._started = datetime.now()
        self._set_status("launching")
        self.logger.info(
            f"Launching {self.device_id}; waiting for '{self.ready_token}'..."
        )
        self._process.start()
        if self._wait_ready_pumping(self.ready_timeout):
            self.logger.info(f"{self.device_id} is ready.")
            self._set_status("ready")
        elif self._process.is_running():
            self.logger.warning(
                f"{self.device_id} not ready after {self.ready_timeout:.0f}s but "
                f"still running; continuing."
            )
            self._set_status("running")
        else:
            self.logger.error(
                f"{self.device_id} exited before reporting '{self.ready_token}'; "
                f"see the log lines above for the cause."
            )
            self._set_status("failed")
        return True

    def _wait_ready_pumping(self, timeout: float) -> bool:
        """Wait for the readiness handshake, pumping Qt events if a GUI is up.

        In the GUI, arm_all/start_all run on the Qt main thread, so a plain
        blocking wait would freeze the window. Pumping ``processEvents`` keeps
        it responsive during the brief boot; headless callers just block.
        """
        if self._process is None:
            return False
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
        except Exception:
            app = None

        if app is None:
            return self._process.wait_ready(timeout=timeout)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process.wait_ready(timeout=0.05):
                return True
            app.processEvents()
        return self._process.wait_ready(timeout=0.0)

    def _on_finished(self, _code: int = 0) -> None:
        # Subprocess exited. Reflect an unexpected mid-run exit without
        # clobbering a "failed" set by the launch path.
        if self._gui_status in ("launching", "running", "ready"):
            self._set_status("stopped")
        self.signals.finished.emit()

    # -- introspection --------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "running": self._process is not None and self._process.is_running(),
            "started": self._started,
            "stopped": self._stopped,
        }

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
        }
