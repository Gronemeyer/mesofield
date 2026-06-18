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
- ``require_ready`` (class attr) -- treat a missing readiness handshake as a
  failure even if the child is still alive (PsychoPy); default False.
- ``enabled`` (instance attr) -- per-run gate; a Procedure may set it False to
  record a stimulus-free task.

Operator presentation hooks (no-op/log by default, so automatic stimuli are
unaffected; a GUI subclass overrides them to drive modal dialogs):
:meth:`present_launching` / :meth:`dismiss_launching`, :meth:`present_failure`,
and :meth:`confirm_ready_to_record` (the post-readiness "press to record" gate).
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
    # When True, a launch that never reports ``ready_token`` is a failure even
    # if the child is still alive (the handshake is mandatory -- PsychoPy). When
    # False (default), a still-running-but-not-ready child is accepted and the
    # run continues, since some apps are slow to print their token (MousePortal).
    require_ready: ClassVar[bool] = False

    def __init__(self, cfg: Dict[str, Any]):
        from psygnal import SignalInstance

        self.signals = DeviceSignals()
        self.cfg = dict(cfg)
        self.device_id: str = cfg.get("id", self.default_device_id)
        self.is_primary: bool = bool(cfg.get("primary", False))
        # Per-run gate. A Procedure may set this False in ``prerun`` for a task
        # that records without a stimulus (e.g. a spontaneous baseline), so
        # neither ``arm`` nor ``start`` launches the subprocess.
        self.enabled: bool = True
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

    # -- operator presentation hooks (override in a GUI-facing subclass) -
    # These let an operator-in-the-loop stimulus (PsychoPy) surface modal
    # dialogs without dragging Qt into this framework-agnostic base: the
    # defaults are silent / log-only, so automatic stimuli (MousePortal) are
    # unaffected. A subclass that overrides them imports its GUI toolkit only
    # in its own module.
    def present_launching(self) -> None:
        """Show a non-blocking 'launching, waiting for readiness' indicator.

        Called right after the subprocess is spawned and before the readiness
        wait (which pumps Qt events, so a shown-but-not-``exec``'d dialog stays
        responsive). Pair with :meth:`dismiss_launching`. No-op by default.
        """
        return None

    def dismiss_launching(self) -> None:
        """Dismiss the indicator shown by :meth:`present_launching`. No-op default."""
        return None

    def present_failure(self, message: str, detail: str = "") -> None:
        """Surface a launch/handshake failure to the operator.

        ``detail`` carries the child's last output (see
        :attr:`SubprocessSupervisor.output_tail`). Logs by default; a GUI
        subclass shows a dialog.
        """
        self.logger.error(message + (f"\n{detail}" if detail else ""))

    def confirm_ready_to_record(self) -> bool:
        """Operator gate after the stimulus reports ready; ``False`` cancels.

        Runs inside :meth:`start` for ``launch_phase == "start"`` devices once
        the readiness handshake has fired. Default proceeds immediately
        (automatic stimuli); PsychoPy shows a focused "press to start
        recording" dialog over its full-screen window.
        """
        return True

    # -- handshake state ------------------------------------------------
    @property
    def handshake_ok(self) -> bool:
        """``True`` once the subprocess reported its ``ready_token``."""
        return self._gui_status == "ready"

    # -- lifecycle ------------------------------------------------------
    def initialize(self) -> bool:
        return True

    def arm(self, config: Any) -> None:
        self._config = config
        if not self.enabled:
            self.logger.info(f"{self.device_id} disabled for this run; not arming.")
            return
        # Reset per-run state so a second run in the same session relaunches
        # cleanly: a prior run leaves a (terminated) supervisor on
        # ``self._process``, which would otherwise make ``start`` a no-op via
        # its "already launched" guard. Terminate defensively in case it is
        # somehow still alive, then drop the handle.
        if self._process is not None and self._process.is_running():
            self._process.terminate()
        self._process = None
        self._started = None
        self._stopped = None
        self._set_status("loaded")
        self.prepare(config)
        if self.launch_phase == "arm":
            self._launch_and_wait()

    def start(self) -> bool:
        if not self.enabled:
            self.logger.info(f"{self.device_id} disabled for this run; skipping launch.")
            return False
        if self.launch_phase == "arm":
            # Launched during arm; just confirm it is alive.
            if self._process is None or not self._process.is_running():
                self.logger.warning(f"{self.device_id} is not running at start().")
                return False
            self._started = self._started or datetime.now()
            return True
        # start-phase: launch now (unless a start gate already launched us).
        if self._process is not None:
            # Already launched (e.g. by a Procedure's start_on_trigger gate,
            # before start_all). Don't spawn a second subprocess.
            return self.handshake_ok
        if not self._launch_and_wait():
            return False
        # Operator gate: only after a clean readiness handshake.
        if self._gui_status == "ready" and not self.confirm_ready_to_record():
            self.logger.info(f"{self.device_id}: operator cancelled at the ready gate.")
            self.stop()
            return False
        return True

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
        """Spawn the subprocess and block (pumping Qt) on its readiness.

        Returns ``True`` if the child reached ``ready`` (or, when
        ``require_ready`` is False, is still running after the timeout) and
        ``False`` on a failed launch -- so a start-phase caller can abort.
        """
        problem = self.preflight()
        if problem is not None:
            self.logger.error(problem)
            self._set_status("failed")
            self.present_failure(problem)
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
        self.present_launching()
        try:
            if self._wait_ready_pumping(self.ready_timeout):
                self.logger.info(f"{self.device_id} is ready.")
                self._set_status("ready")
            elif self._process.is_running() and not self.require_ready:
                self.logger.warning(
                    f"{self.device_id} not ready after {self.ready_timeout:.0f}s but "
                    f"still running; continuing."
                )
                self._set_status("running")
            else:
                self.logger.error(
                    f"{self.device_id} did not report '{self.ready_token}' "
                    f"(timeout {self.ready_timeout:.0f}s); see the log lines above."
                )
                self._set_status("failed")
        finally:
            self.dismiss_launching()

        if self._gui_status == "failed":
            tail = getattr(self._process, "output_tail", "") if self._process else ""
            # A required handshake that timed out can leave the child alive but
            # unusable -- terminate it so we don't orphan the process.
            if self._process is not None and self._process.is_running():
                self._process.terminate()
            self.present_failure(
                f"{self.device_id} did not report ready ('{self.ready_token}').",
                tail,
            )
            return False
        return True

    def _wait_ready_pumping(self, timeout: float) -> bool:
        """Wait for the readiness handshake, pumping Qt events if a GUI is up.

        In the GUI, arm_all/start_all run on the Qt main thread, so a plain
        blocking wait would freeze the window. Pumping ``processEvents`` keeps
        it responsive during the brief boot; headless callers just poll.

        Polls in short slices so it can **fail fast**: if the child exits before
        the handshake (a stimulus script that errors on startup), this returns
        immediately instead of waiting out the full ``timeout``.
        """
        if self._process is None:
            return False
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
        except Exception:
            app = None

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process.wait_ready(timeout=0.05):
                return True
            if not self._process.is_running():
                # Child exited before the handshake -- don't wait out the timeout.
                return self._process.wait_ready(timeout=0.0)
            if app is not None:
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
