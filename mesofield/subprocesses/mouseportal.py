import sys
import time
import json
import os

from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mesofield.config import ExperimentConfig

from PyQt6.QtCore import QProcess

class MousePortal:
    """
    Controller for the MousePortal `runportal.py` subprocess.
    - manages config read/write
    - starts/stops a QProcess
    - sends commands via stdin
    - emits output via a user-provided callback
    """

    def __init__(
        self,
        config: 'ExperimentConfig',
        parent=None,
        python_executable: str = sys.executable,
        script: str = "runportal.py",
        script_args: Optional[list] = None,
    ):
        self.parent = parent
        self.cfg_path = cfg_path
        self.runtime_path = runtime_path
        self.python_executable = python_executable
        self.script = script
        self.script_args = script_args or ["--dev", "--cfg", self.runtime_path]

        self.process = QProcess(self.parent)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_ready_read)
        self.process.finished.connect(self._on_finished)

        self.output_callback: Optional[Callable[[str], None]] = None
        self.finished_callback: Optional[Callable[[int, QProcess.ExitStatus], None]] = None
        self.cfg = {}

    # --------------------
    # Config management
    # --------------------
    def load_cfg(self) -> dict:
        with open(self.cfg_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
        return self.cfg

    def set_cfg(self, cfg: dict) -> None:
        self.cfg = cfg

    def save_runtime(self, cfg: Optional[dict] = None) -> None:
        cfg_to_write = cfg if cfg is not None else self.cfg
        with open(self.runtime_path, "w", encoding="utf-8") as f:
            json.dump(cfg_to_write, f, indent=2)

    def remove_runtime(self) -> None:
        try:
            if os.path.isfile(self.runtime_path):
                os.remove(self.runtime_path)
        except Exception:
            pass

    # --------------------
    # Process control
    # --------------------
    def start(self) -> None:
        if self.is_running:
            return
        self.save_runtime()
        args = [self.script] + list(self.script_args)
        self.process.start(self.python_executable, args)

    def end(self, wait_ms: int = 3000) -> None:
        if not self.is_running:
            self.remove_runtime()
            return
        self.send_cmd("end")
        self.process.waitForFinished(wait_ms)
        self.remove_runtime()

    def terminate(self, wait_ms: int = 3000) -> None:
        if self.is_running:
            self.process.terminate()
            self.process.waitForFinished(wait_ms)
        self.remove_runtime()

    @property
    def is_running(self) -> bool:
        return self.process.state() == QProcess.ProcessState.Running

    # --------------------
    # Command helpers
    # --------------------
    def send_cmd(self, cmd: str) -> None:
        if not self.is_running:
            return
        payload = (cmd + f" {time.time()}\n").encode()
        self.process.write(payload)

    def start_trial(self) -> None:
        self.send_cmd("start_trial")

    def stop_trial(self) -> None:
        self.send_cmd("stop_trial")

    def mark_event(self, label: str = "button") -> None:
        self.send_cmd(f"mark_event {label}")

    # --------------------
    # I/O and callbacks
    # --------------------
    def set_output_callback(self, cb: Callable[[str], None]) -> None:
        self.output_callback = cb

    def set_finished_callback(self, cb: Callable[[int, QProcess.ExitStatus], None]) -> None:
        self.finished_callback = cb

    def _on_ready_read(self) -> None:
        raw = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        if raw and self.output_callback:
            # deliver each line individually
            for line in raw.splitlines():
                self.output_callback(line)
        # If no callback, keep stdout unread (or could buffer)

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        if self.finished_callback:
            self.finished_callback(exit_code, exit_status)
        # cleanup runtime file on finish
        self.remove_runtime()


# Example usage (in a PyQt application):
# manager = PortalSubprocessManager()
# manager.set_output_callback(lambda s: print("OUT:", s))
# manager.start()
# manager.start_trial()
# manager.mark_event("foo")
# manager.stop_trial()
# manager.end()
