import os
import json
import winreg
import base64

from PyQt6.QtCore import QObject, pyqtSignal, QProcess, QTimer, Qt
from PyQt6.QtWidgets import QMessageBox

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mesofield.config import ExperimentConfig

class PsychopyParameters:
    """Lightweight attribute view over the psychopy parameter dict.

    Retained for backwards-compatibility / debugging. The subprocess no
    longer reconstructs *this* class: parameters are now sent as plain JSON
    (see :meth:`PsychoPyProcess.start`) so the PsychoPy interpreter needs
    neither ``dill`` nor the ``mesofield`` package installed to decode them.
    """

    def __init__(self, params: dict):
        for key, value in params.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"<PsychopyParameters {self.__dict__}>"

def get_psychopy_python_exe():
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\PsychoPy", 0, winreg.KEY_READ)
        install_path, _ = winreg.QueryValueEx(key, "InstallPath")
        winreg.CloseKey(key)
        python_exe = os.path.join(install_path, "python.exe")
        if os.path.exists(python_exe):
            return python_exe
    except OSError:
        pass
    return r"C:\Program Files\PsychoPy\python.exe"

def force_foreground(widget) -> None:
    """Force *widget* to the foreground with keyboard focus.

    PsychoPy runs its stimulus in a separate process whose full-screen window
    owns the foreground, and Windows blocks a background process from simply
    calling ``SetForegroundWindow``. The reliable workaround is to momentarily
    attach our input thread to the current foreground window's thread, raise our
    window, then detach. Without this the operator has to click the dialog by
    hand before a keypress reaches it. Best-effort and Windows-specific; any
    failure is swallowed (the dialog still works, it just may not auto-focus).
    """
    from PyQt6.QtCore import Qt

    widget.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    widget.show()
    widget.raise_()
    widget.activateWindow()
    # Best-effort Win32 foreground nudge. We intentionally avoid
    # AttachThreadInput here: coupling our GUI thread's input queue to
    # PsychoPy's full-screen-window thread can wedge input handling, and
    # WindowStaysOnTopHint + activateWindow is enough in practice.
    try:
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = int(widget.winId())
        user32.ShowWindow(hwnd, 5)  # SW_SHOW
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    except Exception:
        pass


def launch(config: 'ExperimentConfig', parent=None):
    """Launches a PsychoPy experiment as a subprocess encapsulated in PsychoPyProcess."""
    proc = PsychoPyProcess(config, parent)
    proc.start()
    return proc

class PsychoPyProcess(QObject):
    ready = pyqtSignal()
    finished = pyqtSignal(int, QProcess.ExitStatus)
    error = pyqtSignal(str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self._handshake_ok = False
        self._error_message = "PsychoPy handshake timed out"
        self._stderr_tail = ""
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.readyReadStandardError.connect(self._on_stderr)
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_process_error)

    def start(self):
        # Serialize parameters as plain JSON (base64 for a single safe argv
        # token). This keeps the contract dependency-free: the PsychoPy
        # interpreter only needs the stdlib (json/base64) to decode it -- no
        # dill, no importable mesofield package. The receiving script rebuilds
        # an attribute namespace, e.g.
        #   config = types.SimpleNamespace(**json.loads(base64.b64decode(sys.argv[1])))
        params = self.config.psychopy_parameters
        b64 = base64.b64encode(
            json.dumps(params).encode("utf-8")
        ).decode("ascii")
        exe = get_psychopy_python_exe()
        script = os.path.join(self.config._save_dir, self.config.psychopy_filename)

        # Handshake timeout. PsychoPy window creation, iohub launch and frame
        # rate measurement can take a while on first start, so allow generous
        # headroom before declaring the handshake failed.
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)
        self._timer.start(60000)

        # show blocking waiting dialog until ready or error
        from PyQt6.QtWidgets import QApplication
        parent_win = self.parent() if callable(getattr(self, 'parent', None)) else None
        waiting = QMessageBox(parent_win)
        waiting.setWindowTitle('Launching PsychoPy')
        waiting.setText('Waiting for PsychoPy script to print(PSYCHOPY_READY, flush=true)...')
        waiting.setStandardButtons(QMessageBox.StandardButton.NoButton)
        waiting.setWindowModality(Qt.WindowModality.ApplicationModal)
        waiting.show()
        waiting.activateWindow()
        waiting.raise_()
        QApplication.processEvents()

        # connect handshake signals to close waiting dialog
        self.ready.connect(waiting.accept)
        self.error.connect(waiting.reject)

        # start process
        self.process.start(exe, [script, b64])

        # block until handshake result (ready or error)
        waiting.exec()
        waiting.close()

        if not self._handshake_ok:
            # handshake failed -> surface the error to the operator
            err = QMessageBox(parent_win)
            err.setIcon(QMessageBox.Icon.Critical)
            err.setWindowTitle('PsychoPy Error')
            err.setText(self._error_message)
            if self._stderr_tail.strip():
                err.setDetailedText(self._stderr_tail.strip())
            err.exec()
            return

        # Handshake done: PsychoPy is loaded and parked in its trigger routine.
        # Show the "ready" gate and *force it to the foreground* over PsychoPy's
        # full-screen window so a spacebar press lands on this dialog (its OK is
        # the default button) instead of PsychoPy. Dismissing it returns control
        # to the procedure, which then starts the recording devices. The user
        # then presses spacebar again in the PsychoPy window to begin the
        # stimulus (cameras lead; timelines are aligned post-hoc).
        ready_box = QMessageBox(parent_win)
        ready_box.setWindowTitle('PsychoPy Ready')
        ready_box.setText('PsychoPy is ready.\nPress spacebar (or click OK) to start recording.')
        ready_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        ready_box.setDefaultButton(QMessageBox.StandardButton.Ok)
        ready_box.setWindowModality(Qt.WindowModality.ApplicationModal)
        force_foreground(ready_box)
        QApplication.processEvents()
        ready_box.exec()

    def _on_stdout(self):
        data = self.process.readAllStandardOutput().data().decode()
        print(data, end="")
        if "PSYCHOPY_READY" in data:
            # handshake succeeded
            self._handshake_ok = True
            if self._timer.isActive():
                self._timer.stop()
            self.ready.emit()

    def _on_stderr(self):
        data = self.process.readAllStandardError().data().decode(errors="replace")
        print(data, end="")
        # Keep the tail so we can surface it if the script dies before READY.
        self._stderr_tail = (self._stderr_tail + data)[-4000:]

    def _on_process_error(self, error):
        # e.g. the PsychoPy python.exe could not be started at all.
        if not self._handshake_ok:
            self._fail_handshake(
                f"Failed to launch PsychoPy subprocess (QProcess error {int(error)}). "
                f"Check the PsychoPy python path."
            )

    def _on_finished(self, exit_code, exit_status):
        # If the subprocess exits before printing PSYCHOPY_READY, the handshake
        # can never arrive -- fail fast instead of waiting out the full timeout.
        if not self._handshake_ok:
            self._fail_handshake(
                f"PsychoPy exited before the handshake (exit code {exit_code}). "
                f"The stimulus script likely errored on startup."
            )
        self.finished.emit(exit_code, exit_status)

    def _on_timeout(self):
        self._fail_handshake("PsychoPy handshake timed out")

    def _fail_handshake(self, message: str) -> None:
        """Record a startup failure and unblock the waiting dialog once."""
        if self._handshake_ok:
            return
        self._error_message = message
        timer = getattr(self, "_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
        self.error.emit(message)