import os
import winreg
import base64
from dataclasses import dataclass

import dill
from PyQt6.QtCore import QObject, pyqtSignal, QProcess, QTimer, QEventLoop, Qt
from PyQt6.QtWidgets import QMessageBox

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mesofield.config import ExperimentConfig

class PsychopyParameters:
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
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.readyReadStandardError.connect(self._on_stderr)
        self.process.finished.connect(self._on_finished)

    def start(self):
        # Serialize parameters
        params = PsychopyParameters(self.config.psychopy_parameters)
        serialized = dill.dumps(params, byref=True)
        b64 = base64.b64encode(serialized).decode('ascii')
        exe = get_psychopy_python_exe()
        script = os.path.join(self.config._save_dir, self.config.psychopy_filename)

        # Handshake timeout
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)
        self._timer.start(29000)

        # show blocking waiting dialog until ready or error
        from PyQt6.QtWidgets import QApplication
        parent_win = self.parent() if callable(getattr(self, 'parent', None)) else None
        waiting = QMessageBox(parent_win)
        waiting.setWindowTitle('Launching PsychoPy')
        waiting.setText('Waiting for PsychoPy script to print(PSYCHOPY_READY, flush=true)...')
        waiting.setStandardButtons(QMessageBox.NoButton)
        waiting.setWindowModality(Qt.ApplicationModal)
        waiting.show()
        waiting.activateWindow()
        waiting.raise_()
        QApplication.processEvents()

        # connect handshake signals to close waiting dialog
        self.ready.connect(waiting.accept)
        self.error.connect(waiting.reject)

        # start process
        self.process.start(exe, [script, b64])

        # block until handshake result
        result = waiting.exec()
        # cleanup waiting dialog
        waiting.close()
        # show ready or error popup
        if self._handshake_ok:
            # show ready popup
            ready_box = QMessageBox(parent_win)
            ready_box.setWindowTitle('PsychoPy Ready')
            ready_box.setText('PsychoPy is ready.\nPress spacebar to start recording.')
            ready_box.setStandardButtons(QMessageBox.Ok)
            ready_box.setWindowModality(Qt.ApplicationModal)
            ready_box.show()
            ready_box.activateWindow()
            ready_box.raise_()
            QApplication.processEvents()
            ready_box.exec()
        else:
            # show timeout error
            err = QMessageBox(parent_win)
            err.setIcon(QMessageBox.Critical)
            err.setWindowTitle('PsychoPy Error')
            err.setText('PsychoPy handshake timed out')
            err.exec()

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
        data = self.process.readAllStandardError().data().decode()
        print(data, end="")

    def _on_finished(self, exit_code, exit_status):
        self.finished.emit(exit_code, exit_status)

    def _on_timeout(self):
        # handshake failed
        self._handshake_ok = False
        self.error.emit("PsychoPy handshake timed out")