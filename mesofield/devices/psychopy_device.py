"""PsychoPy stimulus device.

Launches a PsychoPy experiment script as a subprocess on the shared
:class:`~mesofield.devices.stimulus_base.SubprocessStimulusDevice` engine, so the
:class:`~mesofield.base.Procedure` drives it through the same
``arm/start/stop/shutdown`` lifecycle and stdout readiness handshake as
MousePortal and any other external-app stimulus device.

PsychoPy is *operator-in-the-loop*: unlike MousePortal (which launches silently
at ``arm``), it launches at ``start`` and overrides the base's presentation
hooks to show a "launching" dialog, surface a startup failure with the script's
output, and gate recording behind a focused "press to start" dialog forced over
PsychoPy's full-screen window. The readiness handshake is mandatory
(``require_ready = True``): if ``PSYCHOPY_READY`` never arrives, the run fails
rather than recording against a stimulus that never started.

PsychoPy is *not* a :class:`~mesofield.protocols.DataProducer`: it never emits on
``signals.data``. ``signals.started`` fires on the ``PSYCHOPY_READY`` handshake;
``signals.finished`` fires on ``stop`` / subprocess exit.

Config convention (standardized with MousePortal): the script filename and
parameters live in the ExperimentConfig (``experiment.json`` ->
``psychopy_filename`` / ``psychopy_parameters``); the hardware.yaml ``psychopy``
stanza carries only subprocess plumbing (``type`` / ``python_exe`` /
``ready_timeout``). Parameters are handed to the script as a base64 JSON argv
token so the PsychoPy interpreter needs only the stdlib to decode them (no
``mesofield`` import). The matching offline parser is registered under the
``psychopy`` tag in :mod:`mesofield.datakit.sources` and bound here as
``PsychoPyDevice.Parser`` for the documented dispatch convention.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, ClassVar, Dict, List, Optional

from mesofield import DeviceRegistry
from mesofield.devices.stimulus_base import SubprocessStimulusDevice
from mesofield.devices.subprocesses.psychopy import (
    force_foreground,
    get_psychopy_python_exe,
)


@DeviceRegistry.register("psychopy")
class PsychoPyDevice(SubprocessStimulusDevice):
    """Stimulus device that launches a PsychoPy script as a subprocess."""

    ready_token: ClassVar[str] = "PSYCHOPY_READY"
    launch_phase: ClassVar[str] = "start"
    default_device_id: ClassVar[str] = "psychopy"
    # The PSYCHOPY_READY handshake is mandatory (see module docstring).
    require_ready: ClassVar[bool] = True

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        # PsychoPy's first start (window creation, iohub launch, frame-rate
        # measurement) can be slow; allow generous headroom before declaring
        # the handshake failed.
        self.ready_timeout = float(cfg.get("ready_timeout", 60.0))
        self._script: Optional[str] = None
        self._params_b64: Optional[str] = None
        self._launching_box = None  # QMessageBox shown while waiting for ready

    # -- SubprocessStimulusDevice hooks ---------------------------------
    def prepare(self, config) -> None:
        """Resolve the script path and serialize parameters for the subprocess.

        Parameters are sent as base64-encoded JSON (a single safe argv token) so
        the PsychoPy interpreter decodes them with only the stdlib -- the script
        rebuilds an attribute namespace, e.g.
        ``config = types.SimpleNamespace(**json.loads(base64.b64decode(sys.argv[1])))``.
        """
        self._script = config.psychopy_path
        params = config.psychopy_parameters
        self._params_b64 = base64.b64encode(
            json.dumps(params).encode("utf-8")
        ).decode("ascii")

    def preflight(self) -> Optional[str]:
        if not self._script or not os.path.isfile(self._script):
            return (
                f"PsychoPy script not found: {self._script!r}. Check "
                f"'psychopy_filename' and the experiment save directory."
            )
        return None

    def build_command(self) -> List[str]:
        return [self._resolve_python_exe(), self._script, self._params_b64]

    def _resolve_python_exe(self) -> str:
        """Interpreter that runs the PsychoPy script.

        Honor an explicit ``python_exe:`` plumbing key (the same convention as
        MousePortal) first, else discover the standalone PsychoPy interpreter
        from the Windows registry.
        """
        return self.python_exe or get_psychopy_python_exe()

    # -- operator presentation hooks ------------------------------------
    def present_launching(self) -> None:
        """Show a non-blocking 'waiting for PSYCHOPY_READY' indicator."""
        app = self._qapp()
        if app is None:
            return
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QMessageBox

        box = QMessageBox()
        box.setWindowTitle("Launching PsychoPy")
        box.setText("Waiting for the PsychoPy script to print PSYCHOPY_READY...")
        box.setStandardButtons(QMessageBox.StandardButton.NoButton)
        box.setWindowModality(Qt.WindowModality.ApplicationModal)
        box.show()
        box.raise_()
        box.activateWindow()
        app.processEvents()
        self._launching_box = box

    def dismiss_launching(self) -> None:
        box = self._launching_box
        if box is not None:
            box.close()
            self._launching_box = None

    def present_failure(self, message: str, detail: str = "") -> None:
        super().present_failure(message, detail)  # always log
        app = self._qapp()
        if app is None:
            return
        from PyQt6.QtWidgets import QMessageBox

        box = QMessageBox()
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("PsychoPy Error")
        box.setText(message)
        if detail.strip():
            box.setDetailedText(detail.strip())
        box.exec()

    def confirm_ready_to_record(self) -> bool:
        """Focused 'PsychoPy ready -- press to start recording' gate.

        Forced to the foreground over PsychoPy's full-screen window so a
        spacebar press lands on this dialog (OK is the default button), not on
        PsychoPy. Dismissing it returns control to the Procedure, which then
        starts the recording devices; the operator then presses spacebar in the
        PsychoPy window to begin the stimulus (cameras lead; timelines are
        aligned post-hoc). Returns ``False`` (Cancel) to abort the run.
        """
        app = self._qapp()
        if app is None:
            return True
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QMessageBox

        box = QMessageBox()
        box.setWindowTitle("PsychoPy Ready")
        box.setText(
            "PsychoPy is ready.\nPress spacebar (or click OK) to start recording."
        )
        box.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QMessageBox.StandardButton.Ok)
        box.setWindowModality(Qt.WindowModality.ApplicationModal)
        force_foreground(box)
        app.processEvents()
        return box.exec() == QMessageBox.StandardButton.Ok

    @staticmethod
    def _qapp():
        """Return the live QApplication, or None when running headless."""
        try:
            from PyQt6.QtWidgets import QApplication

            return QApplication.instance()
        except Exception:
            return None


# Manifest-driven dispatch: SOURCE_REGISTRY["psychopy"] resolves to the parser
# in mesofield.datakit.sources.behavior.psychopy. Bind it here too for the
# documented PsychoPyDevice.Parser convention (encoder/treadmill do the same),
# so producer and offline parser are reachable from one place. Imported at the
# bottom to keep device construction independent of import order.
from mesofield.datakit.sources.behavior.psychopy import Psychopy  # noqa: E402

PsychoPyDevice.Parser = Psychopy
