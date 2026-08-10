"""PsychoPy stimulus device.

Launches a PsychoPy experiment script as a subprocess on the shared
:class:`~mesofield.devices.stimulus_base.SubprocessStimulusDevice` engine, so the
:class:`~mesofield.base.Procedure` drives it through the same
``arm/start/stop/shutdown`` lifecycle and stdout readiness handshake as
MousePortal and any other external-app stimulus device.

PsychoPy is *operator-in-the-loop*: unlike MousePortal (which launches silently
at ``arm``), it launches at ``start`` and gates recording behind an operator
confirmation (``confirm_on_ready``). The readiness handshake is mandatory
(``require_ready = True``): if ``PSYCHOPY_READY`` never arrives, the run fails
rather than recording against a stimulus that never started.

PsychoPy is *not* a :class:`~mesofield.protocols.DataProducer`: it never emits on
``signals.data``. ``signals.started`` fires on the ``PSYCHOPY_READY`` handshake;
``signals.finished`` fires on ``stop`` / subprocess exit.

Config convention (standardized with MousePortal): the script and parameters
live in the ExperimentConfig (``experiment.json``); the hardware.yaml ``psychopy``
stanza carries only subprocess plumbing (``type`` / ``python_exe`` /
``ready_timeout``). Which script runs is keyed by the selected ``task``: a
top-level ``PsychoPy`` block maps ``{task: filename}`` (scripts declare their
task by embedding ``task-{name}`` in the filename), and
``ExperimentConfig.psychopy_path`` resolves the entry for the current task,
falling back to the legacy single ``psychopy_filename`` when no map is present.
``prepare()`` reads ``psychopy_path`` / ``psychopy_parameters`` and is unaware of
this resolution. Parameters are handed to the script as a base64 JSON argv
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
from mesofield.devices.subprocesses.psychopy import get_psychopy_python_exe


@DeviceRegistry.register("psychopy")
class PsychoPyDevice(SubprocessStimulusDevice):
    """Stimulus device that launches a PsychoPy script as a subprocess."""

    ready_token: ClassVar[str] = "PSYCHOPY_READY"
    launch_phase: ClassVar[str] = "start"
    default_device_id: ClassVar[str] = "psychopy"
    # The PSYCHOPY_READY handshake is mandatory (see module docstring).
    require_ready: ClassVar[bool] = True
    confirm_on_ready: ClassVar[bool] = True

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        # PsychoPy's first start (window creation, iohub launch, frame-rate
        # measurement) can be slow; allow generous headroom before declaring
        # the handshake failed.
        self.ready_timeout = float(cfg.get("ready_timeout", 60.0))
        self._script: Optional[str] = None
        self._params_b64: Optional[str] = None

    # -- SubprocessStimulusDevice hooks ---------------------------------
    def serves_task(self, task, config) -> bool:
        """Serve a task iff the task->script map has an entry for it.

        With no map (legacy single-script experiments) PsychoPy serves every
        task, preserving the old behavior.
        """
        mapping = config.psychopy
        return task in mapping if mapping else True

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


# Manifest-driven dispatch: SOURCE_REGISTRY["psychopy"] resolves to the parser
# in mesofield.datakit.sources.behavior.psychopy. Bind it here too for the
# documented PsychoPyDevice.Parser convention (encoder/treadmill do the same),
# so producer and offline parser are reachable from one place. Imported at the
# bottom to keep device construction independent of import order.
from mesofield.datakit.sources.behavior.psychopy import Psychopy  # noqa: E402

PsychoPyDevice.Parser = Psychopy
