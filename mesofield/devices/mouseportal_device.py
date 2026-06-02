"""MousePortal stimulus device.

Launches the MousePortal infinite-corridor stimulus (a separate Panda3D app)
as a subprocess and drives it with live treadmill velocity forwarded from
mesofield over a localhost UDP socket.  A
:class:`~mesofield.protocols.StimulusDevice` built on
:class:`~mesofield.devices.stimulus_base.SubprocessStimulusDevice`, which
provides the launch / readiness-handshake / terminate lifecycle shared with
other external-app stimulus devices.  It is *not* a
:class:`~mesofield.protocols.DataProducer` (MousePortal writes its own per-frame
CSV; this device never emits on ``signals.data``).

Synchronisation
---------------
mesofield owns the treadmill serial port.  During ``arm`` (so the corridor is up
before any camera acquires) this device subscribes to the treadmill's
``signals.data`` and forwards each ``{distance, speed, device_us}`` sample as a
UDP datagram (``device_us,distance,speed``) to the MousePortal process, which
runs in ``network`` input mode.  MousePortal logs the ``device_us`` of the
latest sample on every frame, so its corridor data aligns offline to the camera
timeline through the dataqueue affine fit (see
``mesofield.datakit.sources.behavior.mouseportal``).

The matching offline parser is registered separately under the ``mouseportal``
tag in :mod:`mesofield.datakit.sources`; this module intentionally does **not**
import datakit so device construction stays light.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from typing import Any, Callable, ClassVar, Dict, List, Optional

from mesofield import DeviceRegistry
from mesofield.devices.stimulus_base import SubprocessStimulusDevice


# MousePortal application config sections (the experiment design + corridor
# appearance). These live in the ExperimentConfig under the ``mouseportal`` key
# (experiment.json), NOT in hardware.yaml -- the YAML stanza keeps only the
# subprocess plumbing (type/app_dir/python_exe/udp_port/...). Legacy stanzas
# that still carry these sections are honored as a fallback.
_PORTAL_SECTIONS = ("window", "corridor", "camera", "fog", "experiment")


@DeviceRegistry.register("mouseportal")
class MousePortalDevice(SubprocessStimulusDevice):
    """Stimulus device that launches MousePortal and feeds it treadmill velocity."""

    file_type: str = "csv"
    bids_type: Optional[str] = "beh"
    data_type: ClassVar[str] = "corridor"

    # Launch during arm() so the corridor is ready before recording starts.
    ready_token: ClassVar[str] = "MOUSEPORTAL_READY"
    launch_phase: ClassVar[str] = "arm"
    default_device_id: ClassVar[str] = "mouseportal"

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.host: str = str(cfg.get("host", "127.0.0.1"))
        self.udp_port: int = int(cfg.get("udp_port", 8765))
        self.treadmill_id: str = str(cfg.get("treadmill_id", "treadmill"))
        # Recording tail: mesofield devices run this many seconds beyond the
        # MousePortal experiment so MM cameras finish their preallocated frames
        # (an early cutoff leaves black frames in the OME-TIFF). Consumed by the
        # procedure to derive the recording duration.
        self.tail_seconds: float = float(cfg.get("tail_seconds", 5.0))

        self._cfg_path: Optional[str] = None
        self.output_path: Optional[str] = None
        self.metadata_path: Optional[str] = None
        self._sock: Optional[socket.socket] = None
        self._treadmill = None
        self._forward: Optional[Callable[..., None]] = None

    # -- SubprocessStimulusDevice hooks ---------------------------------
    def prepare(self, config) -> None:
        """Generate the MousePortal cfg.json and wire treadmill forwarding.

        The output CSV path is the ExperimentConfig-authoritative path:
        ``DataManager.setup`` assigns ``self.output_path`` from ``DataPaths``
        before arm.  We fall back to ``config.make_path`` only for standalone
        use (no DataManager).  MousePortal is handed this exact file path and
        writes only there -- it never constructs a BIDS directory layout.
        """
        if not self.output_path:
            self.output_path = config.make_path(
                self.device_id, self.file_type, self.bids_type, create_dir=True
            )
        out_dir = os.path.dirname(self.output_path)
        os.makedirs(out_dir, exist_ok=True)

        portal_cfg = self._build_portal_config(config)
        self._cfg_path = os.path.join(out_dir, "mouseportal_cfg.json")
        with open(self._cfg_path, "w", encoding="utf-8") as fh:
            json.dump(portal_cfg, fh, indent=4)
        self.logger.info(f"Wrote MousePortal config -> {self._cfg_path}")

        self._open_forwarder()

    def build_command(self) -> List[str]:
        python_exe = self.python_exe or sys.executable
        return [python_exe, "-m", "mouseportal", "-c", self._cfg_path, "--autostart"]

    def preflight(self) -> Optional[str]:
        """Return an actionable error string if MousePortal cannot launch."""
        python_exe = self.python_exe or sys.executable
        if self.app_dir and not os.path.isdir(self.app_dir):
            return (
                f"MousePortal app_dir does not exist: {self.app_dir!r}. "
                f"Set 'app_dir:' in the mouseportal hardware.yaml stanza to the "
                f"MousePortal repo root."
            )
        # Probe the full app import (panda3d + pyserial + mouseportal) with the
        # same cwd the real launch uses, so a missing *dependency* (not just the
        # package) is caught too.
        try:
            proc = subprocess.run(
                [python_exe, "-c", "import mouseportal.app"],
                cwd=self.app_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
            )
        except Exception as exc:
            return f"Could not probe MousePortal interpreter {python_exe!r}: {exc}"
        if proc.returncode != 0:
            return (
                f"The interpreter mesofield would use to launch MousePortal "
                f"({python_exe!r}) cannot import MousePortal and its dependencies:\n"
                f"    {proc.stdout.strip()}\n"
                f"Fix: set 'python_exe:' in the mouseportal hardware.yaml stanza "
                f"to a Python that has MousePortal + panda3d + pyserial installed "
                f"(e.g. the env where you `pip install -e` the MousePortal repo), "
                f"or install those into mesofield's environment."
            )
        return None

    def on_stop(self) -> None:
        self._close_forwarder()

    # -- treadmill velocity forwarding ----------------------------------
    def _open_forwarder(self) -> None:
        hardware = getattr(self._config, "hardware", None)
        treadmill = None
        if hardware is not None:
            treadmill = hardware.devices.get(self.treadmill_id) or getattr(
                hardware, "encoder", None
            )
        if treadmill is None:
            self.logger.warning(
                f"No treadmill device '{self.treadmill_id}' found; MousePortal "
                f"will receive no velocity (corridor will not move)."
            )
            return

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        addr = (self.host, self.udp_port)

        def _forward(payload: Any, device_ts: Any = None) -> None:
            try:
                if isinstance(payload, dict):
                    device_us = payload.get("device_us", "")
                    distance = payload.get("distance", 0.0)
                    speed = payload.get("speed", 0.0)
                else:
                    # Scalar payload (e.g. a click count): treat as speed only.
                    device_us, distance, speed = "", 0.0, float(payload)
                msg = f"{device_us},{distance},{speed}".encode("ascii", "replace")
                self._sock.sendto(msg, addr)
            except Exception as exc:  # never let the queue thread die
                self.logger.debug(f"treadmill forward failed: {exc}")

        treadmill.signals.data.connect(_forward)
        self._treadmill = treadmill
        self._forward = _forward
        self.logger.info(
            f"Forwarding '{self.treadmill_id}' velocity to MousePortal "
            f"at {self.host}:{self.udp_port}"
        )

    def _close_forwarder(self) -> None:
        if self._treadmill is not None and self._forward is not None:
            try:
                self._treadmill.signals.data.disconnect(self._forward)
            except Exception:
                pass
        self._treadmill = None
        self._forward = None
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    # -- config generation ---------------------------------------------
    def _mouseportal_params(self, config=None) -> Dict[str, Any]:
        """MousePortal application parameters (corridor + gain-trial design).

        Sourced from the ExperimentConfig ``mouseportal`` key (experiment.json)
        so the experiment design lives in the ExperimentConfig, not
        hardware.yaml. Falls back to per-section keys on the hardware.yaml
        stanza for backward compatibility with older configs.
        """
        config = config or self._config or getattr(self, "config", None)
        if config is not None and hasattr(config, "get"):
            params = config.get("mouseportal")
            if isinstance(params, dict) and params:
                return params
        # Legacy fallback: sections authored directly on the hardware stanza.
        return {
            section: self.cfg[section]
            for section in _PORTAL_SECTIONS
            if section in self.cfg
        }

    def _build_portal_config(self, config) -> Dict[str, Any]:
        """Assemble the JSON config handed to the MousePortal subprocess."""
        params = self._mouseportal_params(config)
        portal: Dict[str, Any] = {
            section: params[section]
            for section in _PORTAL_SECTIONS
            if section in params
        }
        portal["input"] = {
            "mode": "network",
            "host": self.host,
            "udp_port": self.udp_port,
        }
        portal["triggers"] = {"enabled": False}
        # Hand MousePortal the exact, ExperimentConfig-allocated file path. It
        # writes there verbatim and never builds its own BIDS directory layout.
        portal["logging"] = {
            "subject": str(config.subject),
            "session": str(config.session),
            "task": str(config.task),
            "output_path": self.output_path,
        }
        return portal

    # -- duration coupling ---------------------------------------------
    def expected_experiment_duration(self) -> float:
        """Estimate the MousePortal experiment length in seconds.

        Mirrors MousePortal's per-trial resolution (condition override → global)
        and sums each trial's duration plus the inter-trial interval that
        follows it.  DURATION-ended trials are exact; DISTANCE/MANUAL trials are
        non-deterministic, so their per-trial ``trial_duration`` (or the global
        default) is used as an estimate -- pair this coupling with
        duration-based trials for a precise camera preallocation.
        """
        exp = self._mouseportal_params().get("experiment") or {}
        num_blocks = int(exp.get("num_blocks", 1))
        trials_per_block = int(exp.get("trials_per_block", 1))
        iti = float(exp.get("iti_duration", 0.0))
        global_dur = float(exp.get("trial_duration", 60.0))
        conditions = {
            c.get("label"): c for c in exp.get("conditions", []) if isinstance(c, dict)
        }
        block_conditions = exp.get("block_conditions", [])

        total = 0.0
        for b in range(num_blocks):
            seq = []
            if b < len(block_conditions) and isinstance(block_conditions[b], dict):
                seq = block_conditions[b].get("condition_sequence", [])
            for t in range(trials_per_block):
                cond = conditions.get(seq[t]) if t < len(seq) else None
                cond = cond or {}
                dur = cond.get("trial_duration")
                trial_seconds = float(dur) if dur is not None else global_dur
                total += trial_seconds + iti
        return total

    # -- introspection (extend base) -----------------------------------
    def status(self) -> Dict[str, Any]:
        st = super().status()
        st["output_path"] = self.output_path
        return st

    @property
    def metadata(self) -> Dict[str, Any]:
        meta = super().metadata
        meta["udp_port"] = self.udp_port
        return meta

    @property
    def calibration(self) -> Dict[str, Any]:
        """Record the corridor/experiment parameters with the session."""
        out: Dict[str, Any] = {"udp_port": self.udp_port}
        params = self._mouseportal_params()
        if params:
            out.update(params)
        return out
