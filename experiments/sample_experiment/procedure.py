"""Sample custom Procedure -- the original Mesofield workflow refactored
as a concrete subclass of the generic :class:`mesofield.base.Procedure`.

Before the 9-phase refactor, the body of ``Procedure.run`` was hard-coded
for the Mesofield rig (two MicroManager cameras + encoder + PsychoPy +
NIDAQ trigger + Arduino LED sequence + fixed ``duration``).  The base
class is now device-agnostic; every rig-specific behaviour belongs in a
subclass like the one below, and is wired through the published
extension hooks:

- :meth:`prerun`      -- prime LED metadata, optionally remap output paths.
- :meth:`on_started`  -- optional NIDAQ trigger gate (``start_on_trigger``)
                         and optional wall-clock duration cap (``duration``).
- :meth:`on_finished` -- reserved for analysis / notification callbacks.
- :meth:`save_data`   -- adds a per-session text log on top of the base.

Discovery: ``experiment.json`` declares ``"procedure_file": "procedure.py"``
and ``"procedure_class": "SampleProcedure"`` so
:func:`mesofield.base.load_procedure_from_config` picks it up
automatically.

Save-path overrides (see :class:`mesofield.base.Procedure` docstring):
- Rename only        -> set ``path_args`` on the device class.
- Different folder   -> mutate ``self.data.save.paths`` in :meth:`prerun`.
- Different format   -> override ``save_data`` on the device.
"""

from __future__ import annotations

import threading
import time

from mesofield.base import Procedure


class SampleProcedure(Procedure):
    """Original Mesofield experiment driver refactored as a subclass.

    Recovers the legacy behaviour that used to live inline in
    ``Procedure.run`` for the Mesoscope + Pupil + Encoder + PsychoPy +
    NIDAQ rig.  Multi-camera sync is now driven by the YAML
    ``primary: true`` flag plus ``HardwareManager.start_all/stop_all``;
    this subclass only adds the truly experiment-specific glue.
    """

    # ------------------------------------------------------------------
    # Subclass hooks

    def prerun(self) -> None:
        """Per-run prep that the old monolithic ``run()`` did inline."""
        # 1. LED pattern publishing.  ``MesoEngine.setup_sequence`` reads
        #    ``led_sequence`` off the MDASequence metadata, and
        #    ``ExperimentConfig.build_sequence`` already injects
        #    ``self.led_pattern`` there -- we just log it for traceability.
        led = self.config.get("led_pattern")
        if led:
            self.logger.info(f"LED pattern primed: {led}")

        # 2. Optional path overrides.  Example: route the queue log into
        #    a sibling directory so it doesn't clutter ``beh/``.
        # self.data.save.paths.queue = self.config.make_path(
        #     "queue", "csv", "session"
        # )

        # 3. Pre-bind camera output paths from the resolved DataPaths so
        #    the base ``arm_all`` finds them already in place.
        for cam in self.hardware.cameras:
            cam_path = self.data.save.paths.hardware.get(cam.device_id)
            if cam_path is not None:
                cam.output_path = cam_path

    def on_started(self) -> None:
        """Replaces the legacy ``start_on_trigger`` branch and ``duration`` timer.

        - If ``start_on_trigger`` is true and a NIDAQ device is present,
          block until its first edge before letting the engines proceed.
          Cameras have already been ``start_all``-ed, so they are armed
          and waiting for the hardware trigger.
        - If ``duration`` is set, arm a wall-clock safety cap that calls
          :meth:`cleanup` if the primary device hasn't finished by then.
        """
        # --- NIDAQ trigger gate ---
        if self.config.get("start_on_trigger"):
            nidaq = getattr(self.hardware, "nidaq", None)
            if nidaq is None:
                self.logger.warning(
                    "start_on_trigger=True but no NIDAQ device configured; "
                    "running ungated."
                )
            else:
                self._wait_for_trigger(nidaq)

        # --- Wall-clock duration cap ---
        duration = self.config.get("duration")
        if duration:
            self.logger.info(f"Duration cap armed: {duration}s")
            self._duration_timer = threading.Timer(float(duration), self.cleanup)
            self._duration_timer.daemon = True
            self._duration_timer.start()

    def on_finished(self) -> None:
        """Hook reserved for analysis / notification callbacks.

        Note: the legacy ``cameras[1].core.stopSequenceAcquisition()`` call
        is gone -- it lives in :meth:`MMCamera.stop` now and is fanned
        out by ``HardwareManager.stop_all``.
        """
        timer = getattr(self, "_duration_timer", None)
        if timer is not None:
            timer.cancel()
            self._duration_timer = None

    # ------------------------------------------------------------------
    # save_data override -- add a per-session text log on top of base.

    def save_data(self) -> None:
        super().save_data()
        try:
            session_log = self.config.make_path("session", "txt", None)
            with open(session_log, "w", encoding="utf-8") as fh:
                fh.write(f"protocol: {self.protocol}\n")
                fh.write(f"experimenter: {self.experimenter}\n")
                fh.write(f"start: {self.start_time}\n")
                fh.write(f"stop: {self.stopped_time}\n")
                if self.start_time and self.stopped_time:
                    delta = (self.stopped_time - self.start_time).total_seconds()
                    fh.write(f"elapsed_s: {delta:.2f}\n")
            self.logger.info(f"Session log written to {session_log}")
        except Exception as exc:
            self.logger.warning(f"Could not write session log: {exc}")

    # ------------------------------------------------------------------
    # Helpers

    def _wait_for_trigger(self, nidaq) -> None:
        """Block on the first ``signals.data`` emission from ``nidaq``."""
        self.logger.info("Waiting for external trigger on NIDAQ ...")
        triggered = {"hit": False}

        def _on_first(*_args, **_kw) -> None:
            triggered["hit"] = True

        nidaq.signals.data.connect(_on_first)
        timeout = float(self.config.get("trigger_timeout", 30))
        deadline = time.time() + timeout
        while not triggered["hit"] and time.time() < deadline:
            time.sleep(0.01)
        try:
            nidaq.signals.data.disconnect(_on_first)
        except Exception:
            pass
        if not triggered["hit"]:
            self.logger.error(
                f"Trigger never arrived within {timeout}s; aborting."
            )
            self.cleanup()
