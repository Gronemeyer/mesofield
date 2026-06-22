"""NI-DAQ device: external start-trigger + TTL edge counter.

Registered as a :class:`~mesofield.devices.base.BaseDataProducer` so its TTL
rising edges flow onto the session dataqueue via :meth:`record` and into the
AcquisitionManifest like any other producer.

Dual purpose, preserved from the original engine-driven design:

1. On :meth:`start` the counter input is armed *first* (begins counting),
   then a single pulse is written on the digital-output line to tell a
   separate system to go. Arming before pulsing means the first TTL returned
   by that system is never missed.
2. A background thread polls the counter and records one row per new rising
   edge (payload ``1``) -- the marker downstream parsers align against
   (``dataqueue_device_match="nidaq"`` in ``psychopy_device.Psychopy``).

``nidaqmx`` is imported lazily so this module loads on machines without the NI
driver; ``development_mode`` short-circuits every hardware call.
"""

from __future__ import annotations

import threading
import time
from typing import Any, ClassVar, Dict, Optional

from mesofield import DeviceRegistry
from mesofield.devices.base import BaseDataProducer


@DeviceRegistry.register("nidaq")
class Nidaq(BaseDataProducer):
    """External start-trigger + TTL edge counter on an NI-DAQ board."""

    device_type: ClassVar[str] = "nidaq"
    file_type: ClassVar[str] = "csv"
    bids_type: ClassVar[Optional[str]] = "beh"
    data_type: ClassVar[str] = "ttl_edges"

    dataqueue_payload_schema: ClassVar[Optional[dict]] = {
        "device_id": "nidaq",
        "payload_format": "scalar",
        "payload_fields": {"pulse": "int"},
        "description": "One row per counter rising edge; payload=1 marks a TTL pulse.",
    }

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        cfg = dict(cfg or {})
        cfg.setdefault("id", "nidaq")
        super().__init__(cfg, **kwargs)

        self.device_name: str = self.cfg.get("device_name", "Dev1")
        self.lines: str = self.cfg.get("lines", "port0/line0")
        self.ctr: str = self.cfg.get("ctr", "ctr0")
        self.io_type: str = self.cfg.get("io_type", "DO")
        self.pulse_width: float = float(self.cfg.get("pulse_width", 0.001))
        self.poll_interval: float = float(self.cfg.get("poll_interval", 0.01))
        self.development_mode: bool = bool(self.cfg.get("development_mode", False))

        self._ci: Any = None
        self._do: Any = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Running tally of TTL rising edges seen since the last start()
        self.pulse_count: int = 0

    # -- hardware helpers ------------------------------------------------
    def _close_tasks(self) -> None:
        """Close and drop any CI/DO tasks, swallowing per-task errors. """
        for name in ("_ci", "_do"):
            task = getattr(self, name, None)
            if task is not None:
                try:
                    task.close()
                except Exception as exc:
                    self.logger.debug(f"{name} close failed: {exc}")
            setattr(self, name, None)

    def _reset_device(self, context: str) -> None:
        """Best-effort hardware reset; logs (never raises) on failure."""
        if self.development_mode:
            return
        try:
            import nidaqmx.system

            nidaqmx.system.Device(self.device_name).reset_device()
        except Exception as exc:
            self.logger.debug(f"device reset ({context}) failed: {exc}")

    def reset(self) -> None:
        """Stop the worker (if running) and reset the NI-DAQ device."""
        if self._thread and self._thread.is_alive():
            self.stop()
        self._close_tasks()
        self._reset_device("reset")

    def test_connection(self) -> None:
        """Pulse the DO line high ~3 s as a wiring check (logs, never raises)."""
        self.logger.info(f"Testing NI-DAQ {self.device_name}/{self.lines}")
        if self.development_mode:
            return
        import nidaqmx

        try:
            with nidaqmx.Task() as task:
                task.do_channels.add_do_chan(f"{self.device_name}/{self.lines}")
                task.write(True)
                time.sleep(3)
                task.write(False)
            self.logger.info("NI-DAQ connection OK")
        except nidaqmx.DaqError as exc:
            self.logger.error(f"NI-DAQ connection failed: {exc}")

    # -- lifecycle -------------------------------------------------------
    def start(self) -> bool:
        # Both start_all() and PupilEngine.exec_sequenced_event call start();
        # the first wins, later calls are no-ops.
        if self._thread and self._thread.is_alive():
            return False

        self._stop_event.clear()
        self.pulse_count = 0

        if not self.development_mode:
            import nidaqmx
            from nidaqmx.constants import Edge

            # Tear down anything lingering and reset the board before creating new tasks
            self._close_tasks()
            self._reset_device("pre-start")

            try:
                # 1) Arm the counter FIRST so no returned TTL is missed.
                self._ci = nidaqmx.Task()
                self._ci.ci_channels.add_ci_count_edges_chan(
                    f"{self.device_name}/{self.ctr}", edge=Edge.RISING, initial_count=0
                )
                self._ci.start()

                # 2) Then pulse the DO line once to start the external system.
                self._do = nidaqmx.Task()
                self._do.do_channels.add_do_chan(f"{self.device_name}/{self.lines}")
                self._do.write(True)
                time.sleep(self.pulse_width)
                self._do.write(False)
            except nidaqmx.DaqError as exc:
                # Hardware start failed -- drop the half-created tasks
                self.logger.error(f"NI-DAQ start failed: {exc}")
                self._close_tasks()
                self._reset_device("failed-start")
                raise RuntimeError(
                    f"NI-DAQ {self.device_name} start failed: {exc}"
                ) from exc

        self._thread = threading.Thread(
            target=self._worker, name=f"Nidaq-{self.device_id}", daemon=True
        )
        self._thread.start()
        return super().start()

    def _worker(self) -> None:
        """Poll the counter and record one row per new rising edge."""
        prev = 0
        while not self._stop_event.is_set():
            if self._ci is not None:
                try:
                    count = int(self._ci.read())
                except Exception as exc:
                    # A device removed/aborted mid-run shouldn't spin the loop
                    self.logger.debug(f"counter read failed: {exc}")
                    self._stop_event.wait(self.poll_interval)
                    continue
                ts = time.time()
                for _ in range(count - prev):
                    self.pulse_count += 1
                    self.record(1, ts)
                prev = count
            self._stop_event.wait(self.poll_interval)

    def stop(self) -> bool:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._close_tasks()
        self._reset_device("stop")
        return super().stop()

    def shutdown(self) -> None:
        if self.is_running:
            self.stop()
