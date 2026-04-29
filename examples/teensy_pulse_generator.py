"""Example custom device: a commandable Teensy 4.0 sync pulse generator.

Targets the ``sync-pulse-generator v3-teensy40`` firmware.  Protocol is
line-based and ``\\n``-terminated; the device USB-CDC speed is fixed at
115200.

Device -> host telemetry
========================

Pulse events (one line per rising LED edge)::

    LED,<device_us>,<event_id>

Banners, status, sync replies, and errors all start with ``#``::

    # sync-pulse-generator v3-teensy40 ts=micros_at_rising_edge
    # status state=idle seq_len=1 run_count=0 run_count_limit=0 epoch=0 next_id=0
    # pattern len=1 (20000,480000)
    # sync token=<t> device_us=<micros>
    # running | # stopped | # auto_stop | # epoch=<n> | # pong
    # err=<reason> | # dropped=<n> | # pattern_set

Host -> device commands (case-insensitive)
==========================================

::

    STATUS                              -> '# status ...'
    PING                                -> '# pong'
    SYNC <token>                        -> '# sync token=<token> device_us=...'

    PATTERN SIMPLE <period_us> <width_us>      (idle only)
    PATTERN SEQ <w1> <g1> <w2> <g2> ...        (idle only, up to 16 pairs)
    PATTERN SHOW                         -> '# pattern ...'

    RUN                                  free run until STOP
    RUN DURATION <us>                    auto-stop after wall-clock duration
    RUN COUNT <n>                        auto-stop after N pulses
    STOP                                 force LED LOW, return to idle

    RESET                                STOP + zero event_id, increment epoch

Limits enforced by the firmware: width >= 100 us, gap >= 100 us,
each interval <= 60_000_000 us, sequence length <= 16 pairs.

Hardware YAML
=============

::

    teensy:
      type: teensy_pulses
      port: COM7              # /dev/ttyACM0 on Linux
      baudrate: 115200
      development_mode: false # true -> no port opened, send_line no-ops
"""

from __future__ import annotations

import re
import time
from typing import Iterable, List, Optional, Sequence, Tuple

from mesofield import DeviceRegistry
from mesofield.devices import BaseSerialDevice


# Pulse line:    LED,<device_us>,<event_id>
_PULSE_RE = re.compile(r"^LED,(\d+),(\d+)$")
# Status line:   # status state=<s> seq_len=<n> run_count=<c> ...
_STATUS_RE = re.compile(r"(\w+)=(\S+)")
# Sync reply:    # sync token=<t> device_us=<u>
_SYNC_RE = re.compile(r"^sync\s+token=(\S+)\s+device_us=(\d+)\s*$")


@DeviceRegistry.register("teensy_pulses")
class TeensyPulseGenerator(BaseSerialDevice):
    """Sync pulse generator driver matching firmware v3-teensy40.

    Each rising LED edge becomes a row::

        timestamp,device_us,event_id,epoch

    where ``timestamp`` is the host wall-clock at line receipt and
    ``device_us`` is the Teensy's ``micros()`` clock latched in the
    rising-edge ISR.  ``epoch`` increments on every ``RESET`` so
    ``event_id`` collisions can be disambiguated across runs.
    """

    device_type = "stimulator"
    file_type = "csv"
    bids_type = "events"

    # Firmware-enforced limits (see sync-pulse-generator v3 source).
    MIN_WIDTH_US: int = 100
    MIN_GAP_US: int = 100
    MAX_INTERVAL_US: int = 60_000_000
    MAX_SEQ_LEN: int = 16

    def __init__(self, cfg=None, **kwargs):
        super().__init__(cfg, **kwargs)
        # Firmware state mirrored from '#'-prefixed lines.
        self.firmware_banner: Optional[str] = None
        self.last_status: dict = {}
        self.last_pattern: List[Tuple[int, int]] = []
        self.last_sync: Optional[Tuple[str, int]] = None  # (token, device_us)
        self.last_dropped: int = 0
        self.epoch: int = 0
        self.is_running_firmware: bool = False

    # ------------------------------------------------------------------
    # High-level command API (callable from a Procedure or the GUI).
    # All methods are fire-and-forget; replies arrive asynchronously
    # through ``parse_line`` and update the ``last_*`` attributes.
    # ------------------------------------------------------------------

    def ping(self) -> None:
        self.send_line("PING")

    def query_status(self) -> None:
        self.send_line("STATUS")

    def query_pattern(self) -> None:
        self.send_line("PATTERN SHOW")

    def sync(self, token: Optional[str] = None) -> str:
        """Send ``SYNC <token>``; the reply lands in :attr:`last_sync`."""
        if token is None:
            token = f"host{int(time.time() * 1000)}"
        self.send_line(f"SYNC {token}")
        return token

    # -- pattern --------------------------------------------------------
    def set_pattern_simple(self, period_us: int, width_us: int) -> None:
        """Repeating single pulse: ``period_us`` cycle, ``width_us`` high.

        Firmware accepts only when device is IDLE.  Validates locally
        against the firmware's documented limits to fail fast.
        """
        period_us = int(period_us)
        width_us = int(width_us)
        if width_us < self.MIN_WIDTH_US:
            raise ValueError(f"width_us must be >= {self.MIN_WIDTH_US}")
        if period_us < width_us + self.MIN_GAP_US:
            raise ValueError(
                f"period_us must be >= width_us + {self.MIN_GAP_US}"
            )
        if period_us > self.MAX_INTERVAL_US:
            raise ValueError(f"period_us must be <= {self.MAX_INTERVAL_US}")
        self.send_line(f"PATTERN SIMPLE {period_us} {width_us}")

    def set_pattern_sequence(
        self, steps: Sequence[Tuple[int, int]]
    ) -> None:
        """Set a (width_us, gap_us) sequence (up to 16 pairs)."""
        steps = list(steps)
        if not steps:
            raise ValueError("steps must contain at least one (width,gap) pair")
        if len(steps) > self.MAX_SEQ_LEN:
            raise ValueError(f"at most {self.MAX_SEQ_LEN} pairs supported")
        for w, g in steps:
            if w < self.MIN_WIDTH_US or g < self.MIN_GAP_US:
                raise ValueError("width/gap below firmware minimum (100 us)")
            if w > self.MAX_INTERVAL_US or g > self.MAX_INTERVAL_US:
                raise ValueError(
                    f"width/gap above firmware maximum ({self.MAX_INTERVAL_US} us)"
                )
        flat: Iterable[int] = (v for pair in steps for v in pair)
        self.send_line("PATTERN SEQ " + " ".join(str(int(v)) for v in flat))

    def set_frequency(self, hz: float, duty: float = 0.5) -> None:
        """Convenience wrapper around :meth:`set_pattern_simple`.

        ``duty`` is the fractional high time (0 < duty < 1).
        """
        if hz <= 0:
            raise ValueError(f"frequency must be > 0, got {hz}")
        if not 0.0 < duty < 1.0:
            raise ValueError(f"duty must be in (0,1), got {duty}")
        period_us = int(round(1_000_000 / hz))
        width_us = max(self.MIN_WIDTH_US, int(round(period_us * duty)))
        self.set_pattern_simple(period_us, width_us)

    # -- run/stop -------------------------------------------------------
    def run(self) -> None:
        """Free-run until :meth:`stop_pulses` (or RESET)."""
        self.send_line("RUN")

    def run_for_duration(self, duration_us: int) -> None:
        if duration_us <= 0:
            raise ValueError("duration_us must be > 0")
        self.send_line(f"RUN DURATION {int(duration_us)}")

    def run_for_count(self, count: int) -> None:
        if count <= 0:
            raise ValueError("count must be > 0")
        self.send_line(f"RUN COUNT {int(count)}")

    def stop_pulses(self) -> None:
        self.send_line("STOP")

    def reset(self) -> None:
        """``STOP`` + zero ``event_id`` + increment ``epoch`` on the device."""
        self.send_line("RESET")

    # ------------------------------------------------------------------
    # BaseSerialDevice hooks
    # ------------------------------------------------------------------

    def setup_serial(self) -> None:
        """Drain the boot banner so the first run starts from a clean slate."""
        # The firmware emits its banner + status + pattern as soon as
        # USB-CDC enumerates.  Give it a moment, then ask for a fresh
        # snapshot so ``self.last_status``/``last_pattern`` are populated
        # before the experiment starts.
        time.sleep(0.1)
        self.send_line("STATUS")
        self.send_line("PATTERN SHOW")

    def parse_line(self, line: bytes) -> Optional[Tuple[dict, Optional[float]]]:
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            return None

        # ---------- pulse event ----------
        m = _PULSE_RE.match(text)
        if m is not None:
            device_us = int(m.group(1))
            event_id = int(m.group(2))
            payload = {
                "device_us": device_us,
                "event_id": event_id,
                "epoch": self.epoch,
                "device_id": self.device_id,
            }
            # ts=None -> BaseDataProducer.record stamps with host time.
            return payload, None

        # ---------- '#'-prefixed control/telemetry lines ----------
        if text.startswith("#"):
            self._handle_meta_line(text[1:].strip())
            return None

        self.logger.debug("unrecognised teensy line: %r", text)
        return None

    # ------------------------------------------------------------------
    # Internal: '#' line dispatch
    # ------------------------------------------------------------------

    def _handle_meta_line(self, body: str) -> None:
        """Update mirrored firmware state from a ``#``-prefixed line."""
        if not body:
            return

        head = body.split(None, 1)[0].lower()

        if head.startswith("sync-pulse-generator"):
            self.firmware_banner = body
            self.logger.info("teensy banner: %s", body)
            return

        if head == "status":
            self.last_status = dict(_STATUS_RE.findall(body))
            self.is_running_firmware = self.last_status.get("state") == "running"
            try:
                self.epoch = int(self.last_status.get("epoch", self.epoch))
            except ValueError:
                pass
            self.logger.debug("teensy status=%s", self.last_status)
            return

        if head == "pattern":
            self.last_pattern = [
                (int(w), int(g))
                for w, g in re.findall(r"\((\d+),(\d+)\)", body)
            ]
            self.logger.debug("teensy pattern=%s", self.last_pattern)
            return

        m = _SYNC_RE.match(body)
        if m is not None:
            self.last_sync = (m.group(1), int(m.group(2)))
            self.logger.debug("teensy sync token=%s device_us=%s",
                              *self.last_sync)
            return

        if body == "running":
            self.is_running_firmware = True
            return
        if body in ("stopped", "auto_stop"):
            self.is_running_firmware = False
            return
        if body == "pong":
            return
        if body == "pattern_set":
            return

        if body.startswith("epoch="):
            try:
                self.epoch = int(body.split("=", 1)[1])
            except ValueError:
                pass
            return

        if body.startswith("dropped="):
            try:
                self.last_dropped += int(body.split("=", 1)[1])
            except ValueError:
                pass
            self.logger.warning("teensy dropped pulses (total=%d)",
                                self.last_dropped)
            return

        if body.startswith("err="):
            self.logger.warning("teensy firmware error: %s", body[4:])
            return

        # Unknown '#' line -> log at debug.
        self.logger.debug("teensy meta: %s", body)
