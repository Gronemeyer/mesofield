"""Minimal Mesofield onboarding example: one serial device, one procedure.

A ``LickDetector`` reads ``LICK,<cap_value>`` lines from an Arduino (the
synthetic-lick firmware) and a ``LickTest`` procedure records them for
``duration`` seconds, then saves a timestamped CSV and an acquisition
manifest under the experiment directory.

This is a self-contained script -- copy it into any folder and run it:

    mesofield serial-ports          # find your Arduino's port
    python lick_detector.py         # opens the Mesofield GUI; press Record
    python lick_detector.py --headless   # run now, no window

The device is constructed like any Python object and handed to the
procedure via ``hardware=[...]``; ``proc.launch()`` opens the GUI right
from the script -- no ``mesofield launch`` needed.
"""

from __future__ import annotations


from mesofield import DeviceRegistry
from mesofield.base import Procedure
from mesofield.devices.base import BaseSerialDevice

# you can use`mesofield tools serial-ports` to find your Arduino's port and set it here
SERIAL_PORT = "/dev/cu.usbmodem101"  # <-- set to your Arduino's port


@DeviceRegistry.register("lick_detector")
class LickDetector(BaseSerialDevice):
    """Parses ``LICK,<cap_value>`` lines into timestamped samples.

    ``BaseSerialDevice`` already owns the port, the read thread, and CSV
    saving; a custom device only turns one raw line into a payload.
    """

    device_id = "lick_detector"   # names the output file + get_device(...)
    device_type = "lick_detector"
    data_type = "licks"

    def parse_line(self, line: bytes):
        text = line.decode("ascii", errors="ignore").strip()
        if not text.startswith("LICK,"):
            return None  # ignore noise / partial lines
        return {"cap_value": int(text.split(",", 1)[1])}, None  # None -> stamp now


class LickTest(Procedure):
    """A custom procedure: record licks, then report how many."""

    def on_finished(self) -> None:
        licks = self.hardware.get_device("lick_detector").get_data()
        self.logger.info(f"Recorded {len(licks)} licks")


if __name__ == "__main__":

    # Construct the device like any object, then hand it to the procedure.
    detector = LickDetector(port=SERIAL_PORT, connect_delay=2.0)

    # Everything else is a plain keyword argument -- no config files, no dict
    # keys to memorize. `experiment_directory` is where data is written
    # (relative to wherever you run this script).
    procedure = LickTest(
        hardware=[detector],
        experiment_directory="lick_test",
        subject="test01",
        session="01",
        task="lick",
        duration=20,
    )

    # Run right now with no window; the `duration` cap ends the run.
    # finished = procedure.run_until_finished()

    # Open the Mesofield GUI; press Record to start the acquisition.
    procedure.launch()
