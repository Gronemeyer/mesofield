# Developer Guide

This guide is for **developers extending mesofield** — writing custom
device adapters, subclassing `Procedure`, building frame processors, or
contributing to the framework itself. If you're just running
acquisitions, see the [User Guide](user_guide.md) instead.

## Architecture in one diagram

```
experiment.json
     │
     ▼
ExperimentConfig ─── HardwareManager ─── Devices (BaseDevice subclasses)
     │                     │                       │
     │                     │                       ├── DataProducer.signals.data ──┐
     │                     │                       │                               ▼
     │                     └── DataManager ◀───── Queues ─────── DataSaver ───--- disk
     │
     ▼
Procedure (orchestrates lifecycle, emits signals, owns the run)
     │
     ▼
MainWindow (Qt GUI; binds widgets to procedure.events)
```

The four backbone classes are:

| Class | Module | Owns |
|-------|--------|------|
| `Procedure` | [`mesofield.base`](api/generated/mesofield.base) | Run lifecycle (initialize → arm → start → finish), hooks, manifest |
| `ExperimentConfig` | [`mesofield.config`](api/generated/mesofield.config) | Parameter registry + JSON I/O + BIDS paths |
| `HardwareManager` | [`mesofield.hardware`](api/generated/mesofield.hardware) | YAML-driven device factory + lifecycle |
| `DataManager` | [`mesofield.data`](api/generated/mesofield.data) | Per-run data queues, notes, timestamps, manifest writes |

## Procedure lifecycle

```
1. initialize_hardware       — bring devices up (one-time)
2. prerun                    — subclass hook (default: no-op)
3. hardware.arm_all          — per-run prep on every device
4. connect primary.signals.finished -> _cleanup_procedure
5. on_started                — subclass hook
6. hardware.start_all
7. on_finished               — subclass hook (after primary fires finished)
8. save_data + cleanup
```

Hooks `prerun`, `on_started`, `on_finished` are no-ops on `Procedure`
itself. Override them in your subclass under
`experiments/<name>/procedure.py`.

```python
from mesofield.base import Procedure


class MyProcedure(Procedure):
    def prerun(self):
        self.logger.info(f"Subject {self.config.subject}, "
                         f"duration {self.config.duration}s")

    def on_started(self):
        # called after every device has started
        pass

    def on_finished(self):
        # called after the primary camera signals finished
        self.logger.info("Run complete; manifest will be written next")
```

`load_procedure_from_config` is the discovery hook called by the CLI; it
reads the optional `procedure_file` and `procedure_class` fields from
`experiment.json` and instantiates your subclass.

## Procedure signals (`procedure.events`)

`Procedure.events` is a [`ProcedureSignals`](api/generated/mesofield.base)
`QObject` exposing four `pyqtSignal`s:

| Signal | Payload | Fires when |
|--------|---------|-----------|
| `procedure_started` | — | After all devices have started |
| `hardware_initialized` | `bool` (success) | After `initialize_hardware` |
| `procedure_finished` | — | After cleanup completes successfully |
| `procedure_error` | `str` (message) | On any uncaught run-time error |
| `data_saved` | — | After `save_data` completes |

Connect from a Qt widget or from another device:

```python
procedure.events.procedure_started.connect(self.lock_form)
procedure.events.procedure_finished.connect(self.unlock_form)
procedure.events.procedure_error.connect(self.show_error_dialog)
```

## Custom hardware devices

A hardware device is any class that satisfies the
[`HardwareDevice`](api/generated/mesofield.protocols) protocol. The
easiest path is to subclass one of the base classes:

| Base | Use when |
|------|----------|
| `BaseDevice` | Generic device with no streaming data |
| `BaseDataProducer` | Streaming source (timeseries / counts) with a buffer |
| `BaseSerialDevice` | Streaming source whose transport is a serial port |
| `BaseCamera` | Anything that produces frames + writes a writer file |

### Minimal example — a serial sensor

```python
from mesofield import DeviceRegistry
from mesofield.devices.base import BaseSerialDevice


@DeviceRegistry.register("thermal")
class ThermalSensor(BaseSerialDevice):
    """One-byte-per-sample thermal probe over serial."""

    file_type = "csv"
    bids_type = "beh"
    data_type = "thermal"

    def parse_line(self, line: bytes):
        """Parse one serial frame.

        Returns:
            ``(payload, timestamp_or_None)`` — the payload is whatever
            you want fanned out on ``self.signals.data``; pass ``None``
            for the timestamp to let the framework stamp it.
        """
        return float(line), None
```

Then in `hardware.yaml`:

```yaml
thermal:
  type: thermal
  port: /dev/ttyUSB1
  baudrate: 115200
  output:
    suffix: thermal
    file_type: csv
    bids_type: beh
```

`@DeviceRegistry.register("thermal")` is what binds the YAML `type:`
string to the class. The decorator also stamps `registry_key` onto the
class so any instance can report its YAML type for hardware export.

### Camera-shaped devices

For anything that produces frames, subclass
[`BaseCamera`](api/generated/mesofield.devices.base_camera) — it
defaults to OME-TIFF output via `CustomWriter`, exposes a `snap()` /
`start_live()` / `stop_live()` contract for the GUI preview, and
plumbs frame metadata into the manifest. The
[`MMCamera`](api/generated/mesofield.devices.cameras) and
[`OpenCVCamera`](api/generated/mesofield.devices.cameras) classes
are the two concrete implementations to read for reference.

## Threading models

Devices can use any concurrency model that respects the lifecycle:

```python
# Qt-thread device (camera, GUI-driven serial)
from PyQt6.QtCore import QThread
from mesofield.protocols import HardwareDevice

class QtDevice(QThread):
    device_type = "qt_device"
    device_id   = "qdev"
    def run(self): ...     # Qt thread loop


# Python threading device
from mesofield.protocols import ThreadedHardwareDevice

class ThreadedDevice(ThreadedHardwareDevice):
    device_type = "thread_device"
    device_id   = "tdev"
    def _run(self): ...    # standard daemon thread


# asyncio device
from mesofield.protocols import AsyncioHardwareDevice

class AsyncDevice(AsyncioHardwareDevice):
    device_type = "async_device"
    device_id   = "adev"
    async def _run(self): ...
```

Pick whichever fits your hardware best — the framework only cares about
the protocol, not the concurrency model.

## Frame processors

For per-frame compute (mean intensity, ROI tracking, anything that
turns an ndarray into a scalar), use the `@processor` decorator on a
`Procedure` method:

```python
from mesofield.base import Procedure, processor


class MyProcedure(Procedure):
    @processor(camera="meso", plot=True, label="Frame mean", y_range=(0, 65535))
    def frame_mean(self, img, idx, ts):
        return float(img.mean())
```

The framework wraps the function in a
[`FrameProcessor`](api/generated/mesofield.processors), attaches it to
the camera whose `device_id` matches `"meso"`, registers it with
`DataManager`, and (when `plot=True`) adds a live
[`SerialWidget`](api/generated/mesofield.gui.speedplotter) to the GUI.

Recognised `plot_kwargs`: `label`, `value_label`, `value_units`,
`y_range`, `value_scale`, `max_points`.

## Scaffolding a new experiment

The CLI scaffold drops a fill-out template:

```bash
mesofield new my-experiment --rig my-rig
cd my-experiment
```

You get:

```
my-experiment/
    README.md
    experiment.json      # subjects, duration, DisplayKeys
    hardware.yaml        # copied from the selected rig
    procedure.py         # your Procedure subclass
    devices/
        __init__.py
        thermal_example.py  # annotated custom-device template
```

`--rig` selects from `mesofield rig list`. Use `--hardware path/to/file`
to use an explicit YAML; omit both to enter an interactive picker.

## Rig store

A `hardware.yaml` is rig-specific (COM ports, camera ids, MM `.cfg`
paths). Each machine keeps a small store of named canonical configs in
the OS config directory:

```bash
mesofield rig new my-rig             # writes a fill-out template
mesofield rig list                   # show registered rigs
mesofield rig add my-rig file.yaml   # adopt an existing yaml
```

The store lives at the platform-default config location (resolved by
`platformdirs`); `mesofield rig where` prints the path.

## Logging

Use the project logger inside your code:

```python
from mesofield.utils._logger import get_logger

logger = get_logger(__name__)
logger.info("...")
```

Every device and processor should use a `__name__`-scoped logger so the
file traces are easy to grep.

## Where to look for examples

- [`mesofield/devices/mocks.py`](api/generated/mesofield.devices.mocks)
  — mock serial encoder + mock camera. Read these first; they're the
  simplest concrete implementations of every base class.
- [`mesofield/devices/cameras.py`](api/generated/mesofield.devices.cameras)
  — `MMCamera` (Micro-Manager backend) and `OpenCVCamera` (capture
  thread + MP4 writer). The two shapes most custom cameras start from.
- [`mesofield/scaffold/experiment.py`](api/generated/mesofield.scaffold)
  — what the `mesofield new` CLI emits. Reading the templates is a
  shortcut to understanding the expected file shape.
- [`mesofield/processors/frame_mean.py`](api/generated/mesofield.processors)
  — three-line frame processor. The minimum viable processor.

## Retrofitting legacy sessions

Sessions acquired before manifests landed can be retrofitted:

```bash
mesofield retrofit-manifest /path/to/experiment
```

This walks the BIDS tree, reads `timestamps.csv` / `configuration.csv`,
and synthesises an `AcquisitionManifest` per session. Calibration
constants aren't recoverable (they weren't written), but everything
else round-trips and downstream tools become happy again.
