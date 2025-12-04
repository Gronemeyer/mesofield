```
 __    __     ______     ______     ______     ______   __     ______     __         _____    
/\ "-./  \   /\  ___\   /\  ___\   /\  __ \   /\  ___\ /\ \   /\  ___\   /\ \       /\  __-.  
\ \ \-./\ \  \ \  __\   \ \___  \  \ \ \/\ \  \ \  __\ \ \ \  \ \  __\   \ \ \____  \ \ \/\ \ 
 \ \_\ \ \_\  \ \_____\  \/\_____\  \ \_____\  \ \_\    \ \_\  \ \_____\  \ \_____\  \ \____- 
  \/_/  \/_/   \/_____/   \/_____/   \/_____/   \/_/     \/_/   \/_____/   \/_____/   \/____/ 
                                                                                  
```

Mesofield is a small framework for managing neuroscience hardware.  It controls
cameras, encoders and other instruments through serial connections and
[MicroManager](https://micro-manager.org/) using
[pymmcore-plus](https://pymmcore-plus.github.io/pymmcore-plus/). The project is aimed at laboratory use and is not a
full production package.


## Overview

Experiments are driven by `mesofield.base.Procedure`.  Each procedure
owns a configuration object, `~mesofield.config.ExperimentConfig`, which
loads the ``hardware.yaml`` via `mesofield.hardware.HardwareManager`.
This registry stores parameters such as ``subject``, ``session`` and user-definable 
parameters stored and loaded via JSON.  GUI widgets like
`mesofield.gui.controller.ConfigController` expose these values for
interactive editing.

Main components:

- **ExperimentConfig** – configuration registry with methods such as
  `load_json`, `build_sequence` and `make_path`.
- **HardwareManager** – creates devices from the YAML configuration and
  exposes them via attributes like `cameras`, `encoder` and `nidaq`.
- **DataManager** – collects device output using a thread safe
    `DataQueue` and can log entries with `start_queue_logger`. Queue logs are
        written as `_dataqueue.json` files defined in `mesofield.schema.output`
        (schema version 2.0), which captures experiment headers, hardware summaries,
        individual samples, and roll-up stats for quick inspection. DataManager owns a `DataSaver` and
    `DataPaths` for managing data saving and pathing, respectively.
- **Procedure** – high level experiment runner defined in
  `mesofield.base`.  It coordinates the configuration, hardware and data
  manager.  Use `create_procedure` to build a procedure instance.

Mesofield uses PyQt6 and has only been tested on various Windows 10/11.
Multi-camera setups prodcuing large experimental files require modern
computing hardware. I recommend having at least 32gb RAM and a 12th
generation i7 core or equivalent.

An attemp tat universal logging and exception handling has been made;
all logs and uncaught exceptions are written to `logs/mesofield.log`
using a standardized logger.

## Development Setup

```bash
conda create -n mesofield python=3.13
conda activate mesofield
pip install -r requirements.txt
```

## Launch from JSON Configuration

In your terminal, launch Mesofield like so:

```python
python -m mesofield launch --config "path/to/json_config"
```

Below is a sample config. The Subjects section is vital,
Mesofield uses this information to build a BIDS-compliant
directory. Also vital (like any good config) are the paths
in the Configuration section:

```json
{
    "Configuration": {
        "experimenter": "you",
        "protocol": "HFSA",
        "experiment_directory": "/where/mesofield/builds_outputs",
        "hardware_config_file": "path/to/hardware.yaml",
        "database_path": "where/mesofield/builds_h5_database/database.h5",
        "duration": 1000,
    },
    "Subjects": {
        "STREHAB07": {
            "sex": "F",
            "session": "01",
            "task": "mesoscope"
        },
        "STREHAB09": {
            "sex": "M",
            "session": "01",
            "task": "mesoscope"
        }
    },
    "DisplayKeys": [
        "subject",
        "session",
        "task",
        "experimenter",
        "protocol",
        "duration",
        "start_on_trigger",
        "led_pattern"
    ]
}
```

The `ConfigController` uses the `DisplayKeys` list, when present,
to load and display in the `ConfigFormWidget`.  Only these values
are written back to the JSON file when a `Procedure` finishes.

Running a procedure will persist any modified configuration values back to
the JSON file, keeping it in sync with the session counter and other fields.

Mesofield supports adding as many parameters as you want, which will
get saved back to the JSON, and used as you so please.

### MousePortal plugin configuration

MousePortal now ships with a single unified YAML file, `portal_config.yaml`, that
contains the former JSON plugin parameters *and* the VR corridor/sequence
definition. Your `ExperimentConfig` JSON only needs to enable the plugin and
point to that YAML:

```json
"Plugins": {
    "mouse_portal": {
        "enabled": true,
        "module": "mesofield.plugins.mouseportal",
        "config": {
            "unified_config_path": "C:/dev/mesofield/mesofield/subprocesses/portal_config.yaml"
        }
    }
}
```

Inside the YAML you'll find:

- `metadata` and `visual_diagram` to document the scene.
- `plugin` for device id, ZeroMQ endpoints, runtime path, and readiness tuning.
- Rendering knobs (`window_width`, `textures`, etc.) used by the portal process.
- `engine` describing the experimental blocks and trials, which is converted to a
    runtime spec automatically when the plugin launches.

See `docs/templates/mouseportal/README.md` for a walkthrough and a friendly
template to fork for your own experiments.

## Custom Hardware

Hardware classes implement the protocols defined in
`mesofield.protocols`.  A minimal data producing device looks like this:

```python
import serial
from mesofield.protocols import DataAcquisitionDevice

class SerialSensor(DataAcquisitionDevice):
    device_type = "arduino"
    device_id = "temp"

    def initialize(self):
        self.ser = serial.Serial("COM3", 9600)

    def start(self) -> bool:
        return True

    def stop(self) -> bool:
        self.ser.close()
        return True

    def get_data(self):
        return float(self.ser.readline())
```

Register the class with `mesofield.DeviceRegistry.register("sensor")` and add an
entry to your `hardware.yaml` so that `HardwareManager` can instantiate it.

## Threading Models

Devices can run in different concurrency models:

```python
from PyQt6.QtCore import QThread
from mesofield.mixins import ThreadingHardwareDeviceMixin

class QtDevice(QThread):
    device_type = "qt_device"
    device_id = "qdev"
    def run(self):
        ...  # Qt loop

class ThreadedDevice(ThreadingHardwareDeviceMixin):
    device_type = "thread_device"
    device_id = "tdev"
    def _run(self):
        ...  # standard thread
```

Asynchronous devices can be implemented with `asyncio` while still
providing the protocol methods.

## Using the Console

Press **Toggle Console** to open the embedded IPython terminal.  The names
inserted by :func:`MainWindow.initialize_console` include:

- ``self`` – the application window
- ``procedure`` – the running :class:`mesofield.base.Procedure`
- ``data`` – the :mod:`mesofield.data` package

Access the configuration as ``procedure.config``.  Type
``procedure.config.hardware`` to inspect loaded devices and use tab completion
on ``procedure.config.`` to discover parameters.
