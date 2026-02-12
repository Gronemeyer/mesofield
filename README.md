```
 __    __     ______     ______     ______     ______   __     ______     __         _____
/\ "-./  \   /\  ___\   /\  ___\   /\  __ \   /\  ___\ /\ \   /\  ___\   /\ \       /\  __-.
\ \ \-./\ \  \ \  __\   \ \___  \  \ \ \/\ \  \ \  __\ \ \ \  \ \  __\   \ \ \____  \ \ \/\ \
 \ \_\ \ \_\  \ \_____\  \/\_____\  \ \_____\  \ \_\    \ \_\  \ \_____\  \ \_____\  \ \____-
  \/_/  \/_/   \/_____/   \/_____/   \/_____/   \/_/     \/_/   \/_____/   \/_____/   \/____/
```

Mesofield is a PyQt6-based framework for running real-time, multi-camera neuroscience experiments. It coordinates hardware via serial connections and MicroManager (through [pymmcore-plus](https://pymmcore-plus.github.io/pymmcore-plus/) custom `MDAEngines` and multi `CMMCorePlus` object instancing) and manages experiment configuration, acquisition orchestration, and data logging. The project is aimed at laboratory use and is not a full production package; some specialized knowledge of device hardware and hardware classes/MicroManager device configuration are necessary to getting started. 

<img width="2454" height="1592" alt="Screenshot 2025-07-10 161939" src="https://github.com/user-attachments/assets/151196ab-2d74-4644-85b7-c4facf3b779a" />

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Key Concepts](#key-concepts)
- [Installation](#installation)
- [Experiment Setup](#experiment-setup)
  - [1) Create Hardware Configuration](#1-create-hardware-configuration)
  - [2) Create Experiment Configuration](#2-create-experiment-configuration)
  - [3) Launch Mesofield](#3-launch-mesofield)
- [Custom Hardware Devices](#custom-hardware-devices)
- [Threading Models](#threading-models)
- [Using the Embedded Console](#using-the-embedded-console)
- [Logging](#logging)
- [System Requirements](#system-requirements)

---

## Architecture Overview

Mesofield organizes experiments around four core components:

1. **`Procedure`** (`mesofield.base.Procedure`)  
   The high-level experiment runner. It coordinates hardware initialization, configuration, acquisition, and data saving.

2. **`ExperimentConfig`** (`mesofield.config.ExperimentConfig`)  
   A registry for all experiment parameters. It loads JSON configuration data, builds subject/session paths, and synchronizes updated values back to disk at the end of each run.

3. **`HardwareManager`** (`mesofield.hardware.HardwareManager`)  
   Builds hardware devices from a YAML configuration and exposes them as attributes (e.g., `cameras`, `encoder`, `nidaq`). It owns the lifecycle of hardware devices (initialize → start → stop).

4. **`DataManager`** (`mesofield.data.DataManager`)  
   Collects data from devices using a thread-safe queue and manages structured saving via `DataSaver` and `DataPaths`. It can also log queue output with `start_queue_logger`.

A typical run flows as:

```
JSON Experiment Config
        ↓
ExperimentConfig
        ↓
HardwareManager → Devices
        ↓
DataManager → DataQueue → DataSaver
        ↓
Procedure (orchestrates everything)
```

The launch command runs:
```python
procedure = Procedure(json_config_path)
mesofield = MainWindow(procedure)
mesofield.show()
splash.finish(mesofield)
app.exec()
```

---

## Key Concepts

- **Hardware configuration (YAML)** defines *what hardware exists* and how to connect to it.
- **Experiment configuration (JSON)** defines *what the experiment does*, including subjects, sessions, paths, and runtime parameters.
    - **DisplayKeys** determine which configuration values are editable in the GUI and saved back to JSON.

---

## Installation

```bash
conda create -n mesofield python=3.13
conda activate mesofield
pip install -r requirements.txt
```

---

## Experiment Setup

### 1) Create Hardware Configuration

Unfortunately, this step requires the most cognitive time and specialized knowledge; it is not yet "user-friendly". Further development here is within the scope of the project as it becomes tailored toward broader distribution. 

Create a `hardware.yaml` that describes the devices you want Mesofield to control. Currently, this requires knowledge of the device classes and/or MicroManager configurations of hardware. [pymmcore-plus](https://pymmcore-plus.github.io/pymmcore-plus/) is used to interface with MicroManager hardware devices. Otherwise, the `HardwareManager` will reference locally defined device classes. 

**Example structure used for development in /tests:**
```yaml
memory_buffer_size: 10000
viewer_type: "static"

encoder:
  type: "wheel"
  port: "COM4"
  baudrate: 57600
  cpr: 2400
  diameter_mm: 80
  sample_interval_ms: 20
  development_mode: False

cameras:
  - id: "dev"
    name: "devcam1"
    backend: "micromanager"
    micromanager_path: "C:/Program Files/Micro-Manager-2.0"

  - id: "dev"
    name: "devcam2"
    backend: "micromanager"
    micromanager_path: "C:/Program Files/Micro-Manager-2.0"
    output:
      bids_type: "func"
      file_type: "mp4"
```

Mesofield will load this file through `HardwareManager` and make the devices available in `procedure.config.hardware`.

---

### 2) Create Experiment Configuration

Create a JSON file that defines experiment metadata, paths, and subject details.

```json
{
    "Configuration": {
        "experimenter": "you",
        "protocol": "HFSA",
        "experiment_directory": "/where/mesofield/builds_outputs", [optional, can be derived from json file parent directory]
        "hardware_config_file": "path/to/hardware.yaml",           [optional, can be derived from json file parent directory]
        "duration": 1000        --> translates to duration x camera.fps = frames sent to MDASequence
    },
    "Subjects": {
        "STREHAB07": {          --> BIDS sub-STREHAB07
            "sex": "F",
            "session": "01",    --> BIDS ses-01
            "task": "mesoscope" --> BIDS task-mesoscope
        },                      --> {datetime}_sub-STREHAB07_ses-01_task-mesoscope_{filetype.suffix}
        "STREHAB09": {
            "sex": "M",
            "session": "01",
            "task": "mesoscope"
        }
    },
    "DisplayKeys": [            --> GUI widget hot config params
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

**Notes:**
- The `Subjects` section is used to create BIDS-style output directories.
- `DisplayKeys` controls what appears in the GUI and what is saved back to the JSON on completion.
- Any new keys you add are preserved and saved back to disk.

---

### 3) Launch Mesofield

```bash
python -m mesofield launch --config "path/to/json_config"
```

Mesofield will load the config, initialize hardware, and open the GUI for interactive control.

---

## Custom Hardware Devices

Hardware classes implement protocols defined in `mesofield.protocols`. A minimal data-producing device might look like:

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

Register the class with `mesofield.DeviceRegistry.register("sensor")` and add a corresponding entry in `hardware.yaml`.

---

## Threading Models

Devices can be implemented using different concurrency models:

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

Asynchronous devices can also be written using `asyncio` while still implementing the expected protocol methods.

---

## Using the Embedded Console

Press **Toggle Console** to open the IPython terminal. Available names include:

- `self` – the application window
- `procedure` – the active `mesofield.base.Procedure`
- `data` – the `mesofield.data` package

Use `procedure.config` to inspect or modify configuration values.

---

## Logging

Logging is a core part of Mesofield and is designed to make long-running, multi-device experiments observable and debuggable.

**What gets logged**
- **All application logs** (from Mesofield and its subsystems) go through a centralized logger.
- **Uncaught exceptions** are captured and written to the same log file via a global exception hook.
- Devices and data-saving operations also emit informational and error messages (e.g., configuration saves, output paths, or failures).

**Where logs live**

```
logs/mesofield.log
```

- By default, logs are written to `logs/mesofield.log` in the project root.
- The log directory is created automatically if it does not exist.

**Rotation & retention**
- Logs rotate **daily at midnight**.

**Console vs file**
- The console shows **colored logs** at the configured level (default: `INFO`).
- The log file captures **all levels** (including `DEBUG`) for full traceability.

**Log format**
Each entry is formatted consistently to support scanning and grepping:
```
HH:MM:SS | LEVEL    | [logger.name] --> message
```

**Noise reduction**
- Commonly chatty libraries (e.g., `matplotlib`, `asyncio`, `traitlets`) are forced to `WARNING` or above to keep logs clean.

If you need to change log location or verbosity, look at `mesofield/utils/_logger.py` where the centralized setup is defined.



---

## System Requirements

Mesofield has been tested on Windows 10/11.  
For multi-camera acquisition with large files, we recommend:

- **≥ 32 GB RAM**
- **12th-gen Intel i7 or equivalent**
```
