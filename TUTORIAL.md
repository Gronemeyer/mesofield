# Mesofield — getting started

This is the new-lab adoption path. The goal is to get from a fresh
conda env to a working acquisition + analysis on **your** hardware in
under an hour.

## 1. Install

Mesofield ships the full pipeline in one package:

- The acquisition layer (the `Procedure` class, hardware devices)
- The ingest layer (`mesofield.datakit`, formerly the separate `datakit` package)
- The processing layer (`mesofield.processing`, wrappers for DLC, mesomap, lab pipelines)
- The shared schema (`mesokit-schema`) is a direct dependency

```sh
conda create -n my-rig python=3.12 -y
conda activate my-rig
pip install -e /path/to/mesofield      # editable install during development
# pip install mesofield                # once it's on PyPI
```

For analysis-only machines (no Qt, no micromanager), the same install
works. The hardware-side deps (`PyQt6`, `pymmcore-plus`, `nidaqmx`,
`tifffile`, `pyserial`) live in the `[rig]` extra:

```sh
pip install -e /path/to/mesofield[rig]
```

## 2. Register this machine's rig (one-time setup)

A `hardware.yaml` is **rig-specific** — COM ports, camera ids, device
indices, Micro-Manager `.cfg` paths all pin it to one computer. Each machine
keeps a small store of named canonical configs (in the OS config directory):

```sh
mesofield rig new my-rig      # writes a fill-out template; edit it
mesofield rig list            # show registered rigs
# or, if you already have a working hardware.yaml:
mesofield rig add my-rig /path/to/hardware.yaml
```

Edit the template to declare this machine's real devices. You only do
this once per computer (or whenever the hardware changes).

## 3. Scaffold an experiment

```sh
mesofield init my-experiment
cd my-experiment
```

`init` asks which hardware config to use:

- a **registered rig** — its `hardware.yaml` is copied into the experiment,
- **`dev`** — mock devices, runs on any machine with no real hardware,
- **`blank`** — a fill-out template (the default).

Skip the prompt with `mesofield init my-experiment --rig my-rig`.

You get:

```
my-experiment/
  README.md          # this file's smaller sibling, customised to your experiment
  experiment.json    # subject + protocol + duration
  hardware.yaml      # copied from the chosen rig (or a template)
  procedure.py       # your Procedure subclass
  devices/
    __init__.py
    thermal_example.py  # annotated custom-device template
```

The copied `hardware.yaml` belongs to the experiment — edit it freely for
experiment-specific tweaks without touching the canonical rig file.
Pick `dev` if you just want the next step to work with no hardware.

## 4. Run the acquisition

```sh
python procedure.py
```

This calls `MyProcedure(experiment.json).run_until_finished()`. After
the duration cap fires (5s by default), mesofield:

1. Stops every device declared in `hardware.yaml`.
2. Calls `save_data()` on each producer (CSV for serial devices,
   OME-TIFF + frame-metadata sidecar for cameras).
3. Writes `data/sub-SUBJ01/ses-01/manifest.json` — the
   `AcquisitionManifest`, a typed pydantic document declaring exactly
   what landed on disk: per-producer `output_path`, `metadata_path`,
   `calibration`, `time_basis`, optional `dataqueue_schema`.

Open the manifest and you can see the contract the rest of the
pipeline reads. No globbing.

## 5. Add your real hardware

Edit `hardware.yaml`:

```yaml
wheel:
  type: wheel             # was mock_wheel
  primary: true
  port: /dev/ttyUSB0
  baudrate: 57600
  cpr: 2400
  diameter_mm: 80
  output:
    suffix: wheel
    file_type: csv
    bids_type: beh

camera:
  type: camera            # MMCamera (micromanager backend)
  name: dev               # micromanager device label
  backend: micromanager
  output:
    suffix: meso
    file_type: ome.tiff
    bids_type: func
```

Built-in device types: `camera`, `opencv_camera`, `wheel`, `encoder`
(treadmill), `psychopy`, `nidaq`, plus the `mock_wheel` and
`mock_camera` examples.

## 6. Add a lab-specific device

The scaffold ships `devices/thermal_example.py` as a working template.
The pattern: one Python file with both halves of the contract — the
producer (what writes data) and the parser (what reads it back).

```python
# devices/thermal_example.py
from mesofield import DeviceRegistry
from mesofield.devices.base import BaseSerialDevice
from mesofield.datakit.sources.register import TimeseriesSource


@DeviceRegistry.register("thermal")
class ThermalSensor(BaseSerialDevice):
    file_type = "csv"
    bids_type = "beh"
    data_type = "thermal"

    def parse_line(self, line):
        return float(line), None  # (payload, timestamp_or_None)


class _ThermalParser(TimeseriesSource):
    tag = "thermal"
    patterns = ("**/*_thermal.csv",)

    def build_timeseries(self, path, *, context=None):
        df = pd.read_csv(path)
        return df["timestamp"].to_numpy(), df, {"source_file": str(path)}


# Bind parser to producer so manifest-driven dispatch finds it.
ThermalSensor.Parser = _ThermalParser
```

In `procedure.py`, import the module to trigger registration:

```python
from devices import thermal_example  # noqa: F401
```

In `hardware.yaml`, add a stanza:

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

Re-run `python procedure.py`. The manifest now includes a third
producer for the thermal sensor; the parser is automatically reached
via `ThermalSensor.Parser` when ingest runs.

## 7. Ingest into a dataset

```python
from mesofield.datakit import Dataset
ds = Dataset.from_directory("./")
ds.save("processed/dataset.h5", format="hdf5")
```

This walks `data/`, reads the manifests, and writes
`processed/<date>_dataset_mvp.h5` — a pandas DataFrame with a
`(Subject, Session, Task)` MultiIndex and `(Source, Signal)` columns.

## 8. Analyze with databench (optional)

```sh
pip install databench
```

```python
from databench import Project
proj = Project(dataset="processed/dataset_mvp.h5")
session = proj.session(subject="SUBJ01", session="01", task="demo")
```

## 9. Intermediate processing (DLC, mesomap, custom)

For any file-to-file transformation between acquisition and ingest:

```python
from mesofield.processing import ProcessorRunner

class MyPreprocessor(ProcessorRunner):
    tool_name = "my_preprocessor"
    tool_version = "0.1.0"

    def run(self, inputs, *, threshold=0.5):
        # ... read inputs[0], compute, write outputs ...
        return [out_path]

# Call it:
outputs, manifest = MyPreprocessor()([tiff_path], threshold=0.7)
```

The runner hashes inputs, captures parameters, and emits
`<tool_name>.process.json` next to its outputs. Re-running with
different parameters produces a different manifest hash; the
provenance chain extends past acquisition.

## Where things live

- `mesofield.base.Procedure` — orchestrates a run (lifecycle, manifest)
- `mesofield.devices.base.BaseDataProducer` / `BaseSerialDevice` — start here for new devices
- `mesofield.datakit.sources.register.TimeseriesSource` — start here for new parsers
- `mesofield.processing.ProcessorRunner` — start here for intermediate processing
- `mesokit_schema` — the manifests themselves; you rarely import these directly

## Retrofitting legacy data

If you have sessions acquired before mesofield wrote manifests:

```sh
mesofield process retrofit-manifest /path/to/experiment
```

This walks the BIDS tree, reads `timestamps.csv` and `configuration.csv`,
and synthesizes an `AcquisitionManifest` for each session. Calibration
constants aren't recoverable (they weren't written), but everything
else round-trips. The legacy sessions become contract-compliant.
