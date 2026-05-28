# User Guide

This guide is for **experimenters** — people running acquisitions on a
configured rig. If you're writing a new device class or subclassing
`Procedure`, see the [Developer Guide](developer_guide.md) instead.

## Overview

A mesofield experiment is described by two files:

| File | Owns | Usually edited by |
|------|------|-------------------|
| `hardware.yaml` | What devices exist on this rig and how to talk to them | Rig maintainer (one-time per machine) |
| `experiment.json` | Subjects, sessions, protocol, duration | Experimenter (per study / per day) |

The `mesofield` CLI loads both, brings up the GUI, and orchestrates the
acquisition.

## Launching an acquisition

The CLI installs as both a console script and a Python module entry
point; either form works:

```bash
mesofield launch path/to/experiment.json
# equivalent
python -m mesofield launch path/to/experiment.json
```

Either form opens the main acquisition window with hardware initialised
and the parameters from `experiment.json` populated in the form.

## Experiment configuration (`experiment.json`)

```json
{
    "Configuration": {
        "experimenter": "you",
        "protocol": "HFSA",
        "experiment_directory": "/where/mesofield/writes_outputs",
        "hardware_config_file": "path/to/hardware.yaml",
        "duration": 1000
    },
    "Subjects": {
        "STREHAB07": {
            "sex": "F",
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

**Field notes:**

- `experiment_directory` and `hardware_config_file` are optional — both
  default to siblings of the JSON file's parent directory.
- `duration` is in seconds. The MDA sequence builds
  `duration × camera.fps` frames.
- `Subjects` keys become BIDS `sub-<key>` directories under
  `experiment_directory/data/`. `session` and `task` become `ses-<id>`
  and `task-<id>`.
- `DisplayKeys` decides which fields appear in the editable form in the
  GUI. Edits persist back to `experiment.json` when the run completes
  (or via the **Save** button).
- Anything you add to the JSON outside of these reserved keys is
  preserved on save.

## The acquisition window

The window has three regions:

1. **Live Viewer (top-left)** — per-camera snap / live / progress
   panels. The mesoscope view sits next to the pupil view by default.
2. **Configuration form (top-right)** — the `DisplayKeys` you declared,
   plus a subject selector, **Record**, **Add Note**, and dynamic
   hardware controls (LED test, NIDAQ pulse, etc.) for whatever your
   `hardware.yaml` requested.
3. **Encoder / processor plots (bottom)** — live traces of any frame
   processor with `plot=True` and any encoder / serial device with
   `start_live_view` enabled.

The **Toggle Console** action in the toolbar opens an embedded IPython
shell with the live `procedure` bound — handy for inspecting state
mid-run.

## Notes during a run

Click **Add Note** at any time. Notes are timestamped and saved to
`data/sub-<id>/ses-<id>/notes.json` when the run completes.

## What ends up on disk

After a run, your experiment directory looks like:

```
<experiment_dir>/
    experiment.json               # updated with any DisplayKeys edits
    hardware.yaml
    data/
        sub-<id>/
            ses-<id>/
                manifest.json     # AcquisitionManifest — the contract
                notes.json
                <task>/
                    *_meso.ome.tiff
                    *_meso_frame_metadata.json
                    *_pupil.mp4
                    *_pupil_frame_metadata.json
                    *_wheel.csv
```

The `manifest.json` is a typed `AcquisitionManifest` (from
`mesokit-schema`) describing every producer, its output path, its
metadata sidecar, and any calibration constants. Downstream analysis
tools read the manifest instead of globbing.

## Embedded IPython console

Toolbar → **Toggle Console**. The kernel pre-binds:

- `self` — the main window (`MainWindow`)
- `procedure` — the active [`Procedure`](api/generated/mesofield.base)
- `data` — the [`mesofield.data`](api/generated/mesofield.data) package

Common one-liners:

```python
procedure.config.items()                    # all configuration values
procedure.config.set("duration", 600)       # change the run length
procedure.hardware.cameras                  # list configured cameras
procedure.hardware.primary                  # the camera that drives MDA
procedure.events.procedure_started.connect(my_callback)
```

## Logging

All application logs flow through one `loguru` logger and land in:

```
logs/mesofield.log
```

- Rotates **daily at midnight**.
- The console shows colourised logs at `INFO`; the file captures
  everything down to `DEBUG`.
- Uncaught exceptions are routed through the same hook so crashes leave
  a trail.
- Chatty third-party libraries (`matplotlib`, `asyncio`, `traitlets`)
  are pinned at `WARNING` or above.

To change the location or verbosity, see
[`mesofield/utils/_logger.py`](api/generated/mesofield.utils).

## System requirements

Mesofield is tested on Windows 10/11. For multi-camera acquisition with
large files we recommend:

- ≥ 32 GB RAM
- 12th-gen Intel i7 or equivalent
- Fast local storage (NVMe SSD) for the experiment directory
