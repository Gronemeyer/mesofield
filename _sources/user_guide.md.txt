# User Guide

This guide is for **experimenters** — people running acquisitions on a
configured rig. If you're writing a new device class or subclassing
`Procedure`, see the [Developer Guide](developer_guide.md) instead.

## Overview

A mesofield experiment is described by up to three files:

| File | Owns | Usually edited by | Required? |
|------|------|-------------------|-----------|
| `hardware.yaml` | What devices exist on this rig and how to talk to them | Rig maintainer (one-time per machine) | Yes — the rig you launch |
| `experiment.json` | Subjects, sessions, protocol, duration | Experimenter (per study / per day) | Optional — load it, author it in the GUI, or script it |
| `procedure.py` | Run lifecycle hooks and custom device imports | Whoever scripts the study | Optional — see the [Developer Guide](developer_guide.md) |

The `hardware.yaml` is all you need to launch. The `mesofield` CLI brings up
the GUI and orchestrates the acquisition; experiment parameters can be loaded,
authored in the Configuration Wizard, or supplied by a scripted `Procedure`.

## Launching an acquisition

Point the CLI at your rig. The argument can be a **registered rig name**
(see `mesofield rig list`), the literal `dev` (mock devices, no hardware),
a `hardware.yaml` (rig only), an `experiment.json` (its adjacent
`hardware.yaml` is auto-detected), a scripted `procedure.py`, or a
directory containing them:

```bash
mesofield launch dev                         # mock rig — runs on any machine
mesofield launch my-rig                      # a rig registered on this machine
mesofield launch path/to/hardware.yaml       # rig only — author params in the GUI
mesofield launch path/to/experiment.json     # rig + params
python -m mesofield launch path/to/hardware.yaml   # module entry point, equivalent
```

For a directory the precedence is `procedure.py` → `experiment.json` →
`hardware.yaml`.

This opens the main acquisition window with hardware initialised. When an
`experiment.json` is supplied its parameters populate the form; otherwise use
the Configuration Wizard to load or author one. Omit the argument entirely,
or pass `--wizard`, to open the wizard on a config that is already complete.

### Registering a rig and scaffolding an experiment

A `hardware.yaml` is machine-specific (COM ports, camera ids,
Micro-Manager `.cfg` paths). Each computer keeps a store of named rigs in
its OS config directory:

```bash
mesofield rig new my-rig                 # write a template to fill out
mesofield rig add my-rig path/to/hardware.yaml   # register an existing file
mesofield rig list                       # what's registered here
mesofield rig show my-rig                # print one
mesofield rig remove my-rig
```

`mesofield init my-experiment` then scaffolds a self-contained experiment
directory (`experiment.json`, `hardware.yaml`, `procedure.py`, `devices/`),
copying in the rig you pick — a registered rig, `dev`, or a blank template.
Use `--rig my-rig` to skip the prompt.

### Reviewing recorded data

```bash
mesofield playback path/to/experiment    # replay a recorded session in the GUI
mesofield viewer                         # standalone TIFF ROI viewer
```

`playback` accepts `--speed` and `--loop/--no-loop`.

## Experiment configuration (`experiment.json`)

```json
{
    "Configuration": {
        "experimenter": "you",
        "protocol": "HFSA",
        "experiment_directory": "/where/mesofield/writes_outputs",
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

- `experiment_directory` is optional — output defaults to the directory
  holding the JSON. A JSON with no embedded rig block falls back to a
  sibling `hardware.yaml`.
- `duration` is in seconds. The MDA sequence builds
  `primary_camera.sampling_rate × duration` frames, unless
  `num_meso_frames` is set, which overrides it.
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

Click **Add Note** at any time. Notes are timestamped and written as a
single `..._notes.txt` file in the session directory when the run
completes (only if at least one note was added).

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
                *_notes.txt
                *_timestamps.csv
                func/             # the `bids_type` declared per device
                    *_meso.ome.tiff
                    *_meso_frame_metadata.json
                beh/
                    *_wheel.csv
```

Each producer's `output.bids_type` in `hardware.yaml` decides its
subdirectory; devices with no `bids_type` write directly into the session
directory. Filenames are
`<timestamp>_sub-<id>_ses-<id>_task-<id>_<suffix>.<ext>`. If a
`manifest.json` already exists, the new one is written with a timestamp
prefix rather than overwriting.

The `manifest.json` is a typed `AcquisitionManifest` (from
`mesokit-schema`) describing every producer, its output path, its
metadata sidecar, and any calibration constants. Downstream analysis
tools read the manifest instead of globbing.

## Embedded IPython console

Toolbar → **Toggle Console**. The kernel pre-binds:

- `self` — the main window (`MainWindow`)
- `procedure` — the active [`Procedure`](api/generated/mesofield.base)
- `data` — the [`mesofield.data`](api/generated/mesofield.data) package
- `what_do()` — prints a short guide to the console

`procedure` is re-pushed whenever the active procedure is swapped, so the
name always refers to the current run.

Common one-liners:

```python
procedure.config.items()                    # all configuration values
procedure.config.set("duration", 600)       # change the run length
procedure.hardware.cameras                  # list configured cameras
procedure.hardware.primary                  # the camera that drives MDA
procedure.events.procedure_started.connect(my_callback)
```

## Logging

All application logs flow through one `loguru` logger and land in
`logs/mesofield.log` inside the installed `mesofield` package directory
(pass `log_dir` to `setup_logging` to relocate it).

- Rotates **daily at midnight**, keeping **7 days**.
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
