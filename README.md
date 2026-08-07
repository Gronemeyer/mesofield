```
 __    __     ______     ______     ______     ______   __     ______     __         _____
/\ "-./  \   /\  ___\   /\  ___\   /\  __ \   /\  ___\ /\ \   /\  ___\   /\ \       /\  __-.
\ \ \-./\ \  \ \  __\   \ \___  \  \ \ \/\ \  \ \  __\ \ \ \  \ \  __\   \ \ \____  \ \ \/\ \
 \ \_\ \ \_\  \ \_____\  \/\_____\  \ \_____\  \ \_\    \ \_\  \ \_____\  \ \_____\  \ \____-
  \/_/  \/_/   \/_____/   \/_____/   \/_____/   \/_/     \/_/   \/_____/   \/_____/   \/____/
```

Mesofield is a PyQt6-based framework for running real-time, multi-camera
neuroscience experiments. It coordinates hardware via serial connections
and MicroManager (through [pymmcore-plus](https://pymmcore-plus.github.io/pymmcore-plus/)
custom `MDAEngine`s and multi-`CMMCorePlus` instancing) and manages
experiment configuration, acquisition orchestration, and data logging.
Cameras run on either the MicroManager or an OpenCV backend.

Acquisitions writes an `AcquisitionManifest` declaring what files saved
according to the `mesokit-schema` type contract.

The project is aimed at laboratory use and is not a full production
package; some specialised knowledge of device hardware and
MicroManager device configuration is necessary to get started.

<img width="1920" height="1080" alt="Mesofield acquisition window" src="https://github.com/user-attachments/assets/151196ab-2d74-4644-85b7-c4facf3b779a" />

---

## Documentation

Documentation lives at **[gronemeyer.github.io/mesofield](https://gronemeyer.github.io/mesofield/)**
and is split by audience:

- **[Tutorial](https://gronemeyer.github.io/mesofield/tutorial.html)** —
  the fastest path from a fresh conda env to a working acquisition on
  your hardware.
- **[User Guide](https://gronemeyer.github.io/mesofield/user_guide.html)** —
  for experimenters running acquisitions: launching the GUI, writing
  `experiment.json`, interpreting the on-disk output.
- **[Developer Guide](https://gronemeyer.github.io/mesofield/developer_guide.html)** —
  for developers extending mesofield: custom devices, `Procedure`
  subclasses, frame processors, threading models.
- **[API Reference](https://gronemeyer.github.io/mesofield/api/index.html)** —
  auto-generated from docstrings.

---

## Quick start

```bash
conda create -n mesofield python=3.12 -y
conda activate mesofield
pip install -e .
```

or

```bash
pip install mesofield
```

Register this machine's hardware once, then scaffold an experiment against it:

```bash
mesofield rig new my-rig          # write a hardware.yaml template to edit
mesofield rig list                # show rigs registered on this machine
mesofield init my-experiment      # scaffold an experiment (--rig my-rig to skip the prompt)
```

Launch the acquisition GUI by pointing at a rig name or a path:

```bash
mesofield launch dev                        # mock rig, no hardware required
mesofield launch my-rig                     # a registered rig by name
mesofield launch path/to/experiment/        # dir: procedure.py + experiment.json + hardware.yaml
mesofield launch path/to/experiment.json    # rig + params (sibling hardware.yaml auto-detected)
```

A scaffolded experiment can also be run headless with `python procedure.py`.

Beyond acquisition, the CLI is grouped by task — run `mesofield <group> --help`:

| Command | Purpose |
| --- | --- |
| `mesofield launch \| init \| playback \| viewer` | acquisition workflow |
| `mesofield rig ...` | manage this machine's canonical `hardware.yaml` rigs |
| `mesofield datakit ...` | build, explore, profile, and inspect datasets |
| `mesofield process ...` | batch-process and convert recorded data |
| `mesofield tools ...` | setup, export, and diagnostic utilities |

For end-to-end setup, follow the
[Tutorial](https://gronemeyer.github.io/mesofield/tutorial.html).

---

## System requirements

Tested on Windows 10/11. For multi-camera acquisition with large files
we recommend ≥ 32 GB RAM, a 12th-gen Intel i7 or equivalent, and fast
local NVMe storage for the experiment directory.

---

## License

MIT — see `LICENSE`.
