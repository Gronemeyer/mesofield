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
The project is aimed at laboratory use and is not a full production
package; some specialised knowledge of device hardware and
MicroManager device configuration is necessary to get started.

<img width="2454" height="1592" alt="Mesofield acquisition window" src="https://github.com/user-attachments/assets/151196ab-2d74-4644-85b7-c4facf3b779a" />

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

Launch an acquisition:

```bash
mesofield launch path/to/experiment.json
```

Scaffold a new experiment:

```bash
mesofield new my-experiment
```

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
