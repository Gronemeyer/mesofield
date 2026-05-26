"""Scaffold a new experiment directory.

The `mesofield init` CLI command (see :mod:`mesofield.__main__`) calls
:func:`scaffold_experiment` to generate a working layout new lab users
can run, edit, and extend without writing anything from scratch:

    my-experiment/
      experiment.json   # session / protocol / duration
      hardware.yaml     # device stanzas
      procedure.py      # Procedure subclass (Python entry point)
      devices/
        __init__.py
        thermal_example.py   # annotated custom-device template

The `hardware.yaml` is supplied by the caller (``mesofield init``):

  - a ``Path`` to a canonical rig file -> copied in verbatim,
  - ``"dev"``    -> a mock config (:class:`MockEncoderDevice`) that runs
    out of the box on any machine,
  - ``"blank"``  -> a commented real-hardware template to fill out.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Union


def scaffold_experiment(
    target: Path,
    *,
    name: Optional[str] = None,
    force: bool = False,
    hardware: Union[Path, str] = "blank",
) -> Path:
    """Generate a runnable experiment layout at `target`. Returns the dir.

    ``hardware`` selects the ``hardware.yaml`` written into the experiment:
    a :class:`~pathlib.Path` (copied verbatim from a canonical rig file),
    ``"dev"`` (mock config), or ``"blank"`` (fill-out template).
    """
    target = Path(target).resolve()
    if target.exists() and any(target.iterdir()):
        if not force:
            raise FileExistsError(
                f"{target} is not empty. Pass --force to overwrite, or pick an empty dir."
            )
    target.mkdir(parents=True, exist_ok=True)
    (target / "devices").mkdir(exist_ok=True)

    protocol = name or _protocol_from_dir(target)
    files = {
        target / "experiment.json": _experiment_json(protocol),
        target / "procedure.py": _procedure_py(protocol),
        target / "devices" / "__init__.py": "",
        target / "devices" / "thermal_example.py": _thermal_example_py(),
        target / "README.md": _readme(protocol, target.name),
    }
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    # hardware.yaml: copy a canonical rig file, or write a built-in template.
    hardware_dst = target / "hardware.yaml"
    if isinstance(hardware, Path):
        shutil.copyfile(hardware, hardware_dst)
    elif hardware == "dev":
        hardware_dst.write_text(_hardware_yaml_mock(), encoding="utf-8")
    elif hardware == "blank":
        hardware_dst.write_text(hardware_yaml_template(), encoding="utf-8")
    else:
        raise ValueError(
            f"hardware must be a Path, 'dev', or 'blank'; got {hardware!r}"
        )
    return target


def _protocol_from_dir(target: Path) -> str:
    return target.name.upper().replace("-", "_").replace(" ", "_")


def _experiment_json(protocol: str) -> str:
    return (
        "{\n"
        '    "Configuration": {\n'
        '        "experimenter": "your-name",\n'
        f'        "protocol": "{protocol}",\n'
        '        "duration": 5,\n'
        '        "start_on_trigger": false\n'
        "    },\n"
        '    "procedure_file": "procedure.py",\n'
        '    "procedure_class": "MyProcedure",\n'
        '    "Subjects": {\n'
        '        "SUBJ01": {\n'
        '            "session": "01",\n'
        '            "task": "demo"\n'
        "        }\n"
        "    },\n"
        '    "DisplayKeys": [\n'
        '        "subject", "session", "task",\n'
        '        "experimenter", "protocol", "duration"\n'
        "    ]\n"
        "}\n"
    )


def _hardware_yaml_mock() -> str:
    """The ``dev`` hardware config: a mock encoder, runs without hardware."""
    return (
        "# Hardware configuration for this experiment (DEV / mock).\n"
        "#\n"
        "# Each top-level stanza (other than the reserved keys at the bottom)\n"
        "# is a device. `type:` selects which DataProducer class instantiates\n"
        "# it (registered via @DeviceRegistry.register(\"...\")).\n"
        "#\n"
        "# Built-in device types include: camera (micromanager), opencv_camera,\n"
        "# wheel, encoder, psychopy, nidaq, mock_wheel, mock_camera.\n"
        "#\n"
        "# Add a lab-specific device by creating devices/<name>.py with a\n"
        "# DataProducer subclass and a nested `Parser(TimeseriesSource)`,\n"
        "# then reference its registered `type:` here.\n"
        "memory_buffer_size: 1000\n"
        "\n"
        "# Mock encoder so this runs without real hardware.\n"
        "# Replace with your real device when ready (`type: wheel` etc.).\n"
        "wheel:\n"
        "  type: mock_wheel\n"
        "  primary: true\n"
        "  sample_interval_ms: 50\n"
        "  cpr: 2400\n"
        "  diameter_mm: 80\n"
        "  output:\n"
        "    suffix: wheel\n"
        "    file_type: csv\n"
        "    bids_type: beh\n"
    )


def hardware_yaml_template() -> str:
    """A commented real-hardware skeleton that must be filled out before use.

    Used both for ``mesofield init`` with the ``blank`` choice and for
    ``mesofield rig new`` (a fresh canonical rig file to edit).
    """
    return (
        "# Canonical hardware configuration for this rig -- FILL THIS OUT.\n"
        "#\n"
        "# Each top-level stanza (other than the reserved keys) is a device.\n"
        "# `type:` selects the DataProducer class (registered via\n"
        "# @DeviceRegistry.register(\"...\")). Built-in types: camera\n"
        "# (micromanager), opencv_camera, wheel, encoder, psychopy, nidaq,\n"
        "# mock_wheel, mock_camera.\n"
        "#\n"
        "# Exactly one device must be flagged `primary: true`.\n"
        "# Replace every <PLACEHOLDER> below with this machine's real values.\n"
        "memory_buffer_size: 1000\n"
        "\n"
        "# --- Running wheel encoder (Arduino/Teensy over USB-serial) --------\n"
        "wheel:\n"
        "  type: wheel\n"
        "  primary: true\n"
        "  port: \"COM?\"            # <PLACEHOLDER> e.g. COM4 (Win) or /dev/ttyUSB0\n"
        "  baudrate: 115200\n"
        "  cpr: 2400\n"
        "  diameter_mm: 80\n"
        "  output:\n"
        "    suffix: wheel\n"
        "    file_type: csv\n"
        "    bids_type: beh\n"
        "\n"
        "# --- Camera (uncomment and adapt ONE of the following) -------------\n"
        "# OpenCV / UVC camera:\n"
        "# mesoscope:\n"
        "#   type: opencv_camera\n"
        "#   id: mesoscope\n"
        "#   name: mesoscope\n"
        "#   backend: opencv\n"
        "#   device_index: 0          # <PLACEHOLDER> OpenCV VideoCapture index\n"
        "#   fps: 30\n"
        "#   output:\n"
        "#     suffix: mesoscope\n"
        "#     file_type: mp4\n"
        "#     bids_type: func\n"
        "#\n"
        "# Micro-Manager camera:\n"
        "# pupil:\n"
        "#   type: camera\n"
        "#   id: pupil\n"
        "#   name: pupil\n"
        "#   backend: micromanager\n"
        "#   configuration_path: \"<PLACEHOLDER>/MMConfig.cfg\"\n"
        "#   output:\n"
        "#     suffix: pupil\n"
        "#     file_type: ome.tiff\n"
        "#     bids_type: func\n"
    )


def _procedure_py(protocol: str) -> str:
    return (
        f'"""Procedure for the `{protocol}` experiment.\n'
        '\n'
        'The base mesofield.Procedure handles the AcquisitionManifest contract\n'
        'for you. Override `prerun`, `on_started`, `on_finished` to add\n'
        'experiment-specific behavior. Override `manifest_extra()` to attach\n'
        'session-level metadata to the manifest (LED pattern, model version,\n'
        'etc).\n'
        '\n'
        'To run::\n'
        '\n'
        '    mesofield launch experiment.json     # GUI\n'
        '    python -m experiment_script          # headless (see __main__)\n'
        '"""\n'
        '\n'
        'from __future__ import annotations\n'
        '\n'
        'import threading\n'
        '\n'
        'from mesofield import DeviceRegistry\n'
        'from mesofield.base import Procedure\n'
        'from mesofield.devices.mocks import MockEncoderDevice\n'
        '\n'
        '# Register any custom devices you have added under devices/ here.\n'
        '# Built-in device types (camera, wheel, encoder, psychopy, nidaq)\n'
        '# are already registered by mesofield itself.\n'
        'DeviceRegistry._registry.setdefault("mock_wheel", MockEncoderDevice)\n'
        '\n'
        '\n'
        'class MyProcedure(Procedure):\n'
        '    """Your experiment\'s procedure logic.\n'
        '\n'
        '    Lifecycle (subclass hooks in **bold**):\n'
        '      1. initialize_hardware\n'
        '      2. **prerun**            -- override for per-run setup\n'
        '      3. hardware.arm_all\n'
        '      4. **on_started**        -- start timers, mark events\n'
        '      5. hardware.start_all\n'
        '      6. ... (acquisition runs until primary device finishes)\n'
        '      7. save_data + _write_acquisition_manifest\n'
        '      8. **on_finished**       -- post-acquisition hook\n'
        '    """\n'
        '\n'
        '    def on_started(self) -> None:\n'
        '        """Called immediately after start_all. Arm a wall-clock cap."""\n'
        '        duration = self.config.get("duration")\n'
        '        if duration:\n'
        '            self.logger.info(f"Duration cap armed: {duration}s")\n'
        '            self._duration_timer = threading.Timer(float(duration), self.cleanup)\n'
        '            self._duration_timer.daemon = True\n'
        '            self._duration_timer.start()\n'
        '\n'
        '    def on_finished(self) -> None:\n'
        '        super().on_finished()\n'
        '        timer = getattr(self, "_duration_timer", None)\n'
        '        if timer is not None:\n'
        '            timer.cancel()\n'
        '            self._duration_timer = None\n'
        '\n'
        '    def manifest_extra(self) -> dict:\n'
        '        """Attach session-level metadata to the AcquisitionManifest."""\n'
        '        return {}\n'
        '\n'
        '\n'
        'def main():\n'
        '    """Run headless: `python procedure.py`."""\n'
        '    import sys\n'
        '    from pathlib import Path\n'
        '    cfg = Path(__file__).parent / "experiment.json"\n'
        '    proc = MyProcedure(str(cfg))\n'
        '    finished = proc.run_until_finished(timeout=30.0)\n'
        '    sys.exit(0 if finished else 1)\n'
        '\n'
        '\n'
        'if __name__ == "__main__":\n'
        '    main()\n'
    )


def _thermal_example_py() -> str:
    return (
        '"""Example custom device: a thermal sensor over USB-serial.\n'
        '\n'
        'This file is the canonical pattern for adding a lab-specific device\n'
        'to mesofield. One file holds both halves of the contract:\n'
        '\n'
        '  - The producer (`ThermalSensor`) writes data during acquisition.\n'
        '  - The parser (`_ThermalParser`) reads it back during ingest.\n'
        '\n'
        'The producer is registered via `@DeviceRegistry.register("thermal")`;\n'
        'after that the device is referenced in hardware.yaml as\n'
        '`type: thermal`. The parser is bound to the producer via\n'
        '`ThermalSensor.Parser = _ThermalParser` so the SOURCE_REGISTRY\n'
        'resolves it through the producer class.\n'
        '\n'
        'To use:\n'
        '  1. Adapt `parse_line()` for your sensor\'s actual serial protocol.\n'
        '  2. Import the module from your procedure.py to trigger the\n'
        '     registration:\n'
        '         from devices import thermal_example  # noqa: F401\n'
        '  3. Add a stanza to hardware.yaml:\n'
        '         thermal:\n'
        '           type: thermal\n'
        '           port: /dev/ttyUSB0\n'
        '           baudrate: 115200\n'
        '           output:\n'
        '             suffix: thermal\n'
        '             file_type: csv\n'
        '             bids_type: beh\n'
        '"""\n'
        '\n'
        'from __future__ import annotations\n'
        '\n'
        'from pathlib import Path\n'
        'from typing import Optional, Tuple\n'
        '\n'
        'import numpy as np\n'
        'import pandas as pd\n'
        '\n'
        'from mesofield import DeviceRegistry\n'
        'from mesofield.devices.base import BaseSerialDevice\n'
        'from mesofield.datakit.sources.register import SourceContext, TimeseriesSource\n'
        '\n'
        '\n'
        '@DeviceRegistry.register("thermal")\n'
        'class ThermalSensor(BaseSerialDevice):\n'
        '    """USB-serial thermal sensor reading float Celsius values."""\n'
        '\n'
        '    file_type = "csv"\n'
        '    bids_type = "beh"\n'
        '    data_type = "thermal"\n'
        '\n'
        '    def parse_line(self, line: bytes) -> Optional[Tuple[float, Optional[float]]]:\n'
        '        text = line.decode("utf-8", errors="replace").strip()\n'
        '        if not text:\n'
        '            return None\n'
        '        try:\n'
        '            return float(text), None\n'
        '        except ValueError:\n'
        '            self.logger.debug("Non-float line: %r", text)\n'
        '            return None\n'
        '\n'
        '\n'
        'class _ThermalParser(TimeseriesSource):\n'
        '    """Ingest-side parser for ThermalSensor CSV output."""\n'
        '\n'
        '    tag = "thermal"\n'
        '    patterns = ("**/*_thermal.csv",)\n'
        '\n'
        '    def build_timeseries(self, path: Path, *, context: SourceContext | None = None):\n'
        '        df = pd.read_csv(path)\n'
        '        t = df["timestamp"].to_numpy(dtype=np.float64)\n'
        '        return t, df, {"source_file": str(path), "n_samples": len(df)}\n'
        '\n'
        '\n'
        '# Bind the parser to the producer so SOURCE_REGISTRY["thermal"]\n'
        '# resolves through ThermalSensor.Parser.\n'
        'ThermalSensor.Parser = _ThermalParser\n'
    )


def _readme(protocol: str, dir_name: str) -> str:
    return (
        f"# {protocol}\n"
        "\n"
        "A mesofield experiment scaffolded by `mesofield init`.\n"
        "\n"
        "## Layout\n"
        "\n"
        "- `experiment.json` — session/protocol/duration; subjects.\n"
        "- `hardware.yaml` — device stanzas. Each `type:` value names a\n"
        "  registered DataProducer class.\n"
        "- `procedure.py` — the `MyProcedure(Procedure)` subclass. Run with\n"
        "  `python procedure.py` (headless) or `mesofield launch experiment.json`.\n"
        "- `devices/thermal_example.py` — annotated template for adding a\n"
        "  custom hardware device. Delete it when you no longer need it.\n"
        "\n"
        "## Quick start\n"
        "\n"
        "```sh\n"
        "python procedure.py\n"
        "```\n"
        "\n"
        f"The acquisition writes a per-session BIDS layout under `data/sub-SUBJ01/ses-01/`,\n"
        "with an `AcquisitionManifest` (`manifest.json`) at the session root.\n"
        "Datakit ingests by reading the manifest -- no globbing of filenames.\n"
        "\n"
        "## Customizing\n"
        "\n"
        f"1. Edit `experiment.json` to set your real subject IDs and duration.\n"
        "2. Edit `hardware.yaml` to declare your rig's devices. Replace the\n"
        "   `mock_wheel` stanza with `type: wheel` (Arduino) or `type: camera`\n"
        "   (micromanager) once you have real hardware connected.\n"
        "3. Copy `devices/thermal_example.py` as a template for any lab-specific\n"
        "   device. Adapt `parse_line()` for your serial protocol; register\n"
        "   the type in `procedure.py` (the `DeviceRegistry._registry.setdefault`\n"
        "   call).\n"
        "4. Add stage-specific behavior by overriding `prerun` / `on_started`\n"
        "   / `on_finished` in `MyProcedure`.\n"
    )
