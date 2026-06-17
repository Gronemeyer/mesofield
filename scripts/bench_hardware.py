"""Benchmark a Mesofield ``hardware.yaml`` end-to-end.

Runs the rig you point it at through the **real** Mesofield pipeline -- the same
:class:`mesofield.base.Procedure` a live acquisition uses: ``HardwareManager``
brings every device up, ``arm_all`` / ``start_all`` run them *together* (no
device is singled out or "switched"), the ``DataManager`` dataqueue logs, the
real writers persist OME-TIFF / MP4 / CSV, and an ``AcquisitionManifest`` is
written at the end. The only thing this script adds is measurement: a lightweight
tap on each device's ``signals.data`` plus a read of the resulting files.

It reports, per device declared in the YAML:

* **frames / samples** captured and **realized rate** (Hz)
* **dropped frames** -- gaps in a camera's frame counter (MM ``ImageNumber``)
* **write throughput** (MB/s on disk) and **data rate** (uncompressed MB/s)
* **cadence jitter** (std-dev of inter-sample interval) and worst gap
* **buffer overflow** (MM circular buffer)

Usage
-----
Benchmark a rig as configured (5 s)::

    python scripts/bench_hardware.py path/to/hardware.yaml

Longer run, keep the written files::

    python scripts/bench_hardware.py hardware.yaml --duration 20 --keep-files

Sweep MicroManager exposure (-> frame rate) and binning (-> frame size); each
combination is a full pipeline run::

    python scripts/bench_hardware.py hardware.yaml --exposures 1,2,5,10 --binnings 1,2

Find the limit -- ramp frame rate up, or shrink the circular buffer, until a
camera drops frames or overflows::

    python scripts/bench_hardware.py hardware.yaml --stress fps
    python scripts/bench_hardware.py hardware.yaml --stress buffer

The bundled example rig runs anywhere (no real hardware needed)::

    python scripts/bench_hardware.py experiments/benchmark/widefield-pupil-treadmill.yaml
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import platform
import shutil
import statistics
import threading
import time
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Per-device metrics
# ---------------------------------------------------------------------------


@dataclass
class DeviceMetrics:
    device_id: str
    device_type: str
    settings: str = ""
    n: int = 0                       # frames / samples captured
    rate_hz: float = 0.0             # realized rate
    target_hz: Optional[float] = None
    n_dropped: int = 0
    drop_pct: Optional[float] = None  # None for non-camera devices
    write_mbps: float = 0.0
    data_mbps: float = 0.0
    jitter_ms: float = 0.0
    max_gap_ms: float = 0.0
    overflow: bool = False
    bytes_on_disk: int = 0
    error: Optional[str] = None


@dataclass
class RunResult:
    label: str
    wall_s: float
    devices: list[DeviceMetrics] = field(default_factory=list)
    manifest_complete: Optional[bool] = None
    total_bytes: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Benchmark Procedure -- a duration-capped run of whatever the YAML declares
# ---------------------------------------------------------------------------


def _make_procedure(duration: float):
    """Build a duration-capped :class:`Procedure` subclass instance (headless)."""
    from mesofield.base import Procedure

    class _BenchProcedure(Procedure):
        """Standard Procedure with a wall-clock duration cap.

        No device-specific logic beyond a clean stop: the base Procedure already
        arms/starts every device in the YAML together and tears them down on the
        primary's ``finished`` signal. The timer caps the run at ``duration``.
        """

        def _cap(self) -> None:
            # Stop any running MicroManager sequence acquisition *before*
            # cleanup. The primary camera's stop() intentionally leaves its MDA
            # running, and real engines (Meso/Pupil) don't stopSequenceAcquisition
            # on an early break -- so cutting a run short would otherwise leave
            # the camera acquiring and deinitialize() fails to unload it
            # ("Cannot switch camera device while sequence acquisition is running").
            for cam in getattr(self.hardware, "cameras", ()):
                core = getattr(cam, "core", None)
                if core is None:
                    continue
                try:
                    core.mda.cancel()
                except Exception:
                    pass
                try:
                    if core.isSequenceRunning():
                        core.stopSequenceAcquisition()
                except Exception:
                    pass
            self.cleanup()

        def on_started(self) -> None:
            self._bench_timer = threading.Timer(float(duration), self._cap)
            self._bench_timer.daemon = True
            self._bench_timer.start()

        def on_finished(self) -> None:
            timer = getattr(self, "_bench_timer", None)
            if timer is not None:
                timer.cancel()
                self._bench_timer = None

    return _BenchProcedure(None)


def _register_mock_types() -> None:
    """Make the mock device types resolvable (demo rigs use them)."""
    from mesofield import DeviceRegistry
    from mesofield.devices.mocks import MockEncoderDevice, MockFrameProducer

    DeviceRegistry._registry.setdefault("mock_wheel", MockEncoderDevice)
    DeviceRegistry._registry.setdefault("mock_camera", MockFrameProducer)


# ---------------------------------------------------------------------------
# Portability sentinels (so the bundled example runs on a dev box)
# ---------------------------------------------------------------------------


def _resolve_yaml(yaml_path: str, outdir: str) -> str:
    """Resolve MicroManager ``AUTO`` / ``DEMO`` sentinels to the managed demo
    install. A real ``hardware.yaml`` naming concrete paths is returned as-is.
    """
    import yaml

    with open(yaml_path, "r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh) or {}

    mm_path = demo_cfg = None
    changed = False
    for cam in spec.get("cameras", []) or []:
        if str(cam.get("backend", "")).lower() != "micromanager":
            continue
        cfg_path = cam.get("configuration_path")
        if (str(cam.get("micromanager_path", "")).upper() == "AUTO"
                or str(cfg_path).upper() == "DEMO"
                or not cfg_path or not os.path.isfile(str(cfg_path))):
            if mm_path is None:
                from pymmcore_plus import find_micromanager
                managed = [p for p in find_micromanager(return_first=False)
                           if "pymmcore-plus" in p]
                if not managed:
                    raise RuntimeError(
                        f"camera '{cam.get('id')}' requests the managed demo install "
                        "but none was found (run `mmcore install`)."
                    )
                mm_path, demo_cfg = managed[0], os.path.join(managed[0], "MMConfig_demo.cfg")
            cam["micromanager_path"] = mm_path
            cam["configuration_path"] = demo_cfg
            changed = True

    if not changed:
        return os.path.abspath(yaml_path)
    resolved = os.path.join(outdir, "_resolved_hardware.yaml")
    with open(resolved, "w", encoding="utf-8") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False)
    print(f"  [note] resolved MicroManager AUTO/DEMO sentinels -> managed demo install")
    return resolved


# ---------------------------------------------------------------------------
# One pipeline run
# ---------------------------------------------------------------------------


def _coerce_int(p: Any) -> Optional[int]:
    try:
        return int(p)
    except (TypeError, ValueError):
        return None


def _geometry(dev) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """(width, height, bytes_per_frame) for a camera device, else (None, None, None)."""
    core = getattr(dev, "core", None)
    if core is not None:
        try:
            w, h = int(core.getImageWidth()), int(core.getImageHeight())
            return w, h, w * h * int(core.getBytesPerPixel())
        except Exception:
            pass
    w = getattr(dev, "_frame_width", None) or getattr(dev, "width", None)
    h = getattr(dev, "_frame_height", None) or getattr(dev, "height", None)
    if w and h:
        bpp = 3 if str(getattr(dev, "file_type", "")) in {"mp4", "avi"} else 2
        return int(w), int(h), int(w) * int(h) * bpp
    return None, None, None


def _settings_str(dev) -> str:
    w, h, _ = _geometry(dev)
    core = getattr(dev, "core", None)
    if getattr(dev, "device_type", "") == "camera":
        parts = []
        if w and h:
            parts.append(f"{w}x{h}")
        if core is not None:
            try:
                parts.append(f"exp={core.getExposure():g}ms")
            except Exception:
                pass
            try:
                parts.append(f"bin={core.getProperty(core.getCameraDevice(), 'Binning')}")
            except Exception:
                pass
        return " ".join(parts)
    rate = getattr(dev, "sampling_rate", 0)
    return f"{rate:g}Hz" if rate else ""


def run_once(yaml_path: str, outdir: str, *, duration: float, label: str,
             exposure_ms: Optional[float] = None, binning: Optional[int] = None,
             buffer_mb: Optional[int] = None, keep_files: bool = False) -> RunResult:
    """Run the full Procedure for ``duration`` seconds and measure each device."""
    from pathlib import Path

    _register_mock_types()
    result = RunResult(label=label, wall_s=0.0)
    proc = None
    taps: dict[str, list[tuple[float, Any]]] = {}
    try:
        proc = _make_procedure(duration)
        proc.load_config(hardware_yaml_path=yaml_path)
        # Redirect all output into the benchmark dir. proc.data_dir is cached
        # at load time, so sync it too (the manifest writer reads it).
        proc.config.experiment_dir = outdir
        proc.data_dir = proc.config.data_dir
        proc.config.set("subject", "bench")
        proc.config.set("session", "01")
        proc.config.set("task", "benchmark")
        proc.config.set("duration", max(1, int(round(duration))))

        devices = proc.config.hardware.devices

        # Apply optional MicroManager overrides on every MM camera (no
        # "switching" -- the same setting is applied uniformly, then all
        # devices run together).
        for dev in proc.config.hardware.cameras:
            if getattr(dev, "backend", None) == "micromanager" and getattr(dev, "core", None):
                core = dev.core
                if binning is not None:
                    try:
                        core.setProperty(core.getCameraDevice(), "Binning", str(binning))
                    except Exception as exc:
                        dev.logger.warning(f"set Binning failed: {exc}")
                if exposure_ms is not None:
                    core.setExposure(float(exposure_ms))
                    dev.sampling_rate = round(1000.0 / exposure_ms, 3)
                if buffer_mb:
                    try:
                        core.setCircularBufferMemoryFootprint(int(buffer_mb))
                    except Exception as exc:
                        dev.logger.warning(f"set buffer failed: {exc}")

        # Measurement tap: record (emit-time, payload) per device. psygnal is
        # synchronous, so this is true emit cadence and adds no pipeline change.
        for dev_id, dev in devices.items():
            rec: list[tuple[float, Any]] = []
            taps[dev_id] = rec
            sig = getattr(getattr(dev, "signals", None), "data", None)
            if sig is not None:
                sig.connect(lambda payload, ts=None, _r=rec: _r.append((time.perf_counter(), payload)))

        # --- run the real pipeline to completion -------------------------
        proc.run_until_finished(timeout=duration * 4 + 15)
        wall = ((proc.stopped_time - proc.start_time).total_seconds()
                if proc.start_time and proc.stopped_time else duration)
        result.wall_s = round(wall, 3)

        # --- measure each device -----------------------------------------
        for dev_id, dev in devices.items():
            m = DeviceMetrics(device_id=dev_id,
                              device_type=getattr(dev, "device_type", "device"),
                              settings=_settings_str(dev))
            rec = taps.get(dev_id, [])
            m.n = len(rec)
            if wall > 0:
                m.rate_hz = round(m.n / wall, 2)
            if getattr(dev, "sampling_rate", 0):
                m.target_hz = round(float(dev.sampling_rate), 2)

            is_camera = m.device_type == "camera"
            w, h, bpf = _geometry(dev)
            if is_camera:
                # Dropped frames = gaps in the camera's frame counter.
                idxs = [v for v in (_coerce_int(p) for _, p in rec) if v is not None]
                if len(idxs) >= 2:
                    span = max(idxs) - min(idxs) + 1
                    m.n_dropped = max(0, span - len(set(idxs)))
                    m.drop_pct = round(100.0 * m.n_dropped / span, 2)
                else:
                    m.drop_pct = 0.0
                core = getattr(dev, "core", None)
                if core is not None:
                    try:
                        m.overflow = bool(core.isBufferOverflowed())
                    except Exception:
                        pass

            # Throughput from the device's real output file(s).
            out = getattr(dev, "output_path", None)
            if out:
                m.bytes_on_disk = sum(os.path.getsize(p) for p in glob.glob(out + "*")
                                      if os.path.isfile(p))
            if wall > 0:
                m.write_mbps = round(m.bytes_on_disk / 1e6 / wall, 2)
                if bpf:
                    m.data_mbps = round(m.n * bpf / 1e6 / wall, 2)

            ts = [t for t, _ in rec]
            if len(ts) >= 3:
                deltas = [(b - a) * 1000.0 for a, b in zip(ts[:-1], ts[1:])]
                m.jitter_ms = round(statistics.pstdev(deltas), 3)
                m.max_gap_ms = round(max(deltas), 3)
            result.devices.append(m)

        # Manifest + total footprint.
        session_root = Path(proc.data_dir) / "sub-bench" / "ses-01"
        manifest = session_root / "manifest.json"
        if manifest.is_file():
            try:
                import json
                result.manifest_complete = bool(
                    json.loads(manifest.read_text()).get("acquisition_complete"))
            except Exception:
                pass
        if session_root.is_dir():
            result.total_bytes = sum(
                os.path.getsize(p) for p in glob.glob(str(session_root / "**" / "*"),
                                                      recursive=True) if os.path.isfile(p))
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        if proc is not None:
            try:
                proc.config.hardware.deinitialize()
            except Exception:
                pass
        if not keep_files:
            shutil.rmtree(os.path.join(outdir, "data"), ignore_errors=True)
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_COLS = [
    ("device_id", "device", 12, "<"),
    ("device_type", "type", 8, "<"),
    ("settings", "settings", 22, "<"),
    ("n", "frames", 7, ">"),
    ("rate_hz", "rate", 8, ">"),
    ("drop_pct", "drop%", 6, ">"),
    ("write_mbps", "MB/s", 8, ">"),
    ("data_mbps", "dataMB/s", 9, ">"),
    ("jitter_ms", "jit(ms)", 8, ">"),
    ("overflow", "ovf", 4, ">"),
]


def print_run(result: RunResult) -> None:
    print(f"\n  {result.label}   (wall {result.wall_s}s)")
    if result.error:
        print(f"    ERROR: {result.error}")
        return
    head = "    " + "  ".join(f"{t:{a}{w}}" for _, t, w, a in _COLS)
    print(head)
    print("    " + "-" * (len(head) - 4))
    for m in result.devices:
        cells = []
        for attr, _, w, a in _COLS:
            v = getattr(m, attr)
            if attr == "drop_pct" and v is None:
                v = "-"          # non-camera devices have no drop concept
            elif attr == "rate_hz" and m.target_hz:
                v = f"{v:g}/{m.target_hz:g}"
            elif isinstance(v, bool):
                v = "!" if v else ""
            cells.append(f"{str(v):{a}{w}}")
        line = "    " + "  ".join(cells)
        if m.error:
            line += f"   ERR:{m.error}"
        print(line)
    mc = {True: "complete", False: "INCOMPLETE", None: "n/a"}[result.manifest_complete]
    print(f"    manifest: {mc}   total written: {result.total_bytes / 1e6:.1f} MB")


# One CSV row per device per run: run-level context + every DeviceMetrics field.
_CSV_RUN_FIELDS = ["run", "wall_s", "manifest_complete", "total_bytes", "run_error"]
_CSV_DEV_FIELDS = [f.name for f in fields(DeviceMetrics)]


def write_csv(results: list[RunResult], path: str) -> None:
    """Write one row per (run, device) to ``path``."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_RUN_FIELDS + _CSV_DEV_FIELDS)
        for r in results:
            run_cols = [r.label, r.wall_s, r.manifest_complete, r.total_bytes, r.error]
            if not r.devices:  # a run that errored before measuring any device
                writer.writerow(run_cols + [""] * len(_CSV_DEV_FIELDS))
                continue
            for m in r.devices:
                writer.writerow(run_cols + [getattr(m, f) for f in _CSV_DEV_FIELDS])
    print(f"\nWrote {sum(max(1, len(r.devices)) for r in results)} rows -> {path}")


def print_env_header(yaml_path: str, outdir: str, duration: float) -> None:
    try:
        from mesofield import _version
        mf = _version.__version__
    except Exception:
        mf = "n/a"
    print("=" * 78)
    print("Mesofield hardware benchmark")
    print("=" * 78)
    print(f"  hardware  : {yaml_path}")
    print(f"  duration  : {duration} s")
    print(f"  platform  : {platform.platform()}")
    print(f"  mesofield : {mf}")
    print(f"  output    : {os.path.abspath(outdir)}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _broke(result: RunResult, threshold: float) -> Optional[DeviceMetrics]:
    """Return the first device that overflowed or dropped past threshold."""
    if result.error:
        return DeviceMetrics(device_id="(run)", device_type="-", error=result.error)
    for m in result.devices:
        if m.overflow or (m.drop_pct is not None and m.drop_pct > threshold):
            return m
    return None


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hardware", help="path to the hardware.yaml to benchmark")
    p.add_argument("--duration", type=float, default=5.0, help="seconds per run")
    p.add_argument("--repeats", type=int, default=1, help="repeat the run N times")
    p.add_argument("--exposures", default=None,
                   help="comma list of MM exposures (ms) to sweep -> frame rate")
    p.add_argument("--binnings", default=None,
                   help="comma list of MM binnings to sweep -> frame size")
    p.add_argument("--buffer-mb", type=int, default=None,
                   help="override MM circular-buffer footprint (MB)")
    p.add_argument("--stress", choices=["fps", "buffer"], default=None,
                   help="ramp until a camera drops/overflows (fps: lower exposure; "
                        "buffer: shrink circular buffer)")
    p.add_argument("--drop-threshold", type=float, default=1.0,
                   help="drop %% that counts as 'broken' in --stress")
    p.add_argument("--keep-files", action="store_true",
                   help="keep the written session data (default: delete after sizing)")
    p.add_argument("--outdir", default=None, help="output dir (default: ./bench_out/<ts>)")
    args = p.parse_args(argv)

    if not os.path.isfile(args.hardware):
        p.error(f"hardware file not found: {args.hardware}")

    outdir = args.outdir or os.path.join("bench_out", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    print_env_header(args.hardware, outdir, args.duration)
    yaml_path = _resolve_yaml(args.hardware, outdir)

    results: list[RunResult] = []

    if args.stress == "fps":
        exposures = sorted(_parse_floats(args.exposures or "10,5,2,1"), reverse=True)
        print(f"\n[stress:fps] ramping exposure down (frame rate up): {exposures} ms")
        last_stable = breaking = None
        for exp in exposures:
            r = run_once(yaml_path, outdir, duration=args.duration,
                         label=f"exposure={exp}ms", exposure_ms=exp,
                         buffer_mb=args.buffer_mb, keep_files=args.keep_files)
            results.append(r)
            print_run(r)
            if _broke(r, args.drop_threshold):
                breaking = exp
                break
            last_stable = exp
        print(f"\n--- limit ---  last stable exposure: {last_stable}ms"
              + (f";  broke at {breaking}ms" if breaking else ";  no limit hit"))

    elif args.stress == "buffer":
        base = args.buffer_mb or 2000
        buffers = [b for b in (2000, 1000, 500, 200, 100, 50, 25, 10) if b <= base]
        # Drive max load (lowest exposure) so a too-small buffer actually overflows.
        load_exp = min(_parse_floats(args.exposures)) if args.exposures else None
        print(f"\n[stress:buffer] shrinking circular buffer: {buffers} MB"
              + (f" at exposure={load_exp}ms" if load_exp else ""))
        last_stable = breaking = None
        for buf in buffers:
            r = run_once(yaml_path, outdir, duration=args.duration,
                         label=f"buffer={buf}MB", exposure_ms=load_exp,
                         buffer_mb=buf, keep_files=args.keep_files)
            results.append(r)
            print_run(r)
            if _broke(r, args.drop_threshold):
                breaking = buf
                break
            last_stable = buf
        print(f"\n--- limit ---  last stable buffer: {last_stable}MB"
              + (f";  overflowed at {breaking}MB" if breaking else ";  no overflow"))

    elif args.exposures or args.binnings:
        exposures = _parse_floats(args.exposures) if args.exposures else [None]
        binnings = _parse_ints(args.binnings) if args.binnings else [None]
        print(f"\n[sweep] exposures={args.exposures or '-'}  binnings={args.binnings or '-'}")
        for binning in binnings:
            for exp in exposures:
                tag = []
                if exp is not None:
                    tag.append(f"exp={exp}ms")
                if binning is not None:
                    tag.append(f"bin={binning}")
                r = run_once(yaml_path, outdir, duration=args.duration,
                             label=" ".join(tag) or "default", exposure_ms=exp,
                             binning=binning, buffer_mb=args.buffer_mb,
                             keep_files=args.keep_files)
                results.append(r)
                print_run(r)

    else:
        for i in range(args.repeats):
            label = "run" if args.repeats == 1 else f"run {i + 1}/{args.repeats}"
            r = run_once(yaml_path, outdir, duration=args.duration, label=label,
                         buffer_mb=args.buffer_mb, keep_files=args.keep_files)
            results.append(r)
            print_run(r)

    write_csv(results, os.path.join(outdir, "results.csv"))
    print()
    return 1 if any(r.error for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
