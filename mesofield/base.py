"""
Base procedure classes for implementing experimental workflows in Mesofield.

This module defines a *generic* :class:`Procedure` orchestrator that contains
zero device-specific logic.  Custom experiment subclasses live outside the
package (typically under ``experiments/<name>/procedure.py``) and are
discovered via :func:`load_procedure_from_config`, which reads the optional
``procedure_file`` and ``procedure_class`` fields from ``experiment.json``.

Lifecycle (subclass hooks shown in **bold**):

1. ``initialize_hardware``  -- bring devices up
2. ``prerun``               -- **subclass hook** (default: no-op)
3. ``hardware.arm_all``     -- per-run prep on every device
4. connect ``hardware.primary.signals.finished`` -> ``_cleanup_procedure``
5. ``on_started``           -- **subclass hook** (default: no-op)
6. ``hardware.start_all``
7. ``on_finished``          -- **subclass hook** (default: no-op)
8. ``save_data`` + cleanup
"""

import importlib.util
import json
import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Type

from PyQt6.QtCore import QObject, pyqtSignal

from mesokit_schema import (
    AcquisitionManifest,
    DataqueuePayloadSchema,
    ProducerEntry,
    SessionIdentity,
    SidecarEntry,
    TimeBasis,
)

from mesofield.config import ExperimentConfig
from mesofield.data.manager import DataManager

try:
    from mesofield._version import __version__ as _MESOFIELD_VERSION
except Exception:  # pragma: no cover
    _MESOFIELD_VERSION = "0.0.0+unknown"
from mesofield.hardware import HardwareManager
from mesofield.utils._logger import get_logger, hyperlink


def processor(*, camera: str, plot: bool = False, **plot_kwargs: Any):
    """Mark a :class:`Procedure` method as a per-frame compute function.

    The decorated function is called as ``func(self, img, idx, ts)`` and
    should return a ``float | None``.  At procedure init the framework
    builds a :class:`~mesofield.processors.FrameProcessor` that wraps it,
    attaches it to the hardware device whose ``device_id`` matches
    ``camera``, registers it on :class:`~mesofield.data.manager.DataManager`,
    and (when ``plot=True``) tells the GUI to add a
    :class:`~mesofield.gui.speedplotter.SerialWidget`.

    ``plot_kwargs`` are forwarded straight to the widget — recognized
    keys: ``label``, ``value_label``, ``value_units``, ``y_range``,
    ``value_scale``, ``max_points``.

    Example::

        class MyProcedure(Procedure):
            @processor(camera="meso", plot=True, label="Frame Mean")
            def frame_mean(self, img, idx, ts):
                return float(img.mean())
    """
    def wrap(fn):
        fn._mesofield_processor = {
            "camera": camera,
            "plot": plot,
            "plot_kwargs": plot_kwargs,
            "name": fn.__name__,
        }
        return fn
    return wrap


class ProcedureSignals(QObject):
    """All procedure-level signals that a Qt GUI can connect to."""
    procedure_started      = pyqtSignal()
    hardware_initialized   = pyqtSignal(bool)     # success
    data_saved             = pyqtSignal()
    procedure_error        = pyqtSignal(str)      # emits error message
    procedure_finished     = pyqtSignal()


class Procedure:
    """Generic orchestrator for a Mesofield experiment.

    Subclass this in ``experiments/<name>/procedure.py`` and override the
    extension hooks (:meth:`prerun`, :meth:`on_started`, :meth:`on_finished`)
    and/or the lifecycle methods (:meth:`run`, :meth:`save_data`,
    :meth:`cleanup`) as needed.  The base class never references a specific
    device type -- multi-camera sync is driven by the YAML ``primary: true``
    flag and ``HardwareManager.start_all/stop_all``.
    """

    # When True (set on subclasses like PlaybackProcedure), `run`/`cleanup`
    # skip writer/queue-logger/manifest side effects so the on-disk session
    # is left untouched.
    playback: bool = False

    # When True (default), a positive ``duration`` (seconds) in the config caps
    # the run: the base class arms a wall-clock timer after ``start_all`` and
    # stops the run when it elapses. Devices that terminate on their own stop 
    # the run early via the primary's ``finished`` signal; whichever fires first wins. 
    # Set ``False`` on a subclass when the primary runs a bounded acquisition 
    # (e.g. an MDA camera capturing an exact frame count) that must complete regardless of wall time.
    stop_after_duration: bool = True

    def __init__(
        self,
        hardware: Optional[Any] = None,
        experiment: Optional[str] = None,
        *,
        experiment_directory: Optional[str] = None,
        **params: Any,
    ):
        """Build a procedure.

        Parameters
        ----------
        hardware:
            Either a path to a ``hardware.yaml`` rig file, **or** a list of
            already-constructed device objects, e.g.::

                proc = Procedure(hardware=[LickDetector(port="COM3")], ...)

            A single device is automatically the primary, so one-device
            scripts never set ``primary=True``. ``None`` falls back to the
            :meth:`define_hardware` hook.
        experiment:
            Path to an ``experiment.json`` parameter file (or ``None``).
        experiment_directory:
            Where acquisition data is written (``<dir>/data/sub-.../ses-...``).
            This is the simplest way for a standalone script to choose its
            output location; relative paths resolve against the current working
            directory. Overrides any value from ``define_config`` / JSON.
        **params:
            Any other experiment parameters (``subject``, ``session``,
            ``task``, ``duration``, ...) set straight onto the config. These
            also override ``define_config`` / JSON, so a script can stay a
            single ``Procedure(...)`` call with no ``define_config`` override.
        """
        # Initialize the processor registry before anything else so the
        # ``__setattr__`` hook can run safely from the first assignment on.
        # We bypass the hook itself with object.__setattr__ to avoid the
        # isinstance import dance on a guaranteed-non-processor value.
        object.__setattr__(self, "processors", [])

        self.events = ProcedureSignals()
        self._finished_event = threading.Event()
        self.events.procedure_finished.connect(self._finished_event.set)
        self.events.procedure_error.connect(lambda _msg: self._finished_event.set())

        # Wall-clock duration cap (see `stop_after_duration`) and a one-shot
        # guard so the timer and the primary device's `finished` signal can
        # both target cleanup without it running twice.
        self._duration_timer: Optional[threading.Timer] = None
        self._cleanup_started = False

        # Optional start gate, injected by a front-end (e.g. the GUI
        # ConfigController). When ``start_on_trigger`` is set, the default
        # :meth:`await_trigger` calls this after arming and before starting any
        # device; it owns whatever "ready / press to start" interaction the
        # front-end wants (launching a stimulus subprocess, a focused dialog,
        # etc.) and returns ``True`` to proceed or ``False`` to cancel the run.
        # Left ``None`` for headless runs, which then do not block.
        self.start_gate: Optional[Any] = None

        # `hardware` is either a path to a hardware.yaml rig or a list of
        # pre-built device objects. Keep the path (if any) for ExperimentConfig;
        # a device list is handed to a HardwareManager below.
        hardware_path = hardware if isinstance(hardware, str) else None
        scripted_devices = (
            None if isinstance(hardware, (str, type(None))) else list(hardware)
        )

        # The rig (hardware.yaml) is the anchor; experiment params are optional
        # and never touch hardware state.
        self.config: ExperimentConfig
        self.config = ExperimentConfig(hardware_path)

        # Scripted config: a subclass may declare parameters in Python
        # (a dataclass or mapping) instead of an experiment.json file.
        config_data = self.define_config()
        if config_data is not None:
            self.config.load_dict(config_data)
        elif experiment:
            self.config.load_json(experiment)

        # Explicit constructor arguments win over define_config / JSON, so a
        # standalone script can set everything in one readable call.
        if experiment_directory is not None:
            self.config.experiment_dir = experiment_directory
        for key, value in params.items():
            self.config.set(key, value)

        # Scripted hardware: devices passed to the constructor (``hardware=[...]``)
        # win; otherwise fall back to the :meth:`define_hardware` hook.
        if scripted_devices is None:
            scripted_devices = self.define_hardware()
        if scripted_devices is not None:
            scripted_devices = list(scripted_devices)
            # A lone device is the primary by default, so single-device scripts
            # don't have to flag ``primary=True``.
            if (
                len(scripted_devices) == 1
                and not getattr(scripted_devices[0], "is_primary", False)
            ):
                scripted_devices[0].is_primary = True
            self.config.hardware = HardwareManager(devices=scripted_devices)

        self.protocol = self.config.get("protocol", "default_experiment")
        self.experimenter = self.config.get("experimenter", "researcher")

        self.data_dir = self.config.data_dir

        self.start_time: Optional[datetime] = None
        self.stopped_time: Optional[datetime] = None

        self.logger = get_logger(f"PROCEDURE.{self.protocol}")
        self.logger.info(f"Initialized procedure: {self.protocol}")

        if self.config.hardware.is_configured:
            self.initialize_hardware()
            self.config.hardware._configure_engines(self.config)
            self._materialize_decorated_processors()
        else:
            self.logger.info(
                "Hardware not configured yet -- launch in default state. "
                "Use load_config() to apply a configuration."
            )

    # ------------------------------------------------------------------
    # Convenience accessors

    @property
    def paths(self):
        return self.data.base.read('datapaths')

    @property
    def hardware(self) -> HardwareManager:
        return self.config.hardware

    # ------------------------------------------------------------------
    # Hardware bring-up

    def initialize_hardware(self) -> None:
        """Boot up hardware and a :class:`DataManager`."""
        try:
            self.config.hardware.initialize(self.config)
            self.data = DataManager()
            # Register devices eagerly so iPython terminals and GUI inspectors
            # see them on `procedure.data.devices` before run() is called.
            # `Procedure.run()` short-circuits the re-registration via its
            # `if not self.data.devices:` guard.
            self.data.register_devices(self.config.hardware.devices.values())
            self.logger.info("Hardware initialized successfully")
        except RuntimeError as e:
            self.logger.error(f"Failed to initialize hardware: {e}")
            self.config.hardware.deinitialize()
            raise

    def load_config(self, hardware: Optional[str] = None,
                    experiment: Optional[str] = None) -> None:
        """Hot-load a hardware YAML and/or experiment JSON into the live config.

        The two inputs are independent: loading experiment params never touches
        hardware. Callers pass explicit paths (the GUI wizard resolves them from
        its pickers).
        """
        if hardware:
            self.config.load_hardware(hardware)
        if experiment:
            self.config.load_json(experiment)

        self.protocol = self.config.get("protocol", "default_experiment")
        self.experimenter = self.config.get("experimenter", "researcher")
        self.data_dir = self.config.data_dir

        if self.config.hardware.is_configured:
            self.initialize_hardware()
            self.config.hardware._configure_engines(self.config)
            self._materialize_decorated_processors()

        self.events.hardware_initialized.emit(self.config.hardware.is_configured)
        self.logger.info("Configuration hot-loaded successfully")

    # ------------------------------------------------------------------
    # Processor auto-discovery
    #
    # Any :class:`FrameProcessor` instance stored on the procedure (whether
    # by direct attribute assignment or by the ``@processor`` decorator)
    # lands on ``self.processors`` and is registered with the DataManager
    # automatically.  Cleanup detaches them.

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        # Avoid importing at module load (the processors package imports
        # PyQt6, which we don't want forced on headless callers that never
        # use processors).
        try:
            from mesofield.processors import FrameProcessor
        except Exception:
            return
        if isinstance(value, FrameProcessor):
            self._register_processor(value, attr_name=name)

    def _register_processor(self, proc: Any, attr_name: Optional[str] = None) -> None:
        """Register ``proc`` on :attr:`processors` and the DataManager.

        Dedupe-safe and idempotent: re-registering the same instance is a
        no-op.  When ``attr_name`` is supplied and a *different* processor
        with the same name is already registered, the old one is detached
        and replaced.
        """
        procs = self.processors  # plain attribute, no __setattr__ recursion
        if proc in procs:
            return
        # If the user used the default name (lowercased class name) and
        # assigned the processor to a specific attribute, prefer the
        # attribute name. Explicit `name=...` at construction wins.
        if attr_name is not None:
            default_name = type(proc).__name__.lower()
            if proc.device_id == default_name and attr_name != default_name:
                proc.device_id = attr_name
                proc.id = attr_name
                proc.name = attr_name
        # Replace any prior processor that shared this attribute name.
        if attr_name is not None:
            for existing in list(procs):
                if existing.device_id == proc.device_id and existing is not proc:
                    try:
                        existing.detach()
                    except Exception:
                        pass
                    procs.remove(existing)
                    break
        procs.append(proc)
        # DataManager may not exist yet during early construction; the
        # later ``Procedure.run`` call also re-registers devices, so this
        # is best-effort.
        dm = getattr(self, "data", None)
        if dm is not None:
            try:
                dm.register_hardware_device(proc)
            except Exception as exc:
                self.logger.warning(
                    f"register_hardware_device({proc.device_id}) failed: {exc}"
                )

    def _materialize_decorated_processors(self) -> None:
        """Instantiate every ``@processor``-decorated method on this class.

        Walks the MRO so decorators on base procedures are honored.
        Skips names that already resolve to a :class:`FrameProcessor` on
        this instance (idempotent for ``load_config`` hot reloads).
        """
        try:
            from mesofield.processors import FrameProcessor
        except Exception:
            return
        seen: set[str] = set()
        for klass in type(self).__mro__:
            for attr_name, member in klass.__dict__.items():
                if attr_name in seen:
                    continue
                meta = getattr(member, "_mesofield_processor", None)
                if meta is None:
                    continue
                seen.add(attr_name)
                existing = getattr(self, attr_name, None)
                if isinstance(existing, FrameProcessor):
                    continue  # already materialized; skip on hot reload
                camera = self._resolve_camera(meta["camera"], attr_name)
                bound = member.__get__(self, type(self))  # bound method
                cls = type(
                    f"_Decorated_{attr_name}",
                    (FrameProcessor,),
                    {"compute": lambda self, img, idx, ts, _fn=bound: _fn(img, idx, ts)},
                )
                kwargs = dict(meta["plot_kwargs"])
                instance = cls(
                    name=attr_name,
                    camera=camera,
                    plot=meta["plot"],
                    **kwargs,
                )
                # Goes through __setattr__ -> _register_processor.
                setattr(self, attr_name, instance)

    def _resolve_camera(self, device_id: str, for_name: str) -> Any:
        """Look up a hardware device by ``device_id``; clear error if missing."""
        devices = getattr(self.hardware, "devices", {}) or {}
        # ``devices`` is a {device_id: device} mapping in HardwareManager.
        if device_id in devices:
            return devices[device_id]
        # Fall back to scanning ``cameras`` tuple by their .device_id / .id.
        for cam in getattr(self.hardware, "cameras", ()) or ():
            if getattr(cam, "device_id", getattr(cam, "id", None)) == device_id:
                return cam
        available = sorted(devices.keys())
        raise ValueError(
            f"@processor(camera={device_id!r}) on {type(self).__name__}.{for_name}: "
            f"no hardware device with that id. Available: {available}"
        )

    # ------------------------------------------------------------------
    # Subclass extension hooks (no-op defaults)

    def define_config(self) -> Any:
        """Subclass hook to declare experiment parameters in Python.

        Override to return a ``@dataclass`` instance or a plain mapping; it is
        applied to :attr:`config` via :meth:`ExperimentConfig.load_dict`,
        superseding any ``experiment.json``. Default ``None`` -> load JSON.
        """
        return None

    def define_hardware(self) -> Any:
        """Subclass hook to construct hardware devices in Python.

        Override to return a list of pre-built device objects (imported and
        instantiated in the procedure file). They are handed to a fresh
        :class:`HardwareManager`, superseding any ``hardware.yaml``. Default
        ``None`` -> load YAML. Device classes should be decorated with
        ``@DeviceRegistry.register(...)`` so the setup can later be exported
        to a ``hardware.yaml`` rig file via ``HardwareManager.to_yaml``.
        """
        return None

    def prerun(self) -> None:
        """Subclass hook called before arming devices.  Override as needed."""
        return None

    def await_trigger(self) -> None:
        """Gate the run after arming and before starting devices.

        When ``start_on_trigger`` is set and a :attr:`start_gate` has been
        injected (e.g. by the GUI ConfigController), it is invoked here to own
        the "ready / press to start" interaction; returning ``False`` cancels
        the run. Devices are armed but nothing has started yet, so blocking here
        holds the whole run. Headless runs with no gate do not block.

        This default contains no device-specific logic. Subclasses may still
        override it for a fully custom trigger.
        """
        if not self.config.start_on_trigger:
            return
        gate = self.start_gate
        if gate is None:
            self.logger.info(
                "start_on_trigger set but no start gate injected; proceeding."
            )
            return
        if not gate(self):
            raise RuntimeError("Run cancelled at the start gate")

    def on_started(self) -> None:
        """Subclass hook called immediately after ``start_all``."""
        return None

    def on_finished(self) -> None:
        """Subclass hook called immediately after the primary device finishes."""
        return None

    # ------------------------------------------------------------------
    # Core lifecycle

    @property
    def is_running(self) -> bool:
        """True while a run is in progress (started and not yet finished).

        Used to refuse destructive actions mid-recording — e.g. a hardware
        hot-reload, which would tear down devices and abandon their writers.
        """
        return self.start_time is not None and not self._finished_event.is_set()

    def run(self) -> None:
        """Drive a standard experiment run.

        Subclasses may override, but the default body is generic and handles
        any combination of devices declared in ``hardware.yaml``.
        """
        self.logger.info("================= Starting experiment ===================")

        # 0. Reset per-run termination state. `_cleanup_procedure` latches
        # `_cleanup_started` so the duration timer and the primary's `finished`
        # signal only tear down once; without resetting it here, a *second*
        # `run()` would short-circuit cleanup at the guard, so `stop_all()`
        # never fires and non-primary capture threads hang with their writers
        # unflushed. Clearing `_finished_event` keeps `run_until_finished`
        # correct on re-runs too.
        self._cleanup_started = False
        self._finished_event.clear()

        # 1. DataManager / queue logger setup
        self.data.setup(self.config)
        if not self.data.devices:
            self.data.register_devices(self.config.hardware.devices.values())
        if not self.playback:
            self.data.start_queue_logger()

        # 2. Subclass pre-run hook
        self.prerun()

        try:
            # 3. Per-run device prep
            self.hardware.arm_all(self.config)

            # 3b. Optional trigger gate (subclass hook): hold here until an
            # external/manual trigger before anything starts.
            self.await_trigger()

            # 4. Wire termination: primary device's finished signal triggers cleanup
            self.hardware.primary.signals.finished.connect(self._cleanup_procedure)

            # 5. Start everything
            self.start_time = datetime.now(timezone.utc)
            self.events.procedure_started.emit()
            self.hardware.start_all()

            # 6. Subclass post-start hook
            self.on_started()

            # 7. Arm the wall-clock duration cap (see `stop_after_duration`).
            self._arm_duration_timer()
        except Exception as e:
            self.logger.error(f"Error during experiment: {e}")
            self.events.procedure_error.emit(str(e))
            raise

    def _arm_duration_timer(self) -> None:
        """Stop the run after the configured ``duration`` (seconds).

        No-op when ``stop_after_duration`` is False or ``duration`` is unset /
        non-positive. The timer is a daemon so it never blocks interpreter
        exit, and cleanup cancels it if the run ends first.
        """
        if not self.stop_after_duration:
            return
        try:
            duration = float(self.config.get("duration") or 0)
        except (TypeError, ValueError):
            duration = 0.0
        if duration <= 0:
            return
        self.logger.info(f"Duration cap armed: {duration:g}s")
        self._duration_timer = threading.Timer(duration, self.cleanup)
        self._duration_timer.daemon = True
        self._duration_timer.start()

    def save_data(self) -> None:
        mgr = getattr(self, "data_manager", self.data)
        mgr.save.configuration()
        mgr.save.all_notes()
        mgr.save.all_hardware()
        mgr.save.save_timestamps(self.protocol, self.start_time, self.stopped_time)
        self.config.save_json()
        self.events.data_saved.emit()
        self.logger.info("Data saved successfully")

    def cleanup(self) -> None:
        """Public cleanup entry-point (manual stop)."""
        self._cleanup_procedure()

    def launch(self, *, splash: bool = True) -> int:
        """Open the Mesofield GUI for this procedure and block until closed.

        The ordinary-Python alternative to ``mesofield launch``: build the
        procedure in a script, then call ``proc.launch()``. The acquisition is
        started from the GUI's Record button (which calls :meth:`run`), so the
        window comes up ready to configure and record.

        Returns the Qt application exit code. ``splash=False`` skips the
        ASCII splash screen. The heavy GUI dependencies are imported lazily so
        headless scripts that never call this keep a light import footprint.
        """
        from mesofield.gui.maingui import run_gui

        return run_gui(self, splash=splash)

    def run_until_finished(self, timeout: Optional[float] = None) -> bool:
        """Run the procedure and block until cleanup completes.

        Starts the procedure via :meth:`run`, then waits for the
        ``procedure_finished`` (or ``procedure_error``) signal.  Handles
        ``KeyboardInterrupt`` and ``timeout`` by invoking :meth:`cleanup`
        automatically, so callers (e.g. ``__main__`` blocks in experiment
        scripts) do not need to wire up their own threading events.

        Parameters
        ----------
        timeout:
            Optional hard ceiling in seconds.  When ``None`` (default), waits
            indefinitely for the primary device's ``finished`` signal.  When
            provided, forces cleanup if the deadline passes.

        Returns
        -------
        bool
            ``True`` if the procedure finished on its own, ``False`` if
            cleanup was forced by timeout or interrupt.
        """
        self._finished_event.clear()
        deadline = None if timeout is None else (datetime.now().timestamp() + timeout)
        try:
            self.run()
            while not self._finished_event.is_set():
                remaining = None
                if deadline is not None:
                    remaining = deadline - datetime.now().timestamp()
                    if remaining <= 0:
                        self.logger.warning(
                            "run_until_finished: timeout reached, forcing cleanup"
                        )
                        break
                self._finished_event.wait(timeout=min(0.5, remaining) if remaining else 0.5)
        except KeyboardInterrupt:
            self.logger.info("run_until_finished: KeyboardInterrupt, cleaning up")
        finally:
            if not self._finished_event.is_set():
                self.cleanup()
        return self._finished_event.is_set()

    def _cleanup_procedure(self):
        # The wall-clock duration timer and the primary's `finished` signal can
        # both reach here; run the teardown exactly once.
        if self._cleanup_started:
            return
        self._cleanup_started = True
        self.logger.info("Cleanup Procedure")
        try:
            if self._duration_timer is not None:
                self._duration_timer.cancel()
                self._duration_timer = None
            try:
                self.hardware.primary.signals.finished.disconnect(self._cleanup_procedure)
            except Exception:
                pass
            # Detach any procedure-authored FrameProcessors so their worker
            # threads exit and the camera frame signal is released before
            # the hardware itself shuts down.
            for proc in list(getattr(self, "processors", [])):
                try:
                    proc.detach()
                except Exception as exc:
                    self.logger.warning(f"processor detach failed: {exc}")
            self.processors.clear()
            self.hardware.stop_all()
            if not self.playback:
                self.data.stop_queue_logger()
            self.stopped_time = datetime.now(timezone.utc)
            if not self.playback:
                self.save_data()
                self._write_acquisition_manifest()
            self.on_finished()
            self.events.procedure_finished.emit()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            self.events.procedure_error.emit(str(e))
        finally:
            # `events.procedure_finished` is a pyqtSignal; its Python-side
            # `.connect`s only run with a live QApplication. Setting the
            # event directly here keeps `run_until_finished` working in
            # headless contexts (tests, CLI smoke runs, batch scripts).
            self._finished_event.set()

    # ------------------------------------------------------------------
    # Acquisition manifest (mesokit-schema contract)

    def manifest_extra(self) -> Dict[str, Any]:
        """Override to inject extra session-level metadata into the
        AcquisitionManifest's ``extra`` block. Default: empty."""
        return {}

    def _write_acquisition_manifest(self) -> None:
        """Emit a `mesokit_schema.AcquisitionManifest` next to the data.

        Called automatically during cleanup. Override to disable or change
        where the manifest lands; use :meth:`manifest_extra` if you only
        want to attach extra session metadata.
        """
        session_root = (
            Path(self.data_dir)
            / f"sub-{self.config.subject}"
            / f"ses-{self.config.session}"
        )
        session_root.mkdir(parents=True, exist_ok=True)

        def _relativise(p: Any) -> Optional[str]:
            if not p:
                return None
            try:
                return str(Path(p).resolve().relative_to(session_root.resolve()))
            except ValueError:
                return str(p)

        def _coerce_sidecars(raw) -> list[SidecarEntry]:
            out: list[SidecarEntry] = []
            for item in raw or []:
                if isinstance(item, SidecarEntry):
                    rel = _relativise(item.path) or item.path
                    out.append(item.model_copy(update={"path": rel}))
                else:
                    data = dict(item)
                    rel = _relativise(data.get("path"))
                    if rel is not None:
                        data["path"] = rel
                    out.append(SidecarEntry.model_validate(data))
            return out

        def _coerce_dataqueue_schema(raw) -> Optional[DataqueuePayloadSchema]:
            if raw is None:
                return None
            if isinstance(raw, DataqueuePayloadSchema):
                return raw
            return DataqueuePayloadSchema.model_validate(raw)

        producers: list[ProducerEntry] = []
        for device_id, device in self.config.hardware.devices.items():
            output_path = getattr(device, "output_path", None)
            if not output_path:
                continue
            sidecars_method = getattr(device, "sidecars", None)
            sidecar_list = sidecars_method() if callable(sidecars_method) else []
            dq_schema_raw = getattr(device, "dataqueue_payload_schema", None)
            producers.append(
                ProducerEntry(
                    device_id=device_id,
                    device_type=getattr(device, "device_type", "device"),
                    data_type=getattr(device, "data_type", device_id),
                    bids_type=getattr(device, "bids_type", None),
                    file_type=getattr(device, "file_type", "csv"),
                    output_path=_relativise(output_path) or str(output_path),
                    metadata_path=_relativise(getattr(device, "metadata_path", None)),
                    sampling_rate_hz=getattr(device, "sampling_rate", None) or None,
                    time_basis=TimeBasis(
                        clock_source=getattr(device, "clock_source", "wall_unix_s"),
                    ),
                    calibration=dict(getattr(device, "calibration", {}) or {}),
                    sidecars=_coerce_sidecars(sidecar_list),
                    dataqueue_schema=_coerce_dataqueue_schema(dq_schema_raw),
                )
            )

        manifest = AcquisitionManifest(
            mesofield_version=str(_MESOFIELD_VERSION),
            acquisition_complete=True,
            started_at=self.start_time,
            ended_at=self.stopped_time,
            session=SessionIdentity(
                subject=str(self.config.subject),
                session=str(self.config.session),
                task=str(self.config.task) if self.config.task else None,
                experimenter=self.experimenter,
                protocol=self.protocol,
            ),
            producers=producers,
            extra=self.manifest_extra(),
        )
        out = session_root / "manifest.json"
        manifest.write(out)
        self.logger.info(f"Wrote AcquisitionManifest {hyperlink(out, 'AcquisitionManifest')}")

    # ------------------------------------------------------------------

    def add_note(self, note: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.config.notes.append(f"{timestamp}: {note}")
        self.logger.info(f"Added note: {note}")



# ----------------------------------------------------------------------
# Procedure discovery

def _resolve_target(
    target: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Map a launch *target* to ``(hardware_yaml, experiment_json, procedure_py)``.

    The single place that owns the file-discovery convention:

    - ``None``               -> nothing (default state)
    - ``*.yaml`` / ``*.yml``  -> hardware only
    - ``*.py``               -> scripted Procedure subclass (params/hardware via ``define_*``)
    - ``*.json``             -> experiment params + sibling ``hardware.yaml`` if present
    - directory              -> ``hardware.yaml`` and/or ``experiment.json`` found inside
    """
    if not target:
        return None, None, None
    p = Path(target)
    ext = p.suffix.lower()
    if ext in (".yaml", ".yml"):
        return target, None, None
    if ext == ".py":
        return None, None, target
    if ext == ".json":
        sibling = p.parent / "hardware.yaml"
        return (str(sibling) if sibling.is_file() else None), target, None
    if p.is_dir():
        hw = p / "hardware.yaml"
        exp = p / "experiment.json"
        return (
            str(hw) if hw.is_file() else None,
            str(exp) if exp.is_file() else None,
            None,
        )
    return None, None, None


def load_procedure_from_config(target: Optional[str]) -> "Procedure":
    """Build the right :class:`Procedure` for a launch *target*.

    ``target`` may be a ``hardware.yaml``, an ``experiment.json``, a scripted
    ``procedure.py``, an experiment directory, or ``None``. Discovery of the
    hardware/experiment pair is delegated to :func:`_resolve_target`.

    When the experiment JSON declares ``procedure_file`` + ``procedure_class``,
    that subclass is imported and used; otherwise a base :class:`Procedure`.
    """
    hardware, experiment, procedure_py = _resolve_target(target)

    if procedure_py:
        return _load_procedure_from_py(procedure_py)

    cls: Type[Procedure] = Procedure
    if experiment:
        declared = _load_declared_procedure_class(experiment)
        if declared is not None:
            cls = declared

    return cls(hardware=hardware, experiment=experiment)


def _load_declared_procedure_class(
    experiment_json: str,
) -> Optional[Type["Procedure"]]:
    """Import the :class:`Procedure` subclass declared by ``procedure_file`` /
    ``procedure_class`` in an experiment JSON, or ``None`` when not declared.

    The user file is loaded via :func:`importlib.util.spec_from_file_location`
    so ``experiments/`` does not need to be on ``sys.path``.
    """
    try:
        with open(experiment_json, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception:
        return None

    proc_file = cfg.get("procedure_file")
    proc_class = cfg.get("procedure_class")
    if not proc_file or not proc_class:
        return None

    # Resolve relative paths against the JSON's directory
    json_dir = os.path.dirname(os.path.abspath(experiment_json))
    if not os.path.isabs(proc_file):
        proc_file = os.path.join(json_dir, proc_file)

    if not os.path.isfile(proc_file):
        raise FileNotFoundError(
            f"procedure_file declared in {experiment_json} not found: {proc_file}"
        )

    mod_name = f"mesofield_user_procedure_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, proc_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load procedure_file: {proc_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

    cls = getattr(module, proc_class, None)
    if cls is None:
        raise AttributeError(f"Class '{proc_class}' not found in {proc_file}")
    if not (isinstance(cls, type) and issubclass(cls, Procedure)):
        raise TypeError(
            f"{proc_class} in {proc_file} must be a subclass of mesofield.base.Procedure"
        )
    return cls


def _load_procedure_from_py(py_path: str) -> "Procedure":
    """Instantiate the Procedure subclass defined in a scripted ``procedure.py``.

    The class is selected from a module-level ``PROCEDURE`` attribute when
    present, otherwise the single :class:`Procedure` subclass *defined in*
    that file. The instance's ``experiment_dir`` is set to the script's
    directory so data lands beside it; the ``define_config`` /
    ``define_hardware`` hooks supply the actual parameters and devices.
    """
    abs_path = os.path.abspath(py_path)
    mod_name = f"mesofield_user_procedure_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load procedure file: {abs_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

    cls = getattr(module, "PROCEDURE", None)
    if cls is None:
        candidates = [
            obj for obj in vars(module).values()
            if isinstance(obj, type)
            and issubclass(obj, Procedure)
            and obj is not Procedure
            and obj.__module__ == mod_name
        ]
        if not candidates:
            raise AttributeError(
                f"No Procedure subclass found in {abs_path}. Define one, or "
                f"set a module-level 'PROCEDURE = <class>'."
            )
        if len(candidates) > 1:
            names = ", ".join(c.__name__ for c in candidates)
            raise AttributeError(
                f"Multiple Procedure subclasses in {abs_path} ({names}); "
                f"set a module-level 'PROCEDURE = <class>' to disambiguate."
            )
        cls = candidates[0]

    if not (isinstance(cls, type) and issubclass(cls, Procedure)):
        raise TypeError(
            f"{cls!r} in {abs_path} must be a subclass of mesofield.base.Procedure"
        )
    proc = cls()
    # Scripted procedures supply params/hardware via define_*; when the
    # config did not declare an `experiment_directory`, default the data
    # output beside the script rather than the current directory.
    if not proc.config.experiment_dir_is_set:
        proc.config.experiment_dir = os.path.dirname(abs_path)
        proc.data_dir = proc.config.data_dir
    return proc


# Legacy constants for backward compatibility
NAME = "mesofield"

