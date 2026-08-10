"""Procedure orchestration -- run() sequence, hooks, gate, cleanup, manifest.

Drives the real Procedure lifecycle with the mock rig (no demo dir, no
hardware). Most tests use ``run()`` + ``cleanup()`` directly -- the same path a
manual abort takes -- so they are fast and deterministic; one test exercises the
wall-clock duration cap end to end. Complements ``test_rerecord`` (re-run) and
``test_workflow`` (data outcomes).
"""

from __future__ import annotations

import json
import time

import pytest

# Register the mock device types used by the hardware_yaml fixture.
import mesofield.devices.mocks  # noqa: F401
from mesofield.base import Procedure, RunState


class _FakeStim:
    device_type = "stimulus"
    launch_phase = "start"
    enabled = True
    device_id = "stim"
    file_type = ""
    bids_type = None

    def __init__(self, ok: bool):
        self._ok = ok
        self.started = False

    def start(self) -> bool:
        self.started = True
        return self._ok

    def stop(self) -> bool:
        return True


# --------------------------------------------------------------------------- #
# Constructor precedence: kwargs > define_config > experiment.json > defaults
# --------------------------------------------------------------------------- #
def test_kwargs_override_json(experiment_json):
    proc = Procedure(config=str(experiment_json(duration=5)), duration=2)
    assert proc.config.get("duration") == 2


def test_define_config_supersedes_json(experiment_json):
    class _ConfigProc(Procedure):
        def define_config(self):
            return {"duration": 7, "task": "hook"}

    # define_config wins over experiment.json; an explicit kwarg still wins over
    # define_config.
    proc = _ConfigProc(config=str(experiment_json(duration=5)), task="kw")
    assert proc.config.get("duration") == 7
    assert proc.config.get("task") == "kw"


# --------------------------------------------------------------------------- #
# Lifecycle hooks + cleanup
# --------------------------------------------------------------------------- #
class _HookProc(Procedure):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls: list[str] = []

    def prerun(self):
        self.calls.append("prerun")

    def on_started(self):
        self.calls.append("on_started")

    def on_finished(self):
        self.calls.append("on_finished")


def _build(cls, hardware_yaml, experiment_json, tmp_path, **cfg):
    return cls(
        hardware=str(hardware_yaml()),
        config=str(experiment_json(**cfg)),
        experiment_directory=str(tmp_path / "out"),
    )


def test_lifecycle_hooks_fire_in_order(hardware_yaml, experiment_json, tmp_path):
    proc = _build(_HookProc, hardware_yaml, experiment_json, tmp_path)
    proc.run()       # prerun -> arm -> start -> on_started (synchronous)
    proc.cleanup()   # stop -> save -> on_finished
    assert proc.calls == ["prerun", "on_started", "on_finished"]
    assert proc.stopped_time is not None


def test_cleanup_runs_only_once(hardware_yaml, experiment_json, tmp_path):
    proc = _build(_HookProc, hardware_yaml, experiment_json, tmp_path)
    proc.run()
    proc.cleanup()
    proc.cleanup()  # second teardown is a no-op (the run is already DONE)
    assert proc.calls.count("on_finished") == 1


# --------------------------------------------------------------------------- #
# Start gate (await_trigger)
# --------------------------------------------------------------------------- #
def test_start_gate_cancel_raises(hardware_yaml, experiment_json, tmp_path, fake_ui):
    proc = _build(
        Procedure, hardware_yaml, experiment_json, tmp_path, start_on_trigger=True
    )
    fake_ui.answer = False  # operator cancels
    with pytest.raises(RuntimeError, match="cancelled at the start gate"):
        proc.run()


def test_start_gate_proceeds_when_accepted(
    hardware_yaml, experiment_json, tmp_path, fake_ui
):
    proc = _build(
        Procedure, hardware_yaml, experiment_json, tmp_path, start_on_trigger=True
    )
    proc.run()
    try:
        assert len(fake_ui.confirmed) == 1     # operator asked exactly once
        assert proc.state is RunState.RUNNING  # run proceeded past the gate
    finally:
        proc.cleanup()


def test_start_gate_defers_to_a_start_phase_stimulus(
    hardware_yaml, experiment_json, tmp_path, fake_ui
):
    """With a start-phase stimulus, its own ready gate is the trigger."""
    proc = _build(
        Procedure, hardware_yaml, experiment_json, tmp_path, start_on_trigger=True
    )
    stim = _FakeStim(ok=True)
    proc.hardware.devices["stim"] = stim
    proc.run()
    try:
        assert stim.started
        assert fake_ui.confirmed == []
    finally:
        proc.cleanup()


def test_start_gate_cancels_when_the_stimulus_fails(
    hardware_yaml, experiment_json, tmp_path, fake_ui
):
    proc = _build(
        Procedure, hardware_yaml, experiment_json, tmp_path, start_on_trigger=True
    )
    proc.hardware.devices["stim"] = _FakeStim(ok=False)
    with pytest.raises(RuntimeError, match="did not start"):
        proc.run()


def test_cancel_at_the_gate_tears_down_the_armed_run(
    hardware_yaml, experiment_json, tmp_path, fake_ui
):
    """Cancelling must not leave the armed rig running.

    Devices and the queue logger are up by the time the gate is consulted and
    nothing below it is self-terminating. Equally, the run never started, so no
    timestamps or acquisition manifest may be written for it.
    """
    out = tmp_path / "out"
    proc = _build(
        Procedure, hardware_yaml, experiment_json, tmp_path, start_on_trigger=True
    )
    fake_ui.answer = False
    stopped: list = []
    for dev in proc.hardware.devices.values():
        original = dev.stop
        dev.stop = lambda *a, _d=dev, _o=original, **k: (
            stopped.append(_d.device_id) or _o(*a, **k)
        )

    with pytest.raises(RuntimeError):
        proc.run()

    assert stopped == [d.device_id for d in proc.hardware.devices.values()]
    logger_thread = proc.data._queue_thread
    assert logger_thread is None or not logger_thread.is_alive()
    assert proc.state is RunState.DONE
    assert proc._finished_event.is_set()
    assert proc.start_time is None
    assert proc.stopped_time is None
    assert not list(out.rglob("manifest.json"))

    # A follow-up cleanup() must not then save against an unset start_time.
    proc.cleanup()
    assert not list(out.rglob("manifest.json"))
    assert proc.stopped_time is None


# --------------------------------------------------------------------------- #
# Wall-clock duration cap
# --------------------------------------------------------------------------- #
def test_duration_cap_stops_the_run(hardware_yaml, experiment_json, tmp_path):
    proc = _build(Procedure, hardware_yaml, experiment_json, tmp_path, duration=1)
    t0 = time.monotonic()
    assert proc.run_until_finished(timeout=10) is True
    elapsed = time.monotonic() - t0
    assert proc.stopped_time is not None
    assert elapsed < 8, "run ended via the 10s timeout, not the 1s duration cap"


# --------------------------------------------------------------------------- #
# manifest_extra injection
# --------------------------------------------------------------------------- #
def test_manifest_extra_is_written(hardware_yaml, experiment_json, tmp_path):
    class _ExtraProc(Procedure):
        def manifest_extra(self):
            return {"rig_label": "bench-A"}

    out = tmp_path / "out"
    proc = _ExtraProc(
        hardware=str(hardware_yaml()),
        config=str(experiment_json()),
        experiment_directory=str(out),
    )
    proc.run()
    proc.cleanup()

    manifest = json.loads(next(out.rglob("manifest.json")).read_text())
    assert manifest["extra"]["rig_label"] == "bench-A"
