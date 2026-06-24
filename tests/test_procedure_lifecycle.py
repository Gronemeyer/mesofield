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
from mesofield.base import Procedure


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
    proc.cleanup()  # second teardown is a no-op (the _cleanup_started guard)
    assert proc.calls.count("on_finished") == 1


# --------------------------------------------------------------------------- #
# Start gate (await_trigger)
# --------------------------------------------------------------------------- #
def test_start_gate_cancel_raises(hardware_yaml, experiment_json, tmp_path):
    proc = _build(
        Procedure, hardware_yaml, experiment_json, tmp_path, start_on_trigger=True
    )
    proc.start_gate = lambda _p: False  # operator cancels
    try:
        with pytest.raises(RuntimeError):
            proc.run()
    finally:
        proc.cleanup()  # stop the queue-logger thread started before the gate


def test_start_gate_proceeds_when_accepted(hardware_yaml, experiment_json, tmp_path):
    gate_calls = []
    proc = _build(
        Procedure, hardware_yaml, experiment_json, tmp_path, start_on_trigger=True
    )
    proc.start_gate = lambda p: (gate_calls.append(p) or True)
    proc.run()
    try:
        assert gate_calls == [proc]      # gate consulted exactly once
        assert proc.start_time is not None  # run proceeded past the gate
    finally:
        proc.cleanup()


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
