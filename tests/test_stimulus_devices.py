"""Tests for the shared subprocess-stimulus pattern.

Covers the reusable :class:`~mesofield.devices.stimulus_base.SubprocessStimulusDevice`
lifecycle (readiness handshake, fail-fast, enabled gate, require_ready, operator
hooks, re-run) and the PsychoPy port that now rides on it. No Qt and no real
hardware are required: a stub interpreter command stands in for the stimulus app.

Each test that spawns a subprocess guarantees teardown via :func:`_run_device`
so no child/reader thread lingers to perturb later tests.
"""

from __future__ import annotations

import sys
import time
import types
from contextlib import contextmanager

from mesofield.devices.stimulus_base import SubprocessStimulusDevice

# Short-but-safe: long enough that the child is still alive while we assert,
# short enough not to slow the suite. Every test stops the device promptly.
_ALIVE = 1.0


def _ready_cmd(token: str, sleep: float = _ALIVE):
    """A command that prints ``token`` then idles (a stimulus that boots OK)."""
    return [sys.executable, "-c", f"import time; print('{token}', flush=True); time.sleep({sleep})"]


def _alive_no_token_cmd(sleep: float = _ALIVE):
    """A command that stays alive but never prints the ready token."""
    return [sys.executable, "-c", f"import time; time.sleep({sleep})"]


def _die_cmd(code: int = 3):
    """A command that exits immediately without printing the ready token."""
    return [sys.executable, "-c", f"import sys; sys.exit({code})"]


class _FakeStim(SubprocessStimulusDevice):
    default_device_id = "fake"
    ready_token = "FAKE_READY"
    launch_phase = "start"
    require_ready = True

    def __init__(self, cfg, cmd):
        super().__init__(cfg)
        self._cmd = cmd

    def build_command(self):
        return self._cmd


@contextmanager
def _run_device(dev):
    """Yield a device, then always tear its subprocess down."""
    try:
        yield dev
    finally:
        try:
            dev.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Base engine
# ---------------------------------------------------------------------------


def test_ready_handshake_sets_status_and_flag():
    with _run_device(_FakeStim({"id": "fake", "ready_timeout": 8}, _ready_cmd("FAKE_READY"))) as d:
        d.arm(None)
        assert d.start() is True
        assert d.gui_status == "ready"
        assert d.handshake_ok is True
    assert d.gui_status == "stopped"


def test_fail_fast_on_early_exit():
    d = _FakeStim({"id": "fake", "ready_timeout": 30}, _die_cmd())
    d.arm(None)
    t0 = time.monotonic()
    ok = d.start()
    elapsed = time.monotonic() - t0
    assert ok is False
    assert d.gui_status == "failed"
    assert not d.handshake_ok
    # Must not wait out the 30s timeout when the child dies immediately.
    assert elapsed < 5.0


def test_enabled_gate_skips_launch():
    d = _FakeStim({"id": "fake"}, _ready_cmd("FAKE_READY"))
    d.enabled = False
    d.arm(None)
    assert d.start() is False
    assert d.gui_status == "loaded"


def test_require_ready_false_accepts_running_child():
    class _Lenient(_FakeStim):
        require_ready = False

    with _run_device(_Lenient({"id": "lenient", "ready_timeout": 1}, _alive_no_token_cmd(sleep=3))) as d:
        d.arm(None)
        assert d.start() is True
        assert d.gui_status == "running"  # alive but no handshake
        assert not d.handshake_ok


def test_rerun_relaunches_same_instance():
    with _run_device(_FakeStim({"id": "fake", "ready_timeout": 8}, _ready_cmd("FAKE_READY"))) as d:
        d.arm(None)
        assert d.start() and d.handshake_ok
        d.stop()
        # Second run on the same instance must relaunch, not no-op on a stale handle.
        d.arm(None)
        assert d.start() and d.handshake_ok


def test_operator_hooks_and_cancel():
    calls = {"launching": 0, "dismiss": 0, "confirm": 0}

    class _Interactive(_FakeStim):
        def present_launching(self):
            calls["launching"] += 1

        def dismiss_launching(self):
            calls["dismiss"] += 1

        def confirm_ready_to_record(self):
            calls["confirm"] += 1
            return False  # operator cancels at the ready gate

    with _run_device(_Interactive({"id": "fake", "ready_timeout": 8}, _ready_cmd("FAKE_READY"))) as d:
        d.arm(None)
        assert d.start() is False  # cancelled at the ready gate
        assert calls == {"launching": 1, "dismiss": 1, "confirm": 1}
        # Cancelling stops the subprocess.
        assert d._process is None or not d._process.is_running()


def test_present_failure_receives_output_tail():
    captured = {}

    class _Reporter(_FakeStim):
        def present_failure(self, message, detail=""):
            captured["message"] = message
            captured["detail"] = detail

    cmd = [sys.executable, "-c", "import sys; sys.stdout.write('boom-marker\\n'); sys.exit(2)"]
    d = _Reporter({"id": "fake", "ready_timeout": 8}, cmd)
    d.arm(None)
    assert d.start() is False
    assert "boom-marker" in captured.get("detail", "")


# ---------------------------------------------------------------------------
# PsychoPy port
# ---------------------------------------------------------------------------


def test_psychopy_is_subclass_with_canonical_parser():
    from mesofield.devices.psychopy_device import PsychoPyDevice
    from mesofield.datakit.sources.behavior.psychopy import Psychopy
    from mesofield.datakit.sources import SOURCE_REGISTRY
    from mesofield import DeviceRegistry

    assert issubclass(PsychoPyDevice, SubprocessStimulusDevice)
    # The .Parser convention and the source registry resolve to the SAME class
    # (no stale duplicate).
    assert PsychoPyDevice.Parser is Psychopy
    assert SOURCE_REGISTRY["psychopy"] is Psychopy
    assert DeviceRegistry.get_class("psychopy") is PsychoPyDevice


def test_psychopy_build_command_and_preflight(tmp_path):
    from mesofield.devices.psychopy_device import PsychoPyDevice

    script = tmp_path / "stim.py"
    script.write_text("print('PSYCHOPY_READY', flush=True)\n")
    cfg_obj = types.SimpleNamespace(
        psychopy_path=str(script),
        psychopy_parameters={"subject": "001", "num_trials": 3},
    )
    d = PsychoPyDevice({"id": "psychopy", "python_exe": "C:/x/python.exe"})
    d.prepare(cfg_obj)
    cmd = d.build_command()
    assert cmd[0] == "C:/x/python.exe"
    assert cmd[1] == str(script)
    # params arrive as a base64 JSON argv token (decodable with only the stdlib)
    import base64 as _b64, json as _json
    assert _json.loads(_b64.b64decode(cmd[2])) == {"subject": "001", "num_trials": 3}
    assert d.preflight() is None

    # missing script -> preflight returns an actionable error
    d.prepare(types.SimpleNamespace(psychopy_path=str(tmp_path / "missing.py"), psychopy_parameters={}))
    assert d.preflight() is not None


def test_psychopy_headless_lifecycle(tmp_path):
    from mesofield.devices.psychopy_device import PsychoPyDevice

    script = tmp_path / "stim.py"
    script.write_text(
        "import sys, json, base64, time\n"
        "json.loads(base64.b64decode(sys.argv[1]))\n"
        "print('PSYCHOPY_READY', flush=True)\n"
        f"time.sleep({_ALIVE})\n"
    )
    cfg_obj = types.SimpleNamespace(
        psychopy_path=str(script), psychopy_parameters={"subject": "001"}
    )
    d = PsychoPyDevice({"id": "psychopy", "python_exe": sys.executable, "ready_timeout": 8})
    with _run_device(d):
        d.arm(cfg_obj)
        assert d.start() is True
        assert d.handshake_ok


def test_psychopy_stanza_loads_via_init_extras():
    from mesofield.hardware import HardwareManager

    hm = HardwareManager()
    hm.yaml = {"psychopy": {"type": "psychopy", "ready_timeout": 45.0, "widgets": ["psychopy"]}}
    hm._init_extras()

    dev = hm.get_device("psychopy")
    assert dev is not None
    assert dev.device_type == "stimulus"
    assert dev.ready_timeout == 45.0
    assert "psychopy" in hm._aggregate_widgets()
