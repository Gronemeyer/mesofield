"""Shared fixtures for the mesofield suite.

Plain helper doubles (``MockSerial``, ``StubConfig``, ``SignalCounter``)
live in ``_helpers.py`` so they can be imported at module scope; the
fixtures below (``mock_rig_procedure``, ``populated_registry``) are
requested by name. See ``docs/developer_guide.md`` (Testing & development
bounds) for how the layers fit together.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def populated_registry():
    """Import every built-in device module so ``@DeviceRegistry.register``
    decorators have run, then return the registry. Used by the protocol
    conformance suite to parametrize over all known device classes."""
    import importlib

    from mesofield import DeviceRegistry

    for mod in (
        "mesofield.devices.cameras",
        "mesofield.devices.mocks",
        "mesofield.devices.encoder",
        "mesofield.devices.treadmill",
        "mesofield.playback",
    ):
        importlib.import_module(mod)
    return DeviceRegistry


@pytest.fixture
def mock_rig_procedure(tmp_path):
    """Factory building a headless :class:`Procedure` driving a mock rig.

    Returns ``make(duration=1.0, primary="camera", extra_devices=None,
    crashing=False)`` -> a Procedure whose hardware is a mock camera +
    mock wheel (no YAML, no Qt). The primary self-terminates at
    ``duration`` seconds, so ``run_until_finished`` returns on its own.
    Data lands under ``tmp_path``.
    """
    from mesofield.base import Procedure
    from mesofield.devices.mocks import MockEncoderDevice, MockFrameProducer

    class _CrashingEncoder(MockEncoderDevice):
        """Records one sample then dies — drives the device-failure path."""

        def _acquire_loop(self):
            self.record(1)
            raise RuntimeError("synthetic acquisition failure")

    def make(
        duration: float = 1.0,
        primary: str = "camera",
        crashing: bool = False,
        extra_devices: Optional[list] = None,
    ) -> "Procedure":
        camera = MockFrameProducer({
            "id": "camera",
            "width": 16,
            "height": 16,
            "frame_interval_ms": 50,
            "primary": primary == "camera",
            "output": {"suffix": "meso", "file_type": "ome.tiff", "bids_type": "func"},
        })
        wheel_cls = _CrashingEncoder if crashing else MockEncoderDevice
        wheel = wheel_cls({
            "id": "wheel",
            "sample_interval_ms": 50,
            "primary": primary == "wheel",
            "output": {"suffix": "wheel", "file_type": "csv", "bids_type": "beh"},
        })
        # define_hardware returns a list of pre-built device objects.
        devices: list[Any] = [camera, wheel]
        if extra_devices:
            devices.extend(extra_devices)

        cfg = {
            "Configuration": {
                "experimenter": "tester",
                "protocol": "mock_rig",
                "duration": duration,
                "start_on_trigger": False,
            },
            "Subjects": {"M1": {"session": "01", "task": "t"}},
            "DisplayKeys": ["duration", "session", "task"],
        }

        class _MockRigProcedure(Procedure):
            def define_config(self):
                return cfg

            def define_hardware(self):
                return devices

        proc = _MockRigProcedure()
        proc.config.experiment_dir = str(tmp_path)
        return proc

    return make
