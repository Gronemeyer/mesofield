"""Device-Protocol conformance -- the mock-drift guard.

The mock devices in ``mesofield/devices/mocks.py`` stand in for real hardware in
the dev rig and across the test suite. The risk (called out by the project
owner) is that a mock silently drifts from the real device interface it
substitutes for. These tests turn that drift into a *caught* failure:

* every registered device class exposes the core lifecycle surface
  ``HardwareManager`` drives (checked at class level, so it works even for
  classes that need real hardware to construct);
* each mock producer exposes every contract method its real counterpart does;
* the constructed mocks structurally satisfy the ``@runtime_checkable``
  Protocols in :mod:`mesofield.protocols`.

If a real device gains a contract method and its mock lags (or vice versa), the
pairing test fails instead of leaving a silent gap.
"""

from __future__ import annotations

import pytest

# Importing the package runs every device module's @DeviceRegistry.register
# decorator (daq, cameras, encoder, treadmill, psychopy, mocks).
import mesofield.devices  # noqa: F401
from mesofield import DeviceRegistry
from mesofield.protocols import DataProducer, HardwareDevice, StimulusDevice

# The lifecycle methods HardwareManager calls on every device, regardless of
# kind (arm_all / start_all / stop_all / deinitialize bring-up + teardown).
_LIFECYCLE_METHODS = ("initialize", "arm", "start", "stop", "shutdown")
# Extra surface DataSaver relies on for data-producing devices.
_PRODUCER_METHODS = ("save_data", "get_data")

# (mock registry key, real registry key) -- a mock must not lag its real twin.
_PRODUCER_PAIRS = [
    ("mock_wheel", "wheel"),
    ("mock_camera", "camera"),
    ("mock_camera", "opencv_camera"),
]

_REGISTERED = sorted(DeviceRegistry._registry.items())


@pytest.mark.parametrize(
    "key,cls", _REGISTERED, ids=[k for k, _ in _REGISTERED]
)
def test_registered_device_exposes_lifecycle_surface(key, cls):
    """Every registered device class has the methods HardwareManager calls."""
    missing = [m for m in _LIFECYCLE_METHODS if not callable(getattr(cls, m, None))]
    assert not missing, f"{key} ({cls.__name__}) missing lifecycle methods: {missing}"


@pytest.mark.parametrize("mock_key,real_key", _PRODUCER_PAIRS)
def test_mock_producer_matches_real_surface(mock_key, real_key):
    """A mock producer exposes every contract method its real counterpart has."""
    mock_cls = DeviceRegistry._registry.get(mock_key)
    real_cls = DeviceRegistry._registry.get(real_key)
    if real_cls is None:
        pytest.skip(f"real device {real_key!r} not registered in this environment")
    assert mock_cls is not None, f"mock device {mock_key!r} not registered"

    for method in _LIFECYCLE_METHODS + _PRODUCER_METHODS:
        if callable(getattr(real_cls, method, None)):
            assert callable(getattr(mock_cls, method, None)), (
                f"{mock_key} ({mock_cls.__name__}) is missing {method}() that real "
                f"{real_key} ({real_cls.__name__}) provides -- mock drift"
            )


def test_mock_devices_satisfy_runtime_protocols():
    """Constructed mocks structurally satisfy HardwareDevice + DataProducer."""
    from mesofield.devices.mocks import MockEncoderDevice, MockFrameProducer

    wheel = MockEncoderDevice({"id": "wheel"})
    camera = MockFrameProducer({"id": "camera"})
    for dev in (wheel, camera):
        assert isinstance(dev, HardwareDevice)
        assert isinstance(dev, DataProducer)
        # The DeviceSignals bundle every producer streams through.
        for sig in ("started", "finished", "data"):
            assert hasattr(dev.signals, sig), f"{dev.device_id}: signals.{sig} missing"


def test_psychopy_is_a_stimulus_device():
    """The PsychoPy device satisfies the StimulusDevice (not DataProducer) role."""
    from mesofield.devices.psychopy_device import PsychoPyDevice

    dev = PsychoPyDevice({"id": "psychopy"})
    assert isinstance(dev, HardwareDevice)
    assert isinstance(dev, StimulusDevice)
    assert dev.device_type == "stimulus"
