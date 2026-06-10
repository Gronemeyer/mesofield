"""Tests for the three device-registration paths and YAML type resolution.

Covers the surface an experimenter touches when adding a custom device
(see ``mesofield/__init__.py``):

  * ``@DeviceRegistry.register("name")`` decorator,
  * ``register_device(Class, "name")`` programmatic form,
  * import-string ``type:`` (``module:Class`` / ``module.Class``) resolved
    on demand without any registration call,
  * and a clean ``None`` (not a spurious import) for an unknown short name.
"""

from __future__ import annotations

import pytest

from mesofield import DeviceRegistry, register_device
from mesofield.devices.base import BaseDevice


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Snapshot/restore the global registry so tests don't leak keys."""
    saved = dict(DeviceRegistry._registry)
    yield
    DeviceRegistry._registry = saved


def test_decorator_registration():
    @DeviceRegistry.register("decorated_dev")
    class Decorated(BaseDevice):
        pass

    assert DeviceRegistry.get_class("decorated_dev") is Decorated
    assert Decorated.registry_key == "decorated_dev"


def test_programmatic_registration_explicit_key():
    class Lick(BaseDevice):
        device_type = "lick"

    ret = register_device(Lick, "lick_detector")
    assert ret is Lick  # returns class unchanged (usable as decorator)
    assert DeviceRegistry.get_class("lick_detector") is Lick
    assert Lick.registry_key == "lick_detector"


def test_programmatic_registration_defaults_to_class_name():
    class Thermistor(BaseDevice):
        pass

    register_device(Thermistor)
    assert DeviceRegistry.get_class("Thermistor") is Thermistor


def test_import_string_colon_form():
    from mesofield.devices.mocks import MockEncoderDevice

    cls = DeviceRegistry.get_class("mesofield.devices.mocks:MockEncoderDevice")
    assert cls is MockEncoderDevice


def test_import_string_dotted_form():
    from mesofield.devices.mocks import MockFrameProducer

    cls = DeviceRegistry.get_class("mesofield.devices.mocks.MockFrameProducer")
    assert cls is MockFrameProducer


def test_import_string_is_cached_without_stamping_registry_key():
    ref = "mesofield.devices.mocks:MockEncoderDevice"
    first = DeviceRegistry.get_class(ref)
    assert ref in DeviceRegistry._registry  # cached under the literal string
    assert DeviceRegistry.get_class(ref) is first


def test_unknown_short_name_returns_none_no_import_attempt():
    # No dot/colon -> never treated as an import string.
    assert DeviceRegistry.get_class("definitely_not_a_device") is None


def test_import_string_with_missing_module_returns_none():
    assert DeviceRegistry.get_class("no_such_module:Thing") is None


def test_import_string_with_missing_attr_returns_none():
    assert DeviceRegistry.get_class("mesofield.devices.mocks:NoSuchClass") is None


def test_import_string_pointing_at_non_class_returns_none():
    # `record` exists in base but is a function/attr, not a class.
    assert DeviceRegistry.get_class("mesofield.devices.base:Optional") is None


def test_hardware_manager_resolves_import_string(tmp_path):
    """End-to-end: a hardware.yaml stanza using an import-string type builds
    the device with no registration call."""
    import yaml
    from mesofield.hardware import HardwareManager

    yaml_path = tmp_path / "hardware.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "wheel": {
                    "type": "mesofield.devices.mocks:MockEncoderDevice",
                    "sample_interval_ms": 50,
                    "primary": True,  # initialize() enforces exactly one primary
                }
            }
        )
    )
    hw = HardwareManager(str(yaml_path))
    hw.initialize(cfg=None)  # materializes devices from the YAML stanzas
    from mesofield.devices.mocks import MockEncoderDevice

    assert isinstance(hw.get_device("wheel"), MockEncoderDevice)
