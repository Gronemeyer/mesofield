"""Editing a rig in the builder must not quietly change what it does not show.

"Edit rig" loads a ``hardware.yaml`` into forms and writes it back out. Three
ways that round trip used to lose information:

* a stanza key the form has no control for (``properties:``, ``widgets:``,
  ``auto_contrast:``) was rebuilt out of existence;
* cameras spelled as a ``cameras:`` list were invisible to the builder, so they
  could neither be edited nor counted when checking that exactly one camera is
  primary;
* a device's ``type:`` is its DeviceRegistry key, which is not always the
  catalog key -- a treadmill written once as ``type: encoder`` stopped matching
  the catalog and dropped out of the form on the next edit.

The last one reached the runtime too: ``_init_encoder`` only accepted
``wheel``/``treadmill``, so a builder-written ``encoder:`` stanza left the rig
with no encoder and only a log line to say so.
"""

from __future__ import annotations

import pytest

from mesofield.gui.config_builder import (
    DEVICE_SPECS,
    HardwareBuilderDialog,
    _DeviceCard,
    camera_spec_key,
    spec_key_for_stanza,
)


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("enc_type", ["treadmill", "encoder"])
def test_init_encoder_accepts_the_registry_type(monkeypatch, enc_type):
    """``type: encoder`` is what the builder writes; it used to fall through to
    "Unknown encoder type" and leave the rig with no encoder."""
    import mesofield.hardware as hardware

    built = {}

    class _FakeEncoder:
        def __init__(self, **kwargs):
            built.update(kwargs)

        def initialize(self):
            built["initialized"] = True

    monkeypatch.setattr(hardware, "EncoderSerialInterface", _FakeEncoder)

    manager = object.__new__(hardware.HardwareManager)
    manager.yaml = {"encoder": {"type": enc_type, "port": "COM8",
                                "baudrate": 192000}}
    manager.devices = {}
    manager.logger = hardware.get_logger("test") if hasattr(hardware, "get_logger") \
        else __import__("logging").getLogger("test")
    manager._apply_output_args = lambda *a, **k: None

    manager._init_encoder()

    assert built.get("port") == "COM8"
    assert built.get("initialized") is True
    assert "encoder" in manager.devices


def test_init_encoder_still_rejects_nonsense(monkeypatch):
    import logging

    import mesofield.hardware as hardware

    manager = object.__new__(hardware.HardwareManager)
    manager.yaml = {"encoder": {"type": "sometimes", "port": "COM8"}}
    manager.devices = {}
    manager.logger = logging.getLogger("test")

    manager._init_encoder()
    assert manager.devices == {}
