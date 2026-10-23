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
# Fixtures
# --------------------------------------------------------------------------- #


def _rig_with_camera_list() -> dict:
    """A rig in the shape the hand-written ones here use."""
    return {
        "memory_buffer_size": 10000,
        "blue_led_power_mw": "8.2",
        "cameras": [
            {
                "id": "Dhyana", "name": "mesoscope", "backend": "micromanager",
                "primary": True,
                "configuration_path": "C:/MM/mm-sipefield.cfg",
                "auto_contrast": True,
                "properties": {
                    "Dhyana": {"Exposure": 20.0, "Gain": "High Dynamic", "fps": 49.9},
                    "Arduino-Switch": {"Sequence": "On"},
                },
                "widgets": ["led_control"],
                "output": {"suffix": "meso", "file_type": "ome.tiff", "bids_type": "func"},
            },
            {
                "id": "ThorCam", "name": "pupil", "backend": "micromanager",
                "configuration_path": "C:/MM/ThorCam.cfg",
                "properties": {"ThorCam": {"Exposure": 5.0, "ROI": [440, 305, 509, 509]}},
                "output": {"suffix": "pupil", "file_type": "mp4", "bids_type": "func"},
            },
            {
                "id": "ArduCam", "name": "bodycam", "type": "opencv_camera",
                "backend": "opencv", "device_index": 0, "fps": 30,
                "output": {"suffix": "cam", "file_type": "mp4", "bids_type": "func"},
            },
        ],
        "encoder": {
            "type": "encoder", "port": "COM8", "baudrate": 192000,
            "output": {"suffix": "treadmill", "file_type": "csv", "bids_type": "beh"},
        },
    }


def _save_doc(dialog: HardwareBuilderDialog) -> dict:
    """What :meth:`HardwareBuilderDialog._save` would write, minus the dialogs."""
    doc = {"memory_buffer_size": int(dialog._buffer_spin.value())}
    doc.update(dialog._passthrough)
    cameras = [c for c in dialog._cards if c.from_camera_list]
    if cameras:
        doc["cameras"] = [c.stanza() for c in cameras]
    for card in dialog._cards:
        if not card.from_camera_list:
            doc[card.name()] = card.stanza()
    return doc


# --------------------------------------------------------------------------- #
# Spec resolution
# --------------------------------------------------------------------------- #


def test_spec_resolves_by_catalog_key():
    assert spec_key_for_stanza({"type": "camera"}) == "camera"
    assert spec_key_for_stanza({"type": "mock_wheel"}) == "mock_wheel"


def test_spec_resolves_by_registry_type():
    """The treadmill is catalogued as ``treadmill`` but registers as
    ``encoder``, which is what the builder writes."""
    assert DEVICE_SPECS["treadmill"].type == "encoder"
    assert spec_key_for_stanza({"type": "encoder"}) == "treadmill"


def test_unknown_type_resolves_to_nothing():
    assert spec_key_for_stanza({"type": "not_a_device"}) is None
    assert spec_key_for_stanza({}) is None


def test_camera_list_entries_fall_back_to_backend():
    """Entries in a ``cameras:`` list usually carry no ``type:`` at all."""
    assert camera_spec_key({"backend": "micromanager"}) == "camera"
    assert camera_spec_key({"backend": "opencv"}) == "opencv_camera"
    assert camera_spec_key({"type": "opencv_camera"}) == "opencv_camera"


# --------------------------------------------------------------------------- #
# A card keeps what it does not show
# --------------------------------------------------------------------------- #


def test_card_preserves_keys_the_form_does_not_model(qapp):
    original = {
        "type": "camera", "backend": "micromanager",
        "configuration_path": "C:/MM/mm-sipefield.cfg",
        "auto_contrast": True,
        "properties": {"Dhyana": {"Exposure": 20.0, "Gain": "High Dynamic"}},
        "widgets": ["led_control"],
        "output": {"suffix": "meso", "file_type": "ome.tiff", "bids_type": "func"},
    }
    card = _DeviceCard(DEVICE_SPECS["camera"])
    card.set_values("mesoscope", original)
    out = card.stanza()

    assert out["properties"] == original["properties"]
    assert out["widgets"] == ["led_control"]
    assert out["auto_contrast"] is True
    assert out["configuration_path"] == "C:/MM/mm-sipefield.cfg"


def test_card_preserves_extra_output_keys(qapp):
    card = _DeviceCard(DEVICE_SPECS["camera"])
    card.set_values("pupil", {
        "type": "camera", "backend": "micromanager",
        "output": {"suffix": "pupil", "file_type": "mp4",
                   "bids_type": "func", "bodycam": None},
    })
    assert "bodycam" in card.stanza()["output"]


def test_clearing_a_modelled_field_still_removes_it(qapp):
    """Preserving the unmodelled must not mean ignoring the user: blanking
    configuration_path means "no .cfg", not "keep the old one"."""
    card = _DeviceCard(DEVICE_SPECS["camera"])
    card.set_values("pupil", {"type": "camera", "backend": "micromanager",
                              "configuration_path": "C:/MM/ThorCam.cfg"})
    card._field_form.set_values({"configuration_path": ""})
    assert "configuration_path" not in card.stanza()


def test_unchecking_primary_removes_the_key(qapp):
    card = _DeviceCard(DEVICE_SPECS["camera"])
    card.set_values("pupil", {"type": "camera", "backend": "micromanager",
                              "primary": True})
    assert card.stanza()["primary"] is True
    card.set_primary(False)
    assert "primary" not in card.stanza()


# --------------------------------------------------------------------------- #
# cameras: lists
# --------------------------------------------------------------------------- #


def test_camera_list_becomes_editable_cards(qapp):
    dialog = HardwareBuilderDialog(None, doc=_rig_with_camera_list())
    names = [c.name() for c in dialog._cards]

    assert names == ["mesoscope", "pupil", "bodycam", "encoder"]
    assert "cameras" not in dialog._passthrough  # no longer invisible
    assert dialog._passthrough == {"blue_led_power_mw": "8.2"}


def test_camera_list_survives_a_round_trip_unchanged(qapp):
    before = _rig_with_camera_list()
    after = _save_doc(HardwareBuilderDialog(None, doc=before))

    assert list(after) == list(before)  # same shape, same order
    assert len(after["cameras"]) == 3
    for was, now in zip(before["cameras"], after["cameras"]):
        assert was.get("id") == now.get("id")
        assert was.get("properties") == now.get("properties")
        assert was.get("widgets") == now.get("widgets")
        assert was.get("output") == now.get("output")


def test_treadmill_stanza_survives_a_round_trip(qapp):
    """It used to drop to passthrough on load and so could never be edited."""
    before = _rig_with_camera_list()
    after = _save_doc(HardwareBuilderDialog(None, doc=before))
    assert after["encoder"] == before["encoder"]


def test_a_new_camera_joins_the_existing_list(qapp):
    """Otherwise the rig would describe cameras two ways at once."""
    dialog = HardwareBuilderDialog(None, doc=_rig_with_camera_list())
    card = dialog._make_card("camera")
    card.from_camera_list = dialog._cameras_from_list
    card.set_values("facecam", {
        "type": "camera", "backend": "micromanager",
        "device": {"library": "TSI", "adapter": "TSICam",
                   "pre_init": {"CameraID": "22801"}},
    })

    doc = _save_doc(dialog)
    assert "facecam" not in doc                     # not a stray stanza
    assert len(doc["cameras"]) == 4
    assert doc["cameras"][-1]["name"] == "facecam"
    assert doc["cameras"][-1]["device"]["pre_init"] == {"CameraID": "22801"}


def test_named_camera_stanza_is_not_pulled_into_the_list(qapp):
    """A rig that already mixes both shapes keeps them as they were."""
    doc = _rig_with_camera_list()
    doc["facecam"] = {"type": "camera", "backend": "micromanager",
                      "output": {"suffix": "face", "file_type": "mp4",
                                 "bids_type": "beh"}}

    out = _save_doc(HardwareBuilderDialog(None, doc=doc))
    assert len(out["cameras"]) == 3
    assert out["facecam"]["type"] == "camera"


def test_every_camera_counts_towards_the_primary_check(qapp):
    """Cameras hidden in the list were excluded from the check, which is how a
    rig could be saved with two primaries."""
    doc = _rig_with_camera_list()
    doc["facecam"] = {"type": "camera", "backend": "micromanager", "primary": True}

    dialog = HardwareBuilderDialog(None, doc=doc)
    primaries = [c.name() for c in dialog._cards if c.is_primary()]
    assert sorted(primaries) == ["facecam", "mesoscope"]  # both now visible


def test_rig_without_a_camera_list_keeps_named_stanzas(qapp):
    doc = {
        "memory_buffer_size": 1000,
        "pupil": {"type": "camera", "backend": "micromanager", "primary": True,
                  "output": {"suffix": "pupil", "file_type": "mp4", "bids_type": "func"}},
    }
    out = _save_doc(HardwareBuilderDialog(None, doc=doc))
    assert "cameras" not in out
    assert out["pupil"]["type"] == "camera"


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


# --------------------------------------------------------------------------- #
# Form ergonomics
# --------------------------------------------------------------------------- #


def _send_wheel(widget) -> bool:
    """Scroll down over *widget*; return whether it accepted the event."""
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtWidgets import QApplication

    event = QWheelEvent(
        QPointF(5, 5), QPointF(5, 5), QPoint(0, -120), QPoint(0, -120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )
    QApplication.sendEvent(widget, event)
    return event.isAccepted()


def test_added_device_appears_at_the_top(qapp):
    """So it is visible without scrolling past the rig already there."""
    dialog = HardwareBuilderDialog(None, doc=_rig_with_camera_list())
    dialog._make_card("camera", at_top=True)._name_edit.setText("facecam")

    layout = dialog._cards_layout
    widgets = [layout.itemAt(i).widget() for i in range(layout.count())]
    names = [w.name() for w in widgets if w is not None]
    assert names[0] == "facecam"


def test_adding_a_device_does_not_reorder_the_file(qapp):
    """Only the widget moves to the top; stanzas keep their order."""
    dialog = HardwareBuilderDialog(None, doc=_rig_with_camera_list())
    card = dialog._make_card("camera", at_top=True)
    card.from_camera_list = True
    card.set_values("facecam", {"type": "camera", "backend": "micromanager"})

    cameras = [c["name"] for c in _save_doc(dialog)["cameras"]]
    assert cameras == ["mesoscope", "pupil", "bodycam", "facecam"]


def test_scrolling_over_a_dropdown_does_not_change_it(qapp):
    from mesofield.gui.config_builder import SchemaForm

    form = SchemaForm(DEVICE_SPECS["opencv_camera"].fields)
    combo = form._editors["cv_backend"]
    before = combo.currentText()

    accepted = _send_wheel(combo)
    assert combo.currentText() == before
    # Refused, not swallowed -- the enclosing scroll area still scrolls.
    assert accepted is False


def test_scrolling_over_a_spinbox_does_not_change_it(qapp):
    from mesofield.gui.config_builder import SchemaForm

    form = SchemaForm(DEVICE_SPECS["opencv_camera"].fields)
    spin = form._editors["device_index"]
    before = spin.value()

    accepted = _send_wheel(spin)
    assert spin.value() == before
    assert accepted is False


def test_scrolling_over_the_add_device_menu_adds_nothing(qapp):
    """A stray wheel here used to add a device outright."""
    dialog = HardwareBuilderDialog(None, doc=_rig_with_camera_list())
    before = len(dialog._cards)

    _send_wheel(dialog._add_combo)
    assert dialog._add_combo.currentIndex() == 0
    assert len(dialog._cards) == before


def test_scrolling_over_the_buffer_size_does_not_change_it(qapp):
    dialog = HardwareBuilderDialog(None, doc=_rig_with_camera_list())
    before = dialog._buffer_spin.value()

    _send_wheel(dialog._buffer_spin)
    assert dialog._buffer_spin.value() == before


# --------------------------------------------------------------------------- #
# Exactly one primary
# --------------------------------------------------------------------------- #


def _primaries(dialog) -> list[str]:
    return [c.name() for c in dialog._cards if c.is_primary()]


def test_added_camera_does_not_steal_primary_from_a_camera(qapp):
    """The reported bug: a new camera defaulted to primary even though the rig
    already had one, because the cameras in a `cameras:` list were invisible
    to the check."""
    dialog = HardwareBuilderDialog(None, doc=_rig_with_camera_list())
    assert _primaries(dialog) == ["mesoscope"]

    dialog._make_card("camera", at_top=True)
    assert _primaries(dialog) == ["mesoscope"]


def test_added_camera_takes_primary_from_a_non_camera(qapp):
    """Cameras drive acquisition timing, so a camera added to an
    encoder-primary rig takes over -- and the encoder gives it up, rather than
    the rig ending up with two."""
    dialog = HardwareBuilderDialog(None, doc={
        "memory_buffer_size": 1000,
        "encoder": {"type": "encoder", "port": "COM8", "primary": True},
    })
    assert _primaries(dialog) == ["encoder"]

    card = dialog._make_card("camera", at_top=True)
    assert _primaries(dialog) == [card.name()]


def test_first_camera_in_an_empty_rig_becomes_primary(qapp):
    dialog = HardwareBuilderDialog(None, doc={"memory_buffer_size": 1000})
    card = dialog._make_card("camera", at_top=True)
    assert card.is_primary() is True


def test_ticking_one_camera_unticks_the_others(qapp):
    """Primary is a choice between cameras, so the boxes behave like radios
    instead of letting two be ticked and complaining only at save time."""
    dialog = HardwareBuilderDialog(None, doc=_rig_with_camera_list())
    pupil = next(c for c in dialog._cards if c.name() == "pupil")

    pupil._primary_check.setChecked(True)
    assert _primaries(dialog) == ["pupil"]


def test_a_file_with_two_primaries_is_reported_not_silently_fixed(qapp):
    """Quietly repairing it would hide that the rig on disk is wrong."""
    doc = _rig_with_camera_list()
    doc["facecam"] = {"type": "camera", "backend": "micromanager", "primary": True}

    dialog = HardwareBuilderDialog(None, doc=doc)
    assert sorted(_primaries(dialog)) == ["facecam", "mesoscope"]


def test_non_camera_keeps_a_primary_flag_it_arrived_with(qapp):
    """A treadmill-only rig has to have a primary, and the builder shows no
    checkbox for one -- so the flag has to be carried, not dropped."""
    card = _DeviceCard(DEVICE_SPECS["treadmill"])
    card.set_values("encoder", {"type": "encoder", "port": "COM8", "primary": True})

    assert card.is_primary() is True
    assert card.stanza()["primary"] is True


def test_non_camera_without_the_flag_does_not_gain_one(qapp):
    card = _DeviceCard(DEVICE_SPECS["treadmill"])
    card.set_values("encoder", {"type": "encoder", "port": "COM8"})
    assert "primary" not in card.stanza()
