"""Editing a rig in the builder must not quietly change what it does not show.

"Edit rig" loads a ``hardware.yaml`` into forms and writes it back out. Three
ways that round trip used to lose information:

* a stanza key the form has no control for (``properties:``, ``widgets:``,
  ``auto_contrast:``) was rebuilt out of existence;
* cameras spelled as a ``cameras:`` list were invisible to the builder, so they
  could neither be edited nor counted when checking that exactly one camera is
  primary -- and the builder wrote top-level ``camera:`` stanzas of its own, a
  second shape the runtime does not treat the same way;
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
    cameras = [c.stanza() for c in dialog._cards if c.list_entry]
    cameras.extend(dialog._camera_list_extras)
    if cameras:
        doc["cameras"] = cameras
    for card in dialog._cards:
        if not card.list_entry:
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
    """An `output:` block may carry keys the form has no control for."""
    card = _DeviceCard(DEVICE_SPECS["camera"])
    card.set_values("pupil", {
        "type": "camera", "backend": "micromanager",
        "output": {"suffix": "pupil", "file_type": "mp4",
                   "bids_type": "func", "compression": "zstd"},
    })
    assert card.stanza()["output"]["compression"] == "zstd"


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

    assert [c.name() for c in dialog._cards] == [
        "Dhyana", "ThorCam", "ArduCam", "encoder"]
    assert [c.display_name() for c in dialog._cards if c.list_entry] == [
        "mesoscope", "pupil", "bodycam"]
    assert "cameras" not in dialog._passthrough  # no longer invisible
    assert dialog._passthrough == {"blue_led_power_mw": "8.2"}
    assert dialog._camera_list_extras == []


def test_camera_list_survives_a_round_trip_unchanged(qapp):
    before = _rig_with_camera_list()
    after = _save_doc(HardwareBuilderDialog(None, doc=before))

    assert list(after) == list(before)  # same shape, same order
    assert len(after["cameras"]) == 3
    for was, now in zip(before["cameras"], after["cameras"]):
        assert was.get("id") == now.get("id")
        assert was.get("name") == now.get("name")
        assert was.get("properties") == now.get("properties")
        assert was.get("widgets") == now.get("widgets")
        assert was.get("output") == now.get("output")


def test_camera_entries_carry_no_catalog_type(qapp):
    """`cameras:` entries dispatch on `backend:`; `type:` is the catalog's."""
    after = _save_doc(HardwareBuilderDialog(None, doc=_rig_with_camera_list()))
    for entry in after["cameras"]:
        assert "type" not in entry
        assert entry["backend"] in {"micromanager", "opencv"}


def test_top_level_camera_stanza_moves_onto_the_list(qapp):
    """A camera an older build wrote as its own stanza settles onto the list,
    keeping the stanza key as its id -- which is what the runtime read it as."""
    doc = {
        "memory_buffer_size": 1000,
        "meso": {"type": "camera", "backend": "micromanager", "primary": True,
                 "configuration_path": "C:/mm/mm.cfg",
                 "output": {"suffix": "meso", "file_type": "ome.tiff",
                            "bids_type": "func"}},
        "wheel": {"type": "wheel", "port": "COM4", "baudrate": 115200},
    }
    after = _save_doc(HardwareBuilderDialog(None, doc=doc))

    assert "meso" not in after              # no longer its own stanza
    assert [c["id"] for c in after["cameras"]] == ["meso"]
    assert after["cameras"][0]["primary"] is True
    assert after["wheel"]["type"] == "wheel"


def test_unmodelled_camera_backend_is_kept_verbatim(qapp):
    """A backend with no card must not be forced into the wrong one."""
    doc = {
        "memory_buffer_size": 1000,
        "cameras": [{"id": "X", "name": "x", "backend": "spinnaker"}],
    }
    dialog = HardwareBuilderDialog(None, doc=doc)

    assert dialog._camera_list_extras == doc["cameras"]
    assert not [c for c in dialog._cards if c.list_entry]
    assert "spinnaker" in dialog._passthrough_label.text()
    assert _save_doc(dialog)["cameras"] == doc["cameras"]


def test_treadmill_stanza_survives_a_round_trip(qapp):
    """It used to drop to passthrough on load and so could never be edited."""
    before = _rig_with_camera_list()
    after = _save_doc(HardwareBuilderDialog(None, doc=before))
    assert after["encoder"] == before["encoder"]


def test_a_new_camera_joins_the_list(qapp):
    """Whatever the rig looked like, a camera is a `cameras:` entry."""
    dialog = HardwareBuilderDialog(None, doc=_rig_with_camera_list())
    card = dialog._make_card("camera")
    card.set_values("ThorCam2", {
        "type": "camera", "backend": "micromanager", "name": "facecam",
        "device": {"library": "TSI", "adapter": "TSICam",
                   "pre_init": {"CameraID": "22801"}},
    })

    doc = _save_doc(dialog)
    assert "ThorCam2" not in doc                    # not a stray stanza
    assert len(doc["cameras"]) == 4
    assert doc["cameras"][-1]["id"] == "ThorCam2"
    assert doc["cameras"][-1]["name"] == "facecam"
    assert doc["cameras"][-1]["device"]["pre_init"] == {"CameraID": "22801"}


def test_cameras_use_the_list_even_in_a_rig_that_had_none(qapp):
    """A fresh rig gets the same shape as a hand-written one."""
    dialog = HardwareBuilderDialog(None, doc={"memory_buffer_size": 1000})

    assert dialog._make_card("camera").list_entry is True
    assert dialog._make_card("opencv_camera").list_entry is True
    assert dialog._make_card("wheel").list_entry is False  # only cameras


def test_a_rig_mixing_both_shapes_is_normalised(qapp):
    """A rig split across both spellings ends up with one."""
    doc = _rig_with_camera_list()
    doc["facecam"] = {"type": "camera", "backend": "micromanager",
                      "output": {"suffix": "face", "file_type": "mp4",
                                 "bids_type": "beh"}}

    out = _save_doc(HardwareBuilderDialog(None, doc=doc))
    assert "facecam" not in out
    assert [c["id"] for c in out["cameras"]] == [
        "Dhyana", "ThorCam", "ArduCam", "facecam"]


def test_every_camera_counts_towards_the_primary_check(qapp):
    """Cameras hidden in the list were excluded from the check, which is how a
    rig could be saved with two primaries."""
    doc = _rig_with_camera_list()
    doc["facecam"] = {"type": "camera", "backend": "micromanager", "primary": True}

    dialog = HardwareBuilderDialog(None, doc=doc)
    primaries = [c.name() for c in dialog._cards if c.is_primary()]
    assert sorted(primaries) == ["Dhyana", "facecam"]  # both now visible


def test_non_cameras_keep_their_own_stanzas(qapp):
    """Only cameras move to a list; everything else is a named stanza."""
    out = _save_doc(HardwareBuilderDialog(None, doc=_rig_with_camera_list()))
    assert out["encoder"]["type"] == "encoder"
    assert "cameras" in out


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
    dialog._make_card("camera", at_top=True)._name_edit.setText("ThorCam2")

    layout = dialog._cards_layout
    widgets = [layout.itemAt(i).widget() for i in range(layout.count())]
    names = [w.name() for w in widgets if w is not None]
    assert names[0] == "ThorCam2"


def test_adding_a_device_does_not_reorder_the_file(qapp):
    """Only the widget moves to the top; stanzas keep their order."""
    dialog = HardwareBuilderDialog(None, doc=_rig_with_camera_list())
    card = dialog._make_card("camera", at_top=True)
    card.set_values("ThorCam2", {"type": "camera", "backend": "micromanager",
                                 "name": "facecam"})

    cameras = [c["id"] for c in _save_doc(dialog)["cameras"]]
    assert cameras == ["Dhyana", "ThorCam", "ArduCam", "ThorCam2"]


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
# device: and configuration_path: are alternatives
# --------------------------------------------------------------------------- #


def _camera_form(stanza: dict):
    card = _DeviceCard(DEVICE_SPECS["camera"])
    card.set_values("mesoscope", {"type": "camera", "backend": "micromanager",
                                  **stanza})
    return card, card._field_form


def _choose_device(form, value: dict) -> None:
    """Stand in for the picker dialog returning *value*."""
    editor = form._editors["device"]
    editor._value = dict(value)
    editor._refresh()
    form._clear_conflicts(next(f for f in form._fields if f.key == "device"))


def _clear_button(form):
    """The picker row's Clear button."""
    return form._editors["device"].layout().itemAt(2).widget()


DEVICE = {"library": "TSI", "adapter": "TSICam"}


def test_choosing_a_device_clears_the_cfg_path(qapp):
    """Otherwise the rig is saved with both and the runtime refuses to load
    it, naming the camera but not where to fix it."""
    card, form = _camera_form({"configuration_path": "C:/MM/mm-sipefield.cfg"})
    _choose_device(form, DEVICE)

    assert form.values()["configuration_path"] == ""
    assert "configuration_path" not in card.stanza()
    assert card.stanza()["device"] == DEVICE


def test_typing_a_cfg_path_clears_the_device(qapp):
    card, form = _camera_form({})
    _choose_device(form, DEVICE)
    form._editors["configuration_path"]._line_edit.setText("C:/MM/rig.cfg")

    assert form.values()["device"] == {}
    assert "device" not in card.stanza()
    assert card.stanza()["configuration_path"] == "C:/MM/rig.cfg"


def test_clear_button_undoes_the_device_choice(qapp):
    """The gap that made this necessary: Choose… had no counterpart, so an
    accidental pick could not be taken back."""
    card, form = _camera_form({})
    _choose_device(form, DEVICE)
    assert card.stanza()["device"] == DEVICE

    _clear_button(form).click()
    assert form.values()["device"] == {}
    assert "device" not in card.stanza()


def test_clear_restores_the_cfg_path_it_displaced(qapp):
    """A real undo: picking a device by accident on a .cfg camera must not
    cost the user the path."""
    card, form = _camera_form({"configuration_path": "C:/MM/mm-sipefield.cfg"})
    _choose_device(form, DEVICE)
    _clear_button(form).click()

    assert card.stanza()["configuration_path"] == "C:/MM/mm-sipefield.cfg"
    assert "device" not in card.stanza()


def test_clear_button_is_enabled_only_when_a_device_is_set(qapp):
    _card, form = _camera_form({})
    assert _clear_button(form).isEnabled() is False

    _choose_device(form, DEVICE)
    assert _clear_button(form).isEnabled() is True

    _clear_button(form).click()
    assert _clear_button(form).isEnabled() is False


def test_loading_a_stanza_with_a_device_enables_clear(qapp):
    _card, form = _camera_form({"device": DEVICE})
    assert _clear_button(form).isEnabled() is True


def test_conflicting_fields_are_declared_on_the_camera_spec():
    fields = {f.key: f for f in DEVICE_SPECS["camera"].fields}
    assert fields["device"].conflicts_with == ("configuration_path",)
    assert fields["configuration_path"].conflicts_with == ("device",)


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
    assert _primaries(dialog) == ["Dhyana"]

    dialog._make_card("camera", at_top=True)
    assert _primaries(dialog) == ["Dhyana"]


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
    pupil = next(c for c in dialog._cards if c.name() == "ThorCam")

    pupil._primary_check.setChecked(True)
    assert _primaries(dialog) == ["ThorCam"]


def test_a_file_with_two_primaries_is_reported_not_silently_fixed(qapp):
    """Quietly repairing it would hide that the rig on disk is wrong."""
    doc = _rig_with_camera_list()
    doc["facecam"] = {"type": "camera", "backend": "micromanager", "primary": True}

    dialog = HardwareBuilderDialog(None, doc=doc)
    assert sorted(_primaries(dialog)) == ["Dhyana", "facecam"]


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


# --------------------------------------------------------------------------- #
# Rendering a test snap
# --------------------------------------------------------------------------- #


def test_snap_renders_a_grayscale_preview(qapp):
    """A 10-bit frame has to be stretched to 8 bits to be visible at all."""
    import numpy as np

    from mesofield.gui.mm_device_picker import snap_to_qimage

    frame = np.linspace(24, 1022, 64 * 48, dtype=np.uint16).reshape(48, 64)
    image = snap_to_qimage(frame, bit_depth=10)

    assert (image.width(), image.height()) == (64, 48)
    assert image.format().name == "Format_Grayscale8"
    # Stretched to the frame's own range, not the sensor's white level: a
    # true-scale 10-bit bench snap renders as a black rectangle.
    assert image.pixelColor(0, 0).value() == 0
    assert image.pixelColor(63, 47).value() == 255


def test_uniform_snap_does_not_render_as_noise(qapp):
    """min == max would divide by zero; show the true level instead."""
    import numpy as np

    from mesofield.gui.mm_device_picker import snap_to_qimage

    image = snap_to_qimage(np.full((8, 8), 512, dtype=np.uint16), bit_depth=10)
    assert image.pixelColor(4, 4).value() == pytest.approx(127, abs=2)


def test_colour_snap_renders_as_rgb(qapp):
    import numpy as np

    from mesofield.gui.mm_device_picker import snap_to_qimage

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[..., 0] = 255
    image = snap_to_qimage(frame, bit_depth=8)

    assert image.format().name == "Format_RGB888"
    assert image.pixelColor(0, 0).red() == 255


# --------------------------------------------------------------------------- #
# What a re-saved rig reads like
# --------------------------------------------------------------------------- #


def test_camera_entries_lead_with_id_and_name(qapp):
    """A migrated camera otherwise appended them and read unlike its
    neighbours, which is what made a re-saved rig look mangled."""
    doc = {
        "memory_buffer_size": 1000,
        "facecam": {"type": "camera", "backend": "micromanager",
                    "device": {"library": "TSI", "adapter": "TSICam"},
                    "engine": "pupil"},
    }
    entry = _save_doc(HardwareBuilderDialog(None, doc=doc))["cameras"][0]
    assert list(entry)[:2] == ["id", "name"]


def test_null_valued_keys_are_dropped(qapp):
    """`output: {bodycam: null}` is residue from an earlier edit, not a
    setting -- preserving the unmodelled should not mean preserving noise."""
    doc = {
        "memory_buffer_size": 1000,
        "cameras": [{"id": "ThorCam", "name": "pupil", "backend": "micromanager",
                     "auto_contrast": None,
                     "output": {"suffix": "pupil", "file_type": "mp4",
                                "bids_type": "func", "bodycam": None}}],
    }
    entry = _save_doc(HardwareBuilderDialog(None, doc=doc))["cameras"][0]
    assert "bodycam" not in entry["output"]
    assert "auto_contrast" not in entry
    assert entry["output"]["suffix"] == "pupil"     # real keys survive


def test_paths_use_one_separator(qapp):
    """Qt's pickers hand back native separators, so a rig could carry both."""
    doc = {
        "memory_buffer_size": 1000,
        "cameras": [{"id": "ThorCam", "name": "pupil", "backend": "micromanager",
                     "micromanager_path": "C:\Program Files\Micro-Manager-2.0"}],
    }
    entry = _save_doc(HardwareBuilderDialog(None, doc=doc))["cameras"][0]
    assert entry["micromanager_path"] == "C:/Program Files/Micro-Manager-2.0"
