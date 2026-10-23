"""Adding a Micro-Manager camera without authoring a ``.cfg`` file.

A single-camera Micro-Manager system config is only a replay script of core
calls, so a rig can carry the same facts in ``hardware.yaml`` under ``device:``
and skip the file. These cover the contract of that route:

* :func:`~mesofield.devices.cameras.load_camera_device` issues the calls in the
  one order MMCore accepts, and turns an initialisation failure into something
  a user can act on;
* ``MMCamera`` rejects stanzas that are ambiguous or incomplete;
* engine selection no longer depends on the camera id, which cannot work once a
  rig holds two of the same camera.

Everything here is hardware-free except the final class, which is gated behind
``--run-hardware`` and needs two Thorlabs cameras plugged in.
"""

from __future__ import annotations

import logging

import pytest

from mesofield.devices.cameras import MMCamera, load_camera_device


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _FakeCore:
    """Records the calls ``load_camera_device`` makes, in order."""

    def __init__(self, *, init_error: Exception | None = None):
        self.calls: list[tuple] = []
        self._init_error = init_error

    def loadDevice(self, label, library, adapter):
        self.calls.append(("loadDevice", label, library, adapter))

    def setProperty(self, label, prop, value):
        self.calls.append(("setProperty", label, prop, value))

    def initializeDevice(self, label):
        self.calls.append(("initializeDevice", label))
        if self._init_error is not None:
            raise self._init_error

    def setCameraDevice(self, label):
        self.calls.append(("setCameraDevice", label))


def _bare_camera(cam_id: str = "ThorCam1") -> MMCamera:
    """An ``MMCamera`` with only what ``_load_mm_system`` touches.

    Bypasses ``__init__`` so no core is built and no hardware is claimed.
    """
    cam = object.__new__(MMCamera)
    cam.id = cam_id
    cam.logger = logging.getLogger(f"test.{cam_id}")
    return cam


# --------------------------------------------------------------------------- #
# load_camera_device
# --------------------------------------------------------------------------- #


def test_pre_init_properties_are_set_between_load_and_initialize():
    """The only window MMCore accepts them in -- the reason they can't live in
    the camera's ordinary ``properties:`` block."""
    core = _FakeCore()
    load_camera_device(core, "ThorCam1", "TSI", "TSICam", {"CameraID": "22801"})

    assert core.calls == [
        ("loadDevice", "ThorCam1", "TSI", "TSICam"),
        ("setProperty", "ThorCam1", "CameraID", "22801"),
        ("initializeDevice", "ThorCam1"),
        ("setCameraDevice", "ThorCam1"),
    ]


def test_pre_init_values_are_stringified():
    """YAML gives ints for things like ``CameraID: 0``; MMCore wants strings."""
    core = _FakeCore()
    load_camera_device(core, "Cam", "TSI", "TSICam", {"CameraID": 0})
    assert ("setProperty", "Cam", "CameraID", "0") in core.calls


def test_missing_pre_init_block_is_fine():
    core = _FakeCore()
    load_camera_device(core, "Cam", "DemoCamera", "DCam", None)
    assert [c[0] for c in core.calls] == [
        "loadDevice", "initializeDevice", "setCameraDevice",
    ]


def test_initialise_failure_explains_the_likely_cause():
    """MMCore answers an already-open Thorlabs camera with a bare error code,
    which tells a user nothing. The wrapper has to."""
    core = _FakeCore(init_error=RuntimeError('Error in device "X": '
                                             '(Error message unavailable) (10014)'))
    with pytest.raises(RuntimeError) as excinfo:
        load_camera_device(core, "ThorCam2", "TSI", "TSICam", {"CameraID": "22801"})

    message = str(excinfo.value)
    assert "TSI/TSICam" in message
    assert "ThorCam2" in message
    assert "CameraID=22801" in message      # what we actually asked for
    assert "already open" in message        # the usual cause
    assert "10014" in message               # the original is not swallowed


# --------------------------------------------------------------------------- #
# Stanza validation
# --------------------------------------------------------------------------- #


def test_device_and_configuration_path_are_mutually_exclusive():
    cam = _bare_camera()
    with pytest.raises(ValueError, match="alternatives"):
        cam._load_mm_system(_FakeCore(), {
            "device": {"library": "TSI", "adapter": "TSICam"},
            "configuration_path": "C:/rig.cfg",
        })


@pytest.mark.parametrize("device", [
    {"library": "TSI"},                 # no adapter
    {"adapter": "TSICam"},              # no library
    {"library": "", "adapter": ""},     # present but blank
])
def test_device_needs_library_and_adapter(device):
    cam = _bare_camera()
    with pytest.raises(ValueError, match="library"):
        cam._load_mm_system(_FakeCore(), {"device": device})


@pytest.mark.parametrize("device", [None, {}])
def test_absent_device_falls_through_to_the_cfg_route(device):
    """``device:`` with nothing under it reads as absent, which is also what
    the rig builder emits for an untouched picker."""
    core = _FakeCore()
    core.loadSystemConfiguration = lambda *a: core.calls.append(("loadSystemConfiguration", *a))

    _bare_camera()._load_mm_system(core, {"device": device,
                                          "configuration_path": "C:/rig.cfg"})
    assert core.calls == [("loadSystemConfiguration", "C:/rig.cfg")]


def test_device_must_be_a_mapping():
    cam = _bare_camera()
    with pytest.raises(ValueError, match="mapping"):
        cam._load_mm_system(_FakeCore(), {"device": "TSI/TSICam"})


def test_device_label_defaults_to_camera_id():
    """So the ``properties:`` block, which is keyed by MM device label, lines
    up without the user knowing the two are the same string."""
    core = _FakeCore()
    _bare_camera("pupil_left")._load_mm_system(
        core, {"device": {"library": "TSI", "adapter": "TSICam"}}
    )
    assert core.calls[0] == ("loadDevice", "pupil_left", "TSI", "TSICam")


def test_device_label_can_be_overridden():
    core = _FakeCore()
    _bare_camera("pupil_left")._load_mm_system(
        core, {"device": {"library": "TSI", "adapter": "TSICam", "label": "ThorCam"}}
    )
    assert core.calls[0] == ("loadDevice", "ThorCam", "TSI", "TSICam")


# --------------------------------------------------------------------------- #
# Engine selection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name,expected", [
    ("pupil", "PupilEngine"),
    ("meso", "MesoEngine"),
    ("dev", "DevEngine"),
    ("PUPIL", "PupilEngine"),  # case-insensitive
])
def test_explicit_engine_wins(name, expected):
    cam = _bare_camera("anything_at_all")
    assert cam._resolve_engine({"engine": name}).__name__ == expected


def test_unknown_engine_is_rejected_by_name():
    cam = _bare_camera()
    with pytest.raises(ValueError, match="bogus"):
        cam._resolve_engine({"engine": "bogus"})


@pytest.mark.parametrize("cam_id,expected", [
    ("ThorCam", "PupilEngine"),
    ("Dhyana", "MesoEngine"),
    ("something_else", "DevEngine"),
])
def test_id_fallback_still_applies_to_existing_configs(cam_id, expected):
    assert _bare_camera(cam_id)._resolve_engine({}).__name__ == expected


def test_second_identical_camera_needs_an_explicit_engine():
    """Rig ids are unique (``HardwareManager`` does ``setattr(self, cam.id,
    cam)``), so a second ThorCam cannot also be called "ThorCam" -- under the
    id fallback alone it would silently drop to DevEngine."""
    assert _bare_camera("ThorCam2")._resolve_engine({}).__name__ == "DevEngine"
    assert _bare_camera("ThorCam2")._resolve_engine(
        {"engine": "pupil"}
    ).__name__ == "PupilEngine"


# --------------------------------------------------------------------------- #
# Builder catalog
# --------------------------------------------------------------------------- #


def test_camera_spec_offers_the_device_picker():
    from mesofield.gui.config_builder import DEVICE_SPECS

    fields = {f.key: f for f in DEVICE_SPECS["camera"].fields}
    assert fields["device"].picker == "mm_device"
    # the .cfg route survives for rigs that genuinely need one
    assert "configuration_path" in fields
    assert set(fields["engine"].choices) == {"", "pupil", "meso", "dev"}


def test_unset_device_picker_is_left_out_of_the_stanza(qapp):
    """An empty mapping must not reach hardware.yaml, or MMCamera would treat
    the camera as using the (incomplete) device route."""
    from mesofield.gui.config_builder import DEVICE_SPECS, _DeviceCard

    card = _DeviceCard(DEVICE_SPECS["camera"])
    card.set_values("pupil", {"configuration_path": "C:/rig.cfg"})
    stanza = card.stanza()

    assert "device" not in stanza
    assert stanza["configuration_path"] == "C:/rig.cfg"


def test_chosen_device_round_trips_through_the_card(qapp):
    from mesofield.gui.config_builder import DEVICE_SPECS, _DeviceCard

    device = {"library": "TSI", "adapter": "TSICam",
              "pre_init": {"CameraID": "22801"}}
    card = _DeviceCard(DEVICE_SPECS["camera"])
    card.set_values("pupil_left", {"device": device, "engine": "pupil"})

    stanza = card.stanza()
    assert stanza["device"] == device
    assert stanza["engine"] == "pupil"
    assert stanza["backend"] == "micromanager"


# --------------------------------------------------------------------------- #
# Real hardware
# --------------------------------------------------------------------------- #


@pytest.mark.hardware
class TestTwoThorlabsCameras:
    """Two identical Thorlabs cameras, told apart only by ``CameraID``.

    Run with ``--run-hardware`` and both cameras plugged in.
    """

    LIBRARY = "TSI"
    ADAPTER = "TSICam"

    @pytest.fixture(scope="class")
    def mm_path(self) -> str:
        from mesofield.gui.mm_device_picker import check_mm_path, installed_micromanagers

        for path in installed_micromanagers():
            if check_mm_path(path) is None:
                return path
        pytest.skip("no usable Micro-Manager installation found")

    @pytest.fixture(scope="class")
    def serials(self, mm_path) -> list[str]:
        from mesofield.gui.mm_device_picker import probe_pre_init_properties

        props = {p.name: p for p in
                 probe_pre_init_properties(mm_path, self.LIBRARY, self.ADAPTER)}
        if "CameraID" not in props:
            pytest.skip("TSI adapter exposes no CameraID")
        # "0" means "first available" -- the real serial numbers follow it
        found = [v for v in props["CameraID"].allowed if v != "0"]
        if len(found) < 2:
            pytest.skip(f"need two cameras attached, found {len(found)}")
        return found[:2]

    def test_serials_are_discoverable(self, serials):
        assert len(set(serials)) == 2

    def test_both_cameras_open_at_once(self, mm_path, serials):
        """The thing that was impossible to express before: two cameras of the
        same model in one rig, no .cfg file for either."""
        cameras = []
        try:
            for index, serial in enumerate(serials, start=1):
                cameras.append(MMCamera({
                    "id": f"ThorCam{index}",
                    "name": f"pupil{index}",
                    "backend": "micromanager",
                    "micromanager_path": mm_path,
                    "engine": "pupil",
                    "device": {
                        "library": self.LIBRARY,
                        "adapter": self.ADAPTER,
                        "pre_init": {"CameraID": serial},
                    },
                }))

            for cam in cameras:
                assert type(cam._engine).__name__ == "PupilEngine"
                assert cam.core.getCameraDevice() == cam.id
                # the driver's own bit depth, not the frame dtype's
                assert cam.white_level == (1 << cam.core.getImageBitDepth()) - 1
                cam.core.snapImage()
                assert cam.core.getImage().size > 0
        finally:
            for cam in cameras:
                try:
                    cam.core.unloadAllDevices()
                except Exception:
                    pass

    def test_reusing_a_serial_reports_a_useful_error(self, mm_path, serials):
        """The mistake a user makes with identical cameras: same ID twice."""
        first = MMCamera({
            "id": "ThorCamA", "backend": "micromanager",
            "micromanager_path": mm_path, "engine": "pupil",
            "device": {"library": self.LIBRARY, "adapter": self.ADAPTER,
                       "pre_init": {"CameraID": serials[0]}},
        })
        try:
            with pytest.raises(RuntimeError, match="already open"):
                MMCamera({
                    "id": "ThorCamB", "backend": "micromanager",
                    "micromanager_path": mm_path, "engine": "pupil",
                    "device": {"library": self.LIBRARY, "adapter": self.ADAPTER,
                               "pre_init": {"CameraID": serials[0]}},
                })
        finally:
            first.core.unloadAllDevices()
