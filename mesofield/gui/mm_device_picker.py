"""Pick a Micro-Manager camera without authoring a ``.cfg`` file.

A Micro-Manager system config is a replay script of core calls, not a data
format. For a single camera with no peripherals the whole file reduces to::

    Device,ThorCam,TSI,TSICam        # label, adapter library, device name
    Property,ThorCam,CameraID,22801  # one pre-init property
    Property,Core,Camera,ThorCam     # make it the camera role

so a rig can carry those three facts in ``hardware.yaml`` instead and skip the
file. This module finds the facts: it enumerates the camera device adapters an
installation offers, probes the pre-init properties a chosen device wants, and
wraps both in a dialog that returns the ``device:`` stanza
:meth:`mesofield.devices.cameras.MMCamera._load_mm_system` consumes.

The deliberate limit is one camera device and nothing else. Rigs with serial
ports, hubs, ``Parent`` lines, labels or config groups (the Dhyana + Arduino
rig) still want a real ``.cfg`` authored in Micro-Manager's own wizard, which
:mod:`mesofield.gui.config_wizard` already embeds.

The enumeration/probe half is plain functions with no Qt import so it can be
called headless and tested without a display.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mesofield.gui import theme

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumeration + probing (no Qt)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MMDevice:
    """One camera device an installed adapter library offers."""

    library: str
    adapter: str
    description: str = ""

    @property
    def display(self) -> str:
        return f"{self.library} / {self.adapter}"


@dataclass
class PreInitProperty:
    """A property that must be set between ``loadDevice`` and ``initializeDevice``.

    ``allowed`` is the adapter's own answer, and it is the useful part: for
    Thorlabs' ``TSI`` adapter ``CameraID`` answers with the serial numbers of
    the cameras currently plugged in, which is the only thing distinguishing
    two otherwise identical ThorCams.
    """

    name: str
    current: str = ""
    allowed: List[str] = field(default_factory=list)


def installed_micromanagers() -> List[str]:
    """Every Micro-Manager installation pymmcore-plus can find."""
    from pymmcore_plus import find_micromanager

    found = find_micromanager(return_first=False) or []
    return [str(p) for p in found]


def _new_core(mm_path: str):
    from pymmcore_plus import CMMCorePlus

    return CMMCorePlus(mm_path)


def enumerate_camera_devices(mm_path: str) -> List[MMDevice]:
    """Return the camera devices offered by the adapters under *mm_path*.

    Adapters that fail to load are skipped: a stock installation ships ~260
    libraries and the 60-odd that need a vendor SDK present on the machine
    raise here, which is normal and not worth reporting. An installation where
    *every* adapter fails is a different matter -- see :func:`check_mm_path`.
    """
    from pymmcore_plus import DeviceType

    core = _new_core(mm_path)
    devices: List[MMDevice] = []
    for library in core.getDeviceAdapterNames():
        try:
            names = core.getAvailableDevices(library)
            types = core.getAvailableDeviceTypes(library)
            descriptions = core.getAvailableDeviceDescriptions(library)
        except Exception:
            continue
        for name, dev_type, description in zip(names, types, descriptions):
            try:
                is_camera = DeviceType(dev_type) is DeviceType.Camera
            except ValueError:
                continue
            if is_camera:
                devices.append(MMDevice(library, name, description))
    devices.sort(key=lambda d: (d.library.lower(), d.adapter.lower()))
    return devices


def check_mm_path(mm_path: str) -> Optional[str]:
    """Return ``None`` if *mm_path* is usable, else a reason it is not.

    Worth doing before showing a device list, because a Micro-Manager build
    whose MMCore ABI does not match the installed pymmcore fails to load every
    single adapter -- including the hardware-free ``DemoCamera`` -- and the
    only symptom is an empty picker.
    """
    try:
        core = _new_core(mm_path)
        libraries = list(core.getDeviceAdapterNames())
    except Exception as exc:  # pragma: no cover - depends on the install
        return f"Cannot read device adapters: {exc}"

    if not libraries:
        return "No device adapters found in this folder."

    for library in libraries:
        try:
            core.getAvailableDevices(library)
            return None
        except Exception:
            continue
    return (
        f"All {len(libraries)} device adapters failed to load. This build is "
        "probably not compatible with the installed pymmcore."
    )


def probe_pre_init_properties(
    mm_path: str, library: str, adapter: str
) -> List[PreInitProperty]:
    """Load *library*/*adapter* far enough to read its pre-init properties.

    Stops short of ``initializeDevice``, so no hardware is claimed: the device
    is loaded, questioned, and unloaded again.
    """
    core = _new_core(mm_path)
    label = "__probe__"
    core.loadDevice(label, library, adapter)
    try:
        props: List[PreInitProperty] = []
        for name in core.getDevicePropertyNames(label):
            if not core.isPropertyPreInit(label, name):
                continue
            try:
                allowed = [str(v) for v in core.getAllowedPropertyValues(label, name)]
            except Exception:
                allowed = []
            try:
                current = str(core.getProperty(label, name))
            except Exception:
                current = ""
            props.append(PreInitProperty(name, current, allowed))
        return props
    finally:
        try:
            core.unloadDevice(label)
        except Exception:
            logger.debug("probe: unloadDevice(%s) failed", label, exc_info=True)


def test_device(
    mm_path: str, library: str, adapter: str, pre_init: Dict[str, Any]
) -> str:
    """Build the camera for real and snap one frame. Returns a summary line.

    Goes through :func:`mesofield.devices.cameras.load_camera_device`, the same
    function the rig uses at run time, so a camera that tests clean here is a
    camera that will load. Raises whatever that raises; the caller reports it.
    This is the only function here that touches hardware, and it answers "did I
    pick the right serial number?" -- the single question a user adding a
    second identical camera actually has.
    """
    from mesofield.devices.cameras import load_camera_device

    core = _new_core(mm_path)
    label = "__test__"
    try:
        load_camera_device(core, label, library, adapter, pre_init)
        core.snapImage()
        image = core.getImage()
        depth = core.getImageBitDepth()
        return (
            f"{image.shape[1]}×{image.shape[0]} {image.dtype}, "
            f"{depth}-bit (white level {(1 << int(depth)) - 1}), "
            f"mean {image.mean():.0f}"
        )
    finally:
        try:
            core.unloadAllDevices()
        except Exception:
            logger.debug("test: unloadAllDevices failed", exc_info=True)


# ---------------------------------------------------------------------------
# Background enumeration
# ---------------------------------------------------------------------------


class _EnumerateThread(QThread):
    """Enumerating ~260 adapter libraries takes a few seconds; keep the UI alive."""

    done = pyqtSignal(object, str)  # devices, error

    def __init__(self, mm_path: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._mm_path = mm_path

    def run(self) -> None:  # pragma: no cover - thread body
        try:
            problem = check_mm_path(self._mm_path)
            if problem:
                self.done.emit([], problem)
                return
            self.done.emit(enumerate_camera_devices(self._mm_path), "")
        except Exception as exc:
            self.done.emit([], str(exc))


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class MMDevicePickerDialog(QDialog):
    """Choose a Micro-Manager camera and return a ``device:`` stanza.

    On accept, :meth:`device_stanza` returns::

        {"library": "TSI", "adapter": "TSICam", "pre_init": {"CameraID": "22801"}}

    which is exactly what ``hardware.yaml`` carries under a camera's
    ``device:`` key.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        mm_path: str = "",
        device: Optional[dict] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Choose Micro-Manager camera")
        self.resize(620, 560)

        self._devices: List[MMDevice] = []
        self._prop_editors: Dict[str, QComboBox | QLineEdit] = {}
        self._thread: Optional[_EnumerateThread] = None
        self._initial = dict(device or {})
        # Installations still worth trying automatically. `find_micromanager`
        # returns them in its own order and the first can easily be a stale
        # build whose adapters are all the wrong interface version, so a failed
        # scan falls forward to the next rather than dead-ending the user.
        self._untried: List[str] = []

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Pick the camera's device adapter. mesofield loads it directly, so "
            "no Micro-Manager .cfg file is needed.\n"
            "Rigs with serial ports, hubs or config groups still need a .cfg."
        ))

        # -- installation -----------------------------------------------
        install_row = QHBoxLayout()
        install_row.addWidget(QLabel("Micro-Manager:"))
        self._mm_combo = QComboBox()
        self._mm_combo.setEditable(True)
        installed = installed_micromanagers()
        for path in installed:
            self._mm_combo.addItem(path)
        # Only fall forward through installations the user did not pick.
        self._untried = [] if mm_path else list(installed[1:])
        if mm_path:
            index = self._mm_combo.findText(mm_path)
            if index >= 0:
                self._mm_combo.setCurrentIndex(index)
            else:
                self._mm_combo.insertItem(0, mm_path)
                self._mm_combo.setCurrentIndex(0)
        install_row.addWidget(self._mm_combo, 1)
        rescan = QPushButton("Scan")
        rescan.setFixedWidth(60)
        rescan.clicked.connect(self._rescan)
        install_row.addWidget(rescan)
        layout.addLayout(install_row)

        # -- device list -------------------------------------------------
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter cameras (e.g. TSI, Thor, Demo)…")
        self._filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self._filter)

        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_device_selected)
        layout.addWidget(self._list, 1)

        self._status = theme.dim_label("Scanning device adapters…")
        layout.addWidget(self._status)

        # -- pre-init properties -----------------------------------------
        self._props_box = QGroupBox("Device settings (set before initialisation)")
        self._props_form = QFormLayout(self._props_box)
        layout.addWidget(self._props_box)

        # -- test ----------------------------------------------------------
        test_row = QHBoxLayout()
        self._test_btn = QPushButton("Test: load and snap a frame")
        self._test_btn.setEnabled(False)
        self._test_btn.clicked.connect(self._on_test)
        test_row.addWidget(self._test_btn)
        layout.addLayout(test_row)

        self._test_result = theme.dim_label("", selectable=True)
        self._test_result.setWordWrap(True)
        layout.addWidget(self._test_result)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_btn.setEnabled(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._start_enumerate()

    # -- enumeration -----------------------------------------------------

    def mm_path(self) -> str:
        return self._mm_combo.currentText().strip()

    def _rescan(self) -> None:
        """A deliberate scan of the chosen installation -- no falling forward."""
        self._untried = []
        self._start_enumerate()

    def _start_enumerate(self) -> None:
        path = self.mm_path()
        if not path:
            self._set_status("Select a Micro-Manager installation.", warn=True)
            return
        self._list.clear()
        self._set_status("Scanning device adapters…")
        self._thread = _EnumerateThread(path, self)
        self._thread.done.connect(self._on_enumerated)
        # Parented to the dialog, so without this each scan would leave a
        # finished QThread behind for the dialog's lifetime.
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_enumerated(self, devices: object, error: str) -> None:
        self._devices = list(devices or [])
        if error:
            if self._untried:
                skipped = self.mm_path()
                nxt = self._untried.pop(0)
                logger.info("%s unusable (%s); trying %s", skipped, error, nxt)
                index = self._mm_combo.findText(nxt)
                if index >= 0:
                    self._mm_combo.setCurrentIndex(index)
                else:
                    self._mm_combo.setEditText(nxt)
                self._start_enumerate()
                return
            self._set_status(f"{self.mm_path()}: {error}", warn=True)
            return
        self._untried = []
        self._set_status(f"{len(self._devices)} camera devices available.")
        self._apply_filter()
        self._select_initial()

    def _select_initial(self) -> None:
        """Preselect the device an existing stanza names, if it is still there."""
        library = self._initial.get("library")
        adapter = self._initial.get("adapter")
        if not (library and adapter):
            return
        for row in range(self._list.count()):
            item = self._list.item(row)
            device = item.data(Qt.ItemDataRole.UserRole)
            if device.library == library and device.adapter == adapter:
                self._list.setCurrentRow(row)
                return

    def _apply_filter(self) -> None:
        needle = self._filter.text().strip().lower()
        self._list.clear()
        for device in self._devices:
            haystack = f"{device.library} {device.adapter} {device.description}".lower()
            if needle and needle not in haystack:
                continue
            item = QListWidgetItem(device.display)
            if device.description:
                item.setToolTip(device.description)
            item.setData(Qt.ItemDataRole.UserRole, device)
            self._list.addItem(item)

    def _set_status(self, text: str, *, warn: bool = False) -> None:
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {theme.WARN if warn else theme.TEXT_DIM};")

    # -- selection -------------------------------------------------------

    def selected_device(self) -> Optional[MMDevice]:
        item = self._list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _on_device_selected(self, *_args) -> None:
        self._clear_props()
        self._test_result.clear()
        device = self.selected_device()
        self._ok_btn.setEnabled(device is not None)
        self._test_btn.setEnabled(device is not None)
        if device is None:
            return
        try:
            props = probe_pre_init_properties(self.mm_path(), device.library, device.adapter)
        except Exception as exc:
            self._props_form.addRow(theme.dim_label(f"Could not read settings: {exc}"))
            return
        if not props:
            self._props_form.addRow(theme.dim_label(
                "This adapter has no pre-initialisation settings. If you have "
                "two identical cameras, it cannot tell them apart — look for "
                "another adapter that offers a CameraID or serial number."
            ))
            return
        existing = self._initial.get("pre_init") or {}
        for prop in props:
            editor = self._make_prop_editor(prop, existing.get(prop.name))
            self._prop_editors[prop.name] = editor
            self._props_form.addRow(prop.name, editor)

    def _make_prop_editor(
        self, prop: PreInitProperty, existing: Any
    ) -> QComboBox | QLineEdit:
        value = str(existing) if existing is not None else prop.current
        if prop.allowed:
            editor = QComboBox()
            editor.addItems(prop.allowed)
            index = editor.findText(value)
            editor.setCurrentIndex(index if index >= 0 else 0)
            if prop.name.lower() == "cameraid" and len(prop.allowed) > 1:
                editor.setToolTip(
                    "Serial numbers of the connected cameras. '0' means "
                    "'first available', which is ambiguous when two of the "
                    "same camera are plugged in — pick a serial instead."
                )
            return editor
        editor = QLineEdit(value)
        return editor

    def _clear_props(self) -> None:
        self._prop_editors.clear()
        while self._props_form.count():
            item = self._props_form.takeAt(0)
            widget = item.widget() if item else None
            if widget is not None:
                widget.deleteLater()

    def _pre_init_values(self) -> Dict[str, str]:
        values: Dict[str, str] = {}
        for name, editor in self._prop_editors.items():
            text = (
                editor.currentText() if isinstance(editor, QComboBox) else editor.text()
            ).strip()
            if text:
                values[name] = text
        return values

    # -- test ------------------------------------------------------------

    def _on_test(self) -> None:
        device = self.selected_device()
        if device is None:
            return
        self._test_btn.setEnabled(False)
        self._test_result.setText("Loading camera…")
        self._test_result.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self._test_result.repaint()
        try:
            summary = test_device(
                self.mm_path(), device.library, device.adapter, self._pre_init_values()
            )
        except Exception as exc:
            self._test_result.setText(f"✗ {exc}")
            self._test_result.setStyleSheet(f"color: {theme.DANGER};")
        else:
            self._test_result.setText(f"✓ Snapped {summary}")
            self._test_result.setStyleSheet(f"color: {theme.ACCENT};")
        finally:
            self._test_btn.setEnabled(True)

    # -- result ------------------------------------------------------------

    def device_stanza(self) -> dict:
        """The ``device:`` mapping for ``hardware.yaml``."""
        device = self.selected_device()
        if device is None:
            return {}
        stanza: dict = {"library": device.library, "adapter": device.adapter}
        pre_init = self._pre_init_values()
        if pre_init:
            stanza["pre_init"] = pre_init
        return stanza
