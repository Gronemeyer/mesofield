"""Reusable, guided builders for authoring mesofield configuration files.

Two things a first-time user must produce -- an ``experiment.json`` and a
``hardware.yaml`` (a rig) -- are built here from one shared atom so neither can
end up malformed:

- :class:`SchemaForm` renders a typed form from a list of :class:`FieldSpec`.
- :class:`ExperimentBuilderDialog` is one such form -> ``experiment.json``.
- :class:`HardwareBuilderDialog` wraps the same form in an "add device" shell,
  assembling a complete ``hardware.yaml`` from the central :data:`DEVICE_SPECS`
  catalog and saving it as a canonical rig.

The device catalog is intentionally small (the onboarding trio: OpenCV camera,
MicroManager camera, serial wheel encoder, plus mocks). Exotic devices are
still hand-edited in YAML -- this builder optimizes for "hard to fail", not for
covering every possible stanza.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Field schema + reusable form atom
# ---------------------------------------------------------------------------


@dataclass
class FieldSpec:
    """One editable field in a :class:`SchemaForm`."""

    key: str
    label: str
    type: type = str  # str | int | bool
    default: Any = ""
    choices: Optional[List[Any]] = None
    help: str = ""
    file_filter: Optional[str] = None  # set -> render a "browse for file" row
    directory: bool = False            # set -> render a "browse for folder" row

    @property
    def is_path(self) -> bool:
        return self.file_filter is not None or self.directory


class SchemaForm(QWidget):
    """Map a list of :class:`FieldSpec` to a typed form and read values back.

    Mirrors the editor-selection idiom used by
    :class:`mesofield.gui.controller.ConfigFormWidget` so the two feel the same:
    ``QSpinBox`` for ints, ``QCheckBox`` for bools, ``QComboBox`` for choices,
    a browse row for files, else ``QLineEdit``.
    """

    def __init__(self, fields: List[FieldSpec], parent: QWidget | None = None):
        super().__init__(parent)
        self._fields = fields
        self._editors: dict[str, QWidget] = {}

        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        for spec in fields:
            editor = self._make_editor(spec)
            self._editors[spec.key] = editor
            if spec.help:
                editor.setToolTip(spec.help)
            form.addRow(spec.label, editor)

    def _make_editor(self, spec: FieldSpec) -> QWidget:
        if spec.is_path:
            return self._make_file_row(spec)
        if spec.type is bool:
            editor = QCheckBox()
            editor.setChecked(bool(spec.default))
            return editor
        if spec.choices:
            editor = QComboBox()
            editor.addItems([str(c) for c in spec.choices])
            idx = editor.findText(str(spec.default))
            if idx >= 0:
                editor.setCurrentIndex(idx)
            return editor
        if spec.type is int:
            editor = QSpinBox()
            editor.setRange(-1_000_000, 1_000_000)
            try:
                editor.setValue(int(spec.default or 0))
            except (TypeError, ValueError):
                editor.setValue(0)
            return editor
        editor = QLineEdit()
        editor.setText("" if spec.default is None else str(spec.default))
        return editor

    @staticmethod
    def _make_file_row(spec: FieldSpec) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        edit.setText("" if spec.default is None else str(spec.default))
        layout.addWidget(edit)
        browse = QPushButton("Browse…")
        browse.setFixedWidth(80)

        def _pick() -> None:
            if spec.directory:
                path = QFileDialog.getExistingDirectory(row, "Select folder")
            else:
                path, _ = QFileDialog.getOpenFileName(row, "Select file", "", spec.file_filter)
            if path:
                edit.setText(path)

        browse.clicked.connect(_pick)
        layout.addWidget(browse)
        row._line_edit = edit  # type: ignore[attr-defined]
        return row

    def set_values(self, values: dict) -> None:
        """Populate editors from a mapping (the inverse of :meth:`values`)."""
        for spec in self._fields:
            if spec.key not in values:
                continue
            editor = self._editors[spec.key]
            val = values[spec.key]
            if spec.is_path:
                editor._line_edit.setText("" if val is None else str(val))  # type: ignore[attr-defined]
            elif isinstance(editor, QCheckBox):
                editor.setChecked(bool(val))
            elif isinstance(editor, QComboBox):
                i = editor.findText(str(val))
                if i >= 0:
                    editor.setCurrentIndex(i)
            elif isinstance(editor, QSpinBox):
                try:
                    editor.setValue(int(val))
                except (TypeError, ValueError):
                    pass
            else:
                editor.setText("" if val is None else str(val))

    def values(self) -> dict:
        """Return ``{key: typed value}`` for every field."""
        out: dict[str, Any] = {}
        for spec in self._fields:
            editor = self._editors[spec.key]
            if spec.is_path:
                out[spec.key] = editor._line_edit.text().strip()  # type: ignore[attr-defined]
            elif isinstance(editor, QCheckBox):
                out[spec.key] = editor.isChecked()
            elif isinstance(editor, QComboBox):
                out[spec.key] = editor.currentText()
            elif isinstance(editor, QSpinBox):
                out[spec.key] = int(editor.value())
            else:
                text = editor.text().strip()
                out[spec.key] = self._coerce(spec, text)
        return out

    @staticmethod
    def _coerce(spec: FieldSpec, text: str) -> Any:
        if spec.type is int and text:
            try:
                return int(text)
            except ValueError:
                return text
        return text


# ---------------------------------------------------------------------------
# Central device catalog (the single maintenance point)
# ---------------------------------------------------------------------------

_BIDS_CHOICES = ["func", "beh", "behav", "anat"]

# OpenCV capture/codec choices + platform defaults — owned by
# mesofield.data.codecs so there's a single dependency-free place that both the
# wizard (here) and the runtime camera/writer agree on. The wrong capture
# backend/format is the single most common rig mistake (a camera opens but
# yields no frames), so they're explicit, platform-defaulted dropdowns
from mesofield.data.codecs import (
    FOURCC_CHOICES,
    DEFAULT_FOURCC,
    CV_BACKENDS as _CV_BACKENDS,
    CAP_FOURCC_CHOICES as _CAP_FOURCCS,
    default_cv_backend as _default_cv_backend,
    default_cap_fourcc as _default_cap_fourcc,
)


def _output_fields(suffix: str, file_type: str, file_choices: List[str], bids: str) -> List[FieldSpec]:
    """Build the shared ``output:`` sub-form for a device."""
    return [
        FieldSpec("suffix", "output.suffix", str, suffix, help="Filename suffix for this stream"),
        FieldSpec("file_type", "output.file_type", str, file_type, choices=file_choices),
        FieldSpec("bids_type", "output.bids_type", str, bids, choices=_BIDS_CHOICES),
    ]


@dataclass
class DeviceSpec:
    """A buildable device type: its label, default stanza name, and fields."""

    type: str
    label: str
    default_name: str
    category: str = "Device"          # groups the "+ Add device" menu
    fields: List[FieldSpec] = field(default_factory=list)
    output: List[FieldSpec] = field(default_factory=list)
    fixed: dict = field(default_factory=dict)  # keys forced into the stanza
    stimulus: bool = False            # subprocess stimulus app: no output, never primary


# Subprocess plumbing shared by every stimulus app (SubprocessStimulusDevice).
# These are transitional: ``app_dir`` / ``python_exe`` point at a separately
# installed app + interpreter today, and can be left blank once mesofield ships
# the stimulus package (pip-installed, same interpreter).
def _stimulus_launch_fields() -> List[FieldSpec]:
    return [
        FieldSpec(
            "app_dir", "app_dir", str, "", directory=True,
            help="Folder of the stimulus app. Optional once the app is pip-installed.",
        ),
        FieldSpec(
            "python_exe", "python_exe", str, "",
            file_filter="Python interpreter (python*);;All Files (*)",
            help="Interpreter that has the stimulus app installed. Leave blank once "
                 "mesofield ships it in the same environment.",
        ),
        FieldSpec("ready_timeout", "ready_timeout (s)", int, 30),
    ]


# Keys mirror what the loaders actually read
# (mesofield/hardware.py, devices/cameras.py, devices/encoder.py, devices/mocks.py,
# devices/stimulus_base.py, devices/mouseportal_device.py).
# The "+ Add device" menu only offers a type once its class is registered in the
# DeviceRegistry, so pip-installing a stimulus app makes it appear automatically.
DEVICE_SPECS: dict[str, DeviceSpec] = {
    "opencv_camera": DeviceSpec(
        type="opencv_camera",
        label="OpenCV / USB camera",
        default_name="camera",
        category="Camera",
        fields=[
            FieldSpec("device_index", "device_index", int, 0, help="cv2.VideoCapture index"),
            FieldSpec(
                "cv_backend", "cv_backend", str, _default_cv_backend(), choices=_CV_BACKENDS,
                help="OpenCV capture backend. AVFOUNDATION=macOS, MSMF/DSHOW=Windows, "
                     "V4L2=Linux, ANY=auto. Wrong value = camera opens but shows nothing.",
            ),
            FieldSpec(
                "cap_fourcc", "cap_fourcc", str, _default_cap_fourcc(), choices=_CAP_FOURCCS,
                help="Capture pixel format forced on the camera (NOT the saved-file "
                     "codec). Blank = camera default; defaults to MJPG on Windows, "
                     "where USB webcams otherwise open but show no live view.",
            ),
            FieldSpec("fps", "fps", int, 30),
            FieldSpec(
                "fourcc", "fourcc", str, DEFAULT_FOURCC, choices=FOURCC_CHOICES,
                help="OpenCV video codec. mp4v=portable default (works everywhere). "
                     "H264/avc1 compress better but need an external codec "
                     "(OpenH264 DLL on Windows, libx264 on Linux); falls back to "
                     "mp4v if unavailable. MJPG/XVID use the .avi container.",
            ),
        ],
        output=_output_fields("cam", "mp4", ["mp4", "ome.tiff"], "func"),
        fixed={"backend": "opencv"},
    ),
    "camera": DeviceSpec(
        type="camera",
        label="MicroManager camera",
        default_name="camera",
        category="Camera",
        fields=[
            FieldSpec(
                "micromanager_path", "micromanager_path", str, "",
                help="Path to the Micro-Manager installation folder (optional)",
                directory=True,
            ),
            FieldSpec(
                "configuration_path", "configuration_path", str, "",
                help="Path to a MicroManager system .cfg (optional)",
                file_filter="MicroManager Config (*.cfg);;All Files (*)",
            ),
        ],
        output=_output_fields("cam", "ome.tiff", ["ome.tiff", "mp4"], "func"),
        fixed={"backend": "micromanager"},
    ),
    "treadmill": DeviceSpec(
        # Teensy treadmill (EncoderSerialInterface) registers under "encoder";
        # the stanza name defaults to "treadmill" so it routes through
        # _init_extras rather than the reserved top-level "encoder" key.
        type="encoder",
        label="Treadmill encoder (Teensy serial)",
        default_name="treadmill",
        category="Encoder",
        fields=[
            FieldSpec("port", "port", str, "COM5", help="e.g. COM5 (Win) or /dev/ttyACM0"),
            FieldSpec("baudrate", "baudrate", int, 192000),
        ],
        output=_output_fields("treadmill", "csv", ["csv"], "beh"),
    ),
    "wheel": DeviceSpec(
        type="wheel",
        label="Wheel encoder (serial)",
        default_name="wheel",
        category="Encoder",
        fields=[
            FieldSpec("port", "port", str, "COM4", help="e.g. COM4 (Win) or /dev/ttyUSB0"),
            FieldSpec("baudrate", "baudrate", int, 115200),
            FieldSpec("cpr", "cpr", int, 2400),
            FieldSpec("diameter_mm", "diameter_mm", int, 80),
        ],
        output=_output_fields("wheel", "csv", ["csv"], "beh"),
    ),
    "nidaq": DeviceSpec(
        type="nidaq",
        label="NI-DAQ (start trigger + TTL)",
        default_name="nidaq",
        category="DAQ",
        fields=[
            FieldSpec("device_name", "device_name", str, "Dev1"),
            FieldSpec("lines", "lines", str, "port1/line1",
                      help="Digital-output line pulsed once at start to signal an external system"),
            FieldSpec("ctr", "ctr", str, "ctr0",
                      help="Counter input that tallies returned TTL rising edges"),
            FieldSpec("io_type", "io_type", str, "DO", choices=["DO"]),
            FieldSpec("development_mode", "development_mode", bool, False),
        ],
        output=_output_fields("nidaq", "csv", ["csv"], "beh"),
    ),
    # --- Stimulus apps (subprocess-backed; design lives in experiment.json) ---
    "psychopy": DeviceSpec(
        type="psychopy",
        label="PsychoPy stimulus",
        default_name="psychopy",
        category="Stimulus",
        stimulus=True,
        fields=_stimulus_launch_fields(),
    ),
    "mouseportal": DeviceSpec(
        type="mouseportal",
        label="MousePortal corridor",
        default_name="mouseportal",
        category="Stimulus",
        stimulus=True,
        fields=_stimulus_launch_fields() + [
            FieldSpec("udp_port", "udp_port", int, 8765),
            FieldSpec("treadmill_id", "treadmill_id", str, "treadmill",
                      help="Device id of the encoder whose velocity feeds MousePortal"),
            FieldSpec("tail_seconds", "tail_seconds", int, 5,
                      help="Seconds to keep recording past the last trial so cameras "
                           "finish their preallocated frames"),
        ],
    ),
    "mock_wheel": DeviceSpec(
        type="mock_wheel",
        label="Mock wheel (no hardware)",
        default_name="wheel",
        category="Mock",
        fields=[
            FieldSpec("sample_interval_ms", "sample_interval_ms", int, 50),
            FieldSpec("cpr", "cpr", int, 2400),
            FieldSpec("diameter_mm", "diameter_mm", int, 80),
        ],
        output=_output_fields("wheel", "csv", ["csv"], "beh"),
    ),
    "mock_camera": DeviceSpec(
        type="mock_camera",
        label="Mock camera (no hardware)",
        default_name="camera",
        category="Mock",
        fields=[
            FieldSpec("width", "width", int, 64),
            FieldSpec("height", "height", int, 64),
            FieldSpec("frame_interval_ms", "frame_interval_ms", int, 50),
        ],
        output=_output_fields("meso", "ome.tiff", ["ome.tiff", "mp4"], "func"),
    ),
}


# ---------------------------------------------------------------------------
# Hardware (rig) builder
# ---------------------------------------------------------------------------


class _DeviceCard(QFrame):
    """One device stanza in the hardware builder: name + fields + primary."""

    def __init__(self, spec: DeviceSpec, parent: QWidget | None = None):
        super().__init__(parent)
        self.spec = spec
        self.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        tag = "stimulus" if spec.stimulus else f"type: {spec.type}"
        header.addWidget(QLabel(f"<b>{spec.label}</b>  <span style='color:gray'>({tag})</span>"))
        header.addStretch()
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.setFixedWidth(80)
        header.addWidget(self.remove_btn)
        layout.addLayout(header)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("name:"))
        self._name_edit = QLineEdit(spec.default_name)
        self._name_edit.setToolTip("The hardware.yaml stanza key for this device")
        name_row.addWidget(self._name_edit)
        self._primary_check = QCheckBox("primary")
        self._primary_check.setToolTip("The device that drives acquisition timing (exactly one rig-wide)")
        # Stimulus apps are not data producers; they never drive timing.
        self._primary_check.setVisible(not spec.stimulus)
        name_row.addWidget(self._primary_check)
        layout.addLayout(name_row)

        self._field_form = SchemaForm(spec.fields) if spec.fields else None
        if self._field_form is not None:
            layout.addWidget(self._field_form)

        self._output_form = SchemaForm(spec.output) if spec.output else None
        if self._output_form is not None:
            out_label = QLabel("<i>output</i>")
            layout.addWidget(out_label)
            layout.addWidget(self._output_form)

    # -- accessors -----------------------------------------------------------

    def name(self) -> str:
        return self._name_edit.text().strip()

    def set_values(self, name: str, stanza: dict) -> None:
        """Populate this card from an existing ``hardware.yaml`` stanza."""
        self._name_edit.setText(str(name))
        self._primary_check.setChecked(bool(stanza.get("primary", False)))
        if self._field_form is not None:
            self._field_form.set_values(stanza)
        if self._output_form is not None and isinstance(stanza.get("output"), dict):
            self._output_form.set_values(stanza["output"])

    def is_primary(self) -> bool:
        return self._primary_check.isChecked()

    def set_primary(self, value: bool) -> None:
        self._primary_check.setChecked(value)

    def stanza(self) -> dict:
        """Assemble this card's ``hardware.yaml`` stanza."""
        out: dict[str, Any] = {"type": self.spec.type}
        out.update(self.spec.fixed)
        if self._field_form is not None:
            for key, val in self._field_form.values().items():
                if val == "":  # drop empty optionals (e.g. configuration_path)
                    continue
                out[key] = val
        if self.is_primary():
            out["primary"] = True
        if self._output_form is not None:
            out["output"] = self._output_form.values()
        return out


class HardwareBuilderDialog(QDialog):
    """Guided builder that assembles a complete ``hardware.yaml`` rig.

    On success, the rig is written to this machine's rig store and its name is
    available as :attr:`rig_name`.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        doc: Optional[dict] = None,
        rig_name: Optional[str] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Edit rig (hardware.yaml)" if doc else "New rig (hardware.yaml)")
        self.resize(560, 600)
        self.rig_name: Optional[str] = None
        self._initial_name = rig_name
        self._cards: list[_DeviceCard] = []
        # Stanzas/keys the builder doesn't model (custom devices, cameras: lists,
        # viewer_type, etc.) are preserved verbatim so editing never drops them.
        self._passthrough: dict = {}

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Add the devices on this rig. Each becomes a stanza in hardware.yaml.\n"
            "Exactly one recording device drives timing (“primary”); stimulus "
            "apps (PsychoPy, MousePortal) run alongside."
        ))

        top = QHBoxLayout()
        top.addWidget(QLabel("memory_buffer_size:"))
        self._buffer_spin = QSpinBox()
        self._buffer_spin.setRange(1, 1_000_000)
        self._buffer_spin.setValue(1000)
        top.addWidget(self._buffer_spin)
        top.addStretch()
        self._add_combo = QComboBox()
        self._populate_add_combo()
        self._add_combo.currentIndexChanged.connect(self._on_add)
        top.addWidget(self._add_combo)
        layout.addLayout(top)

        # Scrollable device-card area
        self._cards_container = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_container)
        self._cards_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._cards_container)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox()
        self._save_btn = buttons.addButton("Save as rig…", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        self._save_btn.clicked.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if doc:
            self._load_doc(doc)

    _CATEGORY_ORDER = ["Camera", "Encoder", "DAQ", "Stimulus", "Mock"]

    def _populate_add_combo(self) -> None:
        """Group the add-device menu by category, registration-aware.

        Types whose class isn't registered in this environment (e.g. a stimulus
        app that isn't installed yet) are still selectable -- you may be
        authoring a rig for another machine -- but are tagged so the future
        pip-install story is visible: install the package, it registers, the
        tag disappears.
        """
        import mesofield.hardware  # noqa: F401  (triggers built-in device registration)
        from mesofield import DeviceRegistry

        self._add_combo.clear()
        self._add_combo.addItem("+ Add device…")
        by_cat: dict[str, list] = {}
        for type_key, spec in DEVICE_SPECS.items():
            by_cat.setdefault(spec.category, []).append((type_key, spec))
        order = self._CATEGORY_ORDER + [c for c in by_cat if c not in self._CATEGORY_ORDER]
        for cat in order:
            items = by_cat.get(cat)
            if not items:
                continue
            self._add_combo.addItem(f"—  {cat}  —")
            self._add_combo.model().item(self._add_combo.count() - 1).setEnabled(False)
            for type_key, spec in items:
                registered = DeviceRegistry.get_class(type_key) is not None
                label = spec.label if registered else f"{spec.label}  (not installed)"
                self._add_combo.addItem(label, userData=type_key)
                if not registered:
                    self._add_combo.model().item(self._add_combo.count() - 1).setToolTip(
                        "Not available in this environment yet — install the package "
                        "(or fix its import) to run it here."
                    )
        self._add_combo.setCurrentIndex(0)

    # -- populate from an existing rig ---------------------------------------

    def _load_doc(self, doc: dict) -> None:
        """Pre-fill the builder from an existing ``hardware.yaml`` mapping."""
        self._buffer_spin.setValue(int(doc.get("memory_buffer_size", 1000) or 1000))
        for key, val in doc.items():
            if key == "memory_buffer_size":
                continue
            type_key = val.get("type") if isinstance(val, dict) else None
            if type_key in DEVICE_SPECS:
                card = self._make_card(type_key)
                card.set_values(key, val)
            else:
                self._passthrough[key] = val  # preserve what we don't model

    def _make_card(self, type_key: str) -> _DeviceCard:
        card = _DeviceCard(DEVICE_SPECS[type_key])
        card.remove_btn.clicked.connect(lambda _=False, c=card: self._remove_card(c))
        self._cards_layout.insertWidget(self._cards_layout.count() - 1, card)
        self._cards.append(card)
        return card

    # -- slots ---------------------------------------------------------------

    def _on_add(self, index: int) -> None:
        if index <= 0:
            return
        type_key = self._add_combo.itemData(index)
        self._add_combo.setCurrentIndex(0)
        if type_key:
            self._make_card(type_key)

    def _remove_card(self, card: _DeviceCard) -> None:
        if card in self._cards:
            self._cards.remove(card)
        self._cards_layout.removeWidget(card)
        card.deleteLater()

    def _save(self) -> None:
        if not self._cards:
            QMessageBox.information(self, "No devices", "Add at least one device first.")
            return

        names = [c.name() for c in self._cards]
        if any(not n for n in names):
            QMessageBox.warning(self, "Missing name", "Every device needs a name.")
            return
        if len(set(names)) != len(names):
            QMessageBox.warning(self, "Duplicate names", "Device names must be unique.")
            return

        # Primary is a recording device's job; stimulus apps never drive timing.
        recording = [c for c in self._cards if not c.spec.stimulus]
        if not recording:
            QMessageBox.warning(
                self, "No recording device",
                "Add at least one camera or encoder — a stimulus app alone can't "
                "drive acquisition.",
            )
            return
        primaries = [c for c in recording if c.is_primary()]
        if len(primaries) > 1:
            QMessageBox.warning(self, "Too many primaries", "Only one device can be primary.")
            return
        if not primaries:  # hard-to-fail: pick one for the user
            recording[0].set_primary(True)

        doc: dict[str, Any] = {"memory_buffer_size": int(self._buffer_spin.value())}
        doc.update(self._passthrough)  # preserve stanzas the builder doesn't model
        for card in self._cards:
            doc[card.name()] = card.stanza()

        import mesofield.hardware  # noqa: F401  (triggers built-in device registration)
        from mesofield import DeviceRegistry
        from mesofield.scaffold import rigs

        unavailable = sorted({
            c.spec.type for c in self._cards
            if DeviceRegistry.get_class(c.spec.type) is None
        })
        if unavailable:
            QMessageBox.information(
                self, "Heads up",
                "These device types aren't available in this environment yet and "
                f"will be skipped until installed: {', '.join(unavailable)}.\n"
                "The rig is still saved (useful when authoring for another machine).",
            )

        name, ok = QInputDialog.getText(
            self, "Save rig", "Rig name:", text=self._initial_name or ""
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        force = False
        if rigs.rig_path(name).exists():
            reply = QMessageBox.question(
                self, "Overwrite rig?",
                f"A rig named {name!r} already exists. Overwrite it?",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            force = True

        try:
            rigs.save_rig(name, doc, force=force)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save rig:\n\n{exc}")
            return

        self.rig_name = name
        self.accept()


# ---------------------------------------------------------------------------
# Experiment.json builder
# ---------------------------------------------------------------------------

# Session-level parameters shared across every subject.
SESSION_FIELDS: List[FieldSpec] = [
    FieldSpec("experimenter", "experimenter", str, "your-name"),
    FieldSpec("protocol", "protocol", str, "PROTOCOL"),
    FieldSpec("duration", "duration (s)", int, 5),
    FieldSpec("start_on_trigger", "start_on_trigger", bool, False),
]

# Columns always present in the subjects table (before any custom variables).
_FIXED_COLUMNS = ("subject", "session", "task")


def build_experiment_doc(
    configuration: dict,
    tasks: List[str],
    variables: List[tuple],  # (name, type, show_in_app)
    subjects: List[dict],
) -> dict:
    """Assemble the ``Configuration`` / ``Subjects`` / ``DisplayKeys`` shape.

    Blank per-subject cells are omitted, so a subject may carry full details,
    a few, or none. When more than one task is defined it is written as a list
    in ``Configuration`` (which mesofield surfaces as a runtime dropdown).
    ``DisplayKeys`` (the fields shown/editable in the app) includes the fixed
    columns, the session params, and only the variables flagged *show in app*.
    """
    config = {
        "experimenter": configuration.get("experimenter", ""),
        "protocol": configuration.get("protocol", ""),
        "duration": int(configuration.get("duration") or 0),
        "start_on_trigger": bool(configuration.get("start_on_trigger")),
    }
    tasks = [t for t in tasks if t]
    if len(tasks) == 1:
        config["task"] = tasks[0]
    elif len(tasks) > 1:
        config["task"] = list(tasks)

    subjects_out: dict[str, dict] = {}
    for row in subjects:
        sid = str(row.get("subject", "")).strip()
        if not sid:
            continue
        entry: dict[str, Any] = {}
        session = str(row.get("session", "")).strip()
        if session:
            entry["session"] = session
        task = str(row.get("task", "")).strip()
        if task:
            entry["task"] = task
        for name, typ, _show in variables:
            raw = row.get(name, "")
            val = "" if raw is None else str(raw).strip()
            if not val:
                continue
            if typ is int:
                try:
                    val = int(val)
                except ValueError:
                    pass
            entry[name] = val
        subjects_out[sid] = entry

    display = list(_FIXED_COLUMNS) + [n for n, _, show in variables if show] + \
        ["experimenter", "protocol", "duration"]
    return {"Configuration": config, "Subjects": subjects_out, "DisplayKeys": display}


class ExperimentBuilderDialog(QDialog):
    """Author a fresh ``experiment.json`` from a guided, multi-subject form.

    The user sets session-level parameters, defines the task(s) and any
    per-subject variables (extra columns), then fills a subjects table -- as
    much or as little per subject as they like. On success the written path is
    available as :attr:`json_path`.
    """

    _COL_LABELS = {"subject": "Subject ID", "session": "session", "task": "task"}

    def __init__(self, default_dir: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("New experiment.json")
        self.resize(640, 720)
        self._default_dir = default_dir or os.getcwd()
        self.json_path: Optional[str] = None

        self._tasks: List[str] = []                # defined by the user (no confusing default)
        self._variables: List[tuple] = [           # (name, type, show_in_app) — examples to edit
            ("sex", str, True),
            ("age", int, True),
        ]
        self._subjects: List[dict] = []            # row-dicts keyed by column name
        self._rendering = False

        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        layout = QVBoxLayout(body)

        layout.addWidget(self._session_group())
        layout.addWidget(self._tasks_group())
        layout.addWidget(self._variables_group())
        layout.addWidget(self._subjects_group())

        self._summary = QLabel()
        self._summary.setStyleSheet("color: #7d8893;")
        outer.addWidget(self._summary)

        buttons = QDialogButtonBox()
        save_btn = buttons.addButton("Save experiment.json…", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        save_btn.clicked.connect(self._save)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self._add_subject_row()  # start with one subject
        self._refresh_summary()

    # -- section builders ----------------------------------------------------

    def _session_group(self) -> QGroupBox:
        box = QGroupBox("①  Session")
        v = QVBoxLayout(box)
        self._session_form = SchemaForm(SESSION_FIELDS)
        v.addWidget(self._session_form)
        return box

    def _tasks_group(self) -> QGroupBox:
        box = QGroupBox("②  Tasks")
        v = QVBoxLayout(box)
        tasks_hint = QLabel("Task names a subject can be assigned (picked per subject below).")
        tasks_hint.setWordWrap(True)
        v.addWidget(tasks_hint)
        self._task_list = QListWidget()
        self._task_list.setMaximumHeight(96)
        self._task_list.addItems(self._tasks)
        v.addWidget(self._task_list)
        row = QHBoxLayout()
        self._task_edit = QLineEdit()
        self._task_edit.setPlaceholderText("e.g. baseline, freeview, stim…")
        self._task_edit.returnPressed.connect(self._add_task)
        row.addWidget(self._task_edit)
        add = QPushButton("✚ Add")
        add.clicked.connect(self._add_task)
        row.addWidget(add)
        rm = QPushButton("Remove")
        rm.clicked.connect(self._remove_task)
        row.addWidget(rm)
        v.addLayout(row)
        return box

    def _variables_group(self) -> QGroupBox:
        box = QGroupBox("③  Subject variables")
        v = QVBoxLayout(box)
        vars_hint = QLabel(
            "Extra columns recorded per subject (e.g. sex, genotype, weight). "
            "“Show in app” makes a variable editable in the ExperimentConfig panel."
        )
        vars_hint.setWordWrap(True)
        v.addWidget(vars_hint)
        self._var_list = QListWidget()
        self._var_list.setMaximumHeight(96)
        for name, typ, show in self._variables:
            self._var_list.addItem(self._var_label(name, typ, show))
        v.addWidget(self._var_list)
        row = QHBoxLayout()
        self._var_name = QLineEdit()
        self._var_name.setPlaceholderText("variable name…")
        self._var_name.returnPressed.connect(self._add_variable)
        row.addWidget(self._var_name)
        self._var_type = QComboBox()
        self._var_type.addItems(["text", "number"])
        self._var_type.setFixedWidth(90)
        row.addWidget(self._var_type)
        self._var_show = QCheckBox("Show in app")
        self._var_show.setChecked(True)
        self._var_show.setToolTip(
            "Show this variable in the ExperimentConfig panel so it can be "
            "edited before each recording."
        )
        row.addWidget(self._var_show)
        add = QPushButton("✚ Add")
        add.clicked.connect(self._add_variable)
        row.addWidget(add)
        rm = QPushButton("Remove")
        rm.clicked.connect(self._remove_variable)
        row.addWidget(rm)
        v.addLayout(row)
        return box

    @staticmethod
    def _var_label(name: str, typ: type, show: bool) -> str:
        type_name = "number" if typ is int else "text"
        return f"{name}  ({type_name})  ·  {'shown in app' if show else 'hidden'}"

    def _subjects_group(self) -> QGroupBox:
        box = QGroupBox("④  Subjects")
        v = QVBoxLayout(box)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Number of subjects:"))
        self._count_spin = QSpinBox()
        self._count_spin.setRange(1, 999)
        self._count_spin.setValue(1)
        self._count_spin.valueChanged.connect(self._on_count_changed)
        controls.addWidget(self._count_spin)
        controls.addStretch()
        fill = QPushButton("⤓ Fill down")
        fill.setToolTip("Copy the first subject's session / variable values to all subjects")
        fill.clicked.connect(self._fill_down)
        controls.addWidget(fill)
        rm = QPushButton("Remove selected")
        rm.clicked.connect(self._remove_selected_subjects)
        controls.addWidget(rm)
        v.addLayout(controls)

        # Subject ID autofill: type IDs by hand, or set a prefix + start number
        # and let them fill (e.g. STREHAB01..STREHAB08).
        id_row = QHBoxLayout()
        id_row.addWidget(QLabel("ID prefix:"))
        self._id_prefix = QLineEdit("SUBJ")
        self._id_prefix.setPlaceholderText("e.g. STREHAB or GS (leave blank for none)")
        id_row.addWidget(self._id_prefix, 1)
        id_row.addWidget(QLabel("start #:"))
        self._id_start = QSpinBox()
        self._id_start.setRange(0, 9999)
        self._id_start.setValue(1)
        id_row.addWidget(self._id_start)
        apply_ids = QPushButton("Apply IDs")
        apply_ids.setToolTip("Number every subject as <prefix><NN> from the start number")
        apply_ids.clicked.connect(self._apply_ids)
        id_row.addWidget(apply_ids)
        v.addLayout(id_row)

        self._table = QTableWidget(0, 0)
        self._table.setMinimumHeight(180)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.verticalHeader().setVisible(False)
        self._table.itemChanged.connect(self._on_item_changed)
        v.addWidget(self._table)
        self._render_table()
        return box

    # -- tasks ---------------------------------------------------------------

    def _add_task(self) -> None:
        name = self._task_edit.text().strip()
        if not name or name in self._tasks:
            return
        self._tasks.append(name)
        self._task_list.addItem(name)
        self._task_edit.clear()
        self._render_table()  # refresh per-row task dropdowns
        self._refresh_summary()

    def _remove_task(self) -> None:
        row = self._task_list.currentRow()
        if row < 0:
            return
        removed = self._tasks.pop(row)
        self._task_list.takeItem(row)
        fallback = self._tasks[0] if self._tasks else ""
        for subj in self._subjects:  # reassign subjects left pointing at the removed task
            if subj.get("task") == removed:
                subj["task"] = fallback
        self._render_table()
        self._refresh_summary()

    # -- variables -----------------------------------------------------------

    def _add_variable(self) -> None:
        name = self._var_name.text().strip()
        if not name or any(name == n for n, _, _ in self._variables) or name in _FIXED_COLUMNS:
            return
        typ = int if self._var_type.currentText() == "number" else str
        show = self._var_show.isChecked()
        self._variables.append((name, typ, show))
        self._var_list.addItem(self._var_label(name, typ, show))
        self._var_name.clear()
        self._render_table()
        self._refresh_summary()

    def _remove_variable(self) -> None:
        row = self._var_list.currentRow()
        if row < 0:
            return
        name, _, _ = self._variables.pop(row)
        self._var_list.takeItem(row)
        for subj in self._subjects:
            subj.pop(name, None)
        self._render_table()
        self._refresh_summary()

    # -- subjects ------------------------------------------------------------

    def _columns(self) -> List[str]:
        return list(_FIXED_COLUMNS) + [n for n, _, _ in self._variables]

    def _next_id(self) -> str:
        """Auto-generate the next subject ID from the current prefix + start #."""
        prefix = self._id_prefix.text() if hasattr(self, "_id_prefix") else "SUBJ"
        start = self._id_start.value() if hasattr(self, "_id_start") else 1
        return f"{prefix}{start + len(self._subjects):02d}"

    def _add_subject_row(self, subject_id: Optional[str] = None) -> None:
        self._subjects.append({
            "subject": subject_id or self._next_id(),
            "session": "01",
            "task": self._tasks[0] if self._tasks else "",
        })
        self._render_table()
        self._sync_count_spin()

    def _apply_ids(self) -> None:
        """Renumber every subject as ``<prefix><NN>`` from the start number."""
        prefix = self._id_prefix.text()
        start = self._id_start.value()
        for i, subj in enumerate(self._subjects):
            subj["subject"] = f"{prefix}{start + i:02d}"
        self._render_table()

    def _on_count_changed(self, value: int) -> None:
        if self._rendering:
            return
        while len(self._subjects) < value:
            self._add_subject_row()
        if len(self._subjects) > value:
            del self._subjects[value:]
            self._render_table()

    def _remove_selected_subjects(self) -> None:
        rows = sorted({i.row() for i in self._table.selectedIndexes()}, reverse=True)
        if not rows or len(self._subjects) - len(rows) < 1:
            return
        for r in rows:
            del self._subjects[r]
        self._render_table()
        self._sync_count_spin()
        self._refresh_summary()

    def _fill_down(self) -> None:
        if len(self._subjects) < 2:
            return
        src = self._subjects[0]
        for subj in self._subjects[1:]:
            subj["session"] = src.get("session", "")
            subj["task"] = src.get("task", "")
            for name, _, _ in self._variables:
                subj[name] = src.get(name, "")
        self._render_table()

    def _sync_count_spin(self) -> None:
        self._count_spin.blockSignals(True)
        self._count_spin.setValue(len(self._subjects))
        self._count_spin.blockSignals(False)

    def _render_table(self) -> None:
        self._rendering = True
        cols = self._columns()
        self._table.clear()
        self._table.setColumnCount(len(cols))
        self._table.setRowCount(len(self._subjects))
        self._table.setHorizontalHeaderLabels(
            [self._COL_LABELS.get(c, c) for c in cols]
        )
        for r, subj in enumerate(self._subjects):
            for c, col in enumerate(cols):
                if col == "task":
                    # Editable so a task can be typed even before any are
                    # defined in the Tasks section (which seeds the dropdown).
                    combo = QComboBox()
                    combo.setEditable(True)
                    combo.addItems(self._tasks)
                    combo.setCurrentText(subj.get("task", ""))
                    combo.currentTextChanged.connect(
                        lambda text, row=r: self._subjects[row].__setitem__("task", text)
                    )
                    self._table.setCellWidget(r, c, combo)
                else:
                    item = QTableWidgetItem(str(subj.get(col, "")))
                    self._table.setItem(r, c, item)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._rendering = False
        self._refresh_summary()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._rendering:
            return
        cols = self._columns()
        col = cols[item.column()]
        self._subjects[item.row()][col] = item.text()
        if col == "subject":
            self._refresh_summary()

    # -- summary + save ------------------------------------------------------

    def _refresh_summary(self) -> None:
        if not hasattr(self, "_summary"):
            return
        self._summary.setText(
            f"{len(self._subjects)} subject(s)  ·  {len(self._tasks)} task(s)  ·  "
            f"{len(self._variables)} variable(s)"
        )

    def _save(self) -> None:
        ids = [str(s.get("subject", "")).strip() for s in self._subjects]
        if any(not i for i in ids):
            QMessageBox.warning(self, "Missing ID", "Every subject needs an ID.")
            return
        if len(set(ids)) != len(ids):
            QMessageBox.warning(self, "Duplicate IDs", "Subject IDs must be unique.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "New experiment.json",
            os.path.join(self._default_dir, "experiment.json"),
            "JSON Config (*.json)",
        )
        if not path:
            return
        doc = build_experiment_doc(
            self._session_form.values(), self._tasks, self._variables, self._subjects
        )
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, indent=4)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", f"Could not write JSON:\n\n{exc}")
            return
        self.json_path = os.path.abspath(path)
        self.accept()
