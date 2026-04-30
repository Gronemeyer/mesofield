"""
Configuration wizard for hot-loading experiment and hardware configurations.

Provides a unified widget for selecting and applying:
- Experiment JSON config files
- Hardware YAML config files
- MicroManager system .cfg files (via pymmcore-widgets ConfigurationWidget)
- Full pymmcore-widgets Hardware Configuration Wizard (popup)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, List

from PyQt6.QtCore import pyqtSignal, Qt, QSettings
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QFrame,
)

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from mesofield.base import Procedure
    from mesofield.io.devices.cameras import MMCamera


# ---------------------------------------------------------------------------
# Dark-theme fix for pymmcore-widgets ConfigWizard on Windows
# ---------------------------------------------------------------------------

def _is_dark_palette(widget: QWidget) -> bool:
    """Return True if the widget's palette suggests a dark theme."""
    bg = widget.palette().color(widget.backgroundRole())
    # Use perceived luminance; a value < 128 indicates a dark background
    return bg.lightness() < 128


_DARK_WIZARD_QSS = """
/* ---- top-level wizard and every nested widget ---- */
QWizard, QWizard > QWidget {
    background-color: #2b2b2b;
    color: #e0e0e0;
}

/* ---- wizard-page content area ---- */
QWizardPage, QWizardPage > QWidget, QWizardPage QFrame {
    background-color: #2b2b2b;
    color: #e0e0e0;
}

/* ---- Modern-style header (title / subtitle banner) ---- */
QWizard QWidget#qt_wizard_header {
    background-color: #333333;
    border-bottom: 1px solid #555;
}

/* Side widget (step labels panel) */
QWizard QWidget#qt_wizard_sidebar {
    background-color: #252525;
}

QLabel {
    color: #e0e0e0;
    background: transparent;
}
QComboBox {
    background-color: #3c3c3c;
    color: #e0e0e0;
    border: 1px solid #555;
    padding: 4px;
}
QComboBox QAbstractItemView {
    background-color: #3c3c3c;
    color: #e0e0e0;
    selection-background-color: #0078d4;
    selection-color: #ffffff;
}
QComboBox::drop-down {
    border-left: 1px solid #555;
}
QLineEdit {
    background-color: #3c3c3c;
    color: #e0e0e0;
    border: 1px solid #555;
    padding: 4px;
}
QCheckBox {
    color: #e0e0e0;
}
QCheckBox::indicator {
    border: 1px solid #888;
}
QRadioButton {
    color: #e0e0e0;
}
QPushButton {
    background-color: #3c3c3c;
    color: #e0e0e0;
    border: 1px solid #555;
    padding: 4px 12px;
}
QPushButton:hover {
    background-color: #4a4a4a;
}
QPushButton:pressed {
    background-color: #555;
}
QTableWidget, QTableView, QTreeView {
    background-color: #2b2b2b;
    alternate-background-color: #323232;
    color: #e0e0e0;
    gridline-color: #555;
}
QHeaderView::section {
    background-color: #3c3c3c;
    color: #e0e0e0;
    border: 1px solid #555;
    padding: 4px;
}
QTableWidget::item, QTableView::item, QTreeView::item {
    color: #e0e0e0;
}
QGroupBox {
    color: #e0e0e0;
    border: 1px solid #555;
    margin-top: 8px;
    padding-top: 8px;
}
QGroupBox::title {
    color: #e0e0e0;
}
QSplitter::handle {
    background-color: #555;
}
QFormLayout {
    background: transparent;
}
"""


def _apply_dark_fix(wizard: QWidget) -> None:
    """Apply a dark-theme stylesheet to the wizard if the system palette is dark.

    The pymmcore-widgets ConfigWizard was designed for light themes.  On
    Windows 11 with dark mode, Qt applies dark backgrounds to native controls
    (QComboBox, etc.) but leaves the text colour dark ⇒ invisible text.
    This function detects a dark palette and overlays a comprehensive QSS fix.
    """
    if _is_dark_palette(wizard):
        wizard.setStyleSheet(_DARK_WIZARD_QSS)

class _FilePickerRow(QWidget):
    """A single row: [QLineEdit] [Browse…] [Load]."""

    fileLoaded = pyqtSignal(str)  # emits the chosen path after Load is pressed

    def __init__(
        self,
        label: str = "",
        file_filter: str = "All Files (*)",
        placeholder: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._file_filter = file_filter

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if label:
            layout.addWidget(QLabel(label))

        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText(placeholder)
        layout.addWidget(self.line_edit)

        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse)
        layout.addWidget(browse_btn)

        self.load_btn = QPushButton("Load")
        self.load_btn.setFixedWidth(60)
        self.load_btn.clicked.connect(self._load)
        layout.addWidget(self.load_btn)

    # -- helpers -------------------------------------------------------------

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select file", "", self._file_filter
        )
        if path:
            self.line_edit.setText(path)

    def _load(self) -> None:
        path = self.line_edit.text().strip()
        if path and os.path.isfile(path):
            self.fileLoaded.emit(path)
        elif path:
            QMessageBox.warning(self, "File not found", f"Cannot find:\n{path}")

    def text(self) -> str:
        return self.line_edit.text().strip()


# ---------------------------------------------------------------------------
# Per-camera config card
# ---------------------------------------------------------------------------

class _CameraConfigCard(QFrame):
    """Displays the current .cfg status for a single MicroManager camera
    and provides controls to load a different .cfg or launch the Hardware Wizard."""

    cfgChanged = pyqtSignal()  # emitted after a new .cfg is loaded

    def __init__(
        self,
        cam: MMCamera,
        core: CMMCorePlus,
        index: int,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._cam = cam
        self._core = core
        self.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)

        # Header
        header = QLabel(f"<b>Camera {index + 1}:</b> {cam.name}  "
                        f"<span style='color:gray'>({cam.id} / {cam.backend})</span>")
        layout.addWidget(header)

        # Status label showing which .cfg is loaded
        self._status = QLabel()
        layout.addWidget(self._status)

        # File picker + buttons row
        action_row = QHBoxLayout()
        self._cfg_edit = QLineEdit()
        self._cfg_edit.setPlaceholderText("Select a .cfg file…")
        action_row.addWidget(self._cfg_edit)

        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_cfg)
        action_row.addWidget(browse_btn)

        load_btn = QPushButton("Load .cfg")
        load_btn.setFixedWidth(80)
        load_btn.clicked.connect(self._load_cfg)
        action_row.addWidget(load_btn)

        wizard_btn = QPushButton("🔧 Hardware Wizard…")
        wizard_btn.setToolTip(
            "Open the pymmcore-widgets Hardware Configuration Wizard\n"
            "to inspect/edit devices, roles, delays, and labels."
        )
        wizard_btn.clicked.connect(self._open_hw_wizard)
        action_row.addWidget(wizard_btn)

        layout.addLayout(action_row)

        # Initialise status from the core's current state
        self._refresh_status()

    # -- public --------------------------------------------------------------

    def _refresh_status(self) -> None:
        """Update the status label from the core's loaded config file."""
        cfg_file = self._core.systemConfigurationFile() or ""
        yaml_cfg_path = self._cam.cfg.get("configuration_path", "")

        if yaml_cfg_path:
            self._cfg_edit.setText(yaml_cfg_path)

        if cfg_file:
            display = os.path.basename(cfg_file)
            if "MMConfig_demo" in cfg_file or cfg_file.endswith("MMConfig_demo.cfg"):
                self._status.setText(
                    f"✔ Loaded: <b>{display}</b>  "
                    "<span style='color:#888'>(pymmcore-plus demo default)</span>"
                )
                self._status.setStyleSheet("color: green;")
            else:
                self._status.setText(f"✔ Loaded: <b>{display}</b>")
                self._status.setStyleSheet("color: green;")
        else:
            self._status.setText("⚠ No system configuration loaded")
            self._status.setStyleSheet("color: orange;")

    # -- slots ---------------------------------------------------------------

    def _browse_cfg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select MicroManager Configuration", "",
            "MicroManager Config (*.cfg);;All Files (*)",
        )
        if path:
            self._cfg_edit.setText(path)
            self._load_cfg()  # auto-load after selection

    def _load_cfg(self) -> None:
        path = self._cfg_edit.text().strip()
        if not path:
            QMessageBox.information(self, "No file", "Please select a .cfg file first.")
            return
        if not os.path.isfile(path):
            QMessageBox.warning(self, "File not found", f"Cannot find:\n{path}")
            return
        try:
            self._core.loadSystemConfiguration(path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Load Error",
                f"Failed to load .cfg:\n\n{exc}",
            )
            return
        self._refresh_status()
        self.cfgChanged.emit()

    def _open_hw_wizard(self) -> None:
        """Launch the pymmcore-widgets Hardware Configuration Wizard as a popup dialog."""
        try:
            from pymmcore_widgets import ConfigWizard as _MMConfigWizard
        except ImportError:
            QMessageBox.information(
                self,
                "pymmcore-widgets not available",
                "The Hardware Configuration Wizard requires the\n"
                "pymmcore-widgets package.\n\n"
                "Install it with:\n  pip install pymmcore-widgets",
            )
            return

        current_cfg = self._cfg_edit.text().strip() or ""
        wizard = _MMConfigWizard(
            config_file=current_cfg,
            core=self._core,
            parent=self.window(),
        )
        wizard.setWindowModality(Qt.WindowModality.ApplicationModal)
        _apply_dark_fix(wizard)
        result = wizard.exec()  # blocks until closed

        if result:
            # Wizard accepted – grab the saved .cfg path and auto-load it
            saved_path = wizard.field("dest_config") or ""
            if saved_path and os.path.isfile(saved_path):
                self._cfg_edit.setText(saved_path)
                try:
                    self._core.loadSystemConfiguration(saved_path)
                except Exception:
                    pass  # refresh_status will show current state

        # Refresh status regardless of accept/reject (wizard may have changed state)
        self._refresh_status()
        if result:
            self.cfgChanged.emit()


# ---------------------------------------------------------------------------
# MicroManager config section (container for camera cards)
# ---------------------------------------------------------------------------

class _MMConfigSection(QGroupBox):
    """Holds a :class:`_CameraConfigCard` for each MicroManager camera.

    Shows a placeholder when no cameras have been initialised yet.
    """

    cfgChanged = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__("MicroManager System Config (.cfg)", parent)
        self._layout = QVBoxLayout(self)
        self._cards: List[_CameraConfigCard] = []

        self._placeholder = QLabel(
            "<i>Load a hardware config first to enable MicroManager .cfg loading.</i>"
        )
        self._layout.addWidget(self._placeholder)

    def set_cameras(self, cameras) -> None:
        """Populate the section with one card per MicroManager camera."""
        # Clear existing content
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item is not None:
                w = item.widget()
                if w is not None:
                    w.deleteLater()
        self._cards.clear()

        mm_cams = [
            cam for cam in cameras
            if cam.backend == "micromanager" and hasattr(cam, "core")
        ]

        if not mm_cams:
            self._placeholder = QLabel(
                "<i>No MicroManager cameras detected in hardware config.</i>"
            )
            self._layout.addWidget(self._placeholder)
            return

        for i, cam in enumerate(mm_cams):
            card = _CameraConfigCard(cam, cam.core, index=i)
            card.cfgChanged.connect(self.cfgChanged.emit)
            self._cards.append(card)
            self._layout.addWidget(card)


# ---------------------------------------------------------------------------
# Main ConfigWizard
# ---------------------------------------------------------------------------

class ConfigWizard(QWidget):
    """Configuration wizard for loading experiment and hardware configs.

    Signals
    -------
    configApplied
        Emitted **after** the experiment JSON (and optionally hardware YAML)
        have been successfully applied to the running :class:`Procedure`.
    hardwareReady
        Emitted after hardware has been initialised (cameras available).
    """

    configApplied = pyqtSignal()
    hardwareReady = pyqtSignal()
    procedureChanged = pyqtSignal(object)  # emitted when a JSON declares a different Procedure subclass

    _SETTINGS_KEY_JSON = "ConfigWizard/last_json"
    _SETTINGS_KEY_YAML = "ConfigWizard/last_yaml"

    def __init__(self, procedure: Procedure, parent: QWidget | None = None):
        super().__init__(parent)
        self.procedure = procedure
        self._settings = QSettings("Mesofield", "Mesofield")
        self._build_ui()
        self._restore_recent_paths()

        # If hardware is already configured, pre-populate the MM section
        if self.procedure.config.hardware.is_configured:
            self._mm_section.set_cameras(self.procedure.config.hardware.cameras)

    # -- public API ----------------------------------------------------------

    def refresh_mm_section(self) -> None:
        """Re-populate the MicroManager config section from current hardware."""
        cameras = self.procedure.config.hardware.cameras
        self._mm_section.set_cameras(cameras)

    # -- UI ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # -- Title -----------------------------------------------------------
        title = QLabel("<h3>⚙ Configuration Wizard</h3>")
        layout.addWidget(title)

        help_text = QLabel(
            "Select your experiment config (.json) to get started.\n"
            "The adjacent hardware.yaml will be discovered automatically.\n"
            "You can also specify a hardware YAML or MicroManager .cfg manually."
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        # -- Experiment JSON -------------------------------------------------
        json_group = QGroupBox("Experiment Config (.json)")
        json_layout = QVBoxLayout(json_group)
        self._json_picker = _FilePickerRow(
            file_filter="JSON Config (*.json)",
            placeholder="devcfg.json",
        )
        json_layout.addWidget(self._json_picker)
        layout.addWidget(json_group)

        # -- Hardware YAML ---------------------------------------------------
        yaml_group = QGroupBox("Hardware Config (.yaml)")
        yaml_layout = QVBoxLayout(yaml_group)
        self._yaml_picker = _FilePickerRow(
            file_filter="YAML Config (*.yaml *.yml)",
            placeholder="hardware.yaml  (auto-detected from JSON directory)",
        )
        yaml_layout.addWidget(self._yaml_picker)

        self._yaml_status = QLabel("")
        yaml_layout.addWidget(self._yaml_status)
        layout.addWidget(yaml_group)

        # -- Apply button ----------------------------------------------------
        self._apply_btn = QPushButton("▶  Apply Configuration")
        self._apply_btn.setStyleSheet(
            "QPushButton { padding: 8px 16px; font-weight: bold; }"
        )
        self._apply_btn.clicked.connect(self._apply)
        layout.addWidget(self._apply_btn)

        # -- MicroManager .cfg -----------------------------------------------
        self._mm_section = _MMConfigSection()
        layout.addWidget(self._mm_section)

        # -- Spacer ----------------------------------------------------------
        layout.addStretch()

        # -- Auto-populate YAML when JSON is picked --------------------------
        self._json_picker.line_edit.textChanged.connect(self._on_json_path_changed)

    # -- Recent paths persistence ---------------------------------------------

    def _restore_recent_paths(self) -> None:
        """Fill pickers from QSettings if the files still exist."""
        last_json = self._settings.value(self._SETTINGS_KEY_JSON, "", type=str)
        last_yaml = self._settings.value(self._SETTINGS_KEY_YAML, "", type=str)
        if last_json and os.path.isfile(last_json):
            self._json_picker.line_edit.setText(last_json)
        if last_yaml and os.path.isfile(last_yaml):
            self._yaml_picker.line_edit.setText(last_yaml)

    def _save_recent_paths(self) -> None:
        """Persist current picker values to QSettings."""
        json_path = self._json_picker.text()
        yaml_path = self._yaml_picker.text()
        if json_path:
            self._settings.setValue(self._SETTINGS_KEY_JSON, json_path)
        if yaml_path:
            self._settings.setValue(self._SETTINGS_KEY_YAML, yaml_path)

    # -- Slots ---------------------------------------------------------------

    def _on_json_path_changed(self, text: str) -> None:
        """When the JSON line-edit changes, try to auto-detect hardware.yaml."""
        if not text:
            return
        candidate = os.path.join(os.path.dirname(text), "hardware.yaml")
        if os.path.isfile(candidate):
            self._yaml_picker.line_edit.setText(candidate)
            self._yaml_status.setText("✔ hardware.yaml auto-detected")
            self._yaml_status.setStyleSheet("color: green;")
        else:
            self._yaml_status.setText("⚠ hardware.yaml not found in JSON directory")
            self._yaml_status.setStyleSheet("color: orange;")

    def _apply(self) -> None:
        """Apply the selected configuration files to the Procedure."""
        json_path = self._json_picker.text() or None
        yaml_path = self._yaml_picker.text() or None

        if not json_path and not yaml_path:
            QMessageBox.information(
                self,
                "Nothing to load",
                "Please select at least an experiment JSON or hardware YAML file.",
            )
            return

        try:
            # If the JSON declares a custom Procedure subclass that differs
            # from the currently active class, instantiate the new one and
            # propagate to the parent (MainWindow).  Otherwise just hot-load
            # the new config in place.
            from mesofield.base import load_procedure_from_config

            if json_path and os.path.isfile(json_path):
                candidate = load_procedure_from_config(json_path)
                if type(candidate) is not type(self.procedure):
                    # The candidate's __init__ already loaded the JSON and
                    # initialized hardware from <json_dir>/hardware.yaml. Only
                    # reload if the user explicitly picked a *different* YAML;
                    # otherwise we'd orphan the live HardwareManager (devices
                    # still hold the cameras) and loop on re-initialization.
                    if yaml_path:
                        existing_yaml = getattr(
                            candidate.config, "_hardware_yaml_path", None
                        )
                        picked_yaml = os.path.abspath(yaml_path)
                        if not existing_yaml or picked_yaml != os.path.abspath(existing_yaml):
                            candidate.load_config(hardware_yaml_path=yaml_path)
                    self.procedure = candidate
                    self.procedureChanged.emit(candidate)
                else:
                    self.procedure.load_config(
                        json_path=json_path,
                        hardware_yaml_path=yaml_path,
                    )
            else:
                self.procedure.load_config(
                    json_path=json_path,
                    hardware_yaml_path=yaml_path,
                )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Configuration Error",
                f"Failed to apply configuration:\n\n{exc}",
            )
            return

        # Persist the selected paths for next launch
        self._save_recent_paths()

        # Refresh the MM config section now that cameras are available
        cameras = self.procedure.config.hardware.cameras
        self._mm_section.set_cameras(cameras)

        self.configApplied.emit()

        if self.procedure.config.hardware.is_configured:
            self.hardwareReady.emit()

        self._apply_btn.setText("✔  Configuration Applied")
        self._apply_btn.setStyleSheet(
            "QPushButton { padding: 8px 16px; font-weight: bold; color: green; }"
        )
