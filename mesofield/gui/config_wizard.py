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
import json
import inspect
from typing import TYPE_CHECKING, Optional, List

from PyQt6.QtCore import pyqtSignal, Qt, QSettings, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStyle,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QFrame,
)

from mesofield.gui import theme

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from mesofield.base import Procedure
    from mesofield.devices.cameras import MMCamera


# ---------------------------------------------------------------------------
# Open / reveal the actual config files on disk
# ---------------------------------------------------------------------------

def _open_in_default_app(path: str) -> None:
    """Open *path* in the OS default editor/application."""
    QDesktopServices.openUrl(QUrl.fromLocalFile(path))


def _reveal_in_file_manager(path: str) -> None:
    """Select *path* in the OS file manager (Finder/Explorer), else open its folder."""
    import subprocess
    import sys

    if sys.platform == "darwin":
        subprocess.run(["open", "-R", path], check=False)
    elif os.name == "nt":
        subprocess.run(["explorer", f"/select,{os.path.normpath(path)}"], check=False)
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(path)))


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
                self._status.setStyleSheet(f"color: {theme.ACCENT};")
            else:
                self._status.setText(f"✔ Loaded: <b>{display}</b>")
                self._status.setStyleSheet(f"color: {theme.ACCENT};")
        else:
            self._status.setText("⚠ No system configuration loaded")
            self._status.setStyleSheet(f"color: {theme.WARN};")

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
    hardwareAboutToChange
        Emitted **before** a (re)load tears down the current hardware, so the
        GUI can disconnect live viewers from the outgoing cameras first.
    configApplied
        Emitted **after** the experiment JSON (and optionally hardware YAML)
        have been successfully applied to the running :class:`Procedure`.
    hardwareReady
        Emitted after hardware has been initialised (cameras available).
    """

    hardwareAboutToChange = pyqtSignal()
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

    def _icon(self, sp: QStyle.StandardPixmap):
        """Return a themed standard icon for buttons."""
        return self.style().standardIcon(sp)

    def _file_action_row(self, get_path) -> QHBoxLayout:
        """Open / Reveal buttons acting on the path returned by *get_path*."""
        row = QHBoxLayout()
        open_btn = QPushButton(" Open")
        open_btn.setIcon(self._icon(QStyle.StandardPixmap.SP_FileIcon))
        open_btn.setToolTip("Open this file in your default editor")
        open_btn.clicked.connect(lambda: self._open_file(get_path()))
        reveal_btn = QPushButton(" Reveal")
        reveal_btn.setIcon(self._icon(QStyle.StandardPixmap.SP_DirOpenIcon))
        reveal_btn.setToolTip("Show this file in your file manager")
        reveal_btn.clicked.connect(lambda: self._reveal_file(get_path()))
        row.addWidget(open_btn)
        row.addWidget(reveal_btn)
        row.addStretch()
        return row

    def _open_file(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            QMessageBox.information(self, "No file", "Select a file first.")
            return
        _open_in_default_app(path)

    def _reveal_file(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            QMessageBox.information(self, "No file", "Select a file first.")
            return
        _reveal_in_file_manager(path)

    def _build_ui(self) -> None:
        # Pending selections (no raw path fields in the UI — kept here and shown
        # as friendly status lines with the full path in a tooltip).
        self._hardware_path: str = ""
        self._experiment_json: str = ""

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # -- Title -----------------------------------------------------------
        title = QLabel("<h3>⚙ &nbsp;Configuration Wizard</h3>")
        layout.addWidget(title)
        subtitle = QLabel("Pick a rig to launch. An experiment is optional.")
        subtitle.setStyleSheet(f"color: {theme.TEXT_DIM};")
        layout.addWidget(subtitle)

        # === ① Rig (required) ==============================================
        rig_group = QGroupBox("①   Rig  ·  required")
        rig_layout = QVBoxLayout(rig_group)
        rig_layout.setSpacing(6)
        rig_row = QHBoxLayout()
        rig_lbl = QLabel("🔧")
        rig_lbl.setToolTip("Hardware rig (hardware.yaml)")
        rig_row.addWidget(rig_lbl)
        self._rig_combo = QComboBox()
        self._rig_combo.setToolTip("Bring up a canonical rig from this machine's rig store")
        self._populate_rig_combo()
        self._rig_combo.currentIndexChanged.connect(self._on_rig_selected)
        rig_row.addWidget(self._rig_combo, 1)
        rig_layout.addLayout(rig_row)

        rig_btn_row = QHBoxLayout()
        browse_yaml_btn = QPushButton(" Browse hardware.yaml…")
        browse_yaml_btn.setIcon(self._icon(QStyle.StandardPixmap.SP_DialogOpenButton))
        browse_yaml_btn.clicked.connect(self._browse_yaml)
        rig_btn_row.addWidget(browse_yaml_btn)
        new_rig_btn = QPushButton(" New rig…")
        new_rig_btn.setIcon(self._icon(QStyle.StandardPixmap.SP_FileDialogNewFolder))
        new_rig_btn.setToolTip("Build a new hardware.yaml from a guided device list")
        new_rig_btn.clicked.connect(self._new_rig)
        rig_btn_row.addWidget(new_rig_btn)
        edit_rig_btn = QPushButton(" Edit rig…")
        edit_rig_btn.setIcon(self._icon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        edit_rig_btn.setToolTip("Edit the selected rig's devices (e.g. fix a camera backend)")
        edit_rig_btn.clicked.connect(self._edit_rig)
        rig_btn_row.addWidget(edit_rig_btn)
        rig_layout.addLayout(rig_btn_row)

        self._yaml_status = QLabel("• no rig selected")
        self._yaml_status.setStyleSheet(f"color: {theme.TEXT_DIM};")
        rig_layout.addWidget(self._yaml_status)
        rig_layout.addLayout(self._file_action_row(lambda: self._hardware_path))
        layout.addWidget(rig_group)

        # === ② Experiment (optional) =======================================
        exp_group = QGroupBox("②   Experiment  ·  optional")
        exp_layout = QVBoxLayout(exp_group)
        exp_layout.setSpacing(6)
        out_row = QHBoxLayout()
        dir_lbl = QLabel("📁")
        dir_lbl.setToolTip("Experiment / output directory (where data is written)")
        out_row.addWidget(dir_lbl)
        self._outdir_edit = QLineEdit()
        self._outdir_edit.setPlaceholderText("experiment / output directory")
        self._outdir_edit.setText(self.procedure.config.experiment_dir)
        out_row.addWidget(self._outdir_edit, 1)
        outdir_browse = QPushButton()
        outdir_browse.setIcon(self._icon(QStyle.StandardPixmap.SP_DirOpenIcon))
        outdir_browse.setToolTip("Choose the experiment / output directory")
        outdir_browse.setFixedWidth(40)
        outdir_browse.clicked.connect(self._browse_outdir)
        out_row.addWidget(outdir_browse)
        exp_layout.addLayout(out_row)

        self._json_status = QLabel("")
        self._json_status.setStyleSheet(f"color: {theme.TEXT_DIM};")
        exp_layout.addWidget(self._json_status)
        exp_layout.addLayout(self._file_action_row(lambda: self._experiment_json))

        json_btn_row = QHBoxLayout()
        self._create_json_btn = QPushButton(" Create experiment.json…")
        self._create_json_btn.setIcon(self._icon(QStyle.StandardPixmap.SP_FileDialogNewFolder))
        self._create_json_btn.setToolTip("Author a new experiment.json (subjects, tasks, variables)")
        self._create_json_btn.clicked.connect(self._create_experiment_json)
        json_btn_row.addWidget(self._create_json_btn)
        browse_json_btn = QPushButton(" Load .json…")
        browse_json_btn.setIcon(self._icon(QStyle.StandardPixmap.SP_DialogOpenButton))
        browse_json_btn.clicked.connect(self._browse_json)
        json_btn_row.addWidget(browse_json_btn)
        exp_layout.addLayout(json_btn_row)
        layout.addWidget(exp_group)

        self._outdir_edit.textChanged.connect(self._on_outdir_changed)
        self._on_outdir_changed(self._outdir_edit.text())

        # === Apply (primary CTA) ===========================================
        self._apply_btn = QPushButton("  Apply Configuration")
        self._apply_btn.setIcon(self._icon(QStyle.StandardPixmap.SP_MediaPlay))
        self._apply_btn.setStyleSheet(
            f"QPushButton {{ padding: 10px 16px; font-weight: bold; "
            f"border: 1px solid {theme.ACCENT}; color: {theme.ACCENT}; }}"
            f"QPushButton:hover {{ background-color: {theme.PANEL_HI}; }}"
        )
        self._apply_btn.clicked.connect(self._apply)
        layout.addWidget(self._apply_btn)

        # -- Spacer ----------------------------------------------------------
        layout.addStretch()

        # === MicroManager .cfg (secondary; populated once MM cameras exist) =
        self._mm_section = _MMConfigSection()
        layout.addWidget(self._mm_section)

    # -- Recent paths persistence ---------------------------------------------

    def _restore_recent_paths(self) -> None:
        """Fill pickers from QSettings if the files still exist."""
        last_yaml = self._settings.value(self._SETTINGS_KEY_YAML, "", type=str)
        if last_yaml and os.path.isfile(last_yaml):
            self._set_hardware_path(last_yaml, status="rig restored")
            self._select_rig_in_combo(last_yaml)
        # Restore the experiment.json too 
        last_json = self._settings.value(self._SETTINGS_KEY_JSON, "", type=str)
        if last_json and os.path.isfile(last_json):
            self._set_experiment_json(last_json, "experiment restored")
            if not self._outdir_edit.text().strip():
                self._outdir_edit.setText(os.path.dirname(last_json))

    def _save_recent_paths(self) -> None:
        """Persist current picker values to QSettings."""
        if self._experiment_json:
            self._settings.setValue(self._SETTINGS_KEY_JSON, self._experiment_json)
        if self._hardware_path:
            self._settings.setValue(self._SETTINGS_KEY_YAML, self._hardware_path)

    def _dialog_start_dir(self, settings_key: str) -> str:
        """Folder a Browse dialog should open in: the last picked file's dir."""
        last = self._settings.value(settings_key, "", type=str)
        return os.path.dirname(last) if last else ""

    # -- Helpers -------------------------------------------------------------

    def _set_hardware_path(self, path: str, status: str = "") -> None:
        """Adopt *path* as the pending hardware.yaml and update the status line."""
        self._hardware_path = path
        if status:
            self._yaml_status.setText(f"✔ {status}")
            self._yaml_status.setStyleSheet(f"color: {theme.ACCENT};")
            self._yaml_status.setToolTip(path)

    def _set_experiment_json(self, path: str, status: str) -> None:
        """Adopt *path* as the pending experiment.json and update the status line."""
        self._experiment_json = path
        self._json_status.setText(f"✔ {status}")
        self._json_status.setStyleSheet(f"color: {theme.ACCENT};")
        self._json_status.setToolTip(path)

    def _select_rig_in_combo(self, yaml_path: str) -> None:
        """Highlight the rig-store entry matching *yaml_path*, if any."""
        from mesofield.scaffold import rigs

        for name in rigs.list_rigs():
            if os.path.abspath(str(rigs.rig_path(name))) == os.path.abspath(yaml_path):
                idx = self._rig_combo.findText(name)
                if idx >= 0:
                    self._rig_combo.blockSignals(True)
                    self._rig_combo.setCurrentIndex(idx)
                    self._rig_combo.blockSignals(False)
                return

    # -- Slots ---------------------------------------------------------------

    def _populate_rig_combo(self) -> None:
        """Fill the rig dropdown from the machine's rig store."""
        from mesofield.scaffold import rigs

        self._rig_combo.blockSignals(True)
        self._rig_combo.clear()
        self._rig_combo.addItem("— select rig —")
        for name in rigs.list_rigs():
            self._rig_combo.addItem(name)
        self._rig_combo.addItem("dev (mock devices)")
        self._rig_combo.blockSignals(False)

    def _on_rig_selected(self, index: int) -> None:
        """Resolve the chosen rig to a hardware.yaml path (no copy needed)."""
        if index <= 0:
            return
        label = self._rig_combo.currentText()
        if label.startswith("dev"):
            import tempfile
            from mesofield.scaffold.experiment import _hardware_yaml_mock

            fd, tmp = tempfile.mkstemp(prefix="mesofield_dev_", suffix=".yaml")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(_hardware_yaml_mock())
            self._set_hardware_path(tmp, status="dev (mock devices) selected")
            return
        from mesofield.scaffold import rigs

        try:
            path = str(rigs._resolve_existing(label))
        except FileNotFoundError as exc:
            self._yaml_status.setText(f"⚠ {exc}")
            self._yaml_status.setStyleSheet(f"color: {theme.WARN};")
            return
        self._set_hardware_path(path, status=f"rig '{label}' selected")

    def _browse_yaml(self) -> None:
        """Pick an explicit hardware.yaml outside the rig store."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select hardware.yaml", self._dialog_start_dir(self._SETTINGS_KEY_YAML),
            "YAML Config (*.yaml *.yml);;All Files (*)"
        )
        if not path:
            return
        self._rig_combo.blockSignals(True)
        self._rig_combo.setCurrentIndex(0)
        self._rig_combo.blockSignals(False)
        self._set_hardware_path(path, status="hardware.yaml selected")

    def _new_rig(self) -> None:
        """Build a new rig via the guided hardware builder and select it."""
        from mesofield.gui.config_builder import HardwareBuilderDialog

        dialog = HardwareBuilderDialog(self)
        if dialog.exec() and dialog.rig_name:
            self._populate_rig_combo()
            idx = self._rig_combo.findText(dialog.rig_name)
            if idx >= 0:
                self._rig_combo.setCurrentIndex(idx)  # fires _on_rig_selected

    def _edit_rig(self) -> None:
        """Open the selected hardware.yaml in the builder to tweak it in place.

        The common case is a wrong camera backend: fix it here instead of
        hunting through the YAML by hand.
        """
        import yaml
        from mesofield.gui.config_builder import HardwareBuilderDialog
        from mesofield.scaffold import rigs

        if not self._hardware_path or not os.path.isfile(self._hardware_path):
            QMessageBox.information(
                self, "No rig to edit",
                "Select a rig (or browse a hardware.yaml) first.",
            )
            return
        try:
            with open(self._hardware_path, "r", encoding="utf-8") as fh:
                doc = yaml.safe_load(fh) or {}
        except Exception as exc:
            QMessageBox.warning(self, "Could not read rig", str(exc))
            return

        # Prefill the save name if the current selection is a stored rig.
        name = None
        for rname in rigs.list_rigs():
            if os.path.abspath(str(rigs.rig_path(rname))) == os.path.abspath(self._hardware_path):
                name = rname
                break

        dialog = HardwareBuilderDialog(self, doc=doc, rig_name=name)
        if dialog.exec() and dialog.rig_name:
            self._populate_rig_combo()
            idx = self._rig_combo.findText(dialog.rig_name)
            if idx >= 0:
                self._rig_combo.blockSignals(True)
                self._rig_combo.setCurrentIndex(idx)
                self._rig_combo.blockSignals(False)
                self._on_rig_selected(idx)  # re-resolve path + refresh status

    def _browse_outdir(self) -> None:
        """Pick the directory data will be written into."""
        path = QFileDialog.getExistingDirectory(self, "Select experiment directory")
        if path:
            self._outdir_edit.setText(path)

    def _on_outdir_changed(self, text: str) -> None:
        """Auto-detect an experiment.json in the chosen directory."""
        text = text.strip()
        candidate = os.path.join(text, "experiment.json") if text else ""
        if candidate and os.path.isfile(candidate):
            self._set_experiment_json(candidate, "experiment.json found — will load")
            self._create_json_btn.setText(" Replace experiment.json…")
        else:
            # Drop an auto-detected JSON if we navigated away from its dir;
            # keep one the user explicitly browsed from elsewhere.
            if self._experiment_json and \
                    os.path.dirname(self._experiment_json) == os.path.abspath(text):
                self._experiment_json = ""
            if not self._experiment_json:
                self._json_status.setText(
                    "• no experiment.json here — create one, or run hardware-only"
                )
                self._json_status.setStyleSheet(f"color: {theme.TEXT_DIM};")
                self._json_status.setToolTip("")
            self._create_json_btn.setText(" Create experiment.json…")

    def _create_experiment_json(self) -> None:
        """Author a fresh experiment.json via the guided experiment builder."""
        from mesofield.gui.config_builder import ExperimentBuilderDialog

        start = self._outdir_edit.text().strip() or self.procedure.config.experiment_dir
        dialog = ExperimentBuilderDialog(default_dir=start, parent=self)
        if dialog.exec() and dialog.json_path:
            if not self._outdir_edit.text().strip():
                self._outdir_edit.setText(os.path.dirname(dialog.json_path))
            self._set_experiment_json(dialog.json_path, "experiment.json created — will load")

    def _browse_json(self) -> None:
        """Select an experiment.json from anywhere on disk."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select experiment.json", self._dialog_start_dir(self._SETTINGS_KEY_JSON),
            "JSON Config (*.json);;All Files (*)"
        )
        if not path:
            return
        if not self._outdir_edit.text().strip():
            self._outdir_edit.setText(os.path.dirname(path))
        self._set_experiment_json(path, "experiment.json selected — will load")

    def _apply(self) -> None:
        """Apply the selected configuration files to the Procedure."""
        json_path = self._experiment_json or None
        yaml_path = self._hardware_path or None

        if not json_path and not yaml_path:
            QMessageBox.information(
                self,
                "Nothing to load",
                "Please select at least an experiment JSON or hardware YAML file.",
            )
            return

        # Refuse to reload while a recording is in progress: load_config tears
        # down the live hardware, which would abandon the open writers and
        # truncate the output files. The user must stop the run first.
        if getattr(self.procedure, "is_running", False):
            QMessageBox.warning(
                self,
                "Recording in progress",
                "Stop the current recording before reloading the configuration.",
            )
            return

        # Sever live viewers from the outgoing cameras BEFORE the teardown below
        # deinitializes them, so no in-flight frame lands on a doomed widget.
        self.hardwareAboutToChange.emit()

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            # If the JSON declares a custom Procedure subclass that differs
            # from the currently active class, instantiate the new one and
            # propagate to the parent (MainWindow).  Otherwise just hot-load
            # the new config in place.
            from mesofield.base import load_procedure_from_config

            if json_path and os.path.isfile(json_path):
                with open(json_path, "r", encoding="utf-8") as fh:
                    cfg = json.load(fh)

                proc_file = cfg.get("procedure_file") if isinstance(cfg, dict) else None
                proc_class = cfg.get("procedure_class") if isinstance(cfg, dict) else None

                should_switch_procedure = False
                if proc_file and proc_class:
                    json_dir = os.path.dirname(os.path.abspath(json_path))
                    declared_file = proc_file
                    if not os.path.isabs(declared_file):
                        declared_file = os.path.join(json_dir, declared_file)
                    declared_file = os.path.abspath(declared_file)

                    current_cls = type(self.procedure)
                    current_file = inspect.getsourcefile(current_cls) or inspect.getfile(current_cls) or ""
                    current_file = os.path.abspath(current_file) if current_file else ""

                    same_class_name = current_cls.__name__ == proc_class
                    same_file = (
                        bool(current_file)
                        and os.path.normcase(os.path.normpath(current_file))
                        == os.path.normcase(os.path.normpath(declared_file))
                    )
                    should_switch_procedure = not (same_class_name and same_file)

                if should_switch_procedure:
                    candidate = load_procedure_from_config(json_path)
                    # The candidate's __init__ already loaded the JSON and the
                    # sibling hardware.yaml. Only reload if the user explicitly
                    # picked a *different* YAML; otherwise we'd orphan the live
                    # HardwareManager (devices still hold the cameras) and loop
                    # on re-initialization.
                    if yaml_path:
                        existing_yaml = candidate.config.hardware.config_file
                        picked_yaml = os.path.abspath(yaml_path)
                        if not existing_yaml or picked_yaml != os.path.abspath(existing_yaml):
                            candidate.load_config(hardware=yaml_path)
                    self.procedure = candidate
                    self.procedureChanged.emit(candidate)
                else:
                    self.procedure.load_config(
                        hardware=yaml_path,
                        experiment=json_path,
                    )
            else:
                self.procedure.load_config(
                    hardware=yaml_path,
                    experiment=json_path,
                )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Configuration Error",
                f"Failed to apply configuration:\n\n{exc}",
            )
            return
        finally:
            QApplication.restoreOverrideCursor()

        # An explicit output directory overrides the JSON/cwd default.
        out_dir = self._outdir_edit.text().strip()
        if out_dir:
            self.procedure.config.experiment_dir = out_dir
            self.procedure.data_dir = self.procedure.config.data_dir

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
