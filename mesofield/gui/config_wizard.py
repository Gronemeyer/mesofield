"""
Configuration wizard for hot-loading experiment and hardware configurations.

Provides a unified widget for selecting and applying:
- Experiment JSON config files
- Hardware YAML config files
- MicroManager system .cfg files (via pymmcore-widgets ConfigurationWidget)
- Full pymmcore-widgets Hardware Configuration Wizard (popup)
"""

from __future__ import annotations

import json
import os
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
# Front-facing text and icons
# ---------------------------------------------------------------------------
# Every label, glyph and tooltip the wizard renders lives here, so the wording
# and iconography can be tuned without touching the widget-construction code.
#
# Icon-button specs are ``(standard_pixmap_name, text, tooltip)``.  The first
# item names an attribute of :class:`QStyle.StandardPixmap`, or is ``None`` to
# render the ``text`` label alone.  Text-only buttons size to their label; only
# icon buttons are square.


class UI:
    WINDOW_TITLE = "Mesofield Wizard"

    # -- ① Rig ---------------------------------------------------------------
    RIG_GROUP = "①   Rig  ·  configured hardware connected or software installed to this machine"
    RIG_COMBO_TIP = "Bring up a canonical rig from this machine's rig store"
    RIG_COMBO_EMPTY = "— select rig —"
    RIG_COMBO_DEV = "dev (mock devices)"
    RIG_STATUS_EMPTY = "• no rig selected"

    RIG_BROWSE = (
        "SP_DirOpenIcon", "",
        "Browse for a hardware.yaml — opens at the last-used rig folder",
    )
    RIG_NEW = (
        "SP_FileDialogNewFolder", "",
        "New rig — build a hardware.yaml from a guided device list",
    )
    RIG_EDIT = (
        None, "Edit",
        "Edit the selected rig's devices (e.g. fix a camera backend)",
    )

    # -- ② Experiment --------------------------------------------------------
    EXP_GROUP = "②   Experiment  ·  sequencing and design details of scientific procedure"
    EXP_HINT_TIP = "Experiment / output directory (where data is written)"
    EXP_DIR_PLACEHOLDER = "experiment / output directory"
    EXP_STATUS_NONE = "• no experiment.json here — create one, or run hardware-only"
    EXP_STATUS_SCRIPTED = "• running from a scripted procedure (no experiment.json)"

    EXP_BROWSE = (
        "SP_DirOpenIcon", "",
        "Load an experiment.json — opens at the last-used experiment folder",
    )
    EXP_NEW = (
        "SP_FileDialogNewFolder", "",
        "Create a new experiment.json (subjects, tasks, variables)",
    )
    EXP_NEW_REPLACE_TIP = "Replace the experiment.json in this folder"
    EXP_EDIT = (
        None, "Edit",
        "Edit the selected experiment.json (subjects, tasks, variables)",
    )

    # -- Apply ---------------------------------------------------------------
    APPLY = "  Apply Configuration"
    APPLY_APPLIED = "✔  Configuration Applied"

    ICON_BUTTON_SIZE = 34


# ---------------------------------------------------------------------------
# Open / reveal the actual config files on disk
# ---------------------------------------------------------------------------

def _open_in_default_app(path: str) -> None:
    """Open *path* in the OS default editor/application."""
    QDesktopServices.openUrl(QUrl.fromLocalFile(path))


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
        # Restore the last-used pickers first (helpful on a fresh launch), then
        # let the live procedure's actually-loaded files win: a procedure built
        # from a CLI target, a scripted procedure.py, an experiment directory,
        # or a prior hot-swap never touches this wizard's Apply button, so its
        # real paths must override any stale QSettings value -- otherwise the
        # Setup tab could advertise a different experiment.json than the
        # ExperimentConfig tab is actually editing.
        self._restore_recent_paths()
        self.sync_from_procedure()

        # If hardware is already configured, pre-populate the MM section
        if self.procedure.config.hardware.is_configured:
            self._mm_section.set_cameras(self.procedure.config.hardware.cameras)

    # -- public API ----------------------------------------------------------

    def refresh_mm_section(self) -> None:
        """Re-populate the MicroManager config section from current hardware."""
        cameras = self.procedure.config.hardware.cameras
        self._mm_section.set_cameras(cameras)

    def sync_from_procedure(self) -> None:
        """Reflect the live procedure's actually-loaded config files in the UI.

        ``ExperimentConfig._json_file_path`` (and ``hardware.config_file``) are
        the single source of truth for *what is currently loaded*. The wizard's
        own pickers are only a staging area for the next Apply, so whenever the
        procedure was configured outside this wizard -- a CLI target, a scripted
        ``procedure.py``, an experiment directory, or a hot-swap candidate -- we
        adopt and persist its real paths here. This keeps the Setup tab and the
        ExperimentConfig tab in agreement and makes the last-used experiment.json
        survive a relaunch even when Apply was never clicked.
        """
        cfg = getattr(self.procedure, "config", None)
        if cfg is None:
            return

        json_path = getattr(cfg, "_json_file_path", "") or ""
        if json_path and os.path.isfile(json_path):
            self._set_experiment_json(json_path, "experiment loaded")
            if not self._outdir_edit.text().strip():
                self._outdir_edit.setText(os.path.dirname(json_path))
        elif getattr(cfg.hardware, "is_configured", False):
            # The procedure is live but its parameters came from a scripted
            # define_config (no JSON file on disk). Drop any stale restored path
            # so the Setup tab can't advertise an experiment.json the running
            # config never loaded.
            self._experiment_json = ""
            self._json_status.setText(UI.EXP_STATUS_SCRIPTED)
            self._json_status.setStyleSheet(f"color: {theme.TEXT_DIM};")
            self._json_status.setToolTip("")

        yaml_path = getattr(cfg.hardware, "config_file", "") or ""
        if yaml_path and os.path.isfile(yaml_path):
            self._set_hardware_path(yaml_path, status="rig loaded")
            self._select_rig_in_combo(yaml_path)
        elif getattr(cfg.hardware, "is_configured", False):
            # A rig embedded in experiment.json has no standalone file.
            self._yaml_status.setText("✔ rig embedded in experiment.json")
            self._yaml_status.setStyleSheet(f"color: {theme.ACCENT};")

        # Reflect an already-applied launch config so the Setup tab doesn't
        # look unconfigured.
        if getattr(cfg.hardware, "is_configured", False):
            self._mark_applied()

        # Persist whatever the procedure actually loaded so a relaunch restores
        # the right files even when the user never pressed Apply.
        self._save_recent_paths()

    # -- UI ------------------------------------------------------------------

    def _icon(self, sp: QStyle.StandardPixmap):
        """Return a themed standard icon for buttons."""
        return self.style().standardIcon(sp)

    def _icon_button(self, spec, slot) -> QPushButton:
        """Build a compact square button from a :class:`UI` icon spec."""
        icon, text, tooltip = spec
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        if icon:
            btn.setIcon(self._icon(getattr(QStyle.StandardPixmap, icon)))
            btn.setFixedSize(UI.ICON_BUTTON_SIZE, UI.ICON_BUTTON_SIZE)
        else:
            # Text-only buttons keep the row height but size to their label.
            btn.setFixedHeight(UI.ICON_BUTTON_SIZE)
        btn.clicked.connect(slot)
        return btn

    def _build_ui(self) -> None:
        # Pending selections (no raw path fields in the UI — kept here and shown
        # as friendly status lines with the full path in a tooltip).
        self._hardware_path: str = ""
        self._experiment_json: str = ""

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # === ① Rig (required) ==============================================
        # One row: [browse] [rig picker] [new rig] [edit rig], status beneath.
        rig_group = QGroupBox(UI.RIG_GROUP)
        rig_layout = QVBoxLayout(rig_group)
        rig_layout.setSpacing(6)

        rig_row = QHBoxLayout()
        rig_row.addWidget(self._icon_button(UI.RIG_BROWSE, self._browse_yaml))
        self._rig_combo = QComboBox()
        self._rig_combo.setToolTip(UI.RIG_COMBO_TIP)
        self._populate_rig_combo()
        self._rig_combo.currentIndexChanged.connect(self._on_rig_selected)
        rig_row.addWidget(self._rig_combo, 1)
        rig_row.addWidget(self._icon_button(UI.RIG_NEW, self._new_rig))
        rig_row.addWidget(self._icon_button(UI.RIG_EDIT, self._edit_rig))
        rig_layout.addLayout(rig_row)

        self._yaml_status = QLabel(UI.RIG_STATUS_EMPTY)
        self._yaml_status.setStyleSheet(f"color: {theme.TEXT_DIM};")
        rig_layout.addWidget(self._yaml_status)
        layout.addWidget(rig_group)

        # === ② Experiment (optional) =======================================
        # Same shape: [load .json] [output dir] [create .json] [open .json].
        exp_group = QGroupBox(UI.EXP_GROUP)
        exp_layout = QVBoxLayout(exp_group)
        exp_layout.setSpacing(6)

        out_row = QHBoxLayout()
        out_row.addWidget(self._icon_button(UI.EXP_BROWSE, self._browse_json))
        self._outdir_edit = QLineEdit()
        self._outdir_edit.setPlaceholderText(UI.EXP_DIR_PLACEHOLDER)
        self._outdir_edit.setToolTip(UI.EXP_HINT_TIP)
        self._outdir_edit.setText(self.procedure.config.experiment_dir)
        out_row.addWidget(self._outdir_edit, 1)
        self._create_json_btn = self._icon_button(UI.EXP_NEW, self._create_experiment_json)
        out_row.addWidget(self._create_json_btn)
        out_row.addWidget(self._icon_button(UI.EXP_EDIT, self._edit_experiment_json))
        exp_layout.addLayout(out_row)

        self._json_status = QLabel("")
        self._json_status.setStyleSheet(f"color: {theme.TEXT_DIM};")
        exp_layout.addWidget(self._json_status)
        layout.addWidget(exp_group)

        self._outdir_edit.textChanged.connect(self._on_outdir_changed)
        self._on_outdir_changed(self._outdir_edit.text())

        # === Apply (primary CTA) ===========================================
        self._apply_btn = QPushButton(UI.APPLY)
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

        # === MicroManager .cfg ==============================================
        # Hidden: the .cfg a rig needs is declared in hardware.yaml, so this
        # section only added noise to the Setup tab. The widget is still built
        # (un-parented) so `set_cameras`/`refresh_mm_section` stay live for
        # anything that wants to surface it again.
        self._mm_section = _MMConfigSection()
        self._mm_section.hide()

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
            configured_dir = self._experiment_dir_from_json(last_json)
            self._outdir_edit.setText(configured_dir or os.path.dirname(last_json))

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

    @staticmethod
    def _experiment_dir_from_json(json_path: str) -> str:
        """Return experiment_dir declared in an experiment.json file, if any."""
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception:
            return ""

        if isinstance(doc.get("Configuration"), dict):
            cfg = doc["Configuration"]
            val = cfg.get("experiment_dir") or cfg.get("experiment_directory")
            return str(val).strip() if val else ""

        val = doc.get("experiment_dir") or doc.get("experiment_directory")
        return str(val).strip() if val else ""

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
        self._rig_combo.addItem(UI.RIG_COMBO_EMPTY)
        for name in rigs.list_rigs():
            self._rig_combo.addItem(name)
        self._rig_combo.addItem(UI.RIG_COMBO_DEV)
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

    def _on_outdir_changed(self, text: str) -> None:
        """Auto-detect an experiment.json in the chosen directory."""
        text = text.strip()
        candidate = os.path.join(text, "experiment.json") if text else ""
        if candidate and os.path.isfile(candidate):
            self._set_experiment_json(candidate, "experiment.json found — will load")
            self._create_json_btn.setToolTip(UI.EXP_NEW_REPLACE_TIP)
        else:
            # Drop an auto-detected JSON if we navigated away from its dir;
            # keep one the user explicitly browsed from elsewhere.
            if self._experiment_json and \
                    os.path.dirname(self._experiment_json) == os.path.abspath(text):
                self._experiment_json = ""
            if not self._experiment_json:
                self._json_status.setText(UI.EXP_STATUS_NONE)
                self._json_status.setStyleSheet(f"color: {theme.TEXT_DIM};")
                self._json_status.setToolTip("")
            self._create_json_btn.setToolTip(UI.EXP_NEW[2])

    def _create_experiment_json(self) -> None:
        """Author a fresh experiment.json via the guided experiment builder."""
        from mesofield.gui.config_builder import ExperimentBuilderDialog

        start = self._outdir_edit.text().strip() or self.procedure.config.experiment_dir
        dialog = ExperimentBuilderDialog(default_dir=start, parent=self)
        if dialog.exec() and dialog.json_path:
            # Follow the file we just wrote. The builder stamps the new JSON's
            # own directory into Configuration.experiment_dir, so leaving the
            # pre-seeded (cwd) value here would make Apply overwrite it.
            self._outdir_edit.setText(os.path.dirname(dialog.json_path))
            self._set_experiment_json(dialog.json_path, "experiment.json created — will load")

    def _edit_experiment_json(self) -> None:
        """Re-open the guided builder on the selected experiment.json."""
        from mesofield.gui.config_builder import ExperimentBuilderDialog

        path = self._experiment_json
        if not path or not os.path.isfile(path):
            QMessageBox.information(
                self, "Nothing to edit",
                "Select or create an experiment.json first.",
            )
            return

        dialog = ExperimentBuilderDialog(
            default_dir=os.path.dirname(path), parent=self, json_path=path
        )
        if dialog.exec() and dialog.json_path:
            self._outdir_edit.setText(os.path.dirname(dialog.json_path))
            self._set_experiment_json(dialog.json_path, "experiment.json edited — will load")

    def _browse_json(self) -> None:
        """Select an experiment.json from anywhere on disk."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select experiment.json", self._dialog_start_dir(self._SETTINGS_KEY_JSON),
            "JSON Config (*.json);;All Files (*)"
        )
        if not path:
            return
        configured_dir = self._experiment_dir_from_json(path)
        self._outdir_edit.setText(configured_dir or os.path.dirname(path))
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

        # load_config blocks the GUI thread while hardware comes up, and the
        # host window pumps the event loop to keep its log console painting --
        # so the button has to be latched off, or a second click could land
        # mid-initialisation.
        self._apply_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.procedure.load_config(hardware=yaml_path, experiment=json_path)
            # Fold the live rig into experiment.json so the next launch is
            # self-contained and skips the wizard.
            if json_path and self.procedure.config.hardware.is_configured:
                self.procedure.config.update_hardware(
                    self.procedure.config.hardware.rig_spec()
                )
        except Exception as exc:
            from mesofield.gui.errors import show_error

            show_error(self, "Configuration Error", exc)
            return
        finally:
            QApplication.restoreOverrideCursor()
            self._apply_btn.setEnabled(True)

        # An explicit output directory overrides the JSON/cwd default.
        out_dir = self._outdir_edit.text().strip()
        if out_dir:
            self.procedure.config.experiment_dir = out_dir
            self.procedure.data_dir = self.procedure.config.data_dir
            if json_path:
                self.procedure.config.save_json()

        # Persist the selected paths for next launch
        self._save_recent_paths()

        # Refresh the MM config section now that cameras are available
        cameras = self.procedure.config.hardware.cameras
        self._mm_section.set_cameras(cameras)

        self.configApplied.emit()

        if self.procedure.config.hardware.is_configured:
            self.hardwareReady.emit()

        self._mark_applied()

    def _mark_applied(self) -> None:
        """Show the Apply button in its applied (green) state."""
        self._apply_btn.setText(UI.APPLY_APPLIED)
        self._apply_btn.setStyleSheet(
            "QPushButton { padding: 8px 16px; font-weight: bold; color: green; }"
        )
