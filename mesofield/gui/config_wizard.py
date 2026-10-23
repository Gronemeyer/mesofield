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
from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import pyqtSignal, Qt, QSettings, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication,
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

    # -- public API ----------------------------------------------------------

    def refresh_mm_section(self) -> None:
        """Deprecated no-op, kept so older callers don't break.

        The launcher no longer hosts a MicroManager .cfg section: a camera is
        described by its rig file, and loading a different .cfg into a live
        core changed the running hardware out from under the rig that
        described it. Camera setup now lives in the rig builder.
        """

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
            self._json_status.setText(
                "• running from a scripted procedure (no experiment.json)"
            )
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

    # Every button on this tab is one of three widths, so the rows line up
    # whatever text they carry: a full-width action, a compact secondary, or
    # an icon-only square.
    _BTN_HEIGHT = 28
    _BTN_COMPACT = 96
    _BTN_SQUARE = 32

    def _icon(self, sp: QStyle.StandardPixmap):
        """Return a themed standard icon for buttons."""
        return self.style().standardIcon(sp)

    def _button(self, text: str, sp, tooltip: str = "", *, width: int = 0):
        """A button sized to this tab's shared metrics."""
        button = QPushButton(f"  {text}" if text else "")
        button.setIcon(self._icon(sp))
        button.setFixedHeight(self._BTN_HEIGHT)
        if width:
            button.setFixedWidth(width)
        if tooltip:
            button.setToolTip(tooltip)
        return button

    def _file_actions(self, get_path) -> QWidget:
        """Open / Reveal for *get_path*, as two icon buttons beside the status.

        These were a full-width pair under each section -- four buttons of
        chrome for two rarely-used actions, always visible even with nothing
        selected to act on. The caller hides the widget until a file exists.
        """
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        for text, pixmap, tip, slot in (
            ("", QStyle.StandardPixmap.SP_FileIcon,
             "Open this file in your default editor", self._open_file),
            ("", QStyle.StandardPixmap.SP_DirOpenIcon,
             "Show this file in your file manager", self._reveal_file),
        ):
            button = self._button(text, pixmap, tip, width=self._BTN_SQUARE)
            button.clicked.connect(lambda _=False, s=slot: s(get_path()))
            row.addWidget(button)
        return holder

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

    def _field_label(self, text: str) -> QLabel:
        """A fixed-width row label, so the rig and folder rows line up."""
        label = QLabel(text)
        label.setFixedWidth(52)
        label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        return label

    @staticmethod
    def _status_row(status: QLabel, actions: QWidget) -> QHBoxLayout:
        """A status line with its file actions tucked to the right."""
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(status, 1)
        row.addWidget(actions)
        return row

    def _build_ui(self) -> None:
        # Pending selections (no raw path fields in the UI — kept here and shown
        # as friendly status lines with the full path in a tooltip).
        self._hardware_path: str = ""
        self._experiment_json: str = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)

        # -- Title -----------------------------------------------------------
        title = QLabel("Configuration")
        title.setStyleSheet(
            f"font-size: 15px; font-weight: 600; color: {theme.TEXT};"
        )
        layout.addWidget(title)
        subtitle = theme.dim_label("Pick a rig to launch. An experiment is optional.")
        layout.addWidget(subtitle)

        # === ① Rig (required) ==============================================
        rig_group = QGroupBox("①   Rig")
        rig_layout = QVBoxLayout(rig_group)
        rig_layout.setSpacing(6)
        rig_row = QHBoxLayout()
        rig_row.setSpacing(8)
        rig_row.addWidget(self._field_label("Rig"))
        self._rig_combo = theme.NoScrollComboBox()
        self._rig_combo.setFixedHeight(self._BTN_HEIGHT)
        self._rig_combo.setToolTip("Bring up a canonical rig from this machine's rig store")
        self._populate_rig_combo()
        self._rig_combo.currentIndexChanged.connect(self._on_rig_selected)
        rig_row.addWidget(self._rig_combo, 1)
        rig_layout.addLayout(rig_row)

        # Most-used action first, then the two authoring ones.
        rig_btn_row = QHBoxLayout()
        rig_btn_row.setSpacing(6)
        new_rig_btn = self._button(
            "New…", QStyle.StandardPixmap.SP_FileDialogNewFolder,
            "Build a new hardware.yaml from a guided device list",
            width=self._BTN_COMPACT)
        new_rig_btn.clicked.connect(self._new_rig)
        rig_btn_row.addWidget(new_rig_btn)
        edit_rig_btn = self._button(
            "Edit…", QStyle.StandardPixmap.SP_FileDialogDetailedView,
            "Edit the selected rig's devices",
            width=self._BTN_COMPACT)
        edit_rig_btn.clicked.connect(self._edit_rig)
        rig_btn_row.addWidget(edit_rig_btn)
        browse_yaml_btn = self._button(
            "Browse…", QStyle.StandardPixmap.SP_DialogOpenButton,
            "Use a hardware.yaml from outside the rig store",
            width=self._BTN_COMPACT)
        browse_yaml_btn.clicked.connect(self._browse_yaml)
        rig_btn_row.addWidget(browse_yaml_btn)
        rig_btn_row.addStretch()
        rig_layout.addLayout(rig_btn_row)

        self._yaml_status = theme.dim_label("No rig selected")
        self._yaml_actions = self._file_actions(lambda: self._hardware_path)
        self._yaml_actions.setVisible(False)
        rig_layout.addLayout(self._status_row(self._yaml_status, self._yaml_actions))
        layout.addWidget(rig_group)

        # === ② Experiment (optional) =======================================
        exp_group = QGroupBox("②   Experiment")
        exp_layout = QVBoxLayout(exp_group)
        exp_layout.setSpacing(6)
        out_row = QHBoxLayout()
        out_row.setSpacing(8)
        out_row.addWidget(self._field_label("Folder"))
        self._outdir_edit = QLineEdit()
        self._outdir_edit.setPlaceholderText("where data is written")
        self._outdir_edit.setText(self.procedure.config.experiment_dir)
        self._outdir_edit.setFixedHeight(self._BTN_HEIGHT)
        out_row.addWidget(self._outdir_edit, 1)
        outdir_browse = self._button(
            "", QStyle.StandardPixmap.SP_DirOpenIcon,
            "Choose the experiment / output directory", width=self._BTN_SQUARE)
        outdir_browse.clicked.connect(self._browse_outdir)
        out_row.addWidget(outdir_browse)
        exp_layout.addLayout(out_row)

        json_btn_row = QHBoxLayout()
        json_btn_row.setSpacing(6)
        self._create_json_btn = self._button(
            "New…", QStyle.StandardPixmap.SP_FileDialogNewFolder,
            "Author a new experiment.json (subjects, tasks, variables)",
            width=self._BTN_COMPACT)
        self._create_json_btn.clicked.connect(self._create_experiment_json)
        json_btn_row.addWidget(self._create_json_btn)
        browse_json_btn = self._button(
            "Load…", QStyle.StandardPixmap.SP_DialogOpenButton,
            "Load an existing experiment.json", width=self._BTN_COMPACT)
        browse_json_btn.clicked.connect(self._browse_json)
        json_btn_row.addWidget(browse_json_btn)
        json_btn_row.addStretch()
        exp_layout.addLayout(json_btn_row)

        self._json_status = theme.dim_label("No experiment selected")
        self._json_actions = self._file_actions(lambda: self._experiment_json)
        self._json_actions.setVisible(False)
        exp_layout.addLayout(self._status_row(self._json_status, self._json_actions))
        layout.addWidget(exp_group)

        self._outdir_edit.textChanged.connect(self._on_outdir_changed)
        self._on_outdir_changed(self._outdir_edit.text())

        # === Apply (primary CTA) ===========================================
        self._apply_btn = QPushButton("  Apply Configuration")
        self._apply_btn.setIcon(self._icon(QStyle.StandardPixmap.SP_MediaPlay))
        self._apply_btn.setFixedHeight(36)
        self._apply_btn.setStyleSheet(
            f"QPushButton {{ padding: 10px 16px; font-weight: bold; "
            f"border: 1px solid {theme.ACCENT}; color: {theme.ACCENT}; }}"
            f"QPushButton:hover {{ background-color: {theme.PANEL_HI}; }}"
        )
        self._apply_btn.clicked.connect(self._apply)
        layout.addWidget(self._apply_btn)

        layout.addStretch()

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
        # Open / Reveal only make sense once there is a file to act on.
        self._yaml_actions.setVisible(bool(path))
        if status:
            self._yaml_status.setText(f"✔ {status}")
            self._yaml_status.setStyleSheet(f"color: {theme.ACCENT};")
            self._yaml_status.setToolTip(path)

    def _set_experiment_json(self, path: str, status: str) -> None:
        """Adopt *path* as the pending experiment.json and update the status line."""
        self._experiment_json = path
        self._json_actions.setVisible(bool(path))
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
            if json_path:
                self.procedure.config.save_json()

        # Persist the selected paths for next launch
        self._save_recent_paths()

        self.configApplied.emit()

        if self.procedure.config.hardware.is_configured:
            self.hardwareReady.emit()

        self._mark_applied()

    def _mark_applied(self) -> None:
        """Show the Apply button in its applied (green) state."""
        self._apply_btn.setText("✔  Configuration Applied")
        self._apply_btn.setStyleSheet(
            "QPushButton { padding: 8px 16px; font-weight: bold; color: green; }"
        )
