"""Editable PsychoPy task->script configuration tab.

A dedicated tab (sibling of ExperimentConfig/MousePortal/Terminal/Setup) that
lets the user declare which PsychoPy scripts implement which tasks, mirroring how
:class:`~mesofield.gui.mouseportal_controller.MousePortalController` manages the
MousePortal block:

- scripts are discovered from the experiment directory (or browsed from
  elsewhere) and their task is parsed from the ``task-{name}`` filename
  convention (see :func:`mesofield.config.parse_task_from_filename`),
- edits are collected into a ``{task: filename}`` map,
- and committed through :meth:`ExperimentConfig.update_psychopy`, which updates
  the in-memory registry (so the *next* run's ``arm`` resolves the right script),
  re-derives the ``task`` dropdown choices, and writes the top-level ``PsychoPy``
  block back to experiment.json.

Selecting a task in the ConfigController then runs the matching script -- no
custom Procedure subclass needed. Editing is locked while a Procedure is running.
"""

from __future__ import annotations

import glob
import os
from datetime import datetime
from typing import Any, Dict

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QMessageBox, QHeaderView, QFileDialog,
    QSpinBox,
)

from mesofield.config import parse_task_from_filename

# Files in an experiment directory that are never stimulus scripts.
_SKIP_FILENAMES = {"procedure.py", "__init__.py"}


class PsychoPyController(QWidget):
    """Editable view of the task -> PsychoPy-script map."""

    # Emitted after Save changes the task set, so the ConfigController can
    # rebuild its task dropdown from the new choices.
    tasksChanged = pyqtSignal()

    def __init__(self, procedure, parent=None) -> None:
        super().__init__(parent)
        self.procedure = procedure
        self.config = procedure.config
        self._device = procedure.config.hardware.devices.get("psychopy")

        layout = QVBoxLayout(self)
        self.setMaximumWidth(500)

        intro = QLabel(
            "Map each task to its PsychoPy script. A script declares its task by "
            "embedding <b>task-&lt;name&gt;</b> in its filename; Discover parses "
            "that automatically. The selected task in ExperimentConfig runs the "
            "matching script."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # --- Scripts table -------------------------------------------------
        table_box = QGroupBox("PsychoPy scripts")
        table_layout = QVBoxLayout(table_box)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Task", "Script file", "Trial dur (s)"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setToolTip(
            "Task is the value selected in ExperimentConfig; Script file is "
            "resolved against the experiment directory (full path in the tooltip). "
            "Trial dur (s) is optional: when set, num_trials = duration // trial_dur "
            "and both are passed to the script (0 = unset)."
        )
        table_layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.discover_btn = QPushButton("Discover in experiment dir")
        self.discover_btn.setToolTip(
            "Scan the experiment directory for .py files and add a row per script, "
            "prefilling the task from its task-<name> filename."
        )
        self.add_btn = QPushButton("Add file…")
        self.add_btn.setToolTip("Add a PsychoPy script from elsewhere on disk")
        self.del_btn = QPushButton("− Selected")
        self.discover_btn.clicked.connect(self._discover)
        self.add_btn.clicked.connect(self._add_file)
        self.del_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(self.discover_btn)
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.del_btn)
        btn_row.addStretch(1)
        table_layout.addLayout(btn_row)
        layout.addWidget(table_box)

        # --- Actions -------------------------------------------------------
        action_row = QHBoxLayout()
        self.save_btn = QPushButton("Save")
        self.save_btn.setToolTip("Validate and persist to the PsychoPy block in experiment.json")
        self.reload_btn = QPushButton("Reload")
        self.reload_btn.setToolTip("Discard edits and reload from experiment.json")
        self.save_btn.clicked.connect(self._save)
        self.reload_btn.clicked.connect(self._reload)
        action_row.addWidget(self.save_btn)
        action_row.addWidget(self.reload_btn)
        action_row.addStretch(1)
        layout.addWidget(QLabel("<i>Edits apply to the next run after Save.</i>"))
        layout.addLayout(action_row)
        layout.addStretch(1)

        # The editable area locks while a Procedure is running.
        self._editors = [table_box, self.save_btn]
        # The Procedure outlives this controller -- a fresh one is built on every
        # config/hardware reload while the Procedure (and its `events`) persists.
        # Keep handler refs so cleanup() can sever them; otherwise a stale
        # controller keeps reacting to procedure_started/finished/error and calls
        # _set_editable() on already-deleted widgets (RuntimeError mid-run).
        self._events = getattr(self.procedure, "events", None)
        self._lock_handler = lambda *_: self._set_editable(False)
        self._unlock_handler = lambda *_: self._set_editable(True)
        if self._events is not None:
            self._events.procedure_started.connect(self._lock_handler)
            self._events.procedure_finished.connect(self._unlock_handler)
            self._events.procedure_error.connect(self._unlock_handler)

        self._reload()

    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        """Disconnect from the shared Procedure's events before destruction.

        Must be called before ``deleteLater()`` when this controller is rebuilt
        (config/hardware hot-swap). Idempotent.
        """
        events = self._events
        self._events = None
        if events is None:
            return
        for sig, handler in (
            (events.procedure_started, self._lock_handler),
            (events.procedure_finished, self._unlock_handler),
            (events.procedure_error, self._unlock_handler),
        ):
            try:
                sig.disconnect(handler)
            except (TypeError, RuntimeError):
                # Already disconnected, or never connected.
                pass

    def closeEvent(self, event):  # noqa: N802 - Qt naming
        """Safety net: disconnect if closed without an explicit cleanup()."""
        self.cleanup()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    def _set_editable(self, on: bool) -> None:
        for w in self._editors:
            try:
                w.setEnabled(on)
            except RuntimeError:
                # Underlying C++ widget already deleted (a stale handler firing
                # during teardown before cleanup()). Ignore -- nothing to lock.
                pass

    # ---- table helpers ------------------------------------------------
    def _store_form(self, path: str) -> str:
        """Filename as stored in the map: relative to experiment_dir if inside it, else absolute.

        Matches how :attr:`ExperimentConfig.psychopy_path` resolves an entry.
        """
        abspath = os.path.abspath(path)
        expdir = os.path.abspath(self.config.experiment_dir)
        try:
            rel = os.path.relpath(abspath, expdir)
        except ValueError:
            return abspath  # different drive (Windows)
        if rel.startswith(os.pardir):
            return abspath  # outside the experiment directory
        return rel

    def _resolve_abs(self, stored: str) -> str:
        if os.path.isabs(stored):
            return stored
        return os.path.join(os.path.abspath(self.config.experiment_dir), stored)

    def _add_row(self, task: str, stored_filename: str, trial_duration=None) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(task or "")))
        file_item = QTableWidgetItem(str(stored_filename))
        file_item.setFlags(file_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        file_item.setToolTip(self._resolve_abs(stored_filename))
        self.table.setItem(row, 1, file_item)
        spin = QSpinBox()
        spin.setRange(0, 1_000_000)
        spin.setToolTip("Seconds per trial; 0 = unset (uses the default num_trials)")
        try:
            spin.setValue(int(trial_duration) if trial_duration else 0)
        except (TypeError, ValueError):
            spin.setValue(0)
        self.table.setCellWidget(row, 2, spin)

    def _existing_stored(self) -> set:
        return {
            self.table.item(r, 1).text()
            for r in range(self.table.rowCount())
            if self.table.item(r, 1)
        }

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    # ---- slots --------------------------------------------------------
    def _discover(self) -> None:
        expdir = self.config.experiment_dir
        if not expdir or not os.path.isdir(expdir):
            QMessageBox.information(
                self, "No experiment directory",
                "Set an experiment directory (load or create an experiment.json) first.",
            )
            return
        existing = self._existing_stored()
        found = 0
        for path in sorted(glob.glob(os.path.join(expdir, "*.py"))):
            if os.path.basename(path) in _SKIP_FILENAMES:
                continue
            stored = self._store_form(path)
            if stored in existing:
                continue
            self._add_row(parse_task_from_filename(path) or "", stored)
            found += 1
        if found == 0:
            QMessageBox.information(
                self, "Discover",
                "No new .py scripts found in the experiment directory.",
            )

    def _add_file(self) -> None:
        start = self.config.experiment_dir if os.path.isdir(self.config.experiment_dir or "") else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select PsychoPy script", start, "Python (*.py);;All Files (*)"
        )
        if not path:
            return
        stored = self._store_form(path)
        if stored in self._existing_stored():
            return
        self._add_row(parse_task_from_filename(path) or "", stored)

    def _collect_map(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for r in range(self.table.rowCount()):
            task = self.table.item(r, 0).text().strip() if self.table.item(r, 0) else ""
            filename = self.table.item(r, 1).text() if self.table.item(r, 1) else ""
            spin = self.table.cellWidget(r, 2)
            trial = int(spin.value()) if spin is not None else 0
            # A trial_duration promotes the entry to a dict; 0 keeps the plain
            # filename form (ExperimentConfig normalizes either way).
            out[task] = {"file": filename, "trial_duration": trial} if trial > 0 else filename
        return out

    def _validate(self) -> list:
        errors = []
        tasks = []
        for r in range(self.table.rowCount()):
            task = self.table.item(r, 0).text().strip() if self.table.item(r, 0) else ""
            filename = self.table.item(r, 1).text() if self.table.item(r, 1) else ""
            if not task:
                errors.append(f"Row {r + 1} ({filename or '?'}): task is empty.")
            tasks.append(task)
        dupes = sorted({t for t in tasks if t and tasks.count(t) > 1})
        if dupes:
            errors.append(f"Duplicate task(s): {', '.join(dupes)}. Each task maps to one script.")
        return errors

    # ---- load / save --------------------------------------------------
    def _reload(self) -> None:
        self.table.setRowCount(0)
        block = self.config.psychopy
        if block:
            for task, value in block.items():
                filename, trial = self.config._psychopy_entry(value)
                self._add_row(str(task), filename, trial)
        else:
            # Seed a single row from the legacy single-script config, if any.
            legacy = self.config.get("psychopy_filename") or ""
            if legacy and legacy != "experiment.py":
                self._add_row(parse_task_from_filename(legacy) or self.config.task or "", legacy)

    def _save(self) -> None:
        errors = self._validate()
        if errors:
            QMessageBox.warning(
                self, "PsychoPy config invalid",
                "Fix the following before saving:\n\n• " + "\n• ".join(errors),
            )
            return
        try:
            self.config.update_psychopy(self._collect_map())
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.save_btn.setToolTip(f"Saved {datetime.now().strftime('%H:%M:%S')}")
        self.tasksChanged.emit()
