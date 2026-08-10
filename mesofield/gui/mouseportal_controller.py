"""Editable MousePortal configuration tab.

A dedicated tab (sibling of ExperimentConfig/Terminal/Setup) that lets the user
adjust the corridor + gain-trial parameters between runs and persists them
safely, mirroring how :class:`~mesofield.gui.controller.ConfigController`
manages ExperimentConfig:

- edits are collected from the widgets into a MousePortal config block,
- validated via :mod:`mesofield.gui.mouseportal_config` (errors shown in a
  dialog; nothing is persisted on failure),
- and committed through :meth:`ExperimentConfig.update_mouseportal`, which
  updates the in-memory registry (so the *next* run's ``arm`` reads them) and
  writes the top-level ``MousePortal`` block back to experiment.json.

Editing is locked while a Procedure is running.
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Any, Dict, List

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QSpinBox, QDoubleSpinBox, QComboBox, QPushButton, QTableWidget,
    QTableWidgetItem, QPlainTextEdit, QMessageBox, QHeaderView, QLineEdit,
    QCheckBox, QScrollArea, QFileDialog, QStyle,
)

from mesofield.gui.mouseportal_panel import MousePortalPanel
from mesofield.signals import Bindings
from mesofield.gui.mouseportal_config import (
    KNOWN_TRANSFORMS, TRANSFORM_PARAM, ZERO_PARAM, TRIAL_END_CONDITIONS,
    validate_block, parse_block_sequences, format_block_sequences,
    total_duration,
)

# MousePortal's own default window, mirrored from
# ``mesofield.devices.mouseportal_device._DEFAULT_WINDOW`` so this tab can seed
# its geometry fields before any window block has been authored.
_DEFAULT_WINDOW = {"width": 1920, "height": 1080, "origin_x": 0, "origin_y": 0}

# Qt's "no maximum" sentinel, and the height every collapsed section clamps to.
_QWIDGETSIZE_MAX = 16777215
_COLLAPSED_H = 20


def _collapsible(title: str, content: QWidget) -> QGroupBox:
    """A checkable group box whose check state shows/hides *content*.

    Qt's own checkable group box only *disables* its children; hiding is what
    actually buys back vertical space, which is the point on this tab. The
    themed QGroupBox carries ~18px of padding, so hiding alone still leaves a
    stub box -- clamp the height too, to the same value for every section so
    collapsed headers line up.
    """
    box = QGroupBox(title)
    box.setCheckable(True)
    box.setChecked(True)
    inner = QVBoxLayout(box)
    inner.setContentsMargins(0, 0, 0, 0)
    inner.addWidget(content)

    def _toggle(checked: bool) -> None:
        content.setVisible(checked)
        box.setMaximumHeight(_QWIDGETSIZE_MAX if checked else _COLLAPSED_H)

    box.toggled.connect(_toggle)
    return box


class MousePortalController(QWidget):
    """Editable view of the MousePortal config block."""

    # Emitted after Save changes the bound task, so the ConfigController can
    # rebuild its task dropdown from the new choices.
    tasksChanged = pyqtSignal()

    def __init__(self, procedure, parent=None) -> None:
        super().__init__(parent)
        self.procedure = procedure
        self.config = procedure.config
        self._device = procedure.config.hardware.devices.get("mouseportal")

        self.setMaximumWidth(500)
        # Everything lives in a scroll area: with all sections expanded the tab
        # is taller than a laptop screen, and collapsing is a preference, not a
        # requirement for reaching the Save button.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        # The tab is width-constrained; only ever scroll vertically.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        layout = QVBoxLayout(body)
        scroll.setWidget(body)
        outer.addWidget(scroll)

        # Status header (reuses the same indicator as the DynamicController panel).
        if self._device is not None:
            self._panel = MousePortalPanel(self.config, self._device, parent=self)
            layout.addWidget(self._panel)
        else:
            self._panel = None

        # --- Task binding --------------------------------------------------
        # Each MousePortal configuration corresponds to one task ID. Selecting
        # this task in ExperimentConfig is what launches MousePortal for the run
        # (leave blank to launch for every task, as a single-stimulus rig does).
        task_box = QGroupBox("Task")
        task_form = QFormLayout(task_box)
        self.task_edit = QLineEdit()
        self.task_edit.setPlaceholderText("e.g. corridor (blank = serves every task)")
        self.task_edit.setToolTip(
            "ExperimentConfig task that runs this MousePortal configuration. "
            "On a rig with multiple stimulus apps, only the device bound to the "
            "selected task launches."
        )
        task_form.addRow("task", self.task_edit)
        layout.addWidget(task_box)

        # --- Experiment scalars -------------------------------------------
        exp_content = QWidget()
        form = QFormLayout(exp_content)
        form.setContentsMargins(0, 0, 0, 0)
        # Fields must shrink with the tab rather than force it wider.
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.num_blocks = QSpinBox(); self.num_blocks.setRange(1, 999)
        self.trials_per_block = QSpinBox(); self.trials_per_block.setRange(1, 999)
        self.iti_duration = QDoubleSpinBox(); self.iti_duration.setRange(0, 3600); self.iti_duration.setSuffix(" s")
        self.trial_end = QComboBox(); self.trial_end.addItems(list(TRIAL_END_CONDITIONS))
        self.trial_duration = QDoubleSpinBox(); self.trial_duration.setRange(0, 86400); self.trial_duration.setSuffix(" s")
        self.trial_distance = QDoubleSpinBox(); self.trial_distance.setRange(0, 1_000_000)
        form.addRow("num_blocks", self.num_blocks)
        form.addRow("trials_per_block", self.trials_per_block)
        form.addRow("iti_duration", self.iti_duration)
        form.addRow("trial_end_condition", self.trial_end)
        form.addRow("trial_duration", self.trial_duration)
        form.addRow("trial_distance", self.trial_distance)

        # Window geometry. MousePortal's Panda3D window is opened from these;
        # they are also the only way to place it on a second monitor.
        self.win_width = QSpinBox(); self.win_width.setRange(1, 32767)
        self.win_height = QSpinBox(); self.win_height.setRange(1, 32767)
        self.win_origin_x = QSpinBox(); self.win_origin_x.setRange(-32768, 32767)
        self.win_origin_y = QSpinBox(); self.win_origin_y.setRange(-32768, 32767)
        self.win_origin_x.setToolTip("Window origin in desktop pixels (setOrigin)")
        self.win_origin_y.setToolTip("Window origin in desktop pixels (setOrigin)")
        form.addRow("window.width", self.win_width)
        form.addRow("window.height", self.win_height)
        form.addRow("window.origin_x", self.win_origin_x)
        form.addRow("window.origin_y", self.win_origin_y)

        # Extra asset directory. MousePortal appends this to its Panda3D model
        # path (getModelPath().appendDirectory) so corridor models can live
        # outside the app install.
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("(optional) extra model/asset directory")
        model_browse = QPushButton("…"); model_browse.setFixedWidth(30)
        model_browse.clicked.connect(self._browse_model_path)
        model_row = QHBoxLayout()
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.addWidget(self.model_path_edit, 1); model_row.addWidget(model_browse)
        model_holder = QWidget(); model_holder.setLayout(model_row)
        form.addRow("assets.model_path", model_holder)

        # Duration coupling: MousePortal's own block/trial maths is the real
        # experiment length, so it can drive the ExperimentConfig `duration`
        # that sizes camera preallocation and arms the run timer.
        self.total_label = QLabel()
        self.total_label.setToolTip(
            "num_blocks x trials_per_block x (trial_duration + iti_duration), "
            "honouring per-condition duration overrides, plus the device's "
            "tail_seconds."
        )
        self.override_duration = QCheckBox("override on Save")
        self.override_duration.setChecked(True)
        form.addRow("total duration", self.total_label)
        form.addRow("ExperimentConfig", self.override_duration)
        exp_box = _collapsible("Experiment", exp_content)
        layout.addWidget(exp_box)

        for w in (self.num_blocks, self.trials_per_block):
            w.valueChanged.connect(self._update_total_label)
        for w in (self.iti_duration, self.trial_duration):
            w.valueChanged.connect(self._update_total_label)

        # --- Conditions table ---------------------------------------------
        cond_content = QWidget()
        cond_layout = QVBoxLayout(cond_content)
        cond_layout.setContentsMargins(0, 0, 0, 0)
        self.cond_table = QTableWidget(0, 4)
        self.cond_table.setHorizontalHeaderLabels(["Label", "Transform", "Value", "Dur (s)"])
        header = self.cond_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.cond_table.verticalHeader().setVisible(False)
        self.cond_table.setMinimumHeight(220)
        # Cell widgets have wide size hints; let the table shrink to the tab.
        self.cond_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.cond_table.setToolTip(
            "Value is the single transform parameter (gain/offset/sigma/delay).\n"
            "identity/invert/freeze ignore it; multi-param transforms (clamp) "
            "must be edited in the JSON directly.\n"
            "Duration 0 = use the global trial_duration."
        )
        cond_layout.addWidget(self.cond_table)
        btn_row = QHBoxLayout()
        self.add_cond_btn = QPushButton("+ Condition")
        self.del_cond_btn = QPushButton("− Selected")
        self.up_cond_btn = QPushButton(); self.up_cond_btn.setFixedWidth(30)
        self.down_cond_btn = QPushButton(); self.down_cond_btn.setFixedWidth(30)
        self.up_cond_btn.setIcon(self._icon(QStyle.StandardPixmap.SP_ArrowUp))
        self.down_cond_btn.setIcon(self._icon(QStyle.StandardPixmap.SP_ArrowDown))
        self.up_cond_btn.setToolTip("Move the selected condition up")
        self.down_cond_btn.setToolTip("Move the selected condition down")
        self.add_cond_btn.clicked.connect(lambda: self._add_condition_row("", "gain", 1.0))
        self.del_cond_btn.clicked.connect(self._remove_selected_conditions)
        self.up_cond_btn.clicked.connect(lambda: self._move_condition(-1))
        self.down_cond_btn.clicked.connect(lambda: self._move_condition(1))
        btn_row.addWidget(self.add_cond_btn); btn_row.addWidget(self.del_cond_btn)
        btn_row.addWidget(self.up_cond_btn); btn_row.addWidget(self.down_cond_btn)
        btn_row.addStretch(1)
        cond_layout.addLayout(btn_row)
        cond_box = _collapsible("Conditions (velocity transforms)", cond_content)
        layout.addWidget(cond_box, 1)

        # --- Block sequences ----------------------------------------------
        seq_content = QWidget()
        seq_layout = QVBoxLayout(seq_content)
        seq_layout.setContentsMargins(0, 0, 0, 0)
        self.block_edit = QPlainTextEdit()
        self.block_edit.setPlaceholderText("gain_0p5, normal, gain_1p5, gain_2x")
        self.block_edit.setFixedHeight(80)
        self.block_edit.setToolTip(
            "One block per line; each line is a comma-separated list of "
            "condition labels, in trial order."
        )
        self.block_edit.textChanged.connect(self._update_total_label)
        seq_layout.addWidget(self.block_edit)
        seq_box = _collapsible("Block sequences", seq_content)
        seq_box.setToolTip(self.block_edit.toolTip())
        layout.addWidget(seq_box)

        # --- Actions -------------------------------------------------------
        action_row = QHBoxLayout()
        self.save_btn = QPushButton("Save")
        self.save_btn.setToolTip("Validate and persist to the MousePortal block in experiment.json")
        self.reload_btn = QPushButton("Reload")
        self.reload_btn.setToolTip("Discard edits and reload from experiment.json")
        # Same label + icon pairing the Setup tab uses for its file actions.
        self.open_json_btn = QPushButton(" Open")
        self.open_json_btn.setIcon(self._icon(QStyle.StandardPixmap.SP_FileIcon))
        self.open_json_btn.setToolTip(
            "Open experiment.json — the MousePortal block lives under Configuration.mouseportal"
        )
        self.reveal_json_btn = QPushButton(" Reveal")
        self.reveal_json_btn.setIcon(self._icon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.reveal_json_btn.setToolTip("Show experiment.json in your file manager")
        self.save_btn.clicked.connect(self._save)
        self.reload_btn.clicked.connect(self._reload)
        self.open_json_btn.clicked.connect(lambda: self._json_action(open_it=True))
        self.reveal_json_btn.clicked.connect(lambda: self._json_action(open_it=False))
        action_row.addWidget(self.save_btn); action_row.addWidget(self.reload_btn)
        action_row.addWidget(self.open_json_btn); action_row.addWidget(self.reveal_json_btn)
        action_row.addStretch(1)
        layout.addWidget(QLabel("<i>Edits apply to the next run after Save.</i>"))
        layout.addLayout(action_row)
        # Absorbs the slack when sections are collapsed, so the headers stay
        # stacked at the top instead of drifting apart.
        layout.addStretch(1)

        # The editable area locks while a Procedure is running.
        self._editors = [task_box, exp_box, cond_box, seq_box, self.save_btn]
        # The Procedure outlives this controller -- a fresh one is built on every
        # config/hardware reload while the Procedure (and its `events`) persists.
        # A stale controller left connected calls _set_editable() on its deleted
        # QGroupBoxes, raising out of procedure_started.emit() and aborting the run.
        events = self.procedure.events
        self._binds = Bindings()
        self._binds.connect(events.procedure_started, lambda *_: self._set_editable(False))
        for sig in (events.procedure_finished, events.procedure_error):
            self._binds.connect(sig, lambda *_: self._set_editable(True))

        self._reload()

    # ------------------------------------------------------------------
    def _icon(self, sp: QStyle.StandardPixmap):
        """Themed standard icon, matching the Setup tab's button styling."""
        return self.style().standardIcon(sp)

    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        """Disconnect from the shared Procedure's events before destruction.

        Must run before ``deleteLater()`` on a config/hardware hot-swap.
        Cascades into the status panel we own.
        """
        if self._panel is not None:
            self._panel.cleanup()
        self._binds.close()

    def closeEvent(self, event):  # noqa: N802 - Qt naming
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

    # ---- conditions table helpers ------------------------------------
    def _add_condition_row(
        self, label: str, ttype: str, value: float, duration: float = 0.0, row: int | None = None
    ) -> None:
        if row is None:
            row = self.cond_table.rowCount()
        self.cond_table.insertRow(row)
        self.cond_table.setItem(row, 0, QTableWidgetItem(str(label)))
        combo = QComboBox(); combo.addItems(list(KNOWN_TRANSFORMS))
        idx = combo.findText(ttype)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.cond_table.setCellWidget(row, 1, combo)
        spin = QDoubleSpinBox(); spin.setRange(-1_000_000, 1_000_000); spin.setDecimals(3)
        spin.setValue(float(value))
        self.cond_table.setCellWidget(row, 2, spin)
        # 0 means "no override" -- MousePortal falls back to the global
        # trial_duration for this condition.
        dur = QDoubleSpinBox(); dur.setRange(0, 86400); dur.setDecimals(2)
        dur.setSpecialValueText("global")
        dur.setValue(float(duration))
        self.cond_table.setCellWidget(row, 3, dur)
        # Connect last: setValue() on a half-built row would recompute the
        # total against a cell widget that isn't in the table yet.
        dur.valueChanged.connect(self._update_total_label)

    def _row_values(self, row: int) -> tuple[str, str, float, float]:
        item = self.cond_table.item(row, 0)
        return (
            item.text() if item else "",
            self.cond_table.cellWidget(row, 1).currentText(),
            self.cond_table.cellWidget(row, 2).value(),
            self.cond_table.cellWidget(row, 3).value(),
        )

    def _remove_selected_conditions(self) -> None:
        rows = sorted({i.row() for i in self.cond_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.cond_table.removeRow(r)
        self._update_total_label()

    def _move_condition(self, delta: int) -> None:
        """Move the selected condition one row up (-1) or down (+1).

        Row order is the saved ``conditions`` order, so this is how the list is
        arranged without hand-editing the JSON.
        """
        rows = {i.row() for i in self.cond_table.selectedIndexes()}
        if len(rows) != 1:
            return
        row = rows.pop()
        target = row + delta
        if not 0 <= target < self.cond_table.rowCount():
            return
        values = self._row_values(row)
        self.cond_table.removeRow(row)
        self._add_condition_row(*values, row=target)
        self.cond_table.selectRow(target)

    def _collect_conditions(self) -> List[Dict[str, Any]]:
        conditions: List[Dict[str, Any]] = []
        for row in range(self.cond_table.rowCount()):
            label, ttype, value, duration = self._row_values(row)
            cond: Dict[str, Any] = {"label": label.strip(), "transform_type": ttype}
            if ttype in TRANSFORM_PARAM:
                cond["transform_params"] = {TRANSFORM_PARAM[ttype]: value}
            if duration > 0:
                cond["trial_duration"] = duration
            conditions.append(cond)
        return conditions

    # ---- misc actions -------------------------------------------------
    def _browse_model_path(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select MousePortal model/asset directory",
            self.model_path_edit.text().strip(),
        )
        if path:
            self.model_path_edit.setText(path)

    def _json_action(self, *, open_it: bool) -> None:
        """Open or reveal the experiment.json holding the MousePortal block."""
        from mesofield.gui.config_wizard import (
            _open_in_default_app, _reveal_in_file_manager,
        )

        path = getattr(self.config, "_json_file_path", "")
        if not path or not os.path.isfile(path):
            QMessageBox.information(
                self, "No experiment.json",
                "No experiment.json is loaded. Load or create one in the Setup tab.",
            )
            return
        (_open_in_default_app if open_it else _reveal_in_file_manager)(path)

    def _tail_seconds(self) -> float:
        return float(getattr(self._device, "tail_seconds", 5.0) or 0.0)

    def _total_seconds(self) -> float:
        """MousePortal's own estimated run length, in seconds.

        The same figure :func:`summarize_experiment` puts on the status
        indicator -- the recording adds ``tail_seconds`` on top.
        """
        return total_duration(self._collect_block().get("experiment", {}))

    def _update_total_label(self) -> None:
        # Editors are wired before the conditions table exists during __init__.
        if not hasattr(self, "cond_table"):
            return
        total = self._total_seconds()
        tail = self._tail_seconds()
        self.total_label.setText(f"{total:.0f} s  (+{tail:.0f} s tail = {total + tail:.0f} s)")
        self.override_duration.setToolTip(
            f"Save writes ExperimentConfig.duration = {math.ceil(total + tail)} s "
            f"({total:.0f} s of trials plus the device's {tail:.0f} s tail), which "
            "sizes camera preallocation and arms the run timer."
        )

    # ---- load / save --------------------------------------------------
    def _reload(self) -> None:
        block = self.config.mouseportal
        task = (block or {}).get("task", "")
        self.task_edit.setText("" if task is None else str(task))
        exp = (block or {}).get("experiment", {}) or {}
        self.num_blocks.setValue(int(exp.get("num_blocks", 1)))
        self.trials_per_block.setValue(int(exp.get("trials_per_block", 1)))
        self.iti_duration.setValue(float(exp.get("iti_duration", 0.0)))
        end = exp.get("trial_end_condition", "duration")
        i = self.trial_end.findText(end); self.trial_end.setCurrentIndex(i if i >= 0 else 0)
        self.trial_duration.setValue(float(exp.get("trial_duration", 0.0) or 0.0))
        self.trial_distance.setValue(float(exp.get("trial_distance", 0.0) or 0.0))

        window = dict(_DEFAULT_WINDOW)
        window.update((block or {}).get("window") or {})
        self.win_width.setValue(int(window["width"]))
        self.win_height.setValue(int(window["height"]))
        self.win_origin_x.setValue(int(window["origin_x"]))
        self.win_origin_y.setValue(int(window["origin_y"]))
        assets = (block or {}).get("assets") or {}
        self.model_path_edit.setText(str(assets.get("model_path", "") or ""))

        self.cond_table.setRowCount(0)
        for cond in exp.get("conditions", []) or []:
            ttype = cond.get("transform_type", "identity")
            params = cond.get("transform_params", {}) or {}
            value = params.get(TRANSFORM_PARAM.get(ttype, ""), 1.0 if ttype == "gain" else 0.0)
            self._add_condition_row(
                cond.get("label", ""), ttype, float(value),
                float(cond.get("trial_duration") or 0.0),
            )

        self.block_edit.setPlainText(format_block_sequences(exp.get("block_conditions", [])))
        self._update_total_label()

    def _collect_block(self) -> Dict[str, Any]:
        block = dict(self.config.mouseportal)  # preserve corridor/fog/etc.
        block["window"] = {
            "width": self.win_width.value(),
            "height": self.win_height.value(),
            "origin_x": self.win_origin_x.value(),
            "origin_y": self.win_origin_y.value(),
        }
        model_path = self.model_path_edit.text().strip()
        if model_path:
            assets = dict(block.get("assets") or {})
            assets["model_path"] = model_path
            block["assets"] = assets
        else:
            block.pop("assets", None)
        task = self.task_edit.text().strip()
        if task:
            block["task"] = task
        else:
            block.pop("task", None)
        experiment = dict(block.get("experiment", {}))
        experiment.update({
            "num_blocks": self.num_blocks.value(),
            "trials_per_block": self.trials_per_block.value(),
            "iti_duration": self.iti_duration.value(),
            "trial_end_condition": self.trial_end.currentText(),
            "trial_duration": self.trial_duration.value(),
            "trial_distance": self.trial_distance.value(),
            "conditions": self._collect_conditions(),
            "block_conditions": parse_block_sequences(self.block_edit.toPlainText()),
        })
        block["experiment"] = experiment
        return block

    def _save(self) -> None:
        block = self._collect_block()
        errors = validate_block(block)
        if errors:
            QMessageBox.warning(
                self, "MousePortal config invalid",
                "Fix the following before saving:\n\n• " + "\n• ".join(errors),
            )
            return
        try:
            self.config.update_mouseportal(block)
            if self.override_duration.isChecked():
                # MousePortal's block/trial maths is the authoritative run
                # length; ExperimentConfig.duration sizes camera preallocation
                # and arms the run timer, so keep them in step.
                self.config.set(
                    "duration", int(math.ceil(self._total_seconds() + self._tail_seconds()))
                )
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        if self._panel is not None:
            self._panel.refresh_summary()
        self.save_btn.setToolTip(f"Saved {datetime.now().strftime('%H:%M:%S')}")
        self.tasksChanged.emit()
