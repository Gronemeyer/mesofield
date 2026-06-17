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

from datetime import datetime
from typing import Any, Dict, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QSpinBox, QDoubleSpinBox, QComboBox, QPushButton, QTableWidget,
    QTableWidgetItem, QPlainTextEdit, QMessageBox, QHeaderView,
)

from mesofield.gui.mouseportal_panel import MousePortalPanel
from mesofield.gui.mouseportal_config import (
    KNOWN_TRANSFORMS, TRANSFORM_PARAM, ZERO_PARAM, TRIAL_END_CONDITIONS,
    validate_block, parse_block_sequences, format_block_sequences,
)


class MousePortalController(QWidget):
    """Editable view of the MousePortal config block."""

    def __init__(self, procedure, parent=None) -> None:
        super().__init__(parent)
        self.procedure = procedure
        self.config = procedure.config
        self._device = procedure.config.hardware.devices.get("mouseportal")

        layout = QVBoxLayout(self)
        self.setMaximumWidth(500)

        # Status header (reuses the same indicator as the DynamicController panel).
        if self._device is not None:
            self._panel = MousePortalPanel(self.config, self._device, parent=self)
            layout.addWidget(self._panel)
        else:
            self._panel = None

        # --- Experiment scalars -------------------------------------------
        exp_box = QGroupBox("Experiment")
        form = QFormLayout(exp_box)
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
        layout.addWidget(exp_box)

        # --- Conditions table ---------------------------------------------
        cond_box = QGroupBox("Conditions (velocity transforms)")
        cond_layout = QVBoxLayout(cond_box)
        self.cond_table = QTableWidget(0, 3)
        self.cond_table.setHorizontalHeaderLabels(["Label", "Transform", "Value"])
        self.cond_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.cond_table.setToolTip(
            "Value is the single transform parameter (gain/offset/sigma/delay).\n"
            "identity/invert/freeze ignore it; multi-param transforms (clamp) "
            "must be edited in the JSON directly."
        )
        cond_layout.addWidget(self.cond_table)
        btn_row = QHBoxLayout()
        self.add_cond_btn = QPushButton("+ Condition")
        self.del_cond_btn = QPushButton("− Selected")
        self.add_cond_btn.clicked.connect(lambda: self._add_condition_row("", "gain", 1.0))
        self.del_cond_btn.clicked.connect(self._remove_selected_conditions)
        btn_row.addWidget(self.add_cond_btn); btn_row.addWidget(self.del_cond_btn); btn_row.addStretch(1)
        cond_layout.addLayout(btn_row)
        layout.addWidget(cond_box)

        # --- Block sequences ----------------------------------------------
        seq_box = QGroupBox("Block sequences (one block per line, comma-separated labels)")
        seq_layout = QVBoxLayout(seq_box)
        self.block_edit = QPlainTextEdit()
        self.block_edit.setPlaceholderText("gain_0p5, normal, gain_1p5, gain_2x")
        self.block_edit.setFixedHeight(80)
        seq_layout.addWidget(self.block_edit)
        layout.addWidget(seq_box)

        # --- Actions -------------------------------------------------------
        action_row = QHBoxLayout()
        self.save_btn = QPushButton("Save")
        self.save_btn.setToolTip("Validate and persist to the MousePortal block in experiment.json")
        self.reload_btn = QPushButton("Reload")
        self.reload_btn.setToolTip("Discard edits and reload from experiment.json")
        self.save_btn.clicked.connect(self._save)
        self.reload_btn.clicked.connect(self._reload)
        action_row.addWidget(self.save_btn); action_row.addWidget(self.reload_btn); action_row.addStretch(1)
        layout.addWidget(QLabel("<i>Edits apply to the next run after Save.</i>"))
        layout.addLayout(action_row)
        layout.addStretch(1)

        # The editable area locks while a Procedure is running.
        self._editors = [exp_box, cond_box, seq_box, self.save_btn]
        events = getattr(self.procedure, "events", None)
        if events is not None:
            events.procedure_started.connect(lambda *_: self._set_editable(False))
            events.procedure_finished.connect(lambda *_: self._set_editable(True))
            events.procedure_error.connect(lambda *_: self._set_editable(True))

        self._reload()

    # ------------------------------------------------------------------
    def _set_editable(self, on: bool) -> None:
        for w in self._editors:
            w.setEnabled(on)

    # ---- conditions table helpers ------------------------------------
    def _add_condition_row(self, label: str, ttype: str, value: float) -> None:
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

    def _remove_selected_conditions(self) -> None:
        rows = sorted({i.row() for i in self.cond_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.cond_table.removeRow(r)

    def _collect_conditions(self) -> List[Dict[str, Any]]:
        conditions: List[Dict[str, Any]] = []
        for row in range(self.cond_table.rowCount()):
            label_item = self.cond_table.item(row, 0)
            label = label_item.text().strip() if label_item else ""
            ttype = self.cond_table.cellWidget(row, 1).currentText()
            value = self.cond_table.cellWidget(row, 2).value()
            cond: Dict[str, Any] = {"label": label, "transform_type": ttype}
            if ttype in TRANSFORM_PARAM:
                cond["transform_params"] = {TRANSFORM_PARAM[ttype]: value}
            conditions.append(cond)
        return conditions

    # ---- load / save --------------------------------------------------
    def _reload(self) -> None:
        block = self.config.mouseportal
        exp = (block or {}).get("experiment", {}) or {}
        self.num_blocks.setValue(int(exp.get("num_blocks", 1)))
        self.trials_per_block.setValue(int(exp.get("trials_per_block", 1)))
        self.iti_duration.setValue(float(exp.get("iti_duration", 0.0)))
        end = exp.get("trial_end_condition", "duration")
        i = self.trial_end.findText(end); self.trial_end.setCurrentIndex(i if i >= 0 else 0)
        self.trial_duration.setValue(float(exp.get("trial_duration", 0.0) or 0.0))
        self.trial_distance.setValue(float(exp.get("trial_distance", 0.0) or 0.0))

        self.cond_table.setRowCount(0)
        for cond in exp.get("conditions", []) or []:
            ttype = cond.get("transform_type", "identity")
            params = cond.get("transform_params", {}) or {}
            value = params.get(TRANSFORM_PARAM.get(ttype, ""), 1.0 if ttype == "gain" else 0.0)
            self._add_condition_row(cond.get("label", ""), ttype, float(value))

        self.block_edit.setPlainText(format_block_sequences(exp.get("block_conditions", [])))

    def _collect_block(self) -> Dict[str, Any]:
        block = dict(self.config.mouseportal)  # preserve window/fog/etc.
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
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        if self._panel is not None:
            self._panel.refresh_summary()
        self.save_btn.setToolTip(f"Saved {datetime.now().strftime('%H:%M:%S')}")
