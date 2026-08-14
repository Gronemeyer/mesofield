"""Editable MousePortal configuration tab.

A dedicated tab (sibling of ExperimentConfig/Terminal/Setup) that lets the user
adjust the corridor and trial parameters between runs and persists them safely,
mirroring how :class:`~mesofield.gui.controller.ConfigController` manages
ExperimentConfig: edits are collected into a MousePortal config block,
validated via :mod:`mesofield.gui.mouseportal_config` (errors shown in a
dialog; nothing is persisted on failure), and committed through
:meth:`ExperimentConfig.update_mouseportal`.

Editing display text
--------------------
Every user-visible string is in :data:`TEXT` at the top of this file. Nothing
below it hard-codes wording, so the tab can be reworded without reading the
widget code. Strings the pure helpers produce (list rows, the plan preview)
live in ``mouseportal_config.TEXT``.

Layout
------
The tab lives in a width-constrained right-hand panel, so it is built as
master-detail rather than as wide tables: a compact list of conditions (or
blocks) with a form beneath it for whichever one is selected. A form grows
downward, which the panel can scroll; a table grows sideways, which it cannot.
That is also what makes room for paradigms with more per-condition settings: a
go/no-go cue, response window and reward rule are three more form rows, not
three more columns nobody can see.

Nothing is typed that can be chosen. Transform parameters are one labelled,
ranged spin box each, built from the selected transform. A block's trial order
is assembled from a drop-down of the conditions that actually exist, so a label
cannot be misspelled into a config error.

Editing is locked while a Procedure is running.
"""

from __future__ import annotations

import copy
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QColor, QIcon
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QSpinBox, QDoubleSpinBox, QComboBox, QPushButton, QListWidget,
    QMessageBox, QLineEdit, QCheckBox, QScrollArea, QFileDialog, QStyle,
    QAbstractItemView, QSizePolicy, QColorDialog,
)

from mesofield.gui.mouseportal_panel import MousePortalPanel
from mesofield.signals import Bindings
from mesofield.gui.mouseportal_config import (
    KNOWN_TRANSFORMS, TRANSFORM_PARAMS, TRIAL_END_CONDITIONS, BLOCK_ORDERS,
    UNTIMED_END_CONDITIONS, validate_block, resolve_session, describe_plan,
    default_seed, new_condition, new_block, condition_summary, block_summary,
    blocks_using,
)


# ════════════════════════════════════════════════════════════════════════════
# User-facing text. Every string this tab displays is here. Edit freely; the
# widget code below refers to these keys and hard-codes no wording.
# ════════════════════════════════════════════════════════════════════════════
TEXT = {
    # -- Section headers -----------------------------------------------
    "sec_conditions": "Conditions",
    "sec_blocks": "Blocks",
    "sec_session": "Session",
    "sec_display": "Display and input",

    # -- Condition form ------------------------------------------------
    "cond_name": "name",
    "cond_transform": "does",
    "cond_end": "ends",
    "cond_duration": "after",
    "cond_distance": "after",
    "cond_expected": "typically",
    "cond_iti": "then",
    "cond_iti_check": "pause before the next trial",
    "cond_name_placeholder": "e.g. gain_2x",
    "cond_default_name": "condition",

    # -- Block form ----------------------------------------------------
    "blk_name": "name",
    "blk_trials": "trials",
    "blk_repeat": "repeat",
    "blk_order": "order",
    "blk_name_placeholder": "e.g. order_A",
    "blk_default_name": "block",
    "blk_copy_suffix": "{name}_copy",
    "seq_row": "{n}.  {label}",

    # -- Session form --------------------------------------------------
    "task": "task",
    "task_placeholder": "blank = serves every task",
    "iti": "pause between trials",
    "iti_random": "vary randomly",
    "seed": "random seed",
    "seed_auto": "today ({seed})",
    "length_override": "set the recording duration from this",

    # -- Display form --------------------------------------------------
    "win_size": "window size",
    "win_origin": "window origin",
    "camera_height": "camera height",
    "speed_scaling": "treadmill gain",
    "fog_density": "fog density",
    "fog_color": "fog colour",
    "tex_left": "left wall",
    "tex_right": "right wall",
    "tex_floor": "floor",
    "tex_ceiling": "ceiling",
    "assets": "asset folder",
    "assets_placeholder": "optional extra model/asset directory",
    "assets_dialog": "Select MousePortal model/asset directory",

    # -- Buttons -------------------------------------------------------
    "add": "+ Add",
    "remove": "- Remove",
    "duplicate": "Duplicate",
    "browse": "...",
    "save": "Save",
    "reload": "Reload",
    "open": " Open",
    "reveal": " Reveal",
    "saved_at": "Saved {time}",

    # -- Tooltips ------------------------------------------------------
    "tip_sec_conditions":
        "The kinds of trial this experiment can run. Each condition says what "
        "the corridor does and how its own trials end.",
    "tip_sec_blocks":
        "The session, run top to bottom. Each block has its own trial list, so "
        "blocks may differ in length.",
    "tip_cond_name":
        "Written to every row of the session's CSVs, and what a block's trial "
        "list refers to. Renaming updates the blocks that use it.",
    "tip_cond_up": "Move the selected condition up",
    "tip_cond_down": "Move the selected condition down",
    "tip_cond_expected":
        "How long a trial of this condition usually takes. Used only to "
        "estimate the recording length; MousePortal never reads it. Trials that "
        "end on distance or on a keypress have no length until they happen, so "
        "without this they are left out of the estimate rather than guessed at.",
    "tip_cond_iti":
        "Unchecked, the next trial starts on the same frame this one ends, "
        "which is how two conditions are chained into one perceived trial.",
    "tip_blk_name":
        "Written to the block_name column of trials.csv, so an order effect is "
        "a groupby rather than a lookup against this config.",
    "tip_blk_trials": "One pass of this block, in order.",
    "tip_blk_picker": "Conditions defined above. Pick one and add it.",
    "tip_blk_repeat":
        "How many times to run the trial list. Repeating never changes the "
        "balance: 3 repeats of 4 trials is 12 trials, 3 of each.",
    "tip_blk_order":
        "'shuffle' permutes the repeated list using the session seed. The "
        "counts stay exact, only the order changes, and the order actually run "
        "is written to the session's timing sidecar.",
    "tip_blk_dup":
        "Copy the selected block. The quick way to build a counterbalanced "
        "second order from the first.",
    "tip_blk_up": "Run this block earlier in the session",
    "tip_blk_down": "Run this block later in the session",
    "tip_seq_up": "Move this trial earlier",
    "tip_seq_down": "Move this trial later",
    "tip_plan": (
        "The expanded session. Shuffled blocks show their pre-shuffle order; "
        "the realised order is drawn from the seed at run time."
    ),
    "tip_task":
        "ExperimentConfig task that runs this MousePortal configuration. On a "
        "rig with several stimulus apps, only the device bound to the selected "
        "task launches.",
    "tip_iti":
        "Time between trials. 'vary randomly' draws each pause uniformly from "
        "[min, max] using the seed below instead of using one fixed length.",
    "tip_seed":
        "Seed for the random pauses and for any block set to 'shuffle'. Left on "
        "today's date it becomes YYYYMMDD and is recorded in the run's config "
        "and timing sidecar. Two sessions run on the same day then share a "
        "seed; set it explicitly per subject for independent randomisation.",
    "tip_override": "Save writes ExperimentConfig.duration = {seconds} s, "
                    "which sizes camera preallocation and arms the run timer. "
                    "The length is shown on the MousePortal indicator above.",
    "tip_fog": "Distance fog in the corridor. 0 disables it.",
    "tip_texture":
        "Image used for this surface. The list is the images in the asset "
        "folder below, plus whatever the config already names.",
    "tip_win_origin":
        "Top-left corner in desktop pixels: how the corridor is placed on the "
        "stimulus monitor.",
    "tip_speed_scaling":
        "Corridor units per encoder unit. Applies to the treadmill in both "
        "network mode (samples forwarded by mesofield) and MousePortal's own "
        "serial mode. 1.0 passes the encoder's speed through unchanged.",
    "tip_save": "Validate and persist to the MousePortal block in experiment.json",
    "tip_reload": "Discard edits and reload from experiment.json",
    "tip_open": "Open experiment.json",
    "tip_reveal": "Show experiment.json in your file manager",

    # -- Transform descriptions (tooltip on the 'does' selector) -------
    "transform_help": {
        "identity": "Normal closed loop. The corridor tracks the treadmill.",
        "gain": "Scale the corridor's speed relative to the treadmill.",
        "invert": "Reverse direction: running forward moves the corridor back.",
        "reverse": "Open loop. The corridor runs backward at a fixed speed, "
                   "ignoring input.",
        "freeze": "Open loop. The corridor does not move, whatever the subject does.",
        "offset": "Add a constant drift, so the corridor moves even when stationary.",
        "clamp": "Limit the corridor's speed to a range.",
        "noisy": "Add Gaussian noise to the corridor's speed each frame.",
        "delay": "Replay the subject's input after a fixed lag.",
    },

    # -- End-rule descriptions (tooltip on the 'ends' selector) --------
    "end_help": {
        "duration": "Ends after a set number of seconds.",
        "distance": "Ends after the subject travels a set corridor distance.",
        "manual": "Runs until the experimenter presses Space.",
    },

    # -- Dialogs -------------------------------------------------------
    "dlg_invalid_title": "MousePortal config invalid",
    "dlg_invalid_body": "Fix the following before saving:\n\n- {errors}",
    "dlg_save_failed": "Save failed",
    "dlg_no_json_title": "No experiment.json",
    "dlg_no_json_body":
        "No experiment.json is loaded. Load or create one in the Setup tab.",
    "dlg_in_use_title": "Condition is in use",
    "dlg_in_use_body":
        "'{label}' is used by {blocks}.\n\n"
        "Remove it and delete those trials from those blocks?",

    "footer": "<i>Edits apply to the next run after Save.</i>",
}

# MousePortal's own defaults, mirrored so this tab can seed its fields before
# any such block has been authored.
_DEFAULT_WINDOW = {"width": 1920, "height": 1080, "origin_x": 0, "origin_y": 0}
_DEFAULT_CAMERA = {"height": 2.0, "speed_scaling": 1.0, "keyboard_speed": 20.0}
_DEFAULT_FOG = {"density": 0.06, "color": [0.5, 0.5, 0.5]}

# Image types offered as corridor textures.
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".gif"}

# Qt's "no maximum" sentinel, and the height every collapsed section clamps to.
_QWIDGETSIZE_MAX = 16777215
_COLLAPSED_H = 20


def _reveal_in_file_manager(path: str) -> None:
    """Show *path* selected in the OS file manager.

    ``config_wizard`` only exposes an open-in-default-app helper, so the
    reveal is done here. Every platform spells "select this file" differently;
    the last resort is opening the containing folder, which is still closer
    than doing nothing.
    """
    import subprocess
    import sys

    path = os.path.abspath(path)
    folder = os.path.dirname(path)
    try:
        if sys.platform == "win32":
            subprocess.run(["explorer", "/select,", path], check=False)
        elif sys.platform == "darwin":
            subprocess.run(["open", "-R", path], check=False)
        else:
            subprocess.run(["xdg-open", folder], check=False)
    except OSError:
        from mesofield.gui.config_wizard import _open_in_default_app
        _open_in_default_app(folder)


def _collapsible(title: str, content: QWidget, tooltip: str = "",
                 expanded: bool = True) -> QGroupBox:
    """A checkable group box whose check state shows/hides *content*.

    Qt's own checkable group box only *disables* its children; hiding is what
    actually buys back vertical space, which is the point on this tab. The
    themed QGroupBox carries ~18px of padding, so hiding alone still leaves a
    stub box -- clamp the height too, to the same value for every section so
    collapsed headers line up.
    """
    box = QGroupBox(title)
    if tooltip:
        box.setToolTip(tooltip)
    box.setCheckable(True)
    inner = QVBoxLayout(box)
    inner.setContentsMargins(0, 0, 0, 0)
    inner.addWidget(content)

    def _toggle(checked: bool) -> None:
        content.setVisible(checked)
        box.setMaximumHeight(_QWIDGETSIZE_MAX if checked else _COLLAPSED_H)

    box.toggled.connect(_toggle)
    box.setChecked(expanded)
    _toggle(expanded)
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

        # The edit model. The list widgets show summaries of these mappings and
        # the detail forms read and write them; there is no state in the widgets
        # that is not written straight back here on change.
        self._conditions: List[Dict[str, Any]] = []
        self._blocks: List[Dict[str, Any]] = []
        # Set while a form is being populated from the model, so the change
        # signals that populating fires do not write it straight back.
        self._loading = False
        # Spin boxes for the selected transform's parameters, by parameter name.
        self._param_editors: Dict[str, QDoubleSpinBox] = {}

        self.setMaximumWidth(520)
        # Everything lives in a scroll area: with all sections expanded the tab
        # is taller than a laptop screen, and collapsing is a preference, not a
        # requirement for reaching the Save button.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        # The tab is width-constrained; only ever scroll vertically. Every
        # section below is a vertical form for exactly this reason.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        layout = QVBoxLayout(body)
        # The sections are themed group boxes whose titles sit in a 14px top
        # margin; the default 6px inter-widget spacing runs the previous
        # section's border straight into the next one's title.
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(16)
        scroll.setWidget(body)
        outer.addWidget(scroll)

        if self._device is not None:
            self._panel = MousePortalPanel(self.config, self._device, parent=self)
            layout.addWidget(self._panel)
        else:
            self._panel = None

        # Session first: it frames everything below it. Display and input is
        # rig setup that rarely changes between runs, so it starts collapsed.
        layout.addWidget(self._build_session_section())
        layout.addWidget(self._build_display_section())
        layout.addWidget(self._build_conditions_section())
        layout.addWidget(self._build_blocks_section())
        layout.addWidget(QLabel(TEXT["footer"]))
        layout.addLayout(self._build_actions())
        # Absorbs the slack when sections are collapsed, so the headers stay
        # stacked at the top instead of drifting apart.
        layout.addStretch(1)

        self._editors = [
            self._cond_box, self._blk_box, self._session_box, self._display_box,
            self.save_btn,
        ]
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

    # ─── Section builders ───────────────────────────────────────────────
    def _build_conditions_section(self) -> QGroupBox:
        """Condition palette: a picker over the labels, a form for the selected one."""
        content = QWidget()
        col = QVBoxLayout(content)
        col.setContentsMargins(0, 0, 0, 0)

        self.cond_list = QListWidget()
        self.cond_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.cond_list.setMaximumHeight(110)
        self.cond_list.currentRowChanged.connect(self._on_condition_selected)
        col.addWidget(self.cond_list)

        self.add_cond_btn = QPushButton(TEXT["add"])
        self.del_cond_btn = QPushButton(TEXT["remove"])
        self.up_cond_btn = self._arrow_button(
            QStyle.StandardPixmap.SP_ArrowUp, TEXT["tip_cond_up"])
        self.down_cond_btn = self._arrow_button(
            QStyle.StandardPixmap.SP_ArrowDown, TEXT["tip_cond_down"])
        self.add_cond_btn.clicked.connect(self._add_condition)
        self.del_cond_btn.clicked.connect(self._remove_condition)
        self.up_cond_btn.clicked.connect(lambda: self._move_condition(-1))
        self.down_cond_btn.clicked.connect(lambda: self._move_condition(1))
        col.addLayout(self._button_row(
            self.add_cond_btn, self.del_cond_btn, self.up_cond_btn, self.down_cond_btn))

        self._cond_detail = QWidget()
        form = QFormLayout(self._cond_detail)
        form.setContentsMargins(0, 8, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._cond_form = form

        self.cond_label = QLineEdit()
        self.cond_label.setPlaceholderText(TEXT["cond_name_placeholder"])
        self.cond_label.setToolTip(TEXT["tip_cond_name"])
        self.cond_label.editingFinished.connect(self._commit_condition)
        form.addRow(TEXT["cond_name"], self.cond_label)

        # What each transform does rides on the selector's tooltip rather than a
        # caption under it: the explanation is only wanted while choosing.
        self.cond_transform = QComboBox()
        self.cond_transform.addItems(list(KNOWN_TRANSFORMS))
        self.cond_transform.currentTextChanged.connect(self._on_transform_changed)
        self._transform_row = QLabel(TEXT["cond_transform"])
        form.addRow(self._transform_row, self.cond_transform)

        self.cond_end = QComboBox()
        self.cond_end.addItems(list(TRIAL_END_CONDITIONS))
        self.cond_end.currentTextChanged.connect(self._on_end_rule_changed)
        form.addRow(TEXT["cond_end"], self.cond_end)

        self.cond_duration = QDoubleSpinBox()
        self.cond_duration.setRange(0.01, 86400.0); self.cond_duration.setDecimals(2)
        self.cond_duration.setSuffix(" s")
        self.cond_duration.valueChanged.connect(self._commit_condition)
        self._row_duration = self._add_row(form, TEXT["cond_duration"], self.cond_duration)

        self.cond_distance = QDoubleSpinBox()
        self.cond_distance.setRange(0.01, 1_000_000.0); self.cond_distance.setDecimals(2)
        self.cond_distance.setSuffix(" units")
        self.cond_distance.valueChanged.connect(self._commit_condition)
        self._row_distance = self._add_row(form, TEXT["cond_distance"], self.cond_distance)

        self.cond_expected = QDoubleSpinBox()
        self.cond_expected.setRange(0.0, 86400.0); self.cond_expected.setDecimals(2)
        self.cond_expected.setSuffix(" s")
        self.cond_expected.setSpecialValueText("unknown")
        self.cond_expected.setToolTip(TEXT["tip_cond_expected"])
        self.cond_expected.valueChanged.connect(self._commit_condition)
        self._row_expected = self._add_row(form, TEXT["cond_expected"], self.cond_expected)

        self.cond_iti = QCheckBox(TEXT["cond_iti_check"])
        self.cond_iti.setToolTip(TEXT["tip_cond_iti"])
        self.cond_iti.toggled.connect(self._commit_condition)
        form.addRow(TEXT["cond_iti"], self.cond_iti)

        col.addWidget(self._cond_detail)
        self._cond_box = _collapsible(
            TEXT["sec_conditions"], content, TEXT["tip_sec_conditions"])
        return self._cond_box

    def _build_blocks_section(self) -> QGroupBox:
        """Session structure: a picker over blocks, a builder for the selected one."""
        content = QWidget()
        col = QVBoxLayout(content)
        col.setContentsMargins(0, 0, 0, 0)

        self.block_list = QListWidget()
        self.block_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.block_list.setMaximumHeight(90)
        self.block_list.currentRowChanged.connect(self._on_block_selected)
        col.addWidget(self.block_list)

        self.add_blk_btn = QPushButton(TEXT["add"])
        self.del_blk_btn = QPushButton(TEXT["remove"])
        self.dup_blk_btn = QPushButton(TEXT["duplicate"])
        self.dup_blk_btn.setToolTip(TEXT["tip_blk_dup"])
        self.up_blk_btn = self._arrow_button(
            QStyle.StandardPixmap.SP_ArrowUp, TEXT["tip_blk_up"])
        self.down_blk_btn = self._arrow_button(
            QStyle.StandardPixmap.SP_ArrowDown, TEXT["tip_blk_down"])
        self.add_blk_btn.clicked.connect(self._add_block)
        self.del_blk_btn.clicked.connect(self._remove_block)
        self.dup_blk_btn.clicked.connect(self._duplicate_block)
        self.up_blk_btn.clicked.connect(lambda: self._move_block(-1))
        self.down_blk_btn.clicked.connect(lambda: self._move_block(1))
        col.addLayout(self._button_row(
            self.add_blk_btn, self.del_blk_btn, self.dup_blk_btn,
            self.up_blk_btn, self.down_blk_btn))

        self._blk_detail = QWidget()
        form = QFormLayout(self._blk_detail)
        form.setContentsMargins(0, 8, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.block_name = QLineEdit()
        self.block_name.setPlaceholderText(TEXT["blk_name_placeholder"])
        self.block_name.setToolTip(TEXT["tip_blk_name"])
        self.block_name.editingFinished.connect(self._commit_block)
        form.addRow(TEXT["blk_name"], self.block_name)

        # The trial list is built by picking from the conditions that exist, so
        # a label can never be mistyped into a config that fails validation.
        self.seq_list = QListWidget()
        self.seq_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.seq_list.setMaximumHeight(110)
        self.seq_list.setToolTip(TEXT["tip_blk_trials"])
        form.addRow(TEXT["blk_trials"], self.seq_list)

        self.seq_picker = QComboBox()
        self.seq_picker.setSizePolicy(QSizePolicy.Policy.Expanding,
                                      QSizePolicy.Policy.Fixed)
        self.seq_picker.setToolTip(TEXT["tip_blk_picker"])
        self.seq_add_btn = QPushButton(TEXT["add"])
        self.seq_add_btn.clicked.connect(self._append_trial)
        add_row = QHBoxLayout(); add_row.setContentsMargins(0, 0, 0, 0)
        add_row.addWidget(self.seq_picker, 1)
        add_row.addWidget(self.seq_add_btn)
        add_holder = QWidget(); add_holder.setLayout(add_row)
        form.addRow("", add_holder)

        self.seq_del_btn = QPushButton(TEXT["remove"])
        self.seq_up_btn = self._arrow_button(
            QStyle.StandardPixmap.SP_ArrowUp, TEXT["tip_seq_up"])
        self.seq_down_btn = self._arrow_button(
            QStyle.StandardPixmap.SP_ArrowDown, TEXT["tip_seq_down"])
        self.seq_del_btn.clicked.connect(self._remove_trial)
        self.seq_up_btn.clicked.connect(lambda: self._move_trial(-1))
        self.seq_down_btn.clicked.connect(lambda: self._move_trial(1))
        edit_holder = QWidget()
        edit_holder.setLayout(self._button_row(
            self.seq_del_btn, self.seq_up_btn, self.seq_down_btn))
        form.addRow("", edit_holder)

        self.block_repeat = QSpinBox(); self.block_repeat.setRange(1, 10_000)
        self.block_repeat.setToolTip(TEXT["tip_blk_repeat"])
        self.block_repeat.valueChanged.connect(self._commit_block)
        form.addRow(TEXT["blk_repeat"], self.block_repeat)

        self.block_order = QComboBox()
        self.block_order.addItems(list(BLOCK_ORDERS))
        self.block_order.setToolTip(TEXT["tip_blk_order"])
        self.block_order.currentTextChanged.connect(self._commit_block)
        form.addRow(TEXT["blk_order"], self.block_order)

        col.addWidget(self._blk_detail)

        # The one caption kept on screen: it is the expanded session itself,
        # not an explanation of the controls above it.
        self.plan_label = QLabel()
        self.plan_label.setWordWrap(True)
        self.plan_label.setToolTip(TEXT["tip_plan"])
        self.plan_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        try:
            from mesofield.gui import theme
            self.plan_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        except Exception:
            pass
        col.addWidget(self.plan_label)

        self._blk_box = _collapsible(TEXT["sec_blocks"], content, TEXT["tip_sec_blocks"])
        return self._blk_box

    def _build_session_section(self) -> QGroupBox:
        """Everything that applies to the whole session."""
        content = QWidget()
        form = QFormLayout(content)
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.task_edit = QLineEdit()
        self.task_edit.setPlaceholderText(TEXT["task_placeholder"])
        self.task_edit.setToolTip(TEXT["tip_task"])
        form.addRow(TEXT["task"], self.task_edit)

        self.iti_random = QCheckBox(TEXT["iti_random"])
        self.iti_duration = QDoubleSpinBox(); self.iti_duration.setRange(0, 3600)
        self.iti_duration.setSuffix(" s")
        self.iti_min = QDoubleSpinBox(); self.iti_min.setRange(0, 3600); self.iti_min.setSuffix(" s")
        self.iti_max = QDoubleSpinBox(); self.iti_max.setRange(0, 3600); self.iti_max.setSuffix(" s")
        self.iti_dash = QLabel("to")
        iti_row = QHBoxLayout(); iti_row.setContentsMargins(0, 0, 0, 0)
        iti_row.addWidget(self.iti_duration, 1)
        iti_row.addWidget(self.iti_min, 1); iti_row.addWidget(self.iti_dash)
        iti_row.addWidget(self.iti_max, 1)
        iti_holder = QWidget(); iti_holder.setLayout(iti_row)
        iti_holder.setToolTip(TEXT["tip_iti"])
        form.addRow(TEXT["iti"], iti_holder)
        form.addRow("", self.iti_random)
        for w in (self.iti_duration, self.iti_min, self.iti_max):
            w.valueChanged.connect(self._refresh_totals)
        self.iti_random.toggled.connect(self._on_iti_random)

        # The date the 'auto' seed resolves to is in the field itself, so the
        # value is visible without a caption explaining it.
        self.random_seed = QSpinBox(); self.random_seed.setRange(0, 2_147_483_647)
        self.random_seed.setSpecialValueText(TEXT["seed_auto"].format(seed=default_seed()))
        self.random_seed.setToolTip(TEXT["tip_seed"])
        # A new seed redraws every pause, so the length shown on the indicator
        # has to follow it.
        self.random_seed.valueChanged.connect(self._refresh_totals)
        form.addRow(TEXT["seed"], self.random_seed)

        # The estimated length lives on the MousePortal indicator at the top of
        # the tab, so it is not reported twice in two places that could drift.
        self.override_duration = QCheckBox(TEXT["length_override"])
        self.override_duration.setChecked(True)
        form.addRow("", self.override_duration)

        self._session_box = _collapsible(TEXT["sec_session"], content)
        return self._session_box

    def _build_display_section(self) -> QGroupBox:
        """Window geometry, input gain, and the extra asset directory."""
        content = QWidget()
        form = QFormLayout(content)
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.win_width = QSpinBox(); self.win_width.setRange(1, 32767)
        self.win_height = QSpinBox(); self.win_height.setRange(1, 32767)
        size_row = QHBoxLayout(); size_row.setContentsMargins(0, 0, 0, 0)
        size_row.addWidget(self.win_width, 1); size_row.addWidget(QLabel("x"))
        size_row.addWidget(self.win_height, 1)
        size_holder = QWidget(); size_holder.setLayout(size_row)
        form.addRow(TEXT["win_size"], size_holder)

        self.win_origin_x = QSpinBox(); self.win_origin_x.setRange(-32768, 32767)
        self.win_origin_y = QSpinBox(); self.win_origin_y.setRange(-32768, 32767)
        origin_row = QHBoxLayout(); origin_row.setContentsMargins(0, 0, 0, 0)
        origin_row.addWidget(self.win_origin_x, 1); origin_row.addWidget(QLabel(","))
        origin_row.addWidget(self.win_origin_y, 1)
        origin_holder = QWidget(); origin_holder.setLayout(origin_row)
        origin_holder.setToolTip(TEXT["tip_win_origin"])
        form.addRow(TEXT["win_origin"], origin_holder)

        self.camera_height = QDoubleSpinBox(); self.camera_height.setRange(0, 1000)
        self.camera_height.setDecimals(2)
        form.addRow(TEXT["camera_height"], self.camera_height)

        self.speed_scaling = QDoubleSpinBox()
        self.speed_scaling.setRange(0.0001, 10_000.0); self.speed_scaling.setDecimals(4)
        self.speed_scaling.setToolTip(TEXT["tip_speed_scaling"])
        form.addRow(TEXT["speed_scaling"], self.speed_scaling)

        self.fog_density = QDoubleSpinBox()
        self.fog_density.setRange(0.0, 10.0); self.fog_density.setDecimals(3)
        self.fog_density.setSingleStep(0.01)
        self.fog_density.setToolTip(TEXT["tip_fog"])
        form.addRow(TEXT["fog_density"], self.fog_density)

        self.fog_color_btn = QPushButton()
        self.fog_color_btn.setToolTip(TEXT["tip_fog"])
        self.fog_color_btn.clicked.connect(self._pick_fog_color)
        self._fog_rgb = (0.5, 0.5, 0.5)
        form.addRow(TEXT["fog_color"], self.fog_color_btn)

        # Texture pickers. Each lists the images in the asset folder with a
        # thumbnail, so a surface is chosen by looking at it rather than by
        # typing a path that only fails at render time.
        self.texture_pickers: Dict[str, QComboBox] = {}
        for key, label in (
            ("left_wall_texture", TEXT["tex_left"]),
            ("right_wall_texture", TEXT["tex_right"]),
            ("floor_texture", TEXT["tex_floor"]),
            ("ceiling_texture", TEXT["tex_ceiling"]),
        ):
            box = QComboBox()
            box.setIconSize(QSize(28, 28))
            box.setToolTip(TEXT["tip_texture"])
            self.texture_pickers[key] = box
            form.addRow(label, box)

        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText(TEXT["assets_placeholder"])
        model_browse = QPushButton(TEXT["browse"]); model_browse.setFixedWidth(30)
        model_browse.clicked.connect(self._browse_model_path)
        model_row = QHBoxLayout(); model_row.setContentsMargins(0, 0, 0, 0)
        model_row.addWidget(self.model_path_edit, 1); model_row.addWidget(model_browse)
        model_holder = QWidget(); model_holder.setLayout(model_row)
        form.addRow(TEXT["assets"], model_holder)

        self._display_box = _collapsible(TEXT["sec_display"], content, expanded=False)
        return self._display_box

    # ─── Textures ───────────────────────────────────────────────────────
    def _refresh_texture_pickers(self, corridor: Dict[str, Any]) -> None:
        """Fill each texture picker with the asset folder's images, thumbnailed.

        The value already in the config is kept as an option even when it is
        not in the folder, so pointing the asset folder somewhere else does not
        silently rewrite the corridor's appearance on the next Save.
        """
        folder = self.model_path_edit.text().strip()
        found: List[str] = []
        if folder and os.path.isdir(folder):
            for name in sorted(os.listdir(folder)):
                if os.path.splitext(name)[1].lower() in _IMAGE_SUFFIXES:
                    found.append(os.path.join(folder, name))

        for key, box in self.texture_pickers.items():
            current = str(corridor.get(key, "") or "")
            options = list(found)
            if current and current not in options:
                options.insert(0, current)
            self._loading = True
            try:
                box.clear()
                for path in options:
                    icon = QIcon(path) if os.path.isfile(path) else QIcon()
                    box.addItem(icon, os.path.basename(path), userData=path)
                if current:
                    box.setCurrentIndex(max(0, options.index(current)))
            finally:
                self._loading = False

    def _pick_fog_color(self) -> None:
        colour = QColorDialog.getColor(
            QColor.fromRgbF(*self._fog_rgb), self, TEXT["fog_color"])
        if colour.isValid():
            self._fog_rgb = (colour.redF(), colour.greenF(), colour.blueF())
            self._paint_fog_button()

    def _paint_fog_button(self) -> None:
        r, g, b = (int(round(c * 255)) for c in self._fog_rgb)
        self.fog_color_btn.setText(f"{r}, {g}, {b}")
        self.fog_color_btn.setStyleSheet(f"background-color: rgb({r},{g},{b});")

    def _build_actions(self) -> QHBoxLayout:
        self.save_btn = QPushButton(TEXT["save"])
        self.save_btn.setToolTip(TEXT["tip_save"])
        self.reload_btn = QPushButton(TEXT["reload"])
        self.reload_btn.setToolTip(TEXT["tip_reload"])
        self.open_json_btn = QPushButton(TEXT["open"])
        self.open_json_btn.setIcon(self._icon(QStyle.StandardPixmap.SP_FileIcon))
        self.open_json_btn.setToolTip(TEXT["tip_open"])
        self.reveal_json_btn = QPushButton(TEXT["reveal"])
        self.reveal_json_btn.setIcon(self._icon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.reveal_json_btn.setToolTip(TEXT["tip_reveal"])
        self.save_btn.clicked.connect(self._save)
        self.reload_btn.clicked.connect(self._reload)
        self.open_json_btn.clicked.connect(lambda: self._json_action(open_it=True))
        self.reveal_json_btn.clicked.connect(lambda: self._json_action(open_it=False))
        return self._button_row(
            self.save_btn, self.reload_btn, self.open_json_btn, self.reveal_json_btn)

    # ─── Small helpers ──────────────────────────────────────────────────
    def _icon(self, sp: QStyle.StandardPixmap):
        """Themed standard icon, matching the Setup tab's button styling."""
        return self.style().standardIcon(sp)

    def _arrow_button(self, sp: QStyle.StandardPixmap, tooltip: str) -> QPushButton:
        button = QPushButton()
        button.setFixedWidth(30)
        button.setIcon(self._icon(sp))
        button.setToolTip(tooltip)
        return button

    @staticmethod
    def _button_row(*buttons: QPushButton) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for button in buttons:
            row.addWidget(button)
        row.addStretch(1)
        return row

    @staticmethod
    def _add_row(form: QFormLayout, label: str, field: QWidget) -> QLabel:
        """Add a form row and hand back its label, so the pair can be hidden."""
        tag = QLabel(label)
        form.addRow(tag, field)
        return tag

    @staticmethod
    def _show_row(label: QLabel, field: QWidget, visible: bool) -> None:
        label.setVisible(visible)
        field.setVisible(visible)

    def cleanup(self) -> None:
        """Disconnect from the shared Procedure's events before destruction."""
        if self._panel is not None:
            self._panel.cleanup()
        self._binds.close()

    def closeEvent(self, event):  # noqa: N802 - Qt naming
        self.cleanup()
        super().closeEvent(event)

    def _set_editable(self, on: bool) -> None:
        for w in self._editors:
            try:
                w.setEnabled(on)
            except RuntimeError:
                # Underlying C++ widget already deleted (a stale handler firing
                # during teardown before cleanup()). Ignore -- nothing to lock.
                pass

    # ─── Conditions: model <-> form ─────────────────────────────────────
    @property
    def _cond_index(self) -> int:
        return self.cond_list.currentRow()

    def _refresh_condition_list(self, select: Optional[int] = None) -> None:
        """Rebuild the picker from the model, keeping (or setting) the selection."""
        keep = self._cond_index if select is None else select
        self._loading = True
        try:
            self.cond_list.clear()
            for cond in self._conditions:
                self.cond_list.addItem(condition_summary(cond))
            if self._conditions:
                self.cond_list.setCurrentRow(max(0, min(keep, len(self._conditions) - 1)))
        finally:
            self._loading = False
        self._load_condition()
        self._refresh_trial_picker()

    def _on_condition_selected(self, _row: int) -> None:
        if not self._loading:
            self._load_condition()

    def _load_condition(self) -> None:
        """Populate the detail form from the selected condition."""
        idx = self._cond_index
        has = 0 <= idx < len(self._conditions)
        self._cond_detail.setVisible(has)
        if not has:
            return
        cond = self._conditions[idx]
        self._loading = True
        try:
            self.cond_label.setText(str(cond.get("label", "")))
            ttype = str(cond.get("transform_type", "identity"))
            if self.cond_transform.findText(ttype) < 0:
                self.cond_transform.addItem(ttype)
            self.cond_transform.setCurrentIndex(self.cond_transform.findText(ttype))
            self._rebuild_param_rows(ttype, cond.get("transform_params") or {})

            end = str(cond.get("trial_end_condition", "duration"))
            if self.cond_end.findText(end) < 0:
                self.cond_end.addItem(end)
            self.cond_end.setCurrentIndex(self.cond_end.findText(end))
            self.cond_duration.setValue(float(cond.get("trial_duration") or 30.0))
            self.cond_distance.setValue(float(cond.get("trial_distance") or 50.0))
            self.cond_expected.setValue(float(cond.get("expected_duration") or 0.0))
            self.cond_iti.setChecked(bool(cond.get("iti_after", True)))
            self._apply_end_rule_visibility(end)
        finally:
            self._loading = False

    def _rebuild_param_rows(self, ttype: str, values: Dict[str, Any]) -> None:
        """Replace the transform's parameter rows with ones for *ttype*.

        Each parameter gets its own labelled spin box seeded with MousePortal's
        default, so the names and sensible starting values are on screen instead
        of being something the user has to know and type.
        """
        for editor in self._param_editors.values():
            self._cond_form.removeRow(editor)
        self._param_editors.clear()

        self.cond_transform.setToolTip(TEXT["transform_help"].get(ttype, ""))
        # Looked up rather than cached: inserting and removing rows shifts
        # every index below them.
        anchor = self._cond_form.getWidgetPosition(self.cond_transform)[0] + 1
        for name, default in TRANSFORM_PARAMS.get(ttype, {}).items():
            spin = QDoubleSpinBox()
            spin.setRange(-1_000_000.0, 1_000_000.0)
            spin.setDecimals(3)
            spin.setValue(float(values.get(name, default)))
            spin.valueChanged.connect(self._commit_condition)
            self._cond_form.insertRow(anchor, name, spin)
            self._param_editors[name] = spin
            anchor += 1

    def _on_transform_changed(self, ttype: str) -> None:
        if self._loading:
            return
        if not 0 <= self._cond_index < len(self._conditions):
            return
        # Parameters belong to the transform, so switching transform starts
        # from the new one's defaults rather than carrying over names it does
        # not accept -- which MousePortal would reject at load.
        self._loading = True
        try:
            self._rebuild_param_rows(ttype, {})
        finally:
            self._loading = False
        self._commit_condition()

    def _on_end_rule_changed(self, end: str) -> None:
        self._apply_end_rule_visibility(end)
        if not self._loading:
            self._commit_condition()

    def _apply_end_rule_visibility(self, end: str) -> None:
        """Show only the limit the selected rule actually uses."""
        self._show_row(self._row_duration, self.cond_duration, end == "duration")
        self._show_row(self._row_distance, self.cond_distance, end == "distance")
        # A trial whose length is not fixed needs an explicit estimate before
        # the recording length can mean anything.
        self._show_row(self._row_expected, self.cond_expected,
                       end in UNTIMED_END_CONDITIONS)
        self.cond_end.setToolTip(TEXT["end_help"].get(end, ""))

    def _commit_condition(self) -> None:
        """Write the detail form back into the selected condition."""
        if self._loading:
            return
        idx = self._cond_index
        if not 0 <= idx < len(self._conditions):
            return
        cond = self._conditions[idx]
        old_label = cond.get("label", "")
        new_label = self.cond_label.text().strip()

        cond["label"] = new_label
        cond["transform_type"] = self.cond_transform.currentText()
        params = {n: w.value() for n, w in self._param_editors.items()}
        if params:
            cond["transform_params"] = params
        else:
            cond.pop("transform_params", None)

        end = self.cond_end.currentText()
        cond["trial_end_condition"] = end
        cond.pop("trial_duration", None)
        cond.pop("trial_distance", None)
        cond.pop("expected_duration", None)
        if end == "duration":
            cond["trial_duration"] = self.cond_duration.value()
        elif end == "distance":
            cond["trial_distance"] = self.cond_distance.value()
        if end in UNTIMED_END_CONDITIONS and self.cond_expected.value() > 0:
            cond["expected_duration"] = self.cond_expected.value()

        if self.cond_iti.isChecked():
            cond.pop("iti_after", None)
        else:
            cond["iti_after"] = False

        # A rename must follow through to the blocks that refer to this
        # condition, or renaming would silently break every block using it.
        if new_label and old_label and new_label != old_label:
            for blk in self._blocks:
                blk["sequence"] = [
                    new_label if s == old_label else s for s in blk.get("sequence", [])
                ]

        self._loading = True
        try:
            self.cond_list.item(idx).setText(condition_summary(cond))
        finally:
            self._loading = False
        self._refresh_trial_picker()
        self._refresh_block_views()
        self._refresh_totals()

    def _add_condition(self) -> None:
        existing = {c.get("label") for c in self._conditions}
        base, n = TEXT["cond_default_name"], 1
        while f"{base}{n}" in existing:
            n += 1
        self._conditions.append(new_condition(f"{base}{n}"))
        self._refresh_condition_list(select=len(self._conditions) - 1)
        self._refresh_totals()

    def _remove_condition(self) -> None:
        idx = self._cond_index
        if not 0 <= idx < len(self._conditions):
            return
        label = self._conditions[idx].get("label", "")
        used_by = blocks_using(self._blocks, label)
        if used_by:
            # Deleting out from under a block would leave a dangling reference
            # that only surfaces at Save. Say so first, and remove the trials
            # too if the user goes ahead.
            answer = QMessageBox.question(
                self, TEXT["dlg_in_use_title"],
                TEXT["dlg_in_use_body"].format(label=label, blocks=", ".join(used_by)),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            for blk in self._blocks:
                blk["sequence"] = [s for s in blk.get("sequence", []) if s != label]
        del self._conditions[idx]
        self._refresh_condition_list(select=min(idx, len(self._conditions) - 1))
        self._refresh_block_views()
        self._refresh_totals()

    def _move_condition(self, delta: int) -> None:
        idx = self._cond_index
        target = idx + delta
        if not (0 <= idx < len(self._conditions) and 0 <= target < len(self._conditions)):
            return
        self._conditions[idx], self._conditions[target] = (
            self._conditions[target], self._conditions[idx]
        )
        self._refresh_condition_list(select=target)

    # ─── Blocks: model <-> form ─────────────────────────────────────────
    @property
    def _blk_index(self) -> int:
        return self.block_list.currentRow()

    def _refresh_block_list(self, select: Optional[int] = None) -> None:
        keep = self._blk_index if select is None else select
        self._loading = True
        try:
            self.block_list.clear()
            for i, blk in enumerate(self._blocks):
                self.block_list.addItem(block_summary(blk, i))
            if self._blocks:
                self.block_list.setCurrentRow(max(0, min(keep, len(self._blocks) - 1)))
        finally:
            self._loading = False
        self._load_block()

    def _refresh_block_views(self) -> None:
        """Re-render the block list rows and the plan without changing selection."""
        self._loading = True
        try:
            for i, blk in enumerate(self._blocks):
                item = self.block_list.item(i)
                if item is not None:
                    item.setText(block_summary(blk, i))
        finally:
            self._loading = False
        self._load_sequence()

    def _on_block_selected(self, _row: int) -> None:
        if not self._loading:
            self._load_block()

    def _load_block(self) -> None:
        idx = self._blk_index
        has = 0 <= idx < len(self._blocks)
        self._blk_detail.setVisible(has)
        if not has:
            return
        blk = self._blocks[idx]
        self._loading = True
        try:
            self.block_name.setText(str(blk.get("name", "")))
            self.block_repeat.setValue(int(blk.get("repeat", 1) or 1))
            order = str(blk.get("order", "fixed"))
            if self.block_order.findText(order) < 0:
                self.block_order.addItem(order)
            self.block_order.setCurrentIndex(self.block_order.findText(order))
        finally:
            self._loading = False
        self._load_sequence()

    def _load_sequence(self) -> None:
        """Render the selected block's trial list."""
        idx = self._blk_index
        if not 0 <= idx < len(self._blocks):
            self.seq_list.clear()
            return
        blk = self._blocks[idx]
        keep = self.seq_list.currentRow()
        self._loading = True
        try:
            self.seq_list.clear()
            for i, label in enumerate(blk.get("sequence") or [], start=1):
                self.seq_list.addItem(TEXT["seq_row"].format(n=i, label=label))
            if self.seq_list.count():
                self.seq_list.setCurrentRow(min(max(keep, 0), self.seq_list.count() - 1))
        finally:
            self._loading = False
        self._refresh_totals()

    def _refresh_trial_picker(self) -> None:
        """Repopulate the add-trial drop-down from the conditions that exist."""
        current = self.seq_picker.currentText()
        self._loading = True
        try:
            self.seq_picker.clear()
            self.seq_picker.addItems(
                [c["label"] for c in self._conditions if c.get("label")]
            )
            if current and self.seq_picker.findText(current) >= 0:
                self.seq_picker.setCurrentIndex(self.seq_picker.findText(current))
        finally:
            self._loading = False
        # Nothing to add from means nothing to add.
        self.seq_add_btn.setEnabled(self.seq_picker.count() > 0)

    def _commit_block(self) -> None:
        if self._loading:
            return
        idx = self._blk_index
        if not 0 <= idx < len(self._blocks):
            return
        blk = self._blocks[idx]
        blk["name"] = self.block_name.text().strip()
        blk["repeat"] = self.block_repeat.value()
        blk["order"] = self.block_order.currentText()
        self._refresh_block_views()
        self._refresh_totals()

    def _append_trial(self) -> None:
        idx = self._blk_index
        label = self.seq_picker.currentText()
        if not 0 <= idx < len(self._blocks) or not label:
            return
        self._blocks[idx].setdefault("sequence", []).append(label)
        self._refresh_block_views()
        self.seq_list.setCurrentRow(self.seq_list.count() - 1)

    def _remove_trial(self) -> None:
        idx, row = self._blk_index, self.seq_list.currentRow()
        if not 0 <= idx < len(self._blocks):
            return
        sequence = self._blocks[idx].get("sequence") or []
        if not 0 <= row < len(sequence):
            return
        del sequence[row]
        self._refresh_block_views()
        if self.seq_list.count():
            self.seq_list.setCurrentRow(min(row, self.seq_list.count() - 1))

    def _move_trial(self, delta: int) -> None:
        idx, row = self._blk_index, self.seq_list.currentRow()
        if not 0 <= idx < len(self._blocks):
            return
        sequence = self._blocks[idx].get("sequence") or []
        target = row + delta
        if not (0 <= row < len(sequence) and 0 <= target < len(sequence)):
            return
        sequence[row], sequence[target] = sequence[target], sequence[row]
        self._refresh_block_views()
        self.seq_list.setCurrentRow(target)

    def _add_block(self) -> None:
        self._blocks.append(new_block(self._unique_block_name(TEXT["blk_default_name"])))
        self._refresh_block_list(select=len(self._blocks) - 1)
        self._refresh_totals()

    def _remove_block(self) -> None:
        idx = self._blk_index
        if not 0 <= idx < len(self._blocks):
            return
        del self._blocks[idx]
        self._refresh_block_list(select=min(idx, len(self._blocks) - 1))
        self._refresh_totals()

    def _duplicate_block(self) -> None:
        """Copy the selected block in below itself.

        Block names identify blocks in the data and must stay unique, so the
        copy is suffixed rather than shipped as a duplicate Save would reject.
        """
        idx = self._blk_index
        if not 0 <= idx < len(self._blocks):
            return
        clone = copy.deepcopy(self._blocks[idx])
        if clone.get("name"):
            clone["name"] = self._unique_block_name(
                TEXT["blk_copy_suffix"].format(name=clone["name"])
            )
        self._blocks.insert(idx + 1, clone)
        self._refresh_block_list(select=idx + 1)
        self._refresh_totals()

    def _move_block(self, delta: int) -> None:
        idx = self._blk_index
        target = idx + delta
        if not (0 <= idx < len(self._blocks) and 0 <= target < len(self._blocks)):
            return
        self._blocks[idx], self._blocks[target] = self._blocks[target], self._blocks[idx]
        self._refresh_block_list(select=target)
        self._refresh_totals()

    def _unique_block_name(self, base: str) -> str:
        existing = {b.get("name") for b in self._blocks if b.get("name")}
        if base not in existing:
            return base
        n = 2
        while f"{base}{n}" in existing:
            n += 1
        return f"{base}{n}"

    # ─── Totals ─────────────────────────────────────────────────────────
    def _tail_seconds(self) -> float:
        return float(getattr(self._device, "tail_seconds", 5.0) or 0.0)

    def _total_seconds(self) -> float:
        """The session's length for the configured seed, in seconds."""
        return resolve_session(self._collect_experiment())[0]

    def _refresh_totals(self) -> None:
        """Re-render the plan, and the length on the MousePortal indicator."""
        if self._loading or not hasattr(self, "plan_label"):
            return
        experiment = self._collect_experiment()
        total, _unknown, _itis = resolve_session(experiment)
        tail = self._tail_seconds()
        self.plan_label.setText(describe_plan(experiment))
        self.override_duration.setToolTip(
            TEXT["tip_override"].format(seconds=math.ceil(total + tail)))
        # The indicator is the single place the length is reported. It reads
        # the saved config, so show it the edits in progress.
        if self._panel is not None:
            self._panel.refresh_summary(self._collect_block())

    # ─── Misc actions ───────────────────────────────────────────────────
    def _on_iti_random(self, on: bool) -> None:
        """Show either the fixed pause or the range it is drawn from, never both."""
        self.iti_duration.setVisible(not on)
        for w in (self.iti_min, self.iti_dash, self.iti_max):
            w.setVisible(on)
        self._refresh_totals()

    def _browse_model_path(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, TEXT["assets_dialog"], self.model_path_edit.text().strip())
        if path:
            self.model_path_edit.setText(path)

    def _json_action(self, *, open_it: bool) -> None:
        """Open the experiment.json, or show it selected in the file manager."""
        from mesofield.gui.config_wizard import _open_in_default_app

        path = getattr(self.config, "_json_file_path", "")
        if not path or not os.path.isfile(path):
            QMessageBox.information(
                self, TEXT["dlg_no_json_title"], TEXT["dlg_no_json_body"])
            return
        if open_it:
            _open_in_default_app(path)
        else:
            _reveal_in_file_manager(path)

    # ─── Load / save ────────────────────────────────────────────────────
    def _reload(self) -> None:
        block = self.config.mouseportal
        exp = (block or {}).get("experiment", {}) or {}

        # Deep-copied so editing never mutates the live config before Save.
        self._conditions = [
            copy.deepcopy(c) for c in exp.get("conditions", []) or [] if isinstance(c, dict)
        ]
        self._blocks = [
            copy.deepcopy(b) for b in exp.get("blocks", []) or [] if isinstance(b, dict)
        ]

        self._loading = True
        try:
            task = (block or {}).get("task", "")
            self.task_edit.setText("" if task is None else str(task))
            iti_range = exp.get("iti_range") or []
            self.iti_duration.setValue(float(exp.get("iti_duration", 0.0) or 0.0))
            self.iti_min.setValue(float(iti_range[0]) if iti_range else 0.0)
            self.iti_max.setValue(float(iti_range[1]) if iti_range else 0.0)
            self.iti_random.setChecked(bool(iti_range))
            self.random_seed.setValue(int(exp.get("random_seed") or 0))

            window = dict(_DEFAULT_WINDOW)
            window.update((block or {}).get("window") or {})
            self.win_width.setValue(int(window["width"]))
            self.win_height.setValue(int(window["height"]))
            self.win_origin_x.setValue(int(window["origin_x"]))
            self.win_origin_y.setValue(int(window["origin_y"]))

            camera = dict(_DEFAULT_CAMERA)
            camera.update((block or {}).get("camera") or {})
            self.camera_height.setValue(float(camera["height"]))
            self.speed_scaling.setValue(float(camera["speed_scaling"]))

            fog = dict(_DEFAULT_FOG)
            fog.update((block or {}).get("fog") or {})
            self.fog_density.setValue(float(fog["density"]))
            self._fog_rgb = tuple(float(c) for c in fog["color"])[:3]
            self._paint_fog_button()

            assets = (block or {}).get("assets") or {}
            self.model_path_edit.setText(str(assets.get("model_path", "") or ""))
        finally:
            self._loading = False

        self._refresh_texture_pickers((block or {}).get("corridor") or {})

        self._on_iti_random(self.iti_random.isChecked())
        self._refresh_condition_list(select=0)
        self._refresh_block_list(select=0)
        self._refresh_totals()

    def _collect_experiment(self) -> Dict[str, Any]:
        """Assemble the experiment section from the model plus the session form."""
        experiment: Dict[str, Any] = {
            "iti_duration": self.iti_duration.value(),
            "conditions": copy.deepcopy(self._conditions),
            "blocks": copy.deepcopy(self._blocks),
        }
        if self.iti_random.isChecked():
            experiment["iti_range"] = [self.iti_min.value(), self.iti_max.value()]
        # 0 is the "today's date" sentinel: leaving the seed out lets
        # MousePortal fall back to YYYYMMDD and record what it used.
        if self.random_seed.value():
            experiment["random_seed"] = self.random_seed.value()
        return experiment

    def _collect_block(self) -> Dict[str, Any]:
        block = dict(self.config.mouseportal)  # preserve corridor/fog/etc.
        block["window"] = {
            "width": self.win_width.value(),
            "height": self.win_height.value(),
            "origin_x": self.win_origin_x.value(),
            "origin_y": self.win_origin_y.value(),
        }
        # keyboard_speed is carried through rather than edited: it exists to
        # drive the corridor from the arrow keys when testing without a
        # treadmill, and is not a parameter of the experiment.
        camera = dict((block.get("camera") or {}))
        camera["height"] = self.camera_height.value()
        camera["speed_scaling"] = self.speed_scaling.value()
        block["camera"] = camera

        block["fog"] = {
            "density": self.fog_density.value(),
            "color": list(self._fog_rgb),
        }
        # Only surfaces the picker actually resolved are written; an empty
        # picker leaves whatever the corridor already names alone.
        corridor = dict((block.get("corridor") or {}))
        for key, picker in self.texture_pickers.items():
            path = picker.currentData()
            if path:
                corridor[key] = path
        if corridor:
            block["corridor"] = corridor

        model_path = self.model_path_edit.text().strip()
        if model_path:
            block["assets"] = {"model_path": model_path}
        else:
            block.pop("assets", None)
        task = self.task_edit.text().strip()
        if task:
            block["task"] = task
        else:
            block.pop("task", None)
        block["experiment"] = self._collect_experiment()
        return block

    def _save(self) -> None:
        block = self._collect_block()
        errors = validate_block(block)
        if errors:
            QMessageBox.warning(
                self, TEXT["dlg_invalid_title"],
                TEXT["dlg_invalid_body"].format(errors="\n- ".join(errors)),
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
            QMessageBox.critical(self, TEXT["dlg_save_failed"], str(exc))
            return
        if self._panel is not None:
            self._panel.refresh_summary()
        self.save_btn.setToolTip(TEXT["saved_at"].format(
            time=datetime.now().strftime("%H:%M:%S")))
        self.tasksChanged.emit()
