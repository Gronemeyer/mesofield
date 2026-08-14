"""MousePortal config editing -- validation, duration maths, and GUI round-trip.

The headless half exercises :mod:`mesofield.gui.mouseportal_config` directly.
The GUI half builds the real :class:`MousePortalController` offscreen and
proves that loading a config into the tab and collecting it straight back
returns the same design -- the property that stops the editor from quietly
rewriting a config the user only opened to look at.
"""

from __future__ import annotations

import json

import pytest

import mesofield.devices.mocks  # noqa: F401  (registers mock device types)
from mesofield.gui.mouseportal_config import (
    block_trials, blocks_using, default_seed, describe_plan, planning_duration,
    total_duration, validate_block,
)


# A design exercising every field the tab can edit: multi-parameter transforms,
# each end rule, repeats, chaining, and a shuffled block.
FULL_EXPERIMENT = {
    "iti_duration": 2.0,
    "random_seed": 20260813,
    "conditions": [
        {"label": "normal", "transform_type": "identity",
         "trial_end_condition": "distance", "trial_distance": 50.0,
         "expected_duration": 12.0},
        {"label": "gain_2x", "transform_type": "gain",
         "transform_params": {"gain": 2.0},
         "trial_end_condition": "duration", "trial_duration": 30.0},
        {"label": "limited", "transform_type": "clamp",
         "transform_params": {"lo": 0.0, "hi": 10.0},
         "trial_end_condition": "duration", "trial_duration": 20.0},
        {"label": "freeze", "transform_type": "freeze",
         "trial_end_condition": "duration", "trial_duration": 10.0},
        {"label": "chained", "transform_type": "invert",
         "trial_end_condition": "duration", "trial_duration": 5.0,
         "iti_after": False},
    ],
    "blocks": [
        {"name": "order_A", "sequence": ["normal", "gain_2x", "freeze"],
         "repeat": 3, "order": "fixed"},
        {"name": "order_B", "sequence": ["limited", "chained", "normal"],
         "repeat": 2, "order": "shuffle"},
    ],
}


def _block(experiment=None):
    return {
        "task": "corridor",
        "window": {"width": 1280, "height": 720, "origin_x": 0, "origin_y": 0},
        "camera": {"height": 2.0, "speed_scaling": 0.25, "keyboard_speed": 20.0},
        "experiment": json.loads(json.dumps(experiment or FULL_EXPERIMENT)),
    }


def _cond(label="n", **kw):
    base = {"label": label, "transform_type": "identity",
            "trial_end_condition": "duration", "trial_duration": 10.0}
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #
def test_valid_design_has_no_errors():
    assert validate_block(_block()) == []


def test_block_trial_count_is_sequence_times_repeat():
    assert len(block_trials({"sequence": ["a", "b", "c"], "repeat": 3})) == 9
    assert len(block_trials({"sequence": ["a"]})) == 1


def test_blocks_may_differ_in_length():
    """No global trials-per-block, so a session can mix block sizes."""
    counts = [len(block_trials(b)) for b in FULL_EXPERIMENT["blocks"]]
    assert counts == [9, 6]
    assert describe_plan(FULL_EXPERIMENT).endswith("= 2 block(s), 15 trials")


def test_shuffle_marked_in_plan_but_does_not_change_duration():
    """A permutation cannot change which trials a block contains."""
    fixed = json.loads(json.dumps(FULL_EXPERIMENT))
    for blk in fixed["blocks"]:
        blk["order"] = "fixed"
    assert total_duration(fixed) == total_duration(FULL_EXPERIMENT)
    assert "order drawn at run time" in describe_plan(FULL_EXPERIMENT)


def test_repeat_scales_duration_linearly():
    exp = {"iti_duration": 1.0, "conditions": [_cond("n", trial_duration=2.0)],
           "blocks": [{"sequence": ["n"], "repeat": 5}]}
    assert total_duration(exp) == (pytest.approx(15.0), 0)
    exp["blocks"][0]["repeat"] = 10
    assert total_duration(exp) == (pytest.approx(30.0), 0)


def test_iti_after_false_contributes_no_interval():
    exp = {"iti_duration": 5.0,
           "conditions": [_cond("c", trial_duration=1.0, iti_after=False)],
           "blocks": [{"sequence": ["c"], "repeat": 4}]}
    assert total_duration(exp) == (pytest.approx(4.0), 0)


def test_blocks_using_finds_references():
    assert blocks_using(FULL_EXPERIMENT["blocks"], "normal") == ["order_A", "order_B"]
    assert blocks_using(FULL_EXPERIMENT["blocks"], "freeze") == ["order_A"]
    assert blocks_using(FULL_EXPERIMENT["blocks"], "absent") == []


# --------------------------------------------------------------------------- #
# Trial length is per-condition, and unknowable lengths are reported
# --------------------------------------------------------------------------- #
def test_duration_condition_is_its_own_estimate():
    assert planning_duration(_cond(trial_duration=7.0)) == 7.0


def test_distance_condition_needs_an_explicit_estimate():
    distance = {"label": "d", "transform_type": "identity",
                "trial_end_condition": "distance", "trial_distance": 50.0}
    assert planning_duration(distance) is None
    distance["expected_duration"] = 12.0
    assert planning_duration(distance) == 12.0


def test_untimed_trials_are_counted_not_guessed():
    """A trial with no knowable length must not be costed at some default."""
    exp = {
        "iti_duration": 1.0,
        "conditions": [
            _cond("timed", trial_duration=10.0),
            {"label": "untimed", "transform_type": "identity",
             "trial_end_condition": "manual"},
        ],
        "blocks": [{"sequence": ["timed", "untimed"]}],
    }
    total, unknown = total_duration(exp)
    assert unknown == 1
    assert total == pytest.approx(12.0)  # 10s + two 1s pauses; nothing invented


def test_undefined_condition_counts_as_unknown():
    exp = {"iti_duration": 1.0, "conditions": [_cond("n", trial_duration=30.0)],
           "blocks": [{"sequence": ["n", "ghost"]}]}
    total, unknown = total_duration(exp)
    assert (total, unknown) == (pytest.approx(31.0), 1)


# --------------------------------------------------------------------------- #
# Validation -- each of these used to pass silently
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("experiment,needle", [
    ({"conditions": [_cond("normal")], "blocks": [{"sequence": ["nrmal"]}]},
     "undefined condition"),
    ({"conditions": [_cond("x", transform_type="warp")],
      "blocks": [{"sequence": ["x"]}]}, "unknown transform_type"),
    ({"conditions": [_cond("g", transform_type="gain",
                           transform_params={"gian": 2})],
      "blocks": [{"sequence": ["g"]}]}, "does not take"),
    ({"conditions": [_cond("c", transform_type="clamp",
                           transform_params={"lo": 10, "hi": 0})],
      "blocks": [{"sequence": ["c"]}]}, "lo must be"),
    ({"conditions": [{"label": "f", "transform_type": "freeze",
                      "trial_end_condition": "distance", "trial_distance": 50.0}],
      "blocks": [{"sequence": ["f"]}]}, "ignores the subject"),
    ({"conditions": [{"label": "d", "transform_type": "identity",
                      "trial_end_condition": "duration"}],
      "blocks": [{"sequence": ["d"]}]}, "needs a positive duration"),
    ({"conditions": [{"label": "d", "transform_type": "identity",
                      "trial_end_condition": "distance"}],
      "blocks": [{"sequence": ["d"]}]}, "needs a positive distance"),
    ({"conditions": [{"label": "d", "transform_type": "identity"}],
      "blocks": [{"sequence": ["d"]}]}, "trial_end_condition must be"),
    ({"conditions": [_cond("n")]}, "at least one block"),
    ({"blocks": [{"sequence": ["n"]}]}, "at least one condition"),
    ({"conditions": [_cond("n")], "blocks": [{"sequence": ["n"], "repeat": 0}]},
     "repeat must be"),
    ({"conditions": [_cond("n")], "blocks": [{"sequence": ["n"], "order": "randm"}]},
     "order must be"),
    ({"conditions": [_cond("n")],
      "blocks": [{"name": "a", "sequence": ["n"]}, {"name": "a", "sequence": ["n"]}]},
     "must be unique"),
    ({"conditions": [_cond("n"), _cond("n", transform_type="invert")],
      "blocks": [{"sequence": ["n"]}]}, "duplicate condition label"),
    ({"conditions": [_cond("n")], "blocks": [{"sequence": []}]}, "no trials"),
])
def test_validation_names_the_problem(experiment, needle):
    experiment.setdefault("iti_duration", 1.0)
    errors = validate_block({"experiment": experiment})
    assert any(needle in e for e in errors), errors


def test_default_seed_is_todays_date():
    from datetime import date
    seed = default_seed()
    assert seed == int(date.today().strftime("%Y%m%d"))
    assert 20000101 <= seed <= 21001231


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
@pytest.fixture
def make_tab(qtbot, hardware_yaml, experiment_json, tmp_path):
    """Factory: the real MousePortal tab loaded with a given config block."""
    from mesofield.base import Procedure
    from mesofield.gui.mouseportal_controller import MousePortalController

    def _make(experiment=None):
        exp_path = experiment_json()
        doc = json.loads(exp_path.read_text())
        doc["MousePortal"] = _block(experiment)
        exp_path.write_text(json.dumps(doc))
        proc = Procedure(
            hardware=str(hardware_yaml()),
            config=str(exp_path),
            experiment_directory=str(tmp_path / "out"),
        )
        tab = MousePortalController(proc)
        qtbot.addWidget(tab)
        return tab

    return _make


@pytest.fixture
def portal_tab(make_tab):
    return make_tab()


@pytest.mark.gui
def test_gui_round_trip_preserves_the_design(portal_tab):
    """Load then collect must return the same experiment.

    This is what stops merely opening the tab and saving from rewriting the
    config -- the failure mode that previously erased ``clamp`` parameters.
    """
    collected = portal_tab._collect_block()["experiment"]
    assert collected["blocks"] == FULL_EXPERIMENT["blocks"]
    assert collected["conditions"] == FULL_EXPERIMENT["conditions"]
    assert collected["iti_duration"] == FULL_EXPERIMENT["iti_duration"]
    assert collected["random_seed"] == FULL_EXPERIMENT["random_seed"]
    assert validate_block(portal_tab._collect_block()) == []


@pytest.mark.gui
def test_gui_preserves_camera_speed_scaling(portal_tab):
    assert portal_tab._collect_block()["camera"]["speed_scaling"] == pytest.approx(0.25)


@pytest.mark.gui
def test_gui_has_no_competing_global_trial_settings(portal_tab):
    """Trial length lives on conditions only -- no session-wide rivals."""
    experiment = portal_tab._collect_block()["experiment"]
    for gone in ("trial_end_condition", "trial_duration", "trial_distance",
                 "num_blocks", "trials_per_block"):
        assert gone not in experiment
    for gone in ("trial_end", "trial_duration", "trial_distance",
                 "num_blocks", "trials_per_block"):
        assert not hasattr(portal_tab, gone)


@pytest.mark.gui
def test_gui_params_are_typed_fields_named_after_the_transform(portal_tab):
    """Selecting clamp offers 'lo' and 'hi' as fields -- nothing to type."""
    portal_tab.cond_list.setCurrentRow(2)          # 'limited' (clamp)
    assert set(portal_tab._param_editors) == {"lo", "hi"}
    assert portal_tab._param_editors["hi"].value() == pytest.approx(10.0)

    portal_tab.cond_list.setCurrentRow(1)          # 'gain_2x'
    assert set(portal_tab._param_editors) == {"gain"}
    assert portal_tab._param_editors["gain"].value() == pytest.approx(2.0)

    portal_tab.cond_list.setCurrentRow(0)          # 'normal' (identity)
    assert portal_tab._param_editors == {}


@pytest.mark.gui
def test_gui_switching_transform_offers_that_transforms_parameters(portal_tab):
    portal_tab.cond_list.setCurrentRow(0)
    portal_tab.cond_transform.setCurrentText("delay")
    assert set(portal_tab._param_editors) == {"delay_sec"}
    cond = portal_tab._collect_block()["experiment"]["conditions"][0]
    assert cond["transform_type"] == "delay"
    # Parameters of the previous transform must not linger: MousePortal
    # rejects a transform handed a parameter it does not accept.
    assert set(cond["transform_params"]) == {"delay_sec"}
    assert validate_block(portal_tab._collect_block()) == []


@pytest.mark.gui
def test_gui_end_rule_shows_only_the_field_it_uses(portal_tab):
    portal_tab.cond_list.setCurrentRow(1)          # duration-ended
    assert portal_tab.cond_duration.isVisibleTo(portal_tab)
    assert not portal_tab.cond_distance.isVisibleTo(portal_tab)

    portal_tab.cond_end.setCurrentText("distance")
    assert portal_tab.cond_distance.isVisibleTo(portal_tab)
    assert not portal_tab.cond_duration.isVisibleTo(portal_tab)
    # A distance trial has no knowable length, so the estimate field appears.
    assert portal_tab.cond_expected.isVisibleTo(portal_tab)

    cond = portal_tab._collect_block()["experiment"]["conditions"][1]
    assert cond["trial_end_condition"] == "distance"
    assert "trial_duration" not in cond


@pytest.mark.gui
def test_gui_trial_order_is_picked_not_typed(portal_tab):
    """The add-trial picker offers exactly the conditions that exist."""
    labels = [portal_tab.seq_picker.itemText(i)
              for i in range(portal_tab.seq_picker.count())]
    assert labels == [c["label"] for c in FULL_EXPERIMENT["conditions"]]

    portal_tab.block_list.setCurrentRow(0)
    portal_tab.seq_picker.setCurrentText("freeze")
    portal_tab._append_trial()
    seq = portal_tab._collect_block()["experiment"]["blocks"][0]["sequence"]
    assert seq == ["normal", "gain_2x", "freeze", "freeze"]
    assert validate_block(portal_tab._collect_block()) == []


@pytest.mark.gui
def test_gui_trial_reorder_and_remove(portal_tab):
    portal_tab.block_list.setCurrentRow(0)
    portal_tab.seq_list.setCurrentRow(0)
    portal_tab._move_trial(1)
    assert portal_tab._blocks[0]["sequence"] == ["gain_2x", "normal", "freeze"]
    portal_tab.seq_list.setCurrentRow(2)
    portal_tab._remove_trial()
    assert portal_tab._blocks[0]["sequence"] == ["gain_2x", "normal"]


@pytest.mark.gui
def test_gui_renaming_a_condition_updates_the_blocks_using_it(portal_tab):
    """Renaming must not silently break every block that referenced it."""
    portal_tab.cond_list.setCurrentRow(0)          # 'normal'
    portal_tab.cond_label.setText("baseline")
    portal_tab._commit_condition()

    experiment = portal_tab._collect_block()["experiment"]
    assert experiment["blocks"][0]["sequence"] == ["baseline", "gain_2x", "freeze"]
    assert experiment["blocks"][1]["sequence"] == ["limited", "chained", "baseline"]
    assert validate_block({"experiment": experiment}) == []


@pytest.mark.gui
def test_gui_duplicate_block_keeps_names_unique(portal_tab):
    portal_tab.block_list.setCurrentRow(0)
    portal_tab._duplicate_block()
    names = [b.get("name") for b in portal_tab._collect_block()["experiment"]["blocks"]]
    assert names == ["order_A", "order_A_copy", "order_B"]
    assert validate_block(portal_tab._collect_block()) == []


@pytest.mark.gui
def test_gui_add_block_then_build_it(portal_tab):
    portal_tab._add_block()
    portal_tab.seq_picker.setCurrentText("normal")
    portal_tab._append_trial()
    portal_tab._append_trial()
    portal_tab.block_repeat.setValue(4)
    blocks = portal_tab._collect_block()["experiment"]["blocks"]
    assert len(blocks) == 3
    assert len(block_trials(blocks[2])) == 8
    assert validate_block(portal_tab._collect_block()) == []


@pytest.mark.gui
def test_gui_new_condition_is_valid_immediately(portal_tab):
    """Adding a condition must not create a config that fails to save."""
    portal_tab._add_condition()
    cond = portal_tab._collect_block()["experiment"]["conditions"][-1]
    assert cond["trial_end_condition"] == "duration"
    assert cond["trial_duration"] > 0
    assert validate_block(portal_tab._collect_block()) == []


@pytest.mark.gui
def test_gui_seed_zero_means_todays_date(make_tab):
    """The date the auto seed resolves to is shown in the field itself."""
    tab = make_tab()
    tab.random_seed.setValue(0)
    assert "random_seed" not in tab._collect_block()["experiment"]
    assert str(default_seed()) in tab.random_seed.specialValueText()


@pytest.mark.gui
def test_gui_carries_keyboard_speed_through_untouched(portal_tab):
    """Not editable in the tab, but a round-trip must not drop it."""
    assert not hasattr(portal_tab, "keyboard_speed")
    camera = portal_tab._collect_block()["camera"]
    assert camera["keyboard_speed"] == pytest.approx(20.0)
    assert camera["speed_scaling"] == pytest.approx(0.25)


@pytest.mark.gui
def test_gui_display_strings_all_come_from_the_text_table(portal_tab):
    """Section headers and buttons render the TEXT entries, not inline literals."""
    from mesofield.gui.mouseportal_controller import TEXT

    assert portal_tab._cond_box.title() == TEXT["sec_conditions"]
    assert portal_tab._blk_box.title() == TEXT["sec_blocks"]
    assert portal_tab._session_box.title() == TEXT["sec_session"]
    assert portal_tab._display_box.title() == TEXT["sec_display"]
    assert portal_tab.save_btn.text() == TEXT["save"]
    assert portal_tab.add_cond_btn.text() == TEXT["add"]


@pytest.mark.gui
def test_gui_has_one_caption_and_no_em_dashes(portal_tab):
    """Explanations belong in tooltips; the plan preview is the only caption."""
    from PyQt6.QtWidgets import QGroupBox, QLabel

    from mesofield.gui.mouseportal_controller import TEXT

    # Every visible string the tab renders, excluding tooltips.
    shown = [w.text() for w in portal_tab.findChildren(QLabel) if w.text()]
    shown += [w.title() for w in portal_tab.findChildren(QGroupBox) if w.title()]
    offenders = [s for s in shown if "—" in s or "–" in s]
    assert offenders == [], offenders

    # Wordy captions were folded into tooltips; the plan is data, so it stays.
    captions = [s for s in shown if len(s) > 60 and s != portal_tab.plan_label.text()]
    assert captions == [], captions
    assert portal_tab.plan_label.toolTip() == TEXT["tip_plan"]


@pytest.mark.gui
def test_gui_editing_does_not_mutate_the_live_config_before_save(portal_tab):
    """The tab edits a copy; nothing lands until Save."""
    portal_tab.cond_list.setCurrentRow(0)
    portal_tab.cond_label.setText("scribble")
    portal_tab._commit_condition()
    live = portal_tab.config.mouseportal["experiment"]["conditions"][0]["label"]
    assert live == "normal"
