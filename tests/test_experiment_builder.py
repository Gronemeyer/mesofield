"""Editing an existing ``experiment.json``, not just authoring a new one.

The builder could only ever write a fresh file, so changing one subject meant
retyping the experiment or hand-editing JSON. Reading a document back into the
form is necessarily lossy -- the file records the *result* of the form, not the
form -- so what :func:`experiment_doc_from_json` has to infer is what these
cover: a variable's type, whether it is shown in the app, and which tasks exist.
"""

from __future__ import annotations

import json

import pytest

from mesofield.gui.config_builder import (
    ExperimentBuilderDialog,
    build_experiment_doc,
    experiment_doc_from_json,
)


def _doc() -> dict:
    return {
        "Configuration": {
            "experimenter": "dg", "protocol": "ETOH", "duration": 600,
            "start_on_trigger": True, "task": ["baseline", "stim"],
            "experiment_dir": "F:/ETOH-VEC",
        },
        "Subjects": {
            "M1": {"session": "01", "task": "baseline", "sex": "F", "age": 12},
            "M2": {"session": "02", "task": "stim", "sex": "M", "age": 14},
        },
        "DisplayKeys": ["subject", "session", "task", "sex",
                        "experimenter", "protocol", "duration"],
    }


# --------------------------------------------------------------------------- #
# Reading a document back into the form
# --------------------------------------------------------------------------- #


def test_subjects_round_trip_through_the_form():
    doc = _doc()
    rebuilt = build_experiment_doc(*experiment_doc_from_json(doc),
                                   experiment_dir="F:/ETOH-VEC")
    assert rebuilt["Subjects"] == doc["Subjects"]


def test_session_parameters_are_read_back():
    configuration, _tasks, _variables, _subjects = experiment_doc_from_json(_doc())
    assert configuration["experimenter"] == "dg"
    assert configuration["duration"] == 600
    assert configuration["start_on_trigger"] is True


def test_a_variable_is_typed_from_its_values():
    """`age: 12` has to come back as a number, or saving turns it into "12"."""
    _c, _t, variables, _s = experiment_doc_from_json(_doc())
    kinds = {name: typ for name, typ, _show in variables}
    assert kinds == {"sex": str, "age": int}


def test_a_variable_set_to_text_anywhere_stays_text():
    doc = _doc()
    doc["Subjects"]["M2"]["age"] = "unknown"
    _c, _t, variables, _s = experiment_doc_from_json(doc)
    assert dict((n, t) for n, t, _ in variables)["age"] is str


def test_show_in_app_comes_from_display_keys():
    """`sex` is listed in DisplayKeys, `age` is not."""
    _c, _t, variables, _s = experiment_doc_from_json(_doc())
    shown = {name: show for name, _typ, show in variables}
    assert shown == {"sex": True, "age": False}

    rebuilt = build_experiment_doc(*experiment_doc_from_json(_doc()))
    assert "sex" in rebuilt["DisplayKeys"]
    assert "age" not in rebuilt["DisplayKeys"]


@pytest.mark.parametrize("task,expected", [
    (["baseline", "stim"], ["baseline", "stim"]),   # several
    ("baseline", ["baseline"]),                     # one
])
def test_tasks_are_read_from_configuration(task, expected):
    doc = _doc()
    doc["Configuration"]["task"] = task
    doc["Subjects"] = {"M1": {"session": "01", "task": expected[0]}}
    _c, tasks, _v, _s = experiment_doc_from_json(doc)
    assert tasks == expected


def test_a_task_only_a_subject_uses_is_still_collected():
    """Otherwise reopening the form would silently drop that subject's task."""
    doc = _doc()
    doc["Configuration"]["task"] = "baseline"
    doc["Subjects"]["M2"]["task"] = "washout"
    _c, tasks, _v, _s = experiment_doc_from_json(doc)
    assert tasks == ["baseline", "washout"]


def test_an_empty_document_does_not_explode():
    configuration, tasks, variables, subjects = experiment_doc_from_json({})
    assert (configuration, tasks, variables, subjects) == ({}, [], [], [])


# --------------------------------------------------------------------------- #
# The dialog
# --------------------------------------------------------------------------- #


def test_dialog_opens_on_an_existing_experiment(qapp):
    dialog = ExperimentBuilderDialog(doc=_doc(), json_path="C:/rig/experiment.json")

    assert dialog.windowTitle() == "Edit experiment.json"
    assert dialog._tasks == ["baseline", "stim"]
    assert [s["subject"] for s in dialog._subjects] == ["M1", "M2"]
    assert dialog._session_form.values()["protocol"] == "ETOH"


def test_editing_saves_back_over_the_same_file(qapp):
    """So editing a subject list does not scatter near-identical files."""
    dialog = ExperimentBuilderDialog(doc=_doc(), json_path="C:/rig/experiment.json")
    assert dialog._target_path == "C:/rig/experiment.json"


def test_authoring_a_new_one_still_asks_where(qapp):
    dialog = ExperimentBuilderDialog(default_dir="C:/rig")
    assert dialog.windowTitle() == "New experiment.json"
    assert dialog._target_path == ""
    assert [s["subject"] for s in dialog._subjects]  # seeded with one subject


def test_edit_writes_the_file_back(qapp, tmp_path):
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(_doc()), encoding="utf-8")

    dialog = ExperimentBuilderDialog(doc=json.loads(path.read_text(encoding="utf-8")),
                                     json_path=str(path))
    dialog._subjects[0]["subject"] = "M9"
    dialog._save()

    written = json.loads(path.read_text(encoding="utf-8"))
    assert "M9" in written["Subjects"]
    assert "M1" not in written["Subjects"]
    assert written["Subjects"]["M9"]["sex"] == "F"   # the rest survived
