"""Regression test for the multi-record cleanup bug.

`cleanup` runs once per run, so it no-ops once the run is DONE. That state used
to be latched in `__init__` and never reset, so a *second* `run()` on the same
Procedure short-circuited teardown: `stop_all()` never fired, non-primary
capture threads hung, and writers stayed unflushed until process exit. `run()`
now resets the state each time.

This drives the demo procedure (mock primary camera + wheel) twice on the same
Procedure instance and asserts both runs complete.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from mesofield.base import RunState, load_procedure

DEMO_DIR = Path(__file__).resolve().parents[1] / "experiments" / "pipeline_demo"


@pytest.mark.skipif(
    not (DEMO_DIR / "experiment.json").exists(),
    reason="pipeline_demo experiment is not available",
)
def test_procedure_reruns_cleanly_on_same_instance(tmp_path: Path) -> None:
    shutil.copytree(DEMO_DIR, tmp_path, dirs_exist_ok=True)
    proc = load_procedure(str(tmp_path / "experiment.json"))
    duration = float(proc.config.get("duration", 2))

    # Run 1 — leaves the procedure DONE.
    assert proc.run_until_finished(timeout=duration + 5.0), "first run did not finish"
    assert proc.state is RunState.DONE

    # Run 2 on the SAME proc — the regression. Without the run()-time reset,
    # cleanup short-circuits on the DONE state and this times out -> False.
    assert proc.run_until_finished(timeout=duration + 5.0), (
        "second run did not finish — cleanup short-circuited; "
        "Procedure.run() must reset the run state"
    )
