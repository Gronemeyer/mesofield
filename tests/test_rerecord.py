"""Regression test for the multi-record cleanup bug.

`Procedure._cleanup_procedure` latches `_cleanup_started` so the duration timer
and the primary's `finished` signal only tear down once. That flag used to be
set only in `__init__` and never reset, so a *second* `run()` on the same
Procedure short-circuited cleanup at the guard: `stop_all()` never fired,
non-primary capture threads hung, and writers stayed unflushed until process
exit. `run()` now resets `_cleanup_started` / `_finished_event` each time.

This drives the demo procedure (mock primary camera + wheel) twice on the same
Procedure instance and asserts both runs complete. Before the fix, the second
`run_until_finished` returns False (cleanup no-ops, `_finished_event` never set).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from mesofield.base import load_procedure

DEMO_DIR = Path(__file__).resolve().parents[1] / "experiments" / "pipeline_demo"


@pytest.mark.skipif(
    not (DEMO_DIR / "experiment.json").exists(),
    reason="pipeline_demo experiment is not available",
)
def test_procedure_reruns_cleanly_on_same_instance(tmp_path: Path) -> None:
    shutil.copytree(DEMO_DIR, tmp_path, dirs_exist_ok=True)
    proc = load_procedure(str(tmp_path / "experiment.json"))
    duration = float(proc.config.get("duration", 2))

    # Run 1 — establishes the latched _cleanup_started=True.
    assert proc.run_until_finished(timeout=duration + 5.0), "first run did not finish"
    assert proc._cleanup_started is True

    # Run 2 on the SAME proc — the regression. Without the run()-time reset,
    # cleanup short-circuits at the guard and this times out -> False.
    assert proc.run_until_finished(timeout=duration + 5.0), (
        "second run did not finish — _cleanup_procedure short-circuited; "
        "Procedure.run() must reset _cleanup_started/_finished_event"
    )
