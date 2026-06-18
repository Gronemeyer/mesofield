"""Real pymmcore-plus continuous-acquisition smoke test on the demo config.

Uses the MicroManager **demo** system configuration (no physical hardware), so
it is an integration test rather than a hardware one. Skips cleanly when the MM
demo adapters are not installed.
"""

from __future__ import annotations

import pytest

from pymmcore_plus import CMMCorePlus


pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture
def demo_core():
    core = CMMCorePlus()
    try:
        core.loadSystemConfiguration()  # bundled MM demo config
    except Exception as exc:  # depends on a local MicroManager install
        pytest.skip(f"MicroManager demo config unavailable: {exc}")
    try:
        yield core
    finally:
        try:
            core.stopSequenceAcquisition()
        except Exception:
            pass


def test_mmcore_continuous_acquisition(demo_core, wait_until):
    """A continuous sequence yields retrievable images + metadata."""
    core = demo_core
    core.startContinuousSequenceAcquisition()
    try:
        wait_until(
            lambda: core.getRemainingImageCount() > 0,
            timeout=5.0,
            message="demo camera produced no images",
        )
        img, get_metadata = core.getLastImageAndMD()
        assert img is not None
        assert get_metadata is not None

        wait_until(lambda: core.getRemainingImageCount() > 0, timeout=5.0)
        img2, pop_metadata = core.popNextImageAndMD()
        assert img2 is not None
        assert pop_metadata is not None
    finally:
        core.stopSequenceAcquisition()
