import os

import pytest

# Needs a real MicroManager install + system configuration. Opt in with
# MESOFIELD_HARDWARE_TESTS=1 (see `make test-all`).
pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        not os.environ.get("MESOFIELD_HARDWARE_TESTS"),
        reason="hardware test; set MESOFIELD_HARDWARE_TESTS=1 to run",
    ),
]

from pymmcore_plus import CMMCorePlus
import useq
        

def test_mmcore_mda():
    import time
    
    core = CMMCorePlus()
    core.loadSystemConfiguration()
    
    core.startContinuousSequenceAcquisition()
    time.sleep(1)
    img, get_metadata = core.getLastImageAndMD()
    time.sleep(1)
    img, pop_metadata = core.popNextImageAndMD()
    core.stopSequenceAcquisition()
    
    print(f"getLastImageAndMD Metadata object: {get_metadata}")
    print(F"popNextImageAndMD Metadata object: {pop_metadata}")


