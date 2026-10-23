"""Long acquisitions start at once instead of building an MDAEvent per frame first."""

from __future__ import annotations

import time

import useq
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.core._sequencing import SequencedEvent

from mesofield.engines import DevEngine


def test_a_two_hour_sequence_starts_as_one_burst_without_an_event_per_frame():
    engine = DevEngine(CMMCorePlus())
    sequence = useq.MDASequence(time_plan={"interval": 0, "loops": 360_000})

    started = time.perf_counter()
    (burst,) = engine.event_iterator(sequence)
    assert time.perf_counter() - started < 1  # validating every frame's event took ~15 s
    assert isinstance(burst, SequencedEvent) and len(burst.events) == 360_000
    assert burst.events[123_456].index["t"] == 123_456
