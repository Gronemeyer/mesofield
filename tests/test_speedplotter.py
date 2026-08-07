"""Regression tests for the live serial plotter and its data routing.

These lock in the behaviours that were hand-debugged into the lick-detector
plotter so they can't silently regress:

* **Render throttling** -- in pull mode the redraw is decoupled from data
  arrival; flooding samples must *not* trigger a redraw per sample (the freeze
  that motivated the timer + ring-buffer rewrite).
* **Saturation does not freeze** -- once the ring buffer fills, its length is
  pinned at ``max_points`` forever, so a length-based "is there new data?"
  check stops firing. The plot must keep updating via the monotonic sample
  counter (this is the exact bug that froze the capacitance trace).
* **DeviceChannelSampler** -- pull-based bridge: key/callable sources, time
  rebased to ~0, bounded buffers, monotonic count in the snapshot.
* **queue_payload routing** -- the base default passes through; an override can
  drop/reshape what reaches the dataqueue without touching the device's own
  buffer or the live plots.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mesofield.devices.base import BaseDataProducer
from mesofield.gui.qt_device_adapter import DeviceChannelSampler
from mesofield.gui.speedplotter import SerialWidget

pytestmark = pytest.mark.gui


class _Producer(BaseDataProducer):
    """Minimal data producer; ``record()`` drives the psygnal data stream."""

    device_id = "probe"
    device_type = "probe"


def _make_cfg(device, attr="probe"):
    """A stand-in ``ExperimentConfig`` exposing ``cfg.hardware.<attr>``."""
    hardware = SimpleNamespace()
    setattr(hardware, attr, device)
    return SimpleNamespace(hardware=hardware)


def _count_redraws(widget):
    """Patch ``update_plot`` to count actual repaints; return the counter dict."""
    calls = {"n": 0}
    original = widget.update_plot

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    widget.update_plot = counting
    return calls


# --------------------------------------------------------------------------- #
# DeviceChannelSampler (pull-based bridge)
# --------------------------------------------------------------------------- #
def test_sampler_maps_sources_rebases_time_and_counts():
    dev = _Producer()
    sampler = DeviceChannelSampler(
        dev,
        {"a": "x", "b": lambda p: p["x"] * 2},  # key source and callable source
        max_points=100,
    )

    dev.record({"x": 5}, ts=100.0)
    dev.record({"x": 7}, ts=100.5)

    xs_a, ys_a, n_a = sampler.snapshot("a")
    xs_b, ys_b, _ = sampler.snapshot("b")

    # Time rebased to the first sample (~0), not the absolute device clock.
    assert xs_a == [0.0, 0.5]
    # Key source vs callable source.
    assert ys_a == [5.0, 7.0]
    assert ys_b == [10.0, 14.0]
    # Monotonic per-channel count.
    assert n_a == 2


def test_sampler_buffer_is_bounded_but_count_is_monotonic():
    dev = _Producer()
    sampler = DeviceChannelSampler(dev, {"v": "v"}, max_points=5)

    for i in range(20):
        dev.record({"v": i}, ts=i * 0.001)

    xs, ys, count = sampler.snapshot("v")
    assert len(xs) == 5  # ring buffer pinned at max_points
    assert ys == [15.0, 16.0, 17.0, 18.0, 19.0]  # newest 5 retained
    assert count == 20  # but the counter keeps climbing past max_points


def test_sampler_ignores_non_dict_and_missing_keys():
    dev = _Producer()
    sampler = DeviceChannelSampler(dev, {"v": "v"}, max_points=10)

    dev.record(42, ts=0.0)            # non-dict payload: ignored
    dev.record({"other": 1}, ts=0.1)  # key absent: ignored
    dev.record({"v": 3}, ts=0.2)      # counts

    xs, ys, count = sampler.snapshot("v")
    assert ys == [3.0]
    assert count == 1


# --------------------------------------------------------------------------- #
# SerialWidget -- pull mode (the high-rate / freeze path)
# --------------------------------------------------------------------------- #
def test_pull_mode_decouples_render_from_data_arrival(qtbot):
    """Flooding samples must not redraw per sample (the original freeze)."""
    dev = _Producer()
    sampler = DeviceChannelSampler(dev, {"v": "v"}, max_points=1000)
    widget = SerialWidget(
        _make_cfg(dev),
        device_attr="probe",
        signal_name="vUpdated",
        data_provider=sampler.provider("v"),
        value_scale=1.0,
        max_points=1000,
    )
    qtbot.addWidget(widget)
    widget._render_timer.stop()  # drive renders manually for determinism
    calls = _count_redraws(widget)

    for i in range(5000):
        dev.record({"v": i % 100}, ts=i * 0.001)

    # No redraw happened just from data arriving.
    assert calls["n"] == 0
    widget._render()
    assert calls["n"] == 1            # one tick -> one redraw, regardless of 5000 samples
    widget._render()
    assert calls["n"] == 1            # idle tick (no new samples) is a no-op


def test_pull_mode_keeps_updating_after_buffer_saturates(qtbot):
    """Regression: the capacitance trace froze once the ring buffer filled.

    A length-based "has new data?" guard goes stale at saturation (len is
    pinned at max_points). The monotonic count must keep redraws flowing.
    """
    dev = _Producer()
    sampler = DeviceChannelSampler(dev, {"v": "v"}, max_points=5)
    widget = SerialWidget(
        _make_cfg(dev),
        device_attr="probe",
        signal_name="vUpdated",
        data_provider=sampler.provider("v"),
        value_scale=1.0,
        max_points=5,
    )
    qtbot.addWidget(widget)
    widget._render_timer.stop()
    calls = _count_redraws(widget)

    # Saturate the buffer well past max_points.
    for i in range(20):
        dev.record({"v": i}, ts=i * 0.001)
    assert len(sampler.snapshot("v")[0]) == 5  # length is now pinned

    widget._render()
    assert calls["n"] == 1
    widget._render()
    assert calls["n"] == 1  # no new data -> no redraw

    # More data arrives AFTER saturation: the old length guard would freeze here.
    x_before = sampler.snapshot("v")[0][-1]
    for i in range(20, 25):
        dev.record({"v": i}, ts=i * 0.001)
    widget._render()
    assert calls["n"] == 2  # MUST redraw
    x_after = sampler.snapshot("v")[0][-1]
    assert x_after > x_before  # window advanced while saturated


def test_pull_mode_applies_value_scale_and_reports_latest(qtbot):
    dev = _Producer()
    sampler = DeviceChannelSampler(dev, {"v": "v"}, max_points=10)
    widget = SerialWidget(
        _make_cfg(dev),
        device_attr="probe",
        signal_name="vUpdated",
        data_provider=sampler.provider("v"),
        value_scale=0.5,
        max_points=10,
    )
    qtbot.addWidget(widget)
    widget._render_timer.stop()

    dev.record({"v": 10}, ts=0.0)
    dev.record({"v": 20}, ts=0.1)
    widget._render()

    # Plotted series is scaled; the status label reports the raw latest value.
    xs, ys = widget.data_curve.getData()
    assert list(ys) == [5.0, 10.0]
    assert widget._latest_value == 20.0


# --------------------------------------------------------------------------- #
# SerialWidget -- push mode (legacy signal-driven path, e.g. encoder)
# --------------------------------------------------------------------------- #
def test_push_mode_buffers_then_renders_once(qtbot):
    dev = _Producer()
    widget = SerialWidget(
        _make_cfg(dev),
        device_attr="probe",
        signal_name="vUpdated",  # no such signal -> _signal is None, push buffer only
        value_scale=1.0,
        max_points=100,
    )
    qtbot.addWidget(widget)
    widget._render_timer.stop()
    calls = _count_redraws(widget)

    for i in range(50):
        widget.receive_data(i * 0.001, float(i))

    assert calls["n"] == 0      # buffering does not redraw
    assert widget._dirty is True
    widget._render()
    assert calls["n"] == 1      # one repaint for the batch
    assert widget._dirty is False
    widget._render()
    assert calls["n"] == 1      # nothing new -> no repaint


def test_disconnected_widget_is_safe(qtbot):
    cfg = SimpleNamespace(hardware=SimpleNamespace())  # no matching device attr
    widget = SerialWidget(cfg, device_attr="missing", signal_name="vUpdated")
    qtbot.addWidget(widget)

    assert widget.connected is False
    # Render on an empty disconnected widget must not raise.
    widget._render()


# --------------------------------------------------------------------------- #
# queue_payload routing (dataqueue vs full-fidelity signals.data)
# --------------------------------------------------------------------------- #
def test_queue_payload_default_is_passthrough():
    from mesofield.data.manager import DataManager

    dev = _Producer()
    assert dev.queue_payload({"a": 1}) == {"a": 1}

    dm = DataManager()
    dm.register_hardware_device(dev)
    dev.record(42, ts=0.0)
    pkt = dm.queue.pop(block=False)
    assert pkt.payload == 42  # unchanged


def test_queue_payload_override_filters_and_reshapes():
    from mesofield.data.manager import DataManager

    class _Filtered(BaseDataProducer):
        device_id = "filt"
        device_type = "filt"

        def queue_payload(self, payload):
            if not payload.get("event"):
                return None  # drop from queue
            return {"event": 1, "n": payload["n"]}  # reshape (omit other keys)

    dev = _Filtered()
    dm = DataManager()
    dm.register_hardware_device(dev)

    for i in range(6):
        dev.record({"n": i, "extra": 99, "event": 1 if i == 3 else 0}, ts=i * 0.001)

    rows = []
    while not dm.queue.empty():
        rows.append(dm.queue.pop(block=False).payload)

    # Only the event sample reached the queue, reshaped (no "extra").
    assert rows == [{"event": 1, "n": 3}]
    # The device's own buffer (-> CSV) and plots still see every full sample.
    buffered = dev.get_data()
    assert len(buffered) == 6
    assert all("extra" in p for _ts, p in buffered)
