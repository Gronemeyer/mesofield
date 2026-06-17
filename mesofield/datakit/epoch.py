"""Event-triggered epoching / averaging on the shared master clock.

Once trial onsets are available on the camera timeline (see
``mesofield.datakit.sources.behavior.mouseportal.MousePortalTrials``), an
event-triggered average is just: window each timeseries around every onset onto
a common peri-event grid and average across events.

Works for any time-indexed signal whose first axis is time: a scalar ROI/dF-F
trace ``(T,)``, a multi-region trace ``(T, R)``, or a widefield image stack
``(T, H, W)``.

Example
-------
>>> from mesofield.datakit import load_dataset
>>> from mesofield.datakit.epoch import event_triggered_average
>>> ds = load_dataset("data")
>>> trials = ds.select(source="mouseportal_trials")      # interval table
>>> meso   = ds.select(source="meso_mean")               # (t, trace)
>>> onsets = trials.value.query("phase == 'trial'")["start_s"].to_numpy()
>>> grid, eta, stack = event_triggered_average(meso.t, meso.value, onsets,
...                                             pre=1.0, post=3.0, fs=10.0,
...                                             baseline=(-1.0, 0.0))
>>> eta.shape            # (len(grid), *feature_shape)
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def peri_event_grid(pre: float, post: float, fs: float) -> np.ndarray:
    """Uniform peri-event time grid in seconds: [-pre, post) at ``fs`` Hz."""
    if fs <= 0:
        raise ValueError("fs must be > 0")
    n = int(round((float(pre) + float(post)) * float(fs)))
    return (np.arange(n) / float(fs)) - float(pre)


def epoch(
    t: Sequence[float],
    values: np.ndarray,
    onsets: Sequence[float],
    *,
    pre: float,
    post: float,
    fs: float,
    method: str = "interp",
) -> Tuple[np.ndarray, np.ndarray]:
    """Cut a peri-event stack around each onset.

    Parameters
    ----------
    t : (T,) array of sample times in seconds (master clock; strictly increasing).
    values : (T, ...) array; first axis aligns with ``t``.
    onsets : event times in seconds (same clock as ``t``).
    pre, post : window seconds before/after each onset.
    fs : sampling rate of the returned uniform grid.
    method : "interp" (linear, good for traces) or "nearest" (frame indexing,
        cheap for image stacks).

    Returns
    -------
    grid : (n_bins,) peri-event times.
    stack : (n_onsets, n_bins, *feature_shape); out-of-range samples are NaN
        ("interp") or dropped via NaN fill.
    """
    t = np.asarray(t, dtype=np.float64)
    values = np.asarray(values)
    if t.ndim != 1 or t.shape[0] != values.shape[0]:
        raise ValueError("t must be 1-D and match values' first axis")

    grid = peri_event_grid(pre, post, fs)
    feature_shape = values.shape[1:]
    flat = values.reshape(values.shape[0], -1).astype(np.float64)  # (T, F)
    n_onsets, n_bins, F = len(onsets), grid.size, flat.shape[1]
    stack = np.full((n_onsets, n_bins, F), np.nan, dtype=np.float64)

    t_min, t_max = (t[0], t[-1]) if t.size else (0.0, 0.0)
    for i, onset in enumerate(onsets):
        sample_t = float(onset) + grid
        inside = (sample_t >= t_min) & (sample_t <= t_max)
        if not np.any(inside):
            continue
        if method == "nearest":
            idx = np.searchsorted(t, sample_t[inside])
            idx = np.clip(idx, 0, t.size - 1)
            stack[i, inside, :] = flat[idx, :]
        else:  # linear interpolation per feature column
            for j in range(F):
                stack[i, inside, j] = np.interp(sample_t[inside], t, flat[:, j])

    stack = stack.reshape((n_onsets, n_bins) + feature_shape)
    return grid, stack


def event_triggered_average(
    t: Sequence[float],
    values: np.ndarray,
    onsets: Sequence[float],
    *,
    pre: float,
    post: float,
    fs: float,
    baseline: Optional[Tuple[float, float]] = None,
    method: str = "interp",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Event-triggered average + the underlying per-event stack.

    ``baseline=(b0, b1)`` subtracts each event's mean over that peri-event
    window (seconds, relative to onset) before averaging.

    Returns ``(grid, eta, stack)`` where ``eta`` is the nan-mean across events
    with shape ``(n_bins, *feature_shape)``.
    """
    grid, stack = epoch(t, values, onsets, pre=pre, post=post, fs=fs, method=method)
    if baseline is not None:
        b0, b1 = baseline
        bmask = (grid >= b0) & (grid < b1)
        if np.any(bmask):
            base = np.nanmean(stack[:, bmask, ...], axis=1, keepdims=True)
            stack = stack - base
    eta = np.nanmean(stack, axis=0)
    return grid, eta, stack
