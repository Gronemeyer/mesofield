"""Real-time single-bin frequency-band detector.

Estimates per-frame power at a single target frequency using the
Goertzel algorithm (an O(N) single-bin DFT), then applies an online
noise-floor MAD hysteresis to produce a binary "detected" state.

Two synchronized output channels per frame:

* ``power`` -- Goertzel magnitude-squared at ``target_hz`` over the
  trailing ``window_s`` of frame reductions (default: mean intensity).
* ``detected`` -- ``0.0`` / ``1.0`` based on hysteresis thresholds
  anchored to the power signal's noise floor.  ``high_k`` / ``low_k``
  are MAD multipliers; the recipe matches the user's offline event-
  detection script (``median_quiet + k * 1.4826 * MAD``).

Typical usage::

    self.flash = FrequencyBandDetector(
        camera=self.hardware.primary,
        target_hz=4.0,
        window_s=2.0,
        plot={"power": {"label": "4 Hz Power"},
              "detected": {"label": "Detected", "y_range": (-0.1, 1.1)}},
    )
"""

from __future__ import annotations

import collections
import math
from typing import Any, Callable, ClassVar, Optional, Tuple

import numpy as np

from .base import FrameProcessor

__all__ = ["FrequencyBandDetector"]


class FrequencyBandDetector(FrameProcessor):
    """Goertzel power + hysteresis detected-state for a single target frequency."""

    channels: ClassVar[Tuple[str, ...]] = ("power", "detected")
    data_type: ClassVar[str] = "freq_band"

    def __init__(
        self,
        *,
        target_hz: float,
        window_s: float = 2.0,
        warmup_s: float = 4.0,
        high_k: float = 4.0,
        low_k: float = 2.5,
        threshold_refresh_s: float = 1.0,
        reducer: Optional[Callable[[Any], float]] = None,
        **kw: Any,
    ) -> None:
        """
        Parameters
        ----------
        target_hz
            Frequency to detect (Hz).  Must satisfy
            ``target_hz < sampling_rate / 2`` (Nyquist).
        window_s
            Trailing window length (s) over which Goertzel is evaluated
            each frame.  Longer = finer frequency resolution but slower
            response.  2 s × 4 Hz = 8 cycles is a good default.
        warmup_s
            Seconds of power-history to accumulate before reporting
            ``detected``.  Until then ``detected`` stays at 0 and
            thresholds are undefined.
        high_k, low_k
            MAD multipliers for the hysteresis thresholds.  Defaults
            match the user's offline detector.
        threshold_refresh_s
            How often (seconds of frames) to recompute the noise-floor
            thresholds.  Recomputing every frame is wasteful — the
            noise floor moves slowly.
        reducer
            Callable ``(img) -> float`` collapsing each frame to a
            scalar.  Default: mean intensity.  Pass an ROI mean here
            to detect oscillations in a particular region.
        """
        # Pop the channels-specific kwargs that aren't FrameProcessor's.
        super().__init__(**kw)
        fps = float(self.sampling_rate or 30.0)
        if target_hz <= 0:
            raise ValueError(f"target_hz must be > 0 (got {target_hz})")
        if target_hz >= fps / 2.0:
            raise ValueError(
                f"target_hz={target_hz} is above Nyquist for sampling_rate={fps}"
            )

        self.target_hz = float(target_hz)
        self.window_s = float(window_s)
        self.warmup_s = float(warmup_s)
        self.high_k = float(high_k)
        self.low_k = float(low_k)
        self._fps = fps

        # Goertzel: choose the nearest integer bin to target_hz/fps * N,
        # then derive the constant ``2 * cos(2*pi*k/N)``.
        self._N = max(8, int(round(self.window_s * fps)))
        k_bin = max(1, int(round(self.target_hz * self._N / fps)))
        omega = 2.0 * math.pi * k_bin / self._N
        self._coeff = 2.0 * math.cos(omega)

        # Rolling buffer of frame reductions for the Goertzel window.
        self._buf: collections.deque[float] = collections.deque(maxlen=self._N)
        # Rolling buffer of past power values for threshold estimation.
        # Cap at ~2x warmup so old values eventually age out as the
        # session settles.
        self._hist_max = max(64, int(round(self.warmup_s * fps * 2)))
        self._hist: collections.deque[float] = collections.deque(maxlen=self._hist_max)

        # Hysteresis state.
        self._in_event: bool = False
        self._high_th: float = float("inf")
        self._low_th: float = float("inf")
        # Frame counter for threshold refresh cadence.
        self._frames_since_refresh: int = 0
        self._refresh_period: int = max(1, int(round(threshold_refresh_s * fps)))
        self._min_hist_for_threshold: int = max(16, int(round(self.warmup_s * fps)))

        self._reducer: Callable[[Any], float] = (
            reducer if reducer is not None
            else (lambda im: float(np.asarray(im).mean()))
        )

    # ------------------------------------------------------------------
    def compute(self, img: Any, idx: Any, ts: Any) -> Optional[dict]:
        # 1. Reduce frame to scalar and update the Goertzel window.
        try:
            x = self._reducer(img)
        except Exception:
            return None
        self._buf.append(x)
        if len(self._buf) < self._N:
            return None  # warming the frame buffer; no output yet

        # 2. Goertzel single-bin power via two-state recurrence.
        s_prev = 0.0
        s_prev2 = 0.0
        c = self._coeff
        for v in self._buf:
            s = v + c * s_prev - s_prev2
            s_prev2 = s_prev
            s_prev = s
        power = s_prev * s_prev + s_prev2 * s_prev2 - c * s_prev * s_prev2

        # 3. Track power history.
        self._hist.append(power)

        # 4. Periodically refresh thresholds from the noise floor of
        #    the trailing power history (lower-half MAD).
        self._frames_since_refresh += 1
        if (
            len(self._hist) >= self._min_hist_for_threshold
            and self._frames_since_refresh >= self._refresh_period
        ):
            self._frames_since_refresh = 0
            arr = np.fromiter(self._hist, dtype=float, count=len(self._hist))
            med = float(np.median(arr))
            lower = arr[arr <= med]
            if len(lower) < 2:
                lower = arr
            lower_med = float(np.median(lower))
            mad = float(np.median(np.abs(lower - lower_med)))
            sigma = 1.4826 * mad if mad > 0 else float(np.std(lower))
            if sigma <= 0:
                sigma = max(abs(lower_med), 1e-9) * 0.01
            self._high_th = lower_med + self.high_k * sigma
            self._low_th = lower_med + self.low_k * sigma

        # 5. Hysteresis (returns 0/1 until warmup completes).
        if math.isfinite(self._high_th):
            if not self._in_event and power >= self._high_th:
                self._in_event = True
            elif self._in_event and power <= self._low_th:
                self._in_event = False

        return {
            "power": float(power),
            "detected": 1.0 if self._in_event else 0.0,
        }

    # ------------------------------------------------------------------
    def thresholds(self) -> Tuple[float, float]:
        """Current (high, low) hysteresis thresholds (read-only)."""
        return self._high_th, self._low_th
