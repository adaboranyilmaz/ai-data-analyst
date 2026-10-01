"""Whether what the service now shows still looks like what it was evaluated on.

The confidence meter says how often answers at a given confidence were right on held-out
questions. That holds only while the questions and the answers look like those: a different mix of
questions, a changed model or a changed prompt moves the distribution of calibrated confidence
before anyone has labelled an answer. The monitor compares the last `window` calibrated
confidences with the held-out reference by the population stability index (PSI), and the share of
answers withheld or declined with its held-out share the same way:

    PSI = sum over bins of (p_live - p_ref) * ln(p_live / p_ref)

on ten equal-width bins of calibrated confidence (a declined answer counts at 0, as it does in the
evaluation), each bin's count smoothed by `smoothing` so that an empty bin does not make the value
infinite. By the usual reading, below 0.10 is no shift, 0.10 to 0.25 a moderate one, above 0.25 a
major one; the thresholds are in configs/serving.yaml and fixed before the monitor was tested.

Plain Python: the service image has no numerical library.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any


def psi(reference: list[float], live: list[float], smoothing: float = 0.5) -> float:
    """The population stability index of two count vectors over the same bins."""
    if len(reference) != len(live):
        raise ValueError("the two distributions need the same bins")
    r = [c + smoothing for c in reference]
    l_ = [c + smoothing for c in live]
    rs, ls = sum(r), sum(l_)
    return sum(
        (b / ls - a / rs) * math.log((b / ls) / (a / rs)) for a, b in zip(r, l_, strict=True)
    )


def bin_of(confidence: float, bins: int) -> int:
    """The equal-width bin of a calibrated confidence in [0, 1] (1.0 is in the last bin)."""
    return min(int(max(confidence, 0.0) * bins), bins - 1)


class DriftMonitor:
    def __init__(
        self,
        reference_counts: list[float],
        threshold: float,
        window: int = 200,
        min_window: int = 50,
        smoothing: float = 0.5,
        warn: float = 0.10,
        alert: float = 0.25,
    ):
        self.reference = [float(c) for c in reference_counts]
        self.bins = len(self.reference)
        self.threshold = threshold
        self.min_window, self.smoothing = min_window, smoothing
        self.warn, self.alert = warn, alert
        total = sum(self.reference)
        # held-out share below the decline threshold: bins whose upper edge is at or under it,
        # and the share of the bin the threshold falls in, taken in proportion
        below = 0.0
        for i, c in enumerate(self.reference):
            lo, hi = i / self.bins, (i + 1) / self.bins
            if hi <= threshold:
                below += c
            elif lo < threshold:
                below += c * (threshold - lo) / (hi - lo)
        self.reference_below = below
        self.reference_total = total
        self.recent: deque[float] = deque(maxlen=window)

    @classmethod
    def from_meter(cls, meter, **kw) -> DriftMonitor:
        """The reference is the meter's held-out table: how many held-out answers fell in each bin
        of calibrated confidence (the table lists only the bins that hold answers). Recorded runs
        and live ones are each described by their own meter, so each is compared with its own."""
        width = meter.bands[0].high - meter.bands[0].low
        bins = round(1 / width)
        counts = [0.0] * bins
        for band in meter.bands:
            counts[bin_of((band.low + band.high) / 2, bins)] += band.n
        return cls(counts, meter.threshold, **kw)

    def observe(self, calibrated: float | None) -> None:
        """One answer's calibrated confidence; None (declined) counts as 0, as in the evaluation."""
        self.recent.append(0.0 if calibrated is None else float(calibrated))

    def snapshot(self) -> dict[str, Any]:
        n = len(self.recent)
        ready = n >= self.min_window
        out: dict[str, Any] = {"window": n, "ready": ready, "psi": None, "below_psi": None}
        if n == 0:
            return out
        live = [0.0] * self.bins
        for c in self.recent:
            live[bin_of(c, self.bins)] += 1
        below_live = sum(c < self.threshold for c in self.recent)
        out["below_share"] = below_live / n
        out["reference_below_share"] = self.reference_below / self.reference_total
        if ready:
            out["psi"] = psi(self.reference, live, self.smoothing)
            out["below_psi"] = psi(
                [self.reference_below, self.reference_total - self.reference_below],
                [below_live, n - below_live],
                self.smoothing,
            )
        return out

    def reading(self, value: float | None) -> str:
        if value is None:
            return "not enough answers yet"
        return (
            "major shift"
            if value > self.alert
            else "moderate shift"
            if value > self.warn
            else "no shift"
        )
