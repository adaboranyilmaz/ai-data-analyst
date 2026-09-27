"""Calibration: does a confidence of 0.8 mean the answer is right 80% of the time?

- **Brier score:** the mean squared difference between confidence and correctness (0 or 1).
- **ECE** (expected calibration error; Guo et al., 2017): confidences are grouped into bins,
  and ECE is the average gap between a bin's mean confidence and its accuracy, weighted by the
  bin's share of questions. Two binnings: equal-width (ten bins of 0.1, the usual choice) and
  equal-mass (bins of about equal size). Equal-mass bins never split a group of equal
  confidences: a boundary that would falls after the group instead.
- **Reliability diagram:** per bin, mean confidence against accuracy; the ECE's ingredients.
- **AUROC:** the chance that a correct answer has a higher confidence than a wrong one (ties
  count one half): how well confidence separates right from wrong, whatever its scale.
  Undefined (None) when every answer is right or every answer is wrong.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _arrays(confidence: Sequence[float], correct: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    conf = np.asarray(confidence, dtype=float)
    ok = np.asarray(correct, dtype=float)
    if conf.shape != ok.shape or conf.ndim != 1 or conf.size == 0:
        raise ValueError("confidence and correctness must be equal-length, non-empty sequences")
    if np.isnan(conf).any() or (conf < 0).any() or (conf > 1).any():
        raise ValueError("confidences must lie in [0, 1]")
    if not np.isin(ok, (0, 1)).all():
        raise ValueError("correctness must be 0 or 1")
    return conf, ok


def brier(confidence: Sequence[float], correct: Sequence[int]) -> float:
    conf, ok = _arrays(confidence, correct)
    return float(((conf - ok) ** 2).mean())


def _bin_index(conf: np.ndarray, bins: int, strategy: str) -> np.ndarray:
    if strategy == "width":
        return np.minimum((conf * bins).astype(int), bins - 1)  # 1.0 falls in the last bin
    if strategy != "mass":
        raise ValueError(f"unknown binning {strategy!r}")
    order = np.argsort(conf, kind="stable")
    ranked = conf[order]
    n = conf.size
    cuts = []
    for i in range(1, bins):
        cut = round(i * n / bins)
        while 0 < cut < n and ranked[cut] == ranked[cut - 1]:  # keep ties together
            cut += 1
        if 0 < cut < n and (not cuts or cut > cuts[-1]):
            cuts.append(cut)
    index = np.empty(n, dtype=int)
    index[order] = np.searchsorted(np.array(cuts, dtype=int), np.arange(n), side="right")
    return index


def reliability(
    confidence: Sequence[float], correct: Sequence[int], bins: int = 10, strategy: str = "width"
) -> list[dict]:
    """Non-empty bins: size, mean confidence, accuracy (and, for equal-width, the bin's range)."""
    conf, ok = _arrays(confidence, correct)
    index = _bin_index(conf, bins, strategy)
    out = []
    for b in np.unique(index):
        sel = index == b
        row = {
            "n": int(sel.sum()),
            "mean_confidence": float(conf[sel].mean()),
            "accuracy": float(ok[sel].mean()),
        }
        if strategy == "width":
            row["range"] = [b / bins, (b + 1) / bins]
        out.append(row)
    return out


def ece(
    confidence: Sequence[float], correct: Sequence[int], bins: int = 10, strategy: str = "width"
) -> float:
    rows = reliability(confidence, correct, bins, strategy)
    n = sum(r["n"] for r in rows)
    return float(sum(r["n"] / n * abs(r["accuracy"] - r["mean_confidence"]) for r in rows))


def auroc(confidence: Sequence[float], correct: Sequence[int]) -> float | None:
    conf, ok = _arrays(confidence, correct)
    pos = int(ok.sum())
    neg = ok.size - pos
    if pos == 0 or neg == 0:
        return None
    # Mann-Whitney: the positives' mean rank, tied values sharing their average rank
    _, group, counts = np.unique(conf, return_inverse=True, return_counts=True)
    start = np.concatenate(([0], np.cumsum(counts)[:-1]))
    ranks = (start + (counts + 1) / 2)[group]
    return float((ranks[ok == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
