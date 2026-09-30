"""Calibrating a confidence, and choosing the confidence below which the agent declines.

A calibrator maps a raw confidence to the probability that the answer is correct, fitted on one
set of answers (the calibration split) and applied, unchanged, to others (the held-out split):

- **Platt scaling:** a logistic regression of correctness on the raw confidence,
  p = 1 / (1 + exp(-(a * confidence + b))), fitted by maximum likelihood without a penalty
  (Newton's method). It is monotone, so it keeps the answers' order: it changes calibration
  (ECE, Brier score), never a ranking measure (AUROC, AURC), unless its slope is negative.
- **Isotonic regression:** the non-decreasing step function closest to correctness in squared
  error (pool-adjacent-violators, one weight per answer). Between the confidences it was fitted
  on it interpolates linearly; outside them it takes the nearest end value. Distinct raw values
  can map to one calibrated value, so it can create ties and change AURC a little.
- **Logistic** on several confidences at once (the stated and the critic's): the same
  regression with more than one input, for the combined confidence.

A declined answer returns no result, so its probability of being correct is 0 whatever it
states: `calibrated` sets it to 0, as the calibration measures do (src/eval/summary.py).

**The decline threshold** is chosen on the calibration split: the agent answers a question when
its calibrated confidence is at least the threshold, and declines otherwise. Candidate thresholds
are the distinct calibrated confidences of the answered questions, so a group of equal
confidences is never split. The threshold is the one giving the largest coverage (share of
questions answered) at which the answered questions' accuracy reaches the target. If none
reaches it, the threshold with the highest accuracy among those answering at least
`min_answered` questions is used, and the shortfall is reported. Both rules were fixed before any
calibrated confidence was computed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.eval.bootstrap import Interval


def _xy(confidence: Sequence[float], correct: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=float)
    if x.shape[0] != y.shape[0] or y.ndim != 1 or y.size == 0:
        raise ValueError("confidences and correctness must be equal-length, non-empty")
    if np.isnan(x).any():
        raise ValueError("every answer needs a confidence")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("correctness must be 0 or 1")
    return x, y


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def logistic_fit(
    features: np.ndarray, correct: np.ndarray, max_iter: int = 100, tol: float = 1e-10
) -> np.ndarray:
    """Maximum-likelihood coefficients [intercept, w1, w2, ...] of a logistic regression,
    without a penalty. Raises if the data are separable (no finite maximum) or it does not
    converge."""
    X = np.column_stack([np.ones(len(correct)), features])
    y = np.asarray(correct, dtype=float)
    if y.min() == y.max():
        raise ValueError("a logistic fit needs both correct and wrong answers")
    w = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = _sigmoid(X @ w)
        grad = X.T @ (y - p)
        hess = X.T @ (X * (p * (1 - p))[:, None])
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"the logistic fit is singular: {e}") from e
        w = w + step
        if np.abs(w).max() > 1e6:
            raise ValueError("the logistic fit diverges: the answers are separable")
        if np.abs(step).max() < tol:
            return w
    raise ValueError(f"the logistic fit did not converge in {max_iter} iterations")


@dataclass(frozen=True)
class Platt:
    intercept: float
    slope: float

    @classmethod
    def fit(cls, confidence: Sequence[float], correct: Sequence[int]) -> Platt:
        x, y = _xy(confidence, correct)
        b, a = logistic_fit(x[:, None], y)
        return cls(float(b), float(a))

    def __call__(self, confidence: Sequence[float]) -> np.ndarray:
        return _sigmoid(self.slope * np.asarray(confidence, dtype=float) + self.intercept)

    def to_dict(self) -> dict[str, Any]:
        return {"method": "platt", "intercept": self.intercept, "slope": self.slope}


@dataclass(frozen=True)
class Isotonic:
    x: tuple[float, ...]  # the distinct fitted confidences, increasing
    y: tuple[float, ...]  # the calibrated value at each

    @classmethod
    def fit(cls, confidence: Sequence[float], correct: Sequence[int]) -> Isotonic:
        x, y = _xy(confidence, correct)
        ux, inverse = np.unique(x, return_inverse=True)
        weight = np.bincount(inverse).astype(float)
        mean = np.bincount(inverse, weights=y) / weight
        # pool adjacent violators: blocks of (value, weight, how many distinct x)
        blocks: list[list[float]] = []
        for m, w in zip(mean, weight, strict=True):
            blocks.append([m, w, 1])
            while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
                m2, w2, k2 = blocks.pop()
                m1, w1, k1 = blocks.pop()
                blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, k1 + k2])
        fitted = np.repeat([b[0] for b in blocks], [int(b[2]) for b in blocks])
        return cls(tuple(float(v) for v in ux), tuple(float(v) for v in fitted))

    def __call__(self, confidence: Sequence[float]) -> np.ndarray:
        return np.interp(np.asarray(confidence, dtype=float), self.x, self.y)

    def to_dict(self) -> dict[str, Any]:
        return {"method": "isotonic", "x": list(self.x), "y": list(self.y)}


@dataclass(frozen=True)
class Logistic:
    """Correctness on several confidences (columns), e.g. the stated and the critic's."""

    names: tuple[str, ...]
    intercept: float
    weights: tuple[float, ...]

    @classmethod
    def fit(
        cls, names: Sequence[str], columns: Sequence[Sequence[float]], correct: Sequence[int]
    ) -> Logistic:
        X = np.column_stack([np.asarray(c, dtype=float) for c in columns])
        _, y = _xy(X[:, 0], correct)
        w = logistic_fit(X, y)
        return cls(tuple(names), float(w[0]), tuple(float(v) for v in w[1:]))

    def __call__(self, columns: Sequence[Sequence[float]]) -> np.ndarray:
        X = np.column_stack([np.asarray(c, dtype=float) for c in columns])
        return _sigmoid(X @ np.asarray(self.weights) + self.intercept)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "logistic",
            "intercept": self.intercept,
            "weights": dict(zip(self.names, self.weights, strict=True)),
        }


def calibrated(values: np.ndarray, declined: Sequence[bool]) -> np.ndarray:
    """Calibrated confidences with every declined answer at 0: it returns no result."""
    return np.where(np.asarray(declined, dtype=bool), 0.0, np.asarray(values, dtype=float))


def answered_at(p: np.ndarray, declined: np.ndarray, threshold: float) -> np.ndarray:
    """Which questions are answered at a threshold: not declined, confidence at least it."""
    return ~declined & (p >= threshold)


def choose_threshold(
    p: Sequence[float],
    correct: Sequence[int],
    declined: Sequence[bool],
    target: float,
    min_answered: int,
) -> dict[str, Any]:
    """The decline threshold chosen on the calibration split (module docstring), with every
    candidate's coverage and accuracy."""
    p, y = _xy(p, correct)
    d = np.asarray(declined, dtype=bool)
    n = len(y)
    candidates = []
    for t in sorted(set(p[~d].tolist()), reverse=True):
        answered = answered_at(p, d, t)
        k = int(answered.sum())
        candidates.append(
            {
                "threshold": t,
                "answered": k,
                "coverage": k / n,
                "accuracy": float(y[answered].mean()),
            }
        )
    if not candidates:
        raise ValueError("no answered question to choose a threshold on")
    reaching = [c for c in candidates if c["accuracy"] >= target]
    if reaching:
        chosen = max(reaching, key=lambda c: (c["coverage"], c["threshold"]))
        fallback = False
    else:
        enough = [c for c in candidates if c["answered"] >= min_answered]
        if not enough:
            raise ValueError(f"no threshold answers at least {min_answered} questions")
        chosen = max(enough, key=lambda c: (c["accuracy"], c["coverage"]))
        fallback = True
    return {
        "target_accuracy": target,
        "min_answered": min_answered,
        "threshold": chosen["threshold"],
        "reached_target": not fallback,
        "fallback_used": fallback,
        "chosen": chosen,
        "candidates": candidates,
    }


def threshold_outcome(
    p: Sequence[float],
    correct: Sequence[int],
    declined: Sequence[bool],
    threshold: float,
    boot: Callable[..., Interval],
) -> dict[str, Any]:
    """Coverage and the answered questions' accuracy at a fixed threshold, with bootstrap
    intervals over the questions (the threshold is held fixed). A resample with no answered
    question has no accuracy and is left out (counted as undefined)."""
    p, y = _xy(p, correct)
    d = np.asarray(declined, dtype=bool)
    answered = answered_at(p, d, threshold)

    def accuracy(i: np.ndarray) -> float:
        a = answered[i]
        return float(y[i][a].mean()) if a.any() else float("nan")

    return {
        "threshold": threshold,
        "questions": len(y),
        "answered": int(answered.sum()),
        "coverage": boot(lambda i: answered[i].mean(), len(y)).to_dict(),
        "accuracy": boot(accuracy, len(y)).to_dict(),
    }
