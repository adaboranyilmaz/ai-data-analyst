"""Bootstrap intervals over questions.

A statistic is a function of the questions it is computed on, given as an array of indices
into the evaluated questions. Each resample draws n questions with replacement; the interval
is the percentile interval of the statistic over the resamples (the middle 95% by default).
A paired comparison of two designs evaluated on the same questions resamples the same
questions for both and takes the interval of the difference, so that a question both find
easy counts once, not as noise.

A statistic that is undefined on a resample (AUROC when every resampled answer is right)
returns NaN there; those resamples are left out and counted. The generator is seeded, so an
interval is the same on every run, and two statistics computed with the same seed on the same
questions see the same resamples.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass

import numpy as np

Statistic = Callable[[np.ndarray], float]
CHUNK = 1000


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    confidence: float
    resamples: int
    undefined: int  # resamples on which the statistic was undefined, left out

    def to_dict(self) -> dict:
        """For JSON: an undefined value (NaN) becomes null."""
        return {
            k: None if isinstance(v, float) and math.isnan(v) else v
            for k, v in asdict(self).items()
        }


def resamples(n: int, count: int, seed: int) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(seed)
    done = 0
    while done < count:
        size = min(CHUNK, count - done)
        yield from rng.integers(0, n, size=(size, n))
        done += size


def _interval(estimate: float, values: list[float], confidence: float) -> Interval:
    v = np.asarray(values, dtype=float)
    ok = v[~np.isnan(v)]
    if ok.size == 0:
        low = high = float("nan")
    else:
        alpha = (1 - confidence) / 2
        low, high = (float(x) for x in np.quantile(ok, [alpha, 1 - alpha]))
    return Interval(float(estimate), low, high, confidence, len(values), int(v.size - ok.size))


def bootstrap(
    stat: Statistic, n: int, count: int = 10_000, confidence: float = 0.95, seed: int = 0
) -> Interval:
    if n == 0:
        raise ValueError("no questions to resample")
    estimate = stat(np.arange(n))
    return _interval(estimate, [stat(idx) for idx in resamples(n, count, seed)], confidence)


def paired_difference(
    stat_a: Statistic,
    stat_b: Statistic,
    n: int,
    count: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """stat_a - stat_b, both on the same resampled questions."""
    if n == 0:
        raise ValueError("no questions to resample")
    whole = np.arange(n)
    estimate = stat_a(whole) - stat_b(whole)
    values = [stat_a(idx) - stat_b(idx) for idx in resamples(n, count, seed)]
    return _interval(estimate, values, confidence)
