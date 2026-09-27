"""Selective prediction: how accurate the answers are when only the most confident are kept.

Questions are ranked by confidence, most confident first. Answering the top k of n is a
coverage of k/n, and the share of errors among those k is the selective risk. The
risk-coverage curve is the risk at every coverage 1/n, 2/n, ..., 1; AURC, the area under it,
is its mean (Geifman and El-Yaniv, 2017): lower is better, and it rewards both being right and
ranking the right answers first. E-AURC subtracts the AURC of a perfect ranking with the same
accuracy (every correct answer first), leaving the part due to the ranking alone.

**Ties.** Confidence often takes few distinct values (three samples agree three, two or one
times), and the order inside a group of equal confidences is arbitrary. Every quantity here
is its expected value over a random order within ties: taking the first j of a tied group of
m questions with e errors adds j * e / m expected errors. The curve therefore does not depend
on how a sort breaks ties, and is linear inside each tied group.

**Declines.** A declined question has no answer. It ranks below every answered question,
whatever confidence it reports, and counts as an error wherever the coverage reaches it; so
at full coverage the risk is one minus the execution accuracy, declines counted as wrong.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def _arrays(
    confidence: Sequence[float], correct: Sequence[int], declined: Sequence[bool] | None
) -> tuple[np.ndarray, np.ndarray]:
    conf = np.asarray(confidence, dtype=float)
    ok = np.asarray(correct, dtype=float)
    if conf.shape != ok.shape or conf.ndim != 1 or conf.size == 0:
        raise ValueError("confidence and correctness must be equal-length, non-empty sequences")
    if np.isnan(conf).any():
        raise ValueError("every question needs a confidence")
    if not np.isin(ok, (0, 1)).all():
        raise ValueError("correctness must be 0 or 1")
    key = conf.copy()
    if declined is not None:
        d = np.asarray(declined, dtype=bool)
        key[d] = -np.inf  # below every answered question
        ok = np.where(d, 0.0, ok)  # a declined question is never a correct answer
    return key, 1.0 - ok


def expected_errors(
    confidence: Sequence[float], correct: Sequence[int], declined: Sequence[bool] | None = None
) -> np.ndarray:
    """Expected number of errors among the k most confident questions, for k = 1..n."""
    key, errors = _arrays(confidence, correct, declined)
    _, group, sizes = np.unique(-key, return_inverse=True, return_counts=True)  # best first
    group_errors = np.bincount(group, weights=errors, minlength=sizes.size)
    before = np.concatenate(([0.0], np.cumsum(group_errors)[:-1]))
    start = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    g = np.repeat(np.arange(sizes.size), sizes)  # the group of the k-th question
    k = np.arange(1, key.size + 1)
    return before[g] + (k - start[g]) * group_errors[g] / sizes[g]


def risk_coverage(
    confidence: Sequence[float], correct: Sequence[int], declined: Sequence[bool] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(coverage, selective risk) at k = 1..n."""
    e = expected_errors(confidence, correct, declined)
    k = np.arange(1, e.size + 1)
    return k / e.size, e / k


def aurc(
    confidence: Sequence[float], correct: Sequence[int], declined: Sequence[bool] | None = None
) -> float:
    return float(risk_coverage(confidence, correct, declined)[1].mean())


def oracle_aurc(correct_answers: int, n: int) -> float:
    """AURC of a perfect ranking: every correct answer before every error."""
    k = np.arange(1, n + 1)
    return float((np.maximum(k - correct_answers, 0) / k).mean())


def e_aurc(
    confidence: Sequence[float], correct: Sequence[int], declined: Sequence[bool] | None = None
) -> float:
    _, errors = _arrays(confidence, correct, declined)
    n = errors.size
    return aurc(confidence, correct, declined) - oracle_aurc(int(n - errors.sum()), n)


def accuracy_at_coverage(
    confidence: Sequence[float],
    correct: Sequence[int],
    coverage: float,
    declined: Sequence[bool] | None = None,
) -> float:
    """Accuracy of the ceil(coverage * n) most confident questions."""
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    e = expected_errors(confidence, correct, declined)
    k = max(1, math.ceil(coverage * e.size - 1e-9))
    return float(1 - e[k - 1] / k)
