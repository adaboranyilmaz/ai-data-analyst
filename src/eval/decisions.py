"""What errors cost: the expected cost per question of an answering system, where declining is
worth it, and the operating point that costs least.

A system answers each question or declines it. Per question it has an API cost (USD, from its
recorded tokens at the batch price), and it is either right, wrong (answered wrongly, a failing
query included) or declined. With a cost `C_w` for a wrong answer and `C_d` for declining (a person
answers instead, taken to answer rightly), the expected cost per question is

    E = mean(API cost) + C_w * share wrong + C_d * share declined.

E is linear in the two costs, so which system is cheapest depends only on where (C_w, C_d) lies,
and the boundaries are straight lines. Neither cost is known in general, so results are given
over grids of both, never at a chosen price.

- Two systems that never decline break even at `C_w* = (difference in API cost) / (difference in
  share wrong)`: above it the more accurate one is cheaper.
- A system that declines some questions, against the same system answering them all (the API
  cost is spent either way), is cheaper when `C_d / C_w` is below the share of the declined
  questions it would have answered wrongly: declining pays when a person's answer costs less than
  the risk of the analyst's.
- The cost-optimal rule, given a calibrated probability `p` that an answer is right: answering
  costs `(1 - p) * C_w` in expectation and declining costs `C_d`, so decline when
  `p < 1 - C_d / C_w`. It needs `p` calibrated; the calibrator is fitted on the calibration split
  and the rule evaluated on held-out questions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Outcomes:
    """One system's outcome on each question, in a fixed question order."""

    name: str
    api: np.ndarray  # USD per question
    wrong: np.ndarray  # answered and wrong
    declined: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.api)
        if not (len(self.wrong) == len(self.declined) == n):
            raise ValueError("api, wrong and declined must cover the same questions")
        if np.any(np.asarray(self.wrong) & np.asarray(self.declined)):
            raise ValueError("a declined question cannot also be answered wrongly")

    def shares(self, idx: np.ndarray | None = None) -> tuple[float, float, float]:
        """(mean API cost, share wrong, share declined) over the questions `idx` (all if None)."""
        i = slice(None) if idx is None else idx
        return (
            float(np.mean(self.api[i])),
            float(np.mean(self.wrong[i])),
            float(np.mean(self.declined[i])),
        )


def outcomes(
    name: str, api: Sequence[float], right: Sequence[bool], declined: Sequence[bool] | None = None
) -> Outcomes:
    """From each question's API cost, whether the answer was right, and whether it declined."""
    right = np.asarray(right, dtype=bool)
    d = np.zeros(len(right), dtype=bool) if declined is None else np.asarray(declined, dtype=bool)
    return Outcomes(name, np.asarray(api, dtype=float), ~right & ~d, d)


def expected_cost(
    o: Outcomes, cost_wrong: float, cost_decline: float, idx: np.ndarray | None = None
) -> float:
    api, wrong, declined = o.shares(idx)
    return api + cost_wrong * wrong + cost_decline * declined


def cheapest(
    systems: Sequence[Outcomes], wrong_grid: Sequence[float], decline_grid: Sequence[float]
) -> list[list[str]]:
    """The cheapest system's name at each (C_d, C_w): rows by C_d, columns by C_w; on a tie, the
    first listed."""
    shares = [s.shares() for s in systems]
    out = []
    for cd in decline_grid:
        row = []
        for cw in wrong_grid:
            costs = [a + cw * w + cd * d for a, w, d in shares]
            row.append(systems[int(np.argmin(costs))].name)
        out.append(row)
    return out


def break_even_wrong_cost(a: Outcomes, b: Outcomes, idx: np.ndarray | None = None) -> float:
    """The cost of a wrong answer above which `b` is cheaper than `a`, for systems that do not
    decline: (API b - API a) / (wrong a - wrong b). Infinite when `b` is not more accurate (it
    never pays), 0 or less when `b` is also cheaper to run (it always pays)."""
    api_a, wrong_a, _ = a.shares(idx)
    api_b, wrong_b, _ = b.shares(idx)
    gain = wrong_a - wrong_b
    if gain <= 0:
        return float("inf")
    return (api_b - api_a) / gain


def decline_pays_below(
    answer_all: Outcomes, with_declines: Outcomes, idx: np.ndarray | None = None
) -> float:
    """The ratio C_d / C_w below which declining is cheaper than answering every question: the
    share of the declined questions the system would have answered wrongly. NaN when nothing is
    declined."""
    i = slice(None) if idx is None else idx
    declined = with_declines.declined[i]
    if not declined.any():
        return float("nan")
    return float(np.mean(answer_all.wrong[i][declined]))


def optimal_decline(
    name: str, api: Sequence[float], right: Sequence[bool], p: Sequence[float], ratio: float
) -> Outcomes:
    """The cost-optimal rule at C_d / C_w = `ratio`: decline where the calibrated probability of
    being right is below 1 - ratio."""
    p = np.asarray(p, dtype=float)
    return outcomes(name, api, right, p < 1.0 - ratio)


def cost_in_wrong_answers(o: Outcomes, ratio: float, idx: np.ndarray | None = None) -> float:
    """Expected cost per question in units of C_w, when C_w is large against the API cost (the
    API term vanishes): share wrong + ratio * share declined."""
    _, wrong, declined = o.shares(idx)
    return wrong + ratio * declined
