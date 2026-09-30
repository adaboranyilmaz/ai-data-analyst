"""Near misses: a wrong answer whose result holds the right one, up to its columns or rounding.

Execution accuracy compares result rows exactly, as the benchmark's official evaluator does, so an
answer that returns the right values with one extra column, the columns in another order, or a
number with more decimal places counts as wrong. A wrong answer is a near miss when the expert
result's rows, as a set, can be recovered from the answer's result by:
- **columns:** choosing and ordering some of its columns (it has extra columns, or the right
  columns in another order);
- **rounding:** rounding every number in both results to two decimal places;
- or both.
Everything else is "other". Results over `MAX_ROWS_CHECKED` rows, or with more than `MAX_SEARCH`
column choices to try, are not checked (`not_checked`).
"""

from __future__ import annotations

import itertools
import math
from decimal import Decimal

MAX_ROWS_CHECKED = 5000
MAX_SEARCH = 20_000  # column choices tried per answer
KINDS = ("columns", "rounding", "columns_and_rounding", "other", "not_checked")
NEAR_KINDS = ("columns", "rounding", "columns_and_rounding")


def rounded(row: tuple) -> tuple:
    def r(v):
        if isinstance(v, bool) or v is None:
            return v
        if isinstance(v, int | float | Decimal):
            x = float(v)
            return round(x, 2) if math.isfinite(x) else x
        return v

    return tuple(r(v) for v in row)


def recoverable(pred: list[tuple], gold: list[tuple], width: int) -> bool | None:
    """Whether some choice and order of the prediction's columns gives the gold rows as a set;
    None if there are too many choices to try."""
    target = set(gold)
    n = len(pred[0]) if pred else 0
    if n < width:
        return False
    if math.perm(n, width) > MAX_SEARCH:
        return None
    return any(
        {tuple(row[i] for i in cols) for row in pred} == target
        for cols in itertools.permutations(range(n), width)
    )


def classify(pred: list[tuple], gold: list[tuple], gold_width: int) -> str:
    if len(pred) > MAX_ROWS_CHECKED or len(gold) > MAX_ROWS_CHECKED:
        return "not_checked"
    if set(map(rounded, pred)) == set(map(rounded, gold)):
        return "rounding"
    plain = recoverable(pred, gold, gold_width)
    if plain:
        return "columns"
    both = recoverable([rounded(r) for r in pred], [rounded(r) for r in gold], gold_width)
    if both:
        return "columns_and_rounding"
    if plain is None or both is None:
        return "not_checked"
    return "other"
