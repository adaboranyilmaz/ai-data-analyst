"""Execution accuracy (EX) and Soft-F1, computed as BIRD's official evaluator computes them.

The official evaluator (bird-bench/mini_dev, `evaluation/`) runs the predicted query and then
the gold query on one database cursor and compares the rows they return, as the Python values
the driver gives:

- **EX** is 1 when `set(predicted) == set(gold)`. Row order and repeated rows do not matter;
  column order does (rows are tuples); column names are ignored. Values compare by Python
  equality, so `682`, `682.0` and `Decimal("682")` are equal while `Decimal("0.1")` and `0.1`
  are not, `True == 1`, and a float NaN equals nothing, not even the same query's NaN.
- **Soft-F1** keeps each distinct row once, in the order it first appears, pairs the i-th
  predicted row with the i-th gold row, and counts the values of each row found anywhere in
  the other row of its pair. Unlike EX it depends on row order. Its arithmetic below follows
  the official code step by step, so the scores agree to the last bit.

The official evaluator scores any exception 0: a failed or timed-out query, but also a
comparison that raises. A value that cannot be hashed (an array, a JSON document) makes
`set()` fail, and a gold row with no columns divides Soft-F1 by zero. The functions here
raise in the same cases and leave it to the caller (src/eval/execution.py) to score 0 and
record why.

`PredictionStream` computes both scores from a prediction's rows as they arrive, so a
prediction is scored exactly however many rows it returns: EX keeps only the predicted rows
that are in the gold result, and is settled at 0 by the first row that is not. Soft-F1 needs
the number of distinct predicted rows, so it keeps them up to a set number. Beyond that it is
still exact when none of the paired rows shares a value with its gold row (it is then 0 however
many rows follow); otherwise it is not computed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

Row = tuple[Any, ...]


def ex(predicted: Sequence[Row], gold: Sequence[Row]) -> int:
    return int(set(predicted) == set(gold))


def _row_match(predicted_row: Row, gold_row: Row) -> tuple[float, float, float]:
    """Shares of the gold row's width: values found in the other row, values only in the
    predicted row, values only in the gold row."""
    total_columns = len(gold_row)
    matches = 0
    pred_only = 0
    truth_only = 0
    for value in predicted_row:
        if value in gold_row:
            matches += 1
        else:
            pred_only += 1
    for value in gold_row:
        if value not in predicted_row:
            truth_only += 1
    return matches / total_columns, pred_only / total_columns, truth_only / total_columns


def _soft_f1_distinct(predicted: list[Row], gold: list[Row]) -> float:
    """Soft-F1 of two lists of distinct rows, each in first-appearance order."""
    match_scores: list[float] = []
    pred_only_scores: list[float] = []
    truth_only_scores: list[float] = []
    for i, gold_row in enumerate(gold):
        if i >= len(predicted):  # a gold row with no predicted row to pair with
            match_scores.append(0)
            truth_only_scores.append(1)
            continue
        match, pred_only, truth_only = _row_match(predicted[i], gold_row)
        match_scores.append(match)
        pred_only_scores.append(pred_only)
        truth_only_scores.append(truth_only)
    for _ in range(len(predicted) - len(gold)):  # predicted rows beyond the gold's
        match_scores.append(0)
        pred_only_scores.append(1)
        truth_only_scores.append(0)

    tp = sum(match_scores)
    fp = sum(pred_only_scores)
    fn = sum(truth_only_scores)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0


def soft_f1(predicted: Sequence[Row], gold: Sequence[Row]) -> float:
    if not predicted and not gold:
        return 1.0
    # dict.fromkeys keeps the first appearance of each row; it raises on an unhashable value
    # exactly where the official code's set() does
    return _soft_f1_distinct(list(dict.fromkeys(predicted)), list(dict.fromkeys(gold)))


class PredictionStream:
    """EX and Soft-F1 of a prediction fed in batches of rows, against a gold result.

    Raises as the official comparison would: on construction for a gold value that cannot be
    hashed, in `feed` for such a predicted value, and in `soft_f1` for a gold row with no
    columns.
    """

    def __init__(self, gold: Sequence[Row], max_distinct: int):
        self.gold = list(gold)
        self._gold_set = set(self.gold)
        self._gold_distinct = list(dict.fromkeys(self.gold))
        self._seen_in_gold: set[Row] = set()
        self._all_in_gold = True
        # every distinct predicted row in first-appearance order, until there are too many;
        # then only the first ones, those paired with gold rows
        self._distinct: dict[Row, None] | None = {}
        self._paired_head: list[Row] | None = None
        self.max_distinct = max_distinct
        self.rows = 0

    def feed(self, rows: Iterable[Row]) -> None:
        for row in rows:
            self.rows += 1
            if self._all_in_gold:
                if row in self._gold_set:
                    self._seen_in_gold.add(row)
                else:
                    self._all_in_gold = False
            if self._distinct is not None:
                self._distinct[row] = None
                if len(self._distinct) > self.max_distinct:
                    head = len(self._gold_distinct)
                    self._paired_head = list(self._distinct)[:head]
                    self._distinct = None

    @property
    def settled(self) -> bool:
        """No further row can change either score: EX is 0 and Soft-F1 was given up."""
        return not self._all_in_gold and self._distinct is None

    @property
    def soft_f1_computed(self) -> bool:
        return self._distinct is not None

    def ex(self) -> int:
        # every predicted row was in the gold set, so the predicted set is _seen_in_gold
        return int(self._all_in_gold and self._seen_in_gold == self._gold_set)

    def soft_f1(self) -> float | None:
        """None when the prediction had more distinct rows than `max_distinct` and some
        paired row shares a value with its gold row."""
        if self._distinct is None:
            assert self._paired_head is not None
            if len(self._paired_head) < len(self._gold_distinct):
                return None  # more gold rows than the limit: cannot happen in the benchmark
            tp = sum(
                _row_match(p, g)[0]
                for p, g in zip(self._paired_head, self._gold_distinct, strict=True)
            )
            return 0 if tp == 0 else None
        if self.rows == 0 and not self.gold:
            return 1.0
        return _soft_f1_distinct(list(self._distinct), self._gold_distinct)
