"""The automatic checks on an answer's result, and design 4's agreement between samples.

**Checks** (every design's record carries them; only designs 4 and 5 use them in confidence):
- `empty`: the result has no rows;
- `repeated_rows`: two identical rows;
- `out_of_range`: a value outside its possible range. Which output columns have a range is read
  from the query's select list: a count (a COUNT aggregate, or a name like `count`, `cnt`,
  `num_...`) or an amount (an expression over a column whose name contains `amount`) cannot be
  negative, unless the expression subtracts or negates; a share (a name containing `percent`,
  `pct` or `share`, or an expression multiplying a division by 100) must lie from 0 to 100;
- `too_many_rows`: more rows than the configured limit.

**Agreement** (designs 4 and 5): the samples' results are compared as the benchmark compares a
prediction with its gold, as sets of rows (so 1 and 1.0 are the same value, and order and
repeats do not matter). Samples that declined agree with each other; a sample with no SQL, or
whose SQL failed, agrees with no other. The answer is the one given by the largest group of
agreeing samples; on a tie, the group of the earliest sample. Confidence is the share of
samples in that group, halved if any check fails on its result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import sqlglot
from sqlglot import exp

from src.db.execute import Limits, QueryError, ReadOnlyExecutor
from src.db.guard import QueryGuard

_COUNT_NAME = re.compile(r"(^|_)(count|cnt|num|number)(_|$)|^count", re.IGNORECASE)
_SHARE_NAME = re.compile(r"percent|pct|share", re.IGNORECASE)
_AMOUNT = re.compile(r"amount", re.IGNORECASE)


@dataclass
class FinalResult:
    """The final SQL's result as the checks and the agreement use it."""

    ok: bool
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    truncated: bool = False
    error: dict | None = None

    def row_set(self) -> frozenset:
        try:
            return frozenset(self.rows)
        except TypeError:  # an unhashable value (an array): compare the rows' text instead
            return frozenset(repr(r) for r in self.rows)


def fetch(
    guard: QueryGuard, executor: ReadOnlyExecutor, sql: str | None, limits: Limits
) -> FinalResult:
    if not sql:
        return FinalResult(False, error={"kind": "no_sql", "message": "no SQL"})
    verdict = guard.check(sql)
    if not verdict.allowed:
        return FinalResult(False, error={"kind": "refused", "message": "; ".join(verdict.reasons)})
    try:
        r = executor.execute(verdict.query, limits)
    except QueryError as e:
        return FinalResult(False, error={"kind": e.kind, "message": e.message})
    return FinalResult(True, [c for c, _ in r.columns], [tuple(x) for x in r.rows], r.truncated)


# ------------------------------------------------------------------------------ checks


def _is_percentage(e: exp.Expression) -> bool:
    """A division multiplied by 100, in either nesting (`a * 100 / b` or `a / b * 100`)."""
    hundred = any(
        isinstance(x, exp.Literal) and x.this in ("100", "100.0")
        for m in e.find_all(exp.Mul)
        for x in (m.this, m.expression)
    )
    return hundred and e.find(exp.Div) is not None


def _range_kinds(sql: str, names: list[str]) -> list[str | None]:
    """Per output column: `count`, `amount`, `share` or None."""
    kinds: list[str | None] = [None] * len(names)
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except sqlglot.errors.SqlglotError:
        tree = None
    select = tree if isinstance(tree, exp.Select) else (tree.find(exp.Select) if tree else None)
    projections = list(select.expressions) if select is not None else []
    for i, name in enumerate(names):
        e = projections[i] if i < len(projections) and len(projections) == len(names) else None
        inner = e.this if isinstance(e, exp.Alias) else e
        signed = inner is not None and any(inner.find_all(exp.Sub, exp.Neg))
        if _SHARE_NAME.search(name) or (inner is not None and _is_percentage(inner)):
            kinds[i] = "share"
        elif signed:
            kinds[i] = None
        elif (inner is not None and inner.find(exp.Count) is not None) or _COUNT_NAME.search(name):
            kinds[i] = "count"
        elif inner is not None and any(_AMOUNT.search(c.name) for c in inner.find_all(exp.Column)):
            kinds[i] = "amount"
    return kinds


def _number(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int | float | Decimal):
        return float(v)
    return None


def checks(sql: str | None, result: FinalResult, max_rows: int) -> dict[str, Any]:
    """The four checks on a successful result; `failed` is true if any fails."""
    if not result.ok:
        return {"applicable": False, "failed": False}
    out_of_range = []
    for i, kind in enumerate(_range_kinds(sql or "", result.columns)):
        if kind is None:
            continue
        for row in result.rows:
            x = _number(row[i]) if i < len(row) else None
            if x is None:
                continue
            if (kind in ("count", "amount") and x < 0) or (kind == "share" and not 0 <= x <= 100):
                out_of_range.append({"column": result.columns[i], "kind": kind, "value": x})
                break
    found = {
        "applicable": True,
        "empty": not result.rows,
        "repeated_rows": len(set(map(repr, result.rows))) < len(result.rows),
        "out_of_range": out_of_range,
        "too_many_rows": result.truncated or len(result.rows) > max_rows,
    }
    found["failed"] = bool(
        found["empty"] or found["repeated_rows"] or out_of_range or found["too_many_rows"]
    )
    return found


# ------------------------------------------------------------------------------ agreement


@dataclass
class Vote:
    chosen: int  # index of the sample whose answer is given
    group: list[int]  # the samples agreeing with it
    groups: list[list[int]]
    confidence: float  # the share agreeing, halved if a check failed


def vote(samples: list[tuple[bool, FinalResult]], checks_failed: list[bool]) -> Vote:
    """samples: per sample (declined, its result). checks_failed: per sample."""
    groups: list[list[int]] = []
    keys: list[Any] = []
    for i, (declined, result) in enumerate(samples):
        if declined:
            key: Any = ("declined",)
        elif result.ok:
            key = ("rows", result.row_set())
        else:
            key = ("alone", i)
        if key in keys:
            groups[keys.index(key)].append(i)
        else:
            keys.append(key)
            groups.append([i])
    best = max(groups, key=lambda g: (len(g), -g[0]))  # largest; on a tie, the earliest sample
    chosen = best[0]
    confidence = len(best) / len(samples)
    if checks_failed[chosen]:
        confidence /= 2
    return Vote(chosen, best, groups, confidence)
