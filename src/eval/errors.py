"""Where a wrong answer goes wrong: its SQL compared with the expert SQL, and its rows with the
expert rows.

**The query's shape.** Each query is reduced to comparable parts, read from its syntax tree
(sqlglot) over the whole query, subqueries included:
- `tables`: the tables it reads (intermediate results of a WITH are not tables);
- `joins`: the column pairs it matches with `=` (join keys, in ON or WHERE; USING gives one name);
- `filter_columns` and `filter_values`: the columns and the literal values in its other WHERE and
  HAVING conditions (a condition's own subquery is not part of it);
- `computation`: its aggregates (COUNT(DISTINCT ...) apart) and the arithmetic in its select
  lists, as a multiset, and whether it removes repeated rows;
- `output_columns` and `output_width`: the columns its outermost select list refers to, and how
  many columns it returns;
- `order` and `limit`: the outermost ORDER BY (columns and direction) and LIMIT.
Names are compared in lower case without their table (aliases differ between writers); values
exactly (a code's case matters), numbers as numbers.

**The category.** A wrong answer takes the first that applies:
1. `no_result`: its query was refused by the checker or failed (from the scoring record);
2. `format_only`: a near miss, right up to its columns or rounding (src/eval/near_miss.py);
3. `tables`: it reads different tables (the schema was linked wrongly);
4. `join`: the same tables, matched on different keys;
5. `filter`: different conditions: a condition `missing` or `extra`, a `different_column`, or the
   same columns with a `different_value`;
6. `computation`: a different aggregate, formula or removal of repeats;
7. `output`: a different column (or number of columns) returned;
8. `order_limit`: a different order or limit (which rows a "top N" keeps);
9. `other`: none of these parts differs (the difference is in a function, a type or a date
   handling the parts do not capture), and `unparsed` if either query cannot be read.
Every part that differs is kept as well (`differences`), so an answer wrong in several ways shows.

**The result relation**, from both queries' rows (independent of how the SQL is written):
`same_rows_other_shape` (a near miss), `subset` (it misses some expert rows), `superset` (it has
extra rows), `overlap`, `disjoint`, `empty` (it returned nothing), or, when the widths differ,
`fewer_rows`, `same_row_count` or `more_rows`.

Syntax-level categories can mislabel a different but equivalent formulation (a subquery where the
expert joins), so a sample is checked by hand (`agreement`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import sqlglot
from sqlglot import exp

CATEGORIES = (
    "no_result",
    "format_only",
    "tables",
    "join",
    "filter",
    "computation",
    "output",
    "order_limit",
    "other",
    "unparsed",
)
PARTS = ("tables", "joins", "filters", "computation", "output", "order_limit")
_ARITH = (exp.Add, exp.Sub, exp.Mul, exp.Div)


@dataclass(frozen=True)
class Shape:
    tables: frozenset
    joins: frozenset
    filter_columns: frozenset
    filter_values: frozenset
    computation: tuple
    distinct: bool
    output_columns: frozenset
    output_width: int
    order: tuple
    limit: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            k: sorted(map(str, v)) if isinstance(v, frozenset | tuple) else v
            for k, v in asdict(self).items()
        }


def _own(node: exp.Expression, kind: type) -> list:
    """The nodes of `kind` under `node`, not descending into a nested SELECT."""
    out = []
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, kind):
            out.append(n)
        for child in n.iter_expressions():
            if not isinstance(child, exp.Select | exp.Subquery):
                stack.append(child)
    return out


def _value(e: exp.Expression) -> str | None:
    """A literal as a comparable string (numbers as numbers, text quoted), else None."""
    negative = False
    if isinstance(e, exp.Neg) and isinstance(e.this, exp.Literal):
        negative, e = True, e.this
    if not isinstance(e, exp.Literal):
        return None
    if e.is_string:
        return f"'{e.this}'"
    try:
        x = float(e.this)
    except ValueError:
        return e.this
    x = -x if negative else x
    return str(int(x)) if x.is_integer() else repr(x)


def _conjuncts(e: exp.Expression | None) -> list[exp.Expression]:
    if e is None:
        return []
    while isinstance(e, exp.Paren):
        e = e.this
    if isinstance(e, exp.And):
        return _conjuncts(e.left) + _conjuncts(e.right)
    return [e]


def _is_join_key(e: exp.Expression) -> bool:
    return (
        isinstance(e, exp.EQ) and isinstance(e.left, exp.Column) and isinstance(e.right, exp.Column)
    )


def shape(sql: str) -> Shape | None:
    """The query's comparable parts, or None if sqlglot cannot read it."""
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except sqlglot.errors.SqlglotError:
        return None
    if tree is None:
        return None
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    tables = frozenset(t.name.lower() for t in tree.find_all(exp.Table)) - ctes

    joins: set[frozenset] = set()
    for eq in tree.find_all(exp.EQ):
        if _is_join_key(eq):
            joins.add(frozenset({eq.left.name.lower(), eq.right.name.lower()}))
    for j in tree.find_all(exp.Join):
        for u in j.args.get("using") or []:
            joins.add(frozenset({u.name.lower()}))

    columns: set[str] = set()
    values: set[str] = set()
    for clause in [*tree.find_all(exp.Where), *tree.find_all(exp.Having)]:
        for cond in _conjuncts(clause.this):
            if _is_join_key(cond):
                continue
            columns |= {c.name.lower() for c in _own(cond, exp.Column) if c.name}
            for lit in _own(cond, exp.Literal):
                parent = lit.parent
                v = _value(parent if isinstance(parent, exp.Neg) else lit)
                if v is not None:
                    values.add(v)

    computation: list[str] = []
    for agg in tree.find_all(exp.AggFunc):
        name = type(agg).__name__.lower()
        if isinstance(agg, exp.Count) and isinstance(agg.this, exp.Distinct):
            name = "count_distinct"
        computation.append(name)
    for select in tree.find_all(exp.Select):
        for projection in select.expressions:
            computation += [type(a).__name__.lower() for a in _own(projection, _ARITH)]

    outer = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    projections = list(outer.expressions) if outer is not None else []
    output = frozenset(c.name.lower() for p in projections for c in _own(p, exp.Column) if c.name)
    order_node = tree.args.get("order")
    order = tuple(
        (tuple(sorted(c.name.lower() for c in _own(o, exp.Column))), bool(o.args.get("desc")))
        for o in (order_node.expressions if order_node else [])
    )
    limit_node = tree.args.get("limit")
    limit = None
    if limit_node is not None and isinstance(limit_node.expression, exp.Literal):
        try:
            limit = int(limit_node.expression.this)
        except ValueError:
            limit = None
    return Shape(
        tables=tables,
        joins=frozenset(joins),
        filter_columns=frozenset(columns),
        filter_values=frozenset(values),
        computation=tuple(sorted(computation)),
        distinct=any(s.args.get("distinct") is not None for s in tree.find_all(exp.Select)),
        output_columns=output,
        output_width=len(projections),
        order=order,
        limit=limit,
    )


def differences(pred: Shape, gold: Shape) -> dict[str, dict]:
    """Every part in which the answer's query differs from the expert query."""
    out: dict[str, dict] = {}
    if pred.tables != gold.tables:
        out["tables"] = {
            "missing": sorted(gold.tables - pred.tables),
            "extra": sorted(pred.tables - gold.tables),
        }
    if pred.joins != gold.joins:
        out["joins"] = {
            "missing": sorted("=".join(sorted(k)) for k in gold.joins - pred.joins),
            "extra": sorted("=".join(sorted(k)) for k in pred.joins - gold.joins),
        }
    if pred.filter_columns != gold.filter_columns or pred.filter_values != gold.filter_values:
        missing = sorted(gold.filter_columns - pred.filter_columns)
        extra = sorted(pred.filter_columns - gold.filter_columns)
        if missing and extra:
            kind = "different_column"
        elif missing:
            kind = "missing"
        elif extra:
            kind = "extra"
        else:
            kind = "different_value"
        out["filters"] = {
            "kind": kind,
            "missing_columns": missing,
            "extra_columns": extra,
            "missing_values": sorted(gold.filter_values - pred.filter_values),
            "extra_values": sorted(pred.filter_values - gold.filter_values),
        }
    if pred.computation != gold.computation or pred.distinct != gold.distinct:
        missing = Counter(gold.computation) - Counter(pred.computation)
        extra = Counter(pred.computation) - Counter(gold.computation)
        out["computation"] = {
            "missing": sorted(missing.elements()),
            "extra": sorted(extra.elements()),
            "distinct": [gold.distinct, pred.distinct],
        }
    if pred.output_columns != gold.output_columns or pred.output_width != gold.output_width:
        out["output"] = {
            "missing": sorted(gold.output_columns - pred.output_columns),
            "extra": sorted(pred.output_columns - gold.output_columns),
            "width": [gold.output_width, pred.output_width],
        }
    if pred.order != gold.order or pred.limit != gold.limit:
        out["order_limit"] = {
            "limit": [gold.limit, pred.limit],
            "order_differs": pred.order != gold.order,
        }
    return out


_PART_CATEGORY = {
    "tables": "tables",
    "joins": "join",
    "filters": "filter",
    "computation": "computation",
    "output": "output",
    "order_limit": "order_limit",
}


def categorize(
    outcome: str, near_kind: str | None, pred_sql: str | None, gold_sql: str
) -> dict[str, Any]:
    """The category of one wrong answer, with every difference found.

    outcome: the scoring record's `score_outcome` (`ok` when the query ran); near_kind: its
    near-miss kind (src/eval/near_miss.py), None when not checked."""
    if outcome != "ok" or not pred_sql:
        return {"category": "no_result", "subkind": outcome, "differences": {}}
    pred, gold = shape(pred_sql), shape(gold_sql)
    if pred is None or gold is None:
        return {"category": "unparsed", "subkind": None, "differences": {}}
    found = differences(pred, gold)
    if near_kind in ("columns", "rounding", "columns_and_rounding"):
        return {"category": "format_only", "subkind": near_kind, "differences": found}
    for part in PARTS:
        if part in found:
            sub = found[part].get("kind") if part == "filters" else None
            return {"category": _PART_CATEGORY[part], "subkind": sub, "differences": found}
    return {"category": "other", "subkind": None, "differences": found}


def result_relation(
    pred: Sequence[tuple], gold: Sequence[tuple], pred_width: int, gold_width: int
) -> str:
    """How the answer's rows relate to the expert rows, as sets."""
    if not pred:
        return "empty"
    if pred_width != gold_width:
        if len(pred) < len(gold):
            return "fewer_rows"
        return "same_row_count" if len(pred) == len(gold) else "more_rows"
    p, g = set(map(tuple, pred)), set(map(tuple, gold))
    if p == g:
        return "same_rows_other_shape"
    if p < g:
        return "subset"
    if p > g:
        return "superset"
    return "overlap" if p & g else "disjoint"


def agreement(automatic: Sequence[str], by_hand: Sequence[str]) -> dict[str, Any]:
    """Share of equal labels and Cohen's kappa over any number of categories (None when the
    labels leave no room for agreement beyond chance)."""
    if len(automatic) != len(by_hand) or not automatic:
        raise ValueError("two equal-length, non-empty label lists are needed")
    a, b = np.asarray(automatic), np.asarray(by_hand)
    observed = float(np.mean(a == b))
    labels = sorted(set(a) | set(b))
    expected = float(sum(np.mean(a == c) * np.mean(b == c) for c in labels))
    kappa = None if expected == 1 else (observed - expected) / (1 - expected)
    return {"items": len(a), "same": int((a == b).sum()), "share_same": observed, "kappa": kappa}


def hand_check_sample(rows: Sequence[dict], size: int, seed: int) -> list[dict]:
    """About `size` answers spread over the automatic categories: each category gets a share in
    proportion to its size, at least one, all of it if smaller; drawn with a seeded generator
    from the answers sorted by question id."""
    by_cat: dict[str, list[dict]] = {}
    for r in sorted(rows, key=lambda r: r["question_id"]):
        by_cat.setdefault(r["category"], []).append(r)
    total = len(rows)
    rng = np.random.default_rng(seed)
    chosen = []
    for cat in CATEGORIES:
        members = by_cat.get(cat, [])
        if not members:
            continue
        k = min(len(members), max(1, round(size * len(members) / total)))
        chosen += [members[i] for i in sorted(rng.permutation(len(members))[:k])]
    return sorted(chosen, key=lambda r: r["question_id"])
