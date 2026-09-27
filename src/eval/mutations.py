"""Gold queries altered on purpose, for checking a scorer's verdicts against the official ones.

Each mutation makes one small change of the kind a text-to-SQL model gets wrong: a missing or
extra DISTINCT, a changed LIMIT or sort direction, a dropped filter condition, an off-by-one
comparison or literal, another aggregate, columns swapped, dropped, repeated or cast to another
type, an inner join made a left join. Some changes leave the result as it was (a sort order
under no LIMIT, a DISTINCT over rows that are already distinct, a cast of whole numbers to
REAL), so the scorers are compared on verdicts of 1 as well as 0.

Mutations work on sqlglot's syntax tree of the outermost SELECT and print it back as
PostgreSQL. A mutation that does not apply to a query (no LIMIT to change, one column only) is
skipped; which of the applicable ones a query gets is drawn with a seeded generator.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable

import sqlglot
from sqlglot import exp

logging.getLogger("sqlglot").setLevel(logging.ERROR)

Mutation = Callable[[exp.Select], bool]  # changes the SELECT in place; False if it cannot


def _toggle_distinct(sel: exp.Select) -> bool:
    sel.set("distinct", None if sel.args.get("distinct") else exp.Distinct())
    return True


def _limit_plus_one(sel: exp.Select) -> bool:
    limit = sel.args.get("limit")
    n = limit.expression if isinstance(limit, exp.Limit) else None
    if not (isinstance(n, exp.Literal) and not n.is_string and n.this.isdigit()):
        return False
    limit.set("expression", exp.Literal.number(int(n.this) + 1))
    return True


def _drop_limit(sel: exp.Select) -> bool:
    if not sel.args.get("limit"):
        return False
    sel.set("limit", None)
    sel.set("offset", None)
    return True


def _flip_order(sel: exp.Select) -> bool:
    order = sel.args.get("order")
    if not order or not order.expressions:
        return False
    first = order.expressions[0]
    first.set("desc", not first.args.get("desc"))
    first.set("nulls_first", None)
    return True


def _drop_order(sel: exp.Select) -> bool:
    if not sel.args.get("order"):
        return False
    sel.set("order", None)
    return True


def _drop_condition(sel: exp.Select) -> bool:
    """The last condition of the WHERE clause, or the whole clause if it has one."""
    where = sel.args.get("where")
    if not where:
        return False
    if isinstance(where.this, exp.And):
        where.set("this", where.this.this)
    else:
        sel.set("where", None)
    return True


_COMPARISON_SWAP: dict[type, type] = {
    exp.GT: exp.GTE,
    exp.GTE: exp.GT,
    exp.LT: exp.LTE,
    exp.LTE: exp.LT,
    exp.EQ: exp.NEQ,
}


def _shift_comparison(sel: exp.Select) -> bool:
    where = sel.args.get("where")
    node = next((n for n in where.find_all(*_COMPARISON_SWAP)), None) if where else None
    if node is None:
        return False
    node.replace(_COMPARISON_SWAP[type(node)](this=node.this, expression=node.expression))
    return True


def _shift_literal(sel: exp.Select) -> bool:
    """A number in the WHERE clause plus one, or else a string in it in upper case."""
    where = sel.args.get("where")
    if not where:
        return False
    for lit in where.find_all(exp.Literal):
        if not lit.is_string and lit.this.isdigit():
            lit.replace(exp.Literal.number(int(lit.this) + 1))
            return True
    for lit in where.find_all(exp.Literal):
        if lit.is_string and lit.this.upper() != lit.this:
            lit.replace(exp.Literal.string(lit.this.upper()))
            return True
    return False


def _swap_aggregate(sel: exp.Select) -> bool:
    for e in sel.expressions:
        for node in e.find_all(exp.Max, exp.Min, exp.Sum, exp.Avg, exp.Count):
            if isinstance(node, exp.Count):
                inner = node.this
                if isinstance(inner, exp.Distinct):  # COUNT(DISTINCT x) -> COUNT(x)
                    node.set("this", inner.expressions[0])
                    return True
                if isinstance(inner, exp.Star) or inner is None:
                    continue
                node.set("this", exp.Distinct(expressions=[inner]))
                return True
            swap = {exp.Max: exp.Min, exp.Min: exp.Max, exp.Sum: exp.Avg, exp.Avg: exp.Sum}
            node.replace(swap[type(node)](this=node.this))
            return True
    return False


def _swap_columns(sel: exp.Select) -> bool:
    cols = list(sel.expressions)
    if len(cols) < 2 or any(isinstance(c, exp.Star) for c in cols):
        return False
    cols[0], cols[1] = cols[1], cols[0]
    sel.set("expressions", cols)
    return True


def _drop_column(sel: exp.Select) -> bool:
    cols = list(sel.expressions)
    if len(cols) < 2:
        return False
    sel.set("expressions", cols[:-1])
    return True


def _repeat_column(sel: exp.Select) -> bool:
    cols = list(sel.expressions)
    if not cols or isinstance(cols[0], exp.Star):
        return False
    sel.set("expressions", [*cols, cols[0].copy()])
    return True


def _cast_first_column(to: str) -> Mutation:
    def mutate(sel: exp.Select) -> bool:
        cols = list(sel.expressions)
        if not cols or isinstance(cols[0], exp.Star) or cols[0].find(exp.Star):
            return False
        first = cols[0]
        target = first.this if isinstance(first, exp.Alias) else first
        target.replace(exp.Cast(this=target.copy(), to=exp.DataType.build(to)))
        return True

    return mutate


def _left_join(sel: exp.Select) -> bool:
    for join in sel.args.get("joins") or []:
        if not join.args.get("side") and join.args.get("kind") in (None, "INNER"):
            join.set("kind", None)
            join.set("side", "LEFT")
            return True
    return False


MUTATIONS: dict[str, Mutation] = {
    "toggle_distinct": _toggle_distinct,
    "limit_plus_one": _limit_plus_one,
    "drop_limit": _drop_limit,
    "flip_order": _flip_order,
    "drop_order": _drop_order,
    "drop_condition": _drop_condition,
    "shift_comparison": _shift_comparison,
    "shift_literal": _shift_literal,
    "swap_aggregate": _swap_aggregate,
    "swap_columns": _swap_columns,
    "drop_column": _drop_column,
    "repeat_column": _repeat_column,
    "cast_real": _cast_first_column("REAL"),
    "cast_text": _cast_first_column("TEXT"),
    "left_join": _left_join,
}


def apply(sql: str, kind: str) -> str | None:
    """The query with one mutation, or None if it does not apply or changes nothing."""
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except sqlglot.errors.SqlglotError:
        return None
    sel = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if sel is None:
        return None
    before = tree.sql(dialect="postgres")
    try:
        if not MUTATIONS[kind](sel):
            return None
        after = tree.sql(dialect="postgres")
    except (sqlglot.errors.SqlglotError, AttributeError, IndexError, TypeError, ValueError):
        return None
    return after if after != before else None


def mutants(sql: str, per_query: int, rng: random.Random) -> list[tuple[str, str]]:
    """Up to `per_query` distinct mutants of one query, as (mutation, SQL)."""
    applicable = [(k, m) for k in MUTATIONS if (m := apply(sql, k)) is not None]
    rng.shuffle(applicable)
    out: list[tuple[str, str]] = []
    seen = {sql.strip()}
    for kind, text in applicable:
        if text not in seen:
            out.append((kind, text))
            seen.add(text)
        if len(out) == per_query:
            break
    return out
