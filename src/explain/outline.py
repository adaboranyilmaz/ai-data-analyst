"""A query outline: what a SELECT does, as structured parts that can be put in words.

For a reader who does not read SQL: which tables the query uses and how it links them, which
rows it keeps, how it groups them, what it returns, and how it sorts and limits the result. It
is built from the query's syntax tree (sqlglot), with no model, so the same query always gives
the same outline.

An outline never paraphrases what it cannot express exactly: a condition or value containing a
subquery, a window function or a function without a template here is marked `unrendered`, to
be shown as a part the outline cannot put in words (see the SQL). A query that
combines queries (UNION, EXCEPT, INTERSECT) is outlined as `combined` only.

A phrase is a list of tokens, each a dict:
  {"t": "col", "v": "table.column"}   a column (table aliases resolved to table names)
  {"t": "lit", "v": "0.1"}            a literal, strings in quotes
  {"t": "op", "v": "gt"}              an operator or word, put in words in the reader's language
  {"t": "open"} / {"t": "close"}      grouping brackets
  {"t": "agg", "v": "avg", "arg": phrase}              an aggregate over a phrase
  {"t": "fn", "v": "round", "args": [phrase, ...]}     a function with a template
The outline is JSON-ready (lists and dicts only).
"""

from __future__ import annotations

from typing import Any

import sqlglot
from sqlglot import exp

Phrase = list[dict[str, Any]]

_COMPARE = {
    exp.EQ: "eq",
    exp.NEQ: "neq",
    exp.GT: "gt",
    exp.GTE: "gte",
    exp.LT: "lt",
    exp.LTE: "lte",
}
_ARITH = {exp.Add: "plus", exp.Sub: "minus", exp.Mul: "times", exp.Div: "divided_by"}
_AGG = {exp.Sum: "sum", exp.Avg: "avg", exp.Max: "max", exp.Min: "min"}
_EXTRACT_UNITS = {"YEAR": "year_of", "MONTH": "month_of", "DAY": "day_of"}
_NEGATABLE = (exp.Is, exp.In, exp.Like, exp.ILike, exp.Between)


class Unrendered(Exception):
    """The expression has a part the outline cannot put in words exactly."""


def _aliases(select: exp.Select) -> dict[str, str]:
    """Alias (or bare name) -> table name, for the tables in FROM and the joins."""
    out: dict[str, str] = {}
    for t in select.find_all(exp.Table):
        if t.parent_select is not select:  # a table of a subquery or an intermediate result
            continue
        out[t.alias_or_name] = t.name
        out[t.name] = t.name
    return out


def _literal(e: exp.Literal) -> str:
    return f'"{e.this}"' if e.is_string else str(e.this)


class _Words:
    def __init__(self, aliases: dict[str, str]):
        self.aliases = aliases

    def column(self, c: exp.Column) -> Phrase:
        if isinstance(c.this, exp.Star):
            return [{"t": "op", "v": "all_columns"}]
        table = self.aliases.get(c.table, c.table) if c.table else ""
        return [{"t": "col", "v": f"{table}.{c.name}" if table else c.name}]

    def phrase(self, e: exp.Expression, nested: bool = False) -> Phrase:
        """The expression as a phrase; raises Unrendered. `nested` brackets a combination of
        conditions (AND/OR) inside another."""
        # sqlglot writes some negations as a flag on the node (`x IS NOT NULL` is an Is with
        # negate): a node the outline cannot negate is never shown without its NOT
        if e.args.get("negate") and not isinstance(e, _NEGATABLE):
            raise Unrendered
        if isinstance(e, exp.Paren):
            return self.phrase(e.this, nested)
        if isinstance(e, exp.Alias):
            return self.phrase(e.this, nested)
        if isinstance(e, exp.Column):
            return self.column(e)
        if isinstance(e, exp.Star):
            return [{"t": "op", "v": "all_columns"}]
        if isinstance(e, exp.Literal):
            return [{"t": "lit", "v": _literal(e)}]
        if isinstance(e, exp.Boolean):
            return [{"t": "op", "v": "true" if e.this else "false"}]
        if isinstance(e, exp.Null):
            return [{"t": "op", "v": "null"}]
        if isinstance(e, exp.Cast | exp.TryCast):
            return self.phrase(e.this, nested)
        if isinstance(e, exp.Lower | exp.Upper):
            return self.phrase(e.this, nested)
        if isinstance(e, exp.And | exp.Or):
            word = "and" if isinstance(e, exp.And) else "or"
            inner = [
                *self.phrase(e.left, True),
                {"t": "op", "v": word},
                *self.phrase(e.right, True),
            ]
            return [{"t": "open"}, *inner, {"t": "close"}] if nested else inner
        if isinstance(e, exp.Not):
            if isinstance(e.this, exp.Is):
                if not isinstance(e.this.expression, exp.Null):
                    raise Unrendered
                word = "is_empty" if e.this.args.get("negate") else "is_not_empty"
                return [*self.phrase(e.this.this, True), {"t": "op", "v": word}]
            if isinstance(e.this, exp.In | exp.Like | exp.ILike | exp.Between):
                return self._matching(e.this, negated=True)
            return [{"t": "op", "v": "not"}, {"t": "open"}, *self.phrase(e.this), {"t": "close"}]
        if isinstance(e, exp.Is):
            if isinstance(e.expression, exp.Null):
                word = "is_not_empty" if e.args.get("negate") else "is_empty"
                return [*self.phrase(e.this, True), {"t": "op", "v": word}]
            raise Unrendered
        for kind, word in _COMPARE.items():
            if isinstance(e, kind):
                return [
                    *self.phrase(e.left, True),
                    {"t": "op", "v": word},
                    *self.phrase(e.right, True),
                ]
        for kind, word in _ARITH.items():
            if isinstance(e, kind):
                inner = [
                    *self.phrase(e.left, True),
                    {"t": "op", "v": word},
                    *self.phrase(e.right, True),
                ]
                return [{"t": "open"}, *inner, {"t": "close"}] if nested else inner
        if isinstance(e, exp.Neg):
            return [{"t": "op", "v": "minus"}, *self.phrase(e.this, True)]
        if isinstance(e, exp.In | exp.Like | exp.ILike | exp.Between):
            return self._matching(e, negated=False)
        if isinstance(e, exp.Count):
            arg = e.this
            if arg is None or isinstance(arg, exp.Star):
                return [{"t": "agg", "v": "count_rows", "arg": []}]
            if isinstance(arg, exp.Distinct):
                if len(arg.expressions) != 1:
                    raise Unrendered
                return [{"t": "agg", "v": "count_distinct", "arg": self.phrase(arg.expressions[0])}]
            return [{"t": "agg", "v": "count", "arg": self.phrase(arg)}]
        for kind, word in _AGG.items():
            if isinstance(e, kind):
                if isinstance(e.this, exp.Distinct):
                    raise Unrendered
                return [{"t": "agg", "v": word, "arg": self.phrase(e.this)}]
        if isinstance(e, exp.Round):
            places = e.args.get("decimals")
            args = [self.phrase(e.this)] + ([self.phrase(places)] if places is not None else [])
            return [
                {"t": "fn", "v": "round" if places is not None else "round_whole", "args": args}
            ]
        if isinstance(e, exp.Extract):
            unit = e.this.name.upper() if isinstance(e.this, exp.Var | exp.Identifier) else ""
            if unit not in _EXTRACT_UNITS:
                raise Unrendered
            source = e.expression
            if isinstance(source, exp.Anonymous) and source.name.upper() == "AGE":
                if unit != "YEAR" or len(source.expressions) != 2:
                    raise Unrendered
                end, start = source.expressions  # AGE(end, start)
                return [
                    {"t": "fn", "v": "whole_years", "args": [self.phrase(start), self.phrase(end)]}
                ]
            return [{"t": "fn", "v": _EXTRACT_UNITS[unit], "args": [self.phrase(source)]}]
        if isinstance(e, exp.Nullif):
            return [
                {"t": "fn", "v": "nullif", "args": [self.phrase(e.this), self.phrase(e.expression)]}
            ]
        if isinstance(e, exp.Abs):
            return [{"t": "fn", "v": "abs", "args": [self.phrase(e.this)]}]
        if isinstance(e, exp.DPipe):
            return [{"t": "fn", "v": "concat", "args": [self.phrase(e.left), self.phrase(e.right)]}]
        if isinstance(e, exp.Substring):
            start, length = e.args.get("start"), e.args.get("length")
            if start is None or length is None:
                raise Unrendered
            args = [self.phrase(e.this), self.phrase(start), self.phrase(length)]
            return [{"t": "fn", "v": "substring", "args": args}]
        if isinstance(e, exp.Coalesce):
            rest = e.expressions
            if len(rest) != 1:
                raise Unrendered
            return [
                {"t": "fn", "v": "coalesce", "args": [self.phrase(e.this), self.phrase(rest[0])]}
            ]
        if isinstance(e, exp.Case):
            ifs = e.args.get("ifs") or []
            if len(ifs) != 1 or e.this is not None:
                raise Unrendered
            default = e.args.get("default")
            args = [self.phrase(ifs[0].args["true"]), self.phrase(ifs[0].this)]
            if default is None:
                return [{"t": "fn", "v": "case_no_else", "args": args}]
            return [{"t": "fn", "v": "case", "args": [*args, self.phrase(default)]}]
        raise Unrendered

    def _matching(self, e: exp.Expression, negated: bool) -> Phrase:
        negated = negated != bool(e.args.get("negate"))
        subject = self.phrase(e.this, True)
        if isinstance(e, exp.In):
            if e.args.get("query") is not None or not e.expressions:
                raise Unrendered
            items: Phrase = []
            for i, x in enumerate(e.expressions):
                if i:
                    items.append({"t": "op", "v": "list_sep"})
                items += self.phrase(x, True)
            word = "not_in" if negated else "in"
            return [*subject, {"t": "op", "v": word}, {"t": "open"}, *items, {"t": "close"}]
        if isinstance(e, exp.Like | exp.ILike):
            word = "not_like" if negated else "like"
            return [*subject, {"t": "op", "v": word}, *self.phrase(e.expression, True)]
        word = "not_between" if negated else "between"
        return [
            *subject,
            {"t": "op", "v": word},
            *self.phrase(e.args["low"], True),
            {"t": "op", "v": "and"},
            *self.phrase(e.args["high"], True),
        ]


def _conjuncts(e: exp.Expression | None) -> list[exp.Expression]:
    if e is None:
        return []
    while isinstance(e, exp.Paren):
        e = e.this
    if isinstance(e, exp.And):
        return _conjuncts(e.left) + _conjuncts(e.right)
    return [e]


def _part(words: _Words, e: exp.Expression, nested: bool = False) -> dict[str, Any]:
    """One condition or value: its phrase, or `unrendered`."""
    if e.find(exp.Subquery, exp.Select, exp.Window) is not None:
        return {"unrendered": True}
    try:
        return {"phrase": words.phrase(e, nested)}
    except Unrendered:
        return {"unrendered": True}


def _conditions(words: _Words, e: exp.Expression | None) -> list[dict[str, Any]]:
    """The conditions joined by AND, one part each; a condition combining others (OR) is
    bracketed when it is not alone."""
    parts = _conjuncts(e)
    return [_part(words, c, nested=len(parts) > 1) for c in parts]


def outline(sql: str) -> dict[str, Any]:
    """The outline of one query; `{"kind": "unparsed"}` if sqlglot cannot read it."""
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except sqlglot.errors.SqlglotError:
        return {"kind": "unparsed"}
    ctes: list[str] = []
    if isinstance(tree, exp.Select) and tree.args.get("with_"):
        ctes = [c.alias_or_name for c in tree.args["with_"].expressions]
    if isinstance(tree, exp.Union | exp.Except | exp.Intersect):
        return {"kind": "combined", "how": type(tree).__name__.lower()}
    if not isinstance(tree, exp.Select):
        return {"kind": "unparsed"}

    words = _Words(_aliases(tree))
    # every table the query reads, in its subqueries and intermediate results too
    tables: list[dict[str, Any]] = []
    for t in tree.find_all(exp.Table):
        if t.name not in ctes and {"name": t.name} not in tables:
            tables.append({"name": t.name})
    links = []
    for j in tree.args.get("joins") or []:
        side = (j.side or "").lower()
        kind = j.kind.lower() if j.kind else ""
        links.append(
            {
                "table": j.this.name if isinstance(j.this, exp.Table) else None,
                "keep_unmatched": side in ("left", "right", "full"),
                "side": side or kind or "inner",
                "on": _conditions(words, j.args.get("on")),
                "using": [u.name for u in j.args.get("using") or []],
            }
        )
    where, having = tree.args.get("where"), tree.args.get("having")
    group = tree.args.get("group")
    order = tree.args.get("order")
    return {
        "kind": "select",
        "intermediate_results": ctes,
        "tables": tables,
        "links": links,
        "filters": _conditions(words, where.this if where else None),
        "groups": [_part(words, g) for g in (group.expressions if group else [])],
        "group_filters": _conditions(words, having.this if having else None),
        "distinct": tree.args.get("distinct") is not None,
        "returns": [
            {**_part(words, p), "name": p.alias if isinstance(p, exp.Alias) else None}
            for p in tree.expressions
        ],
        "order": [
            {**_part(words, o.this), "descending": bool(o.args.get("desc"))}
            for o in (order.expressions if order else [])
        ],
        "limit": _count(tree.args.get("limit")),
        "offset": _count(tree.args.get("offset")),
    }


def _count(node: exp.Expression | None) -> int | None:
    if node is None:
        return None
    value = node.expression if isinstance(node, exp.Limit | exp.Offset) else node
    if isinstance(value, exp.Literal) and not value.is_string:
        try:
            return int(value.this)
        except ValueError:
            return None
    return None
