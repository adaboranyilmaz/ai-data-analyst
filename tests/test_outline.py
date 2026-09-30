"""The query outline (src/explain/outline.py): exact where it speaks, silent where it cannot."""

from __future__ import annotations

import pytest

from src.explain.outline import outline


def ops(phrase):
    return [t["v"] for t in phrase if t["t"] == "op"]


def cols(phrase):
    return [t["v"] for t in phrase if t["t"] == "col"]


def test_select_with_join_filter_group_order_limit():
    o = outline(
        "SELECT d.a2, COUNT(DISTINCT c.client_id) AS n FROM client c "
        "LEFT JOIN district d ON c.district_id = d.district_id "
        "WHERE c.gender = 'F' AND d.a3 IN ('Prague', 'north Bohemia') "
        "GROUP BY d.a2 HAVING COUNT(*) > 10 ORDER BY n DESC LIMIT 5"
    )
    assert o["kind"] == "select"
    assert o["tables"] == [{"name": "client"}, {"name": "district"}]
    link = o["links"][0]
    assert link["table"] == "district" and link["keep_unmatched"]
    assert cols(link["on"][0]["phrase"]) == ["client.district_id", "district.district_id"]
    assert len(o["filters"]) == 2
    assert ops(o["filters"][0]["phrase"]) == ["eq"]
    assert o["filters"][0]["phrase"][-1] == {"t": "lit", "v": '"F"'}
    assert ops(o["filters"][1]["phrase"]) == ["in", "list_sep"]
    assert cols(o["groups"][0]["phrase"]) == ["district.a2"]
    assert o["group_filters"][0]["phrase"][0] == {"t": "agg", "v": "count_rows", "arg": []}
    assert o["returns"][1]["name"] == "n"
    assert o["returns"][1]["phrase"][0]["v"] == "count_distinct"
    assert o["order"][0]["descending"] is True
    assert o["limit"] == 5 and o["offset"] is None


def test_an_or_condition_among_others_is_bracketed_and_alone_it_is_not():
    many = outline("SELECT 1 FROM t WHERE (a IS NULL OR a = '*') AND b LIKE '%x%'")
    first = many["filters"][0]["phrase"]
    assert first[0] == {"t": "open"} and first[-1] == {"t": "close"}
    assert ops(first) == ["is_empty", "or", "eq"]
    assert ops(many["filters"][1]["phrase"]) == ["like"]
    alone = outline("SELECT 1 FROM t WHERE a = 1 OR b = 2")["filters"][0]["phrase"]
    assert alone[0] != {"t": "open"}


def test_subqueries_and_windows_are_never_paraphrased():
    o = outline(
        "SELECT name, RANK() OVER (ORDER BY x) FROM t WHERE x > (SELECT AVG(x) FROM t) AND y = 1"
    )
    assert o["filters"][0] == {"unrendered": True}
    assert "phrase" in o["filters"][1]
    assert o["returns"][1] == {"unrendered": True, "name": None}


def test_a_function_without_a_template_is_unrendered():
    o = outline("SELECT TO_CHAR(d, 'YYYY') FROM t")
    assert o["returns"][0]["unrendered"] is True


@pytest.mark.parametrize(
    "sql, fn",
    [
        ("SELECT EXTRACT(YEAR FROM date) FROM t", "year_of"),
        ("SELECT ROUND(x, 2) FROM t", "round"),
        ("SELECT ROUND(x) FROM t", "round_whole"),
        ("SELECT EXTRACT(YEAR FROM AGE(CAST('1998-12-31' AS DATE), birth)) FROM t", "whole_years"),
        ("SELECT CASE WHEN a = 1 THEN 1 ELSE 0 END FROM t", "case"),
        ("SELECT a / NULLIF(b, 0) FROM t", "nullif"),
        ("SELECT SUBSTRING(s FROM 7 FOR 2) FROM t", "substring"),
        ("SELECT COALESCE(a, 0) FROM t", "coalesce"),
        ("SELECT first || ' ' || last FROM t", "concat"),
    ],
)
def test_function_templates(sql, fn):
    phrase = outline(sql)["returns"][0]["phrase"]
    found = [t["v"] for t in _walk(phrase) if t["t"] == "fn"]
    assert fn in found


def _walk(phrase):
    for t in phrase:
        yield t
        if t["t"] == "agg":
            yield from _walk(t["arg"])
        if t["t"] == "fn":
            for a in t["args"]:
                yield from _walk(a)


def test_whole_years_puts_the_start_first():
    phrase = outline("SELECT EXTRACT(YEAR FROM AGE(end_d, start_d)) FROM t")["returns"][0]["phrase"]
    start, end = phrase[0]["args"]
    assert cols(start) == ["start_d"] and cols(end) == ["end_d"]


def test_casts_and_case_changes_are_transparent():
    o = outline("SELECT CAST(SUM(x) AS REAL) / COUNT(*) FROM t WHERE LOWER(s) = 'a'")
    phrase = o["returns"][0]["phrase"]
    assert (
        phrase[0] == {"t": "agg", "v": "sum", "arg": [{"t": "col", "v": "t.x"}]}
        or phrase[0]["v"] == "sum"
    )
    assert ops(o["filters"][0]["phrase"]) == ["eq"]


def test_intermediate_results_and_every_table_read():
    o = outline(
        "WITH r AS (SELECT school, RANK() OVER (ORDER BY s) AS rnk FROM satscores) "
        "SELECT school FROM r JOIN schools ON r.school = schools.name WHERE rnk <= 5"
    )
    assert o["intermediate_results"] == ["r"]
    assert {t["name"] for t in o["tables"]} == {"satscores", "schools"}


def test_tables_inside_subqueries_are_listed():
    o = outline("SELECT (SELECT MAX(w) FROM hero) - (SELECT MIN(w) FROM hero) AS d")
    assert o["tables"] == [{"name": "hero"}]
    assert o["returns"][0] == {"unrendered": True, "name": "d"}


def test_combined_and_unparsed_queries():
    assert outline("SELECT a FROM t EXCEPT SELECT a FROM u") == {
        "kind": "combined",
        "how": "except",
    }
    assert outline("SELECT FROM WHERE (((")["kind"] == "unparsed"


def test_between_and_negations():
    o = outline("SELECT 1 FROM t WHERE a BETWEEN 1 AND 3 AND b NOT IN (1, 2) AND c IS NOT NULL")
    assert ops(o["filters"][0]["phrase"]) == ["between", "and"]
    assert ops(o["filters"][1]["phrase"]) == ["not_in", "list_sep"]
    assert ops(o["filters"][2]["phrase"]) == ["is_not_empty"]


@pytest.mark.parametrize(
    "condition, words",
    [
        ("c IS NULL", ["is_empty"]),
        ("c IS NOT NULL", ["is_not_empty"]),
        ("NOT c IS NULL", ["is_not_empty"]),
        ("NOT c IS NOT NULL", ["is_empty"]),
        ("c NOT LIKE 'a%'", ["not_like"]),
        ("NOT c LIKE 'a%'", ["not_like"]),
        ("c NOT ILIKE 'a%'", ["not_like"]),
        ("c NOT BETWEEN 1 AND 2", ["not_between", "and"]),
        ("c NOT IN (1)", ["not_in"]),
        ("NOT c IN (1)", ["not_in"]),
        ("NOT (a = 1 OR b = 2)", ["not", "eq", "or", "eq"]),
        ("c <> 1", ["neq"]),
    ],
)
def test_negations_keep_their_meaning(condition, words):
    part = outline(f"SELECT 1 FROM t WHERE {condition}")["filters"][0]
    assert ops(part["phrase"]) == words


# ------------------------------------------------------------------------------ round trip
# Every rendered phrase is written back as SQL and compared with the query's own expression, so
# a phrase that reads well but means something else fails. The outline leaves out casts and
# case changes on purpose (they are shown in the SQL), so both sides drop them.

_SQL_OP = {
    "eq": "=",
    "neq": "<>",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
    "and": "AND",
    "or": "OR",
    "not": "NOT",
    "is_empty": "IS NULL",
    "is_not_empty": "IS NOT NULL",
    "in": "IN",
    "not_in": "NOT IN",
    "list_sep": ",",
    "like": "LIKE",
    "not_like": "NOT LIKE",
    "between": "BETWEEN",
    "not_between": "NOT BETWEEN",
    "plus": "+",
    "minus": "-",
    "times": "*",
    "divided_by": "/",
    "all_columns": "*",
    "true": "TRUE",
    "false": "FALSE",
    "null": "NULL",
}
_SQL_AGG = {
    "count": "COUNT({})",
    "count_distinct": "COUNT(DISTINCT {})",
    "sum": "SUM({})",
    "avg": "AVG({})",
    "max": "MAX({})",
    "min": "MIN({})",
}
_SQL_FN = {
    "round": "ROUND({0}, {1})",
    "round_whole": "ROUND({0})",
    "year_of": "EXTRACT(YEAR FROM {0})",
    "month_of": "EXTRACT(MONTH FROM {0})",
    "day_of": "EXTRACT(DAY FROM {0})",
    "whole_years": "EXTRACT(YEAR FROM AGE({1}, {0}))",
    "coalesce": "COALESCE({0}, {1})",
    "case": "CASE WHEN {1} THEN {0} ELSE {2} END",
    "case_no_else": "CASE WHEN {1} THEN {0} END",
    "nullif": "NULLIF({0}, {1})",
    "abs": "ABS({0})",
    "concat": "({0}) || ({1})",
    "substring": "SUBSTRING({0} FROM {1} FOR {2})",
}


def to_sql(phrase) -> str:
    out = []
    for t in phrase:
        k = t["t"]
        if k == "col":
            out.append(".".join(f'"{p}"' for p in t["v"].split(".", 1)))
        elif k == "lit":
            v = t["v"]
            out.append("'" + v[1:-1].replace("'", "''") + "'" if v.startswith('"') else v)
        elif k == "op":
            out.append(_SQL_OP[t["v"]])
        elif k == "open":
            out.append("(")
        elif k == "close":
            out.append(")")
        elif k == "agg":
            out.append(
                "COUNT(*)" if t["v"] == "count_rows" else _SQL_AGG[t["v"]].format(to_sql(t["arg"]))
            )
        elif k == "fn":
            out.append(_SQL_FN[t["v"]].format(*(f"({to_sql(a)})" for a in t["args"])))
    return " ".join(out)


def _canon(e, aliases):
    import sqlglot.expressions as ex

    wrappers = (ex.Paren, ex.Cast, ex.TryCast, ex.Lower, ex.Upper, ex.Alias)
    root = ex.Tuple(expressions=[e.copy()])  # a holder, so the top node can be replaced too
    for c in list(root.find_all(ex.Column)):
        if c.table:
            c.set("table", ex.to_identifier(aliases.get(c.table, c.table)))
    while (w := next((n for n in root.find_all(*wrappers)), None)) is not None:
        w.replace(w.this)
    for n in list(root.find_all(ex.Is, ex.In, ex.Like, ex.ILike, ex.Between)):
        if n.args.get("negate"):
            n.set("negate", None)
            n.replace(ex.Not(this=n.copy()))
    for n in list(root.find_all(ex.ILike)):  # the outline says "matches the pattern" for both
        n.replace(ex.Like(this=n.this, expression=n.expression))
    for i in list(root.find_all(ex.Identifier)):
        i.set("this", i.this.lower())
        i.set("quoted", False)
    return root.expressions[0].sql(dialect="postgres")


def round_trip_problems(sql: str) -> list[str]:
    import sqlglot
    import sqlglot.expressions as ex

    from src.explain.outline import _aliases, _conjuncts

    o = outline(sql)
    if o["kind"] != "select":
        return []
    tree = sqlglot.parse_one(sql, read="postgres")
    aliases = _aliases(tree)
    originals = {
        "filters": _conjuncts(tree.args["where"].this) if tree.args.get("where") else [],
        "returns": list(tree.expressions),
        "order": [x.this for x in tree.args["order"].expressions] if tree.args.get("order") else [],
        "groups": list(tree.args["group"].expressions) if tree.args.get("group") else [],
        "group_filters": _conjuncts(tree.args["having"].this) if tree.args.get("having") else [],
    }
    found = []
    for section, exprs in originals.items():
        for part, original in zip(o[section], exprs, strict=True):
            if "phrase" not in part:
                continue
            back = sqlglot.parse_one(f"SELECT {to_sql(part['phrase'])}", read="postgres")
            want, got = _canon(original, aliases), _canon(back.expressions[0], aliases)
            if want != got:
                found.append(f"{section}: {want!r} rendered as {got!r}")
    for link, j in zip(o["links"], tree.args.get("joins") or [], strict=True):
        for part, original in zip(link["on"], _conjuncts(j.args.get("on")), strict=True):
            if "phrase" in part:
                back = sqlglot.parse_one(f"SELECT {to_sql(part['phrase'])}", read="postgres")
                if _canon(original, aliases) != _canon(back.expressions[0], aliases):
                    found.append(f"link: {original.sql()!r}")
    del ex
    return found


def test_round_trip_on_hand_written_queries():
    for sql in (
        "SELECT a.x, COUNT(DISTINCT b.y) AS n FROM a JOIN b ON a.id = b.id "
        "WHERE (a.p IS NULL OR a.p = '*') AND b.q NOT LIKE '%z%' AND a.r NOT BETWEEN 1 AND 2 "
        "ORDER BY n DESC",
        "SELECT CAST(SUM(CASE WHEN t.k = 'A' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM t",
        "SELECT EXTRACT(YEAR FROM AGE(CAST('1998-12-31' AS DATE), c.b)) FROM c "
        "WHERE c.x <> 'it''s'",
        "SELECT ROUND(AVG(x), 2), -y, a / NULLIF(b, 0) FROM t WHERE NOT (a = 1 OR b = 2)",
    ):
        assert round_trip_problems(sql) == [], sql


def test_round_trip_catches_a_wrong_rendering(monkeypatch):
    from src.explain import outline as module

    monkeypatch.setitem(module._COMPARE, module.exp.GT, "lt")  # a deliberately wrong template
    assert round_trip_problems("SELECT 1 FROM t WHERE a > 1")


def test_round_trip_on_every_answer_query_of_the_winning_run():
    """The committed run records hold the analyst's 500 queries, so this runs without the
    benchmark files."""
    from src.agent.confidence import answers_path, confidence_config
    from src.eval.records import read_records

    path, _ = answers_path(confidence_config())
    sqls = [r["final_sql"] for r in read_records(path) if r["final_sql"]]
    assert len(sqls) > 400
    problems = {s: p for s in sqls if (p := round_trip_problems(s))}
    assert problems == {}


@pytest.mark.usefixtures("bird_ready")
@pytest.mark.bird
def test_round_trip_on_every_expert_query():
    from src.data.bird import questions

    problems = {q["SQL"]: p for q in questions() if (p := round_trip_problems(q["SQL"]))}
    assert problems == {}
