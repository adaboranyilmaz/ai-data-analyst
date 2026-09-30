"""Where a wrong answer goes wrong (src/eval/errors.py), on query pairs whose one difference is
known, and the near-miss helpers moved to src/eval/near_miss.py."""

from __future__ import annotations

import math

import pytest

from src.eval import errors, near_miss

GOLD = (
    "SELECT T1.name FROM client AS T1 INNER JOIN account AS T2 ON T1.id = T2.client_id "
    "WHERE T2.type = 'OWNER' AND T1.age > 30 ORDER BY T1.name LIMIT 5"
)


def category(pred: str, gold: str = GOLD, outcome: str = "ok", near: str | None = "other"):
    return errors.categorize(outcome, near, pred, gold)


def test_the_same_query_written_differently_has_no_difference():
    same = (
        "SELECT c.name FROM client c JOIN account a ON a.client_id = c.id "
        "WHERE c.age > 30 AND a.type = 'OWNER' ORDER BY c.name LIMIT 5"
    )
    out = category(same)
    assert out["category"] == "other" and out["differences"] == {}


@pytest.mark.parametrize(
    "pred, expected, subkind",
    [
        # reads another table
        (
            "SELECT T1.name FROM client AS T1 INNER JOIN disp AS T2 ON T1.id = T2.client_id "
            "WHERE T2.type = 'OWNER' AND T1.age > 30 ORDER BY T1.name LIMIT 5",
            "tables",
            None,
        ),
        # the same tables, matched on another key
        (
            "SELECT T1.name FROM client AS T1 INNER JOIN account AS T2 ON T1.id = T2.id "
            "WHERE T2.type = 'OWNER' AND T1.age > 30 ORDER BY T1.name LIMIT 5",
            "join",
            None,
        ),
        # a condition missing
        (
            "SELECT T1.name FROM client AS T1 INNER JOIN account AS T2 ON T1.id = T2.client_id "
            "WHERE T2.type = 'OWNER' ORDER BY T1.name LIMIT 5",
            "filter",
            "missing",
        ),
        # an extra condition
        (
            "SELECT T1.name FROM client AS T1 INNER JOIN account AS T2 ON T1.id = T2.client_id "
            "WHERE T2.type = 'OWNER' AND T1.age > 30 AND T1.sex = 'F' ORDER BY T1.name LIMIT 5",
            "filter",
            "extra",
        ),
        # the same columns, another value (a code's case matters)
        (
            "SELECT T1.name FROM client AS T1 INNER JOIN account AS T2 ON T1.id = T2.client_id "
            "WHERE T2.type = 'owner' AND T1.age > 30 ORDER BY T1.name LIMIT 5",
            "filter",
            "different_value",
        ),
        # a condition on another column
        (
            "SELECT T1.name FROM client AS T1 INNER JOIN account AS T2 ON T1.id = T2.client_id "
            "WHERE T2.type = 'OWNER' AND T1.birth > 30 ORDER BY T1.name LIMIT 5",
            "filter",
            "different_column",
        ),
        # removes repeats where the expert does not
        (
            "SELECT DISTINCT T1.name FROM client AS T1 INNER JOIN account AS T2 "
            "ON T1.id = T2.client_id WHERE T2.type = 'OWNER' AND T1.age > 30 "
            "ORDER BY T1.name LIMIT 5",
            "computation",
            None,
        ),
        # returns another column
        (
            "SELECT T1.surname FROM client AS T1 INNER JOIN account AS T2 ON T1.id = T2.client_id "
            "WHERE T2.type = 'OWNER' AND T1.age > 30 ORDER BY T1.name LIMIT 5",
            "output",
            None,
        ),
        # keeps another number of rows
        (
            "SELECT T1.name FROM client AS T1 INNER JOIN account AS T2 ON T1.id = T2.client_id "
            "WHERE T2.type = 'OWNER' AND T1.age > 30 ORDER BY T1.name LIMIT 1",
            "order_limit",
            None,
        ),
    ],
)
def test_one_known_difference_gives_its_category(pred, expected, subkind):
    out = category(pred)
    assert out["category"] == expected
    assert out["subkind"] == subkind


def test_the_first_category_wins_and_every_difference_is_kept():
    pred = (
        "SELECT T1.surname FROM client AS T1 INNER JOIN disp AS T2 ON T1.id = T2.client_id "
        "WHERE T2.type = 'OWNER' ORDER BY T1.name LIMIT 1"
    )
    out = category(pred)
    assert out["category"] == "tables"
    assert set(out["differences"]) == {"tables", "filters", "output", "order_limit"}
    assert out["differences"]["tables"] == {"missing": ["account"], "extra": ["disp"]}


def test_no_result_format_only_and_unparsed():
    assert category("SELECT 1", outcome="refused")["category"] == "no_result"
    assert category("SELECT 1", outcome="refused")["subkind"] == "refused"
    assert category(None, outcome="error")["category"] == "no_result"
    near = category(GOLD.replace("T1.name FROM", "T1.name, T1.id FROM"), near="columns")
    assert near["category"] == "format_only" and near["subkind"] == "columns"
    assert category("SELECT FROM WHERE ((")["category"] == "unparsed"


def test_aggregates_count_distinct_and_arithmetic_are_part_of_the_computation():
    a = errors.shape("SELECT COUNT(DISTINCT id), SUM(x) * 100 / COUNT(*) FROM t")
    assert a.computation == ("count", "count_distinct", "div", "mul", "sum")
    b = errors.shape("SELECT COUNT(id) FROM t")
    assert "count_distinct" not in b.computation


def test_values_compare_numbers_as_numbers_and_text_exactly():
    a = errors.shape("SELECT 1 FROM t WHERE x = 1.0 AND y = 'F' AND z > -2")
    assert a.filter_values == frozenset({"1", "'F'", "-2"})
    b = errors.shape("SELECT 1 FROM t WHERE x = 1 AND y = 'f' AND z > -2")
    assert a.filter_values != b.filter_values  # 'F' and 'f' differ


def test_a_condition_s_own_subquery_is_not_part_of_it():
    s = errors.shape("SELECT a FROM t WHERE a IN (SELECT b FROM u WHERE c = 5)")
    assert s.filter_columns == frozenset({"a", "c"})  # b is the subquery's output, not a filter
    assert s.tables == frozenset({"t", "u"})


def test_intermediate_results_are_not_tables_and_join_keys_are_not_filters():
    s = errors.shape(
        "WITH r AS (SELECT id FROM client) SELECT r.id FROM r JOIN account a ON a.cid = r.id "
        "WHERE a.x = 1"
    )
    assert s.tables == frozenset({"client", "account"})
    assert s.joins == frozenset({frozenset({"cid", "id"})})
    assert s.filter_columns == frozenset({"x"})


@pytest.mark.parametrize(
    "pred, gold, pw, gw, expected",
    [
        ([], [(1,)], 1, 1, "empty"),
        ([(1,), (2,)], [(1,), (2,)], 1, 1, "same_rows_other_shape"),
        ([(1,)], [(1,), (2,)], 1, 1, "subset"),
        ([(1,), (2,), (3,)], [(1,), (2,)], 1, 1, "superset"),
        ([(1,), (3,)], [(1,), (2,)], 1, 1, "overlap"),
        ([(3,)], [(1,)], 1, 1, "disjoint"),
        ([(1, 2)], [(1,), (2,)], 2, 1, "fewer_rows"),
        ([(1, 2)], [(1,)], 2, 1, "same_row_count"),
        ([(1, 2), (3, 4)], [(1,)], 2, 1, "more_rows"),
    ],
)
def test_result_relation(pred, gold, pw, gw, expected):
    assert errors.result_relation(pred, gold, pw, gw) == expected


def test_agreement_and_kappa():
    a = ["x", "x", "y", "y"]
    assert errors.agreement(a, a)["kappa"] == pytest.approx(1.0)
    # observed 2/4; expected 0.5*0.5 + 0.5*0.5 = 0.5 -> kappa 0
    out = errors.agreement(a, ["x", "y", "x", "y"])
    assert out["share_same"] == 0.5 and out["kappa"] == pytest.approx(0.0)
    assert errors.agreement(["x"], ["x"])["kappa"] is None
    with pytest.raises(ValueError):
        errors.agreement(["x"], [])


def test_hand_check_sample_spreads_over_categories_and_is_seeded():
    rows = [{"question_id": i, "category": "filter" if i < 60 else "tables"} for i in range(100)]
    rows += [{"question_id": 200, "category": "join"}]
    sample = errors.hand_check_sample(rows, 10, 7)
    cats = [r["category"] for r in sample]
    assert cats.count("filter") == 6 and cats.count("tables") == 4 and cats.count("join") == 1
    assert sample == errors.hand_check_sample(list(reversed(rows)), 10, 7)
    assert [r["question_id"] for r in sample] == sorted(r["question_id"] for r in sample)


def test_near_miss_helpers():
    assert near_miss.classify([(1.004,)], [(1.0,)], 1) == "rounding"
    assert near_miss.classify([(1, "a")], [("a",)], 1) == "columns"
    assert near_miss.classify([(1.004, "a")], [("a", 1.0)], 2) == "columns_and_rounding"
    assert near_miss.classify([(2,)], [(1,)], 1) == "other"
    assert near_miss.recoverable([(1,)], [(1, 2)], 2) is False
    assert near_miss.rounded((math.inf, None, True)) == (math.inf, None, True)
