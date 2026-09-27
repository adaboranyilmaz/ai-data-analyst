"""EX and Soft-F1 as the official evaluator computes them, on hand-worked cases, and the
streaming computation against the whole-result one on random results."""

from __future__ import annotations

import random
from decimal import Decimal

import pytest

from src.eval.ex import PredictionStream, ex, soft_f1


def test_ex_ignores_row_order_repeats_and_nothing_else():
    gold = [(1, "a"), (2, "b")]
    assert ex([(2, "b"), (1, "a")], gold) == 1
    assert ex([(1, "a"), (1, "a"), (2, "b")], gold) == 1
    assert ex([("a", 1), ("b", 2)], gold) == 0  # column order matters
    assert ex([(1, "a")], gold) == 0
    assert ex([(1, "a"), (2, "b"), (3, "c")], gold) == 0


def test_ex_compares_values_by_python_equality():
    assert ex([(682.0,)], [(682,)]) == 1
    assert ex([(Decimal("682"),)], [(682,)]) == 1
    assert ex([(Decimal("1.50"),)], [(Decimal("1.5"),)]) == 1
    assert ex([(0.1,)], [(Decimal("0.1"),)]) == 0
    assert ex([(True,)], [(1,)]) == 1
    assert ex([(None,)], [(None,)]) == 1
    assert ex([("682",)], [(682,)]) == 0


def test_ex_nan_equals_nothing_not_even_itself():
    assert ex([(float("nan"),)], [(float("nan"),)]) == 0


def test_ex_empty_results():
    assert ex([], []) == 1
    assert ex([], [(1,)]) == 0
    assert ex([(1,)], []) == 0


def test_unhashable_values_raise_as_in_the_official_comparison():
    with pytest.raises(TypeError):
        ex([([1, 2],)], [(1,)])
    with pytest.raises(TypeError):
        soft_f1([({"a": 1},)], [(1,)])


def test_soft_f1_hand_worked():
    assert soft_f1([(1, "a")], [(1, "a")]) == 1.0
    assert soft_f1([], []) == 1.0
    assert soft_f1([], [(1,)]) == 0
    assert soft_f1([(1,)], []) == 0
    # pair 1: one of two values found (match 1/2, predicted-only 1/2, gold-only 1/2); pair 2:
    # both found; the third predicted row has no gold row (predicted-only 1).
    # tp 1.5, fp 1.5, fn 0.5: precision 0.5, recall 0.75, F1 0.6
    got = soft_f1([(1, "x"), (2, "b"), (3, "c")], [(1, "a"), (2, "b")])
    assert got == pytest.approx(0.6)


def test_soft_f1_depends_on_row_order_and_not_on_column_position():
    # rows paired by position: reversed, only the middle pair matches
    assert soft_f1([(3,), (2,), (1,)], [(1,), (2,), (3,)]) == pytest.approx(1 / 3)
    # within a pair, a value counts wherever it sits
    assert soft_f1([(2, 1)], [(1, 2)]) == 1.0
    # repeats are dropped keeping the first appearance
    assert soft_f1([(1,), (1,), (2,)], [(1,), (2,)]) == 1.0
    assert soft_f1([(2,), (1,), (2,)], [(1,), (2,)]) == 0


def test_soft_f1_of_a_gold_row_with_no_columns_divides_by_zero():
    with pytest.raises(ZeroDivisionError):
        soft_f1([()], [()])
    assert ex([()], [()]) == 1  # EX has no division: the official EX run scores this 1


VALUES = [0, 1, 2, 1.0, 2.5, Decimal("2.5"), Decimal("0.1"), 0.1, "a", "b", "", None, True]


def random_rows(rng: random.Random, width: int) -> list[tuple]:
    return [tuple(rng.choice(VALUES) for _ in range(width)) for _ in range(rng.randint(0, 8))]


def test_streaming_matches_the_whole_result_computation():
    rng = random.Random(7)
    for _ in range(3000):
        width = rng.randint(1, 3)
        gold = random_rows(rng, width)
        pred = random_rows(rng, rng.choice([width, width, width + 1]))
        stream = PredictionStream(gold, max_distinct=1000)
        i = 0
        while i < len(pred):  # random batch sizes
            j = i + rng.randint(1, 3)
            stream.feed(pred[i:j])
            i = j
        assert stream.ex() == ex(pred, gold)
        assert stream.soft_f1() == soft_f1(pred, gold)
        assert stream.rows == len(pred)


def test_streaming_ex_stays_exact_when_soft_f1_gives_up():
    gold = [(1,), (2,), (3,)]
    stream = PredictionStream(gold, max_distinct=2)
    stream.feed([(1,), (3,), (2,), (1,)] * 1000)
    assert stream.ex() == 1
    # the first pair matches, so Soft-F1 needs the count of distinct rows it no longer has
    assert stream.soft_f1() is None and not stream.soft_f1_computed
    assert not stream.settled  # every row so far is a gold row: EX could still be 1


def test_soft_f1_above_the_limit_is_exact_when_no_pair_matches():
    gold = [(1,), (2,), (3,)]
    pred = [(3,), (1,), (2,), (4,), (5,)]
    stream = PredictionStream(gold, max_distinct=3)
    stream.feed(pred)
    assert not stream.soft_f1_computed
    assert stream.soft_f1() == 0 == soft_f1(pred, gold)


def test_soft_f1_above_the_limit_is_exact_or_none():
    rng = random.Random(8)
    for _ in range(3000):
        gold = random_rows(rng, 1)
        pred = random_rows(rng, 1) * rng.randint(1, 3)
        stream = PredictionStream(gold, max_distinct=rng.randint(1, 4))
        stream.feed(pred)
        got = stream.soft_f1()
        assert got is None or got == soft_f1(pred, gold)


def test_streaming_settles_at_the_first_row_outside_the_gold_once_soft_f1_gave_up():
    stream = PredictionStream([(1,)], max_distinct=1)
    stream.feed([(1,), (2,)])
    assert stream.settled and stream.ex() == 0


def test_streaming_raises_where_the_official_comparison_does():
    with pytest.raises(TypeError):
        PredictionStream([([1],)], max_distinct=10)
    stream = PredictionStream([(1,)], max_distinct=10)
    with pytest.raises(TypeError):
        stream.feed([([1],)])
    stream = PredictionStream([()], max_distinct=10)
    stream.feed([()])
    assert stream.ex() == 1
    with pytest.raises(ZeroDivisionError):
        stream.soft_f1()
