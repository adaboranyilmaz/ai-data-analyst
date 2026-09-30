"""What errors cost (src/eval/decisions.py), on small systems whose costs are worked out by hand."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.eval import decisions as dec

# four questions; the cheap model is wrong on two, the dear model on one
CHEAP = dec.outcomes("cheap", [0.01] * 4, [True, True, False, False])
DEAR = dec.outcomes("dear", [0.03] * 4, [True, True, True, False])


def test_outcomes_and_shares():
    assert CHEAP.shares() == pytest.approx((0.01, 0.5, 0.0))
    declining = dec.outcomes("d", [0.01] * 4, [True, True, False, False], [False] * 3 + [True])
    assert declining.shares() == pytest.approx((0.01, 0.25, 0.25))  # a declined answer is not wrong
    with pytest.raises(ValueError):
        dec.Outcomes("x", np.array([1.0]), np.array([True]), np.array([True]))
    with pytest.raises(ValueError):
        dec.Outcomes("x", np.array([1.0, 2.0]), np.array([True]), np.array([False]))


def test_expected_cost():
    # 0.01 + 10 * 0.5 + 3 * 0
    assert dec.expected_cost(CHEAP, 10, 3) == pytest.approx(5.01)
    assert dec.expected_cost(CHEAP, 10, 3, idx=np.array([0, 1])) == pytest.approx(0.01)


def test_break_even_between_two_systems_that_never_decline():
    # (0.03 - 0.01) / (0.5 - 0.25) = 0.08: above 8 cents a wrong answer, the dear model is cheaper
    cw = dec.break_even_wrong_cost(CHEAP, DEAR)
    assert cw == pytest.approx(0.08)
    assert dec.expected_cost(CHEAP, cw, 0) == pytest.approx(dec.expected_cost(DEAR, cw, 0))
    assert dec.break_even_wrong_cost(DEAR, CHEAP) == math.inf  # the cheap one is never better
    better_and_cheaper = dec.outcomes("b", [0.005] * 4, [True] * 4)
    assert dec.break_even_wrong_cost(CHEAP, better_and_cheaper) < 0  # it always pays


def test_cheapest_on_a_grid():
    grid = dec.cheapest([CHEAP, DEAR], wrong_grid=[0.01, 1.0], decline_grid=[1.0])
    assert grid == [["cheap", "dear"]]


def test_declining_pays_below_the_error_rate_of_the_declined_questions():
    all_answered = dec.outcomes("all", [0.01] * 4, [True, True, False, True])
    # declines questions 2 and 3: wrong on one of them if answered
    declining = dec.outcomes(
        "dec", [0.01] * 4, [True, True, False, True], [False, False, True, True]
    )
    r = dec.decline_pays_below(all_answered, declining)
    assert r == pytest.approx(0.5)
    # at that ratio both cost the same; below it declining is cheaper
    assert dec.expected_cost(all_answered, 1, r) == pytest.approx(
        dec.expected_cost(declining, 1, r)
    )
    assert dec.expected_cost(declining, 1, 0.4) < dec.expected_cost(all_answered, 1, 0.4)
    nothing = dec.outcomes("none", [0.01] * 4, [True] * 4)
    assert math.isnan(dec.decline_pays_below(all_answered, nothing))


def test_cost_optimal_rule_declines_below_one_minus_the_ratio():
    p = [0.95, 0.85, 0.6, 0.3]
    right = [True, True, False, False]
    o = dec.optimal_decline("opt", [0.01] * 4, right, p, ratio=0.2)  # decline where p < 0.8
    assert o.declined.tolist() == [False, False, True, True]
    assert dec.cost_in_wrong_answers(o, 0.2) == pytest.approx(0 + 0.2 * 0.5)
    # a ratio of 1 (declining costs as much as a wrong answer) never declines
    assert not dec.optimal_decline("x", [0.01] * 4, right, p, ratio=1.0).declined.any()
    # a ratio of 0 declines everything not certain
    assert dec.optimal_decline("x", [0.01] * 4, right, p, ratio=0.0).declined.all()


def test_the_optimal_rule_is_never_worse_than_a_fixed_rule_when_p_is_exact():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    right = rng.uniform(0, 1, 5000) < p  # p is the true probability of being right
    api = np.full(5000, 0.01)
    for ratio in (0.05, 0.2, 0.5):
        best = dec.cost_in_wrong_answers(dec.optimal_decline("o", api, right, p, ratio), ratio)
        for fixed in (0.3, 0.5, 0.9):
            other = dec.outcomes("f", api, right, p < fixed)
            assert best <= dec.cost_in_wrong_answers(other, ratio) + 0.01
