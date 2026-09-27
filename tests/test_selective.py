"""Risk-coverage, AURC and accuracy at coverage on hand-worked cases, and the tie handling
against a brute force over every order of tied questions."""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from src.eval.selective import (
    accuracy_at_coverage,
    aurc,
    e_aurc,
    oracle_aurc,
    risk_coverage,
)


def test_hand_worked_curve_without_ties():
    conf, ok = [0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0]
    coverage, risk = risk_coverage(conf, ok)
    assert coverage.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert risk.tolist() == pytest.approx([0, 1 / 2, 1 / 3, 2 / 4])
    assert aurc(conf, ok) == pytest.approx((0 + 1 / 2 + 1 / 3 + 1 / 2) / 4)
    # perfect ranking of 2 right out of 4: risks 0, 0, 1/3, 1/2
    assert oracle_aurc(2, 4) == pytest.approx((1 / 3 + 1 / 2) / 4)
    assert e_aurc(conf, ok) == pytest.approx(aurc(conf, ok) - oracle_aurc(2, 4))


def test_input_order_does_not_matter():
    conf, ok = [0.6, 0.9, 0.7, 0.8], [0, 1, 1, 0]
    assert aurc(conf, ok) == pytest.approx(aurc([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0]))


def test_a_perfect_ranking_has_no_excess():
    assert e_aurc([0.9, 0.8, 0.3, 0.1], [1, 1, 0, 0]) == pytest.approx(0)


def test_one_tied_group_has_the_error_rate_at_every_coverage():
    _, risk = risk_coverage([0.5] * 4, [1, 0, 0, 1])
    assert risk.tolist() == pytest.approx([0.5] * 4)


def expected_by_brute_force(conf, ok, coverage):
    """Average over every order that keeps higher confidence first."""
    n = len(conf)
    orders = [p for p in itertools.permutations(range(n))
              if all(conf[p[i]] >= conf[p[i + 1]] for i in range(n - 1))]  # fmt: skip
    aurcs, accs = [], []
    k_cov = max(1, int(np.ceil(coverage * n - 1e-9)))
    for p in orders:
        errors = np.cumsum([1 - ok[i] for i in p])
        k = np.arange(1, n + 1)
        aurcs.append(float((errors / k).mean()))
        accs.append(1 - errors[k_cov - 1] / k_cov)
    return float(np.mean(aurcs)), float(np.mean(accs))


def test_ties_are_the_expectation_over_random_order():
    rng = random.Random(3)
    for _ in range(200):
        n = rng.randint(1, 7)
        conf = [rng.choice([0.2, 0.5, 0.9]) for _ in range(n)]
        ok = [rng.randint(0, 1) for _ in range(n)]
        coverage = rng.choice([0.5, 0.8, 1.0])
        want_aurc, want_acc = expected_by_brute_force(conf, ok, coverage)
        assert aurc(conf, ok) == pytest.approx(want_aurc)
        assert accuracy_at_coverage(conf, ok, coverage) == pytest.approx(want_acc)


def test_declined_questions_rank_last_and_count_as_errors():
    # the declined question reports the higher confidence but still comes last
    conf, ok, declined = [0.1, 0.9], [1, 1], [False, True]
    _, risk = risk_coverage(conf, ok, declined)
    assert risk.tolist() == pytest.approx([0, 1 / 2])
    assert accuracy_at_coverage(conf, ok, 1.0, declined) == pytest.approx(0.5)
    assert accuracy_at_coverage(conf, ok, 0.5, declined) == pytest.approx(1.0)


def test_accuracy_at_coverage_takes_the_ceiling():
    conf = [1 - i / 10 for i in range(10)]
    ok = [1] * 8 + [0, 0]
    assert accuracy_at_coverage(conf, ok, 0.8) == 1.0  # the top 8
    assert accuracy_at_coverage(conf, ok, 0.81) == pytest.approx(8 / 9)  # the top 9


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        aurc([], [])
    with pytest.raises(ValueError):
        aurc([0.5, float("nan")], [1, 0])
    with pytest.raises(ValueError):
        aurc([0.5], [2])
    with pytest.raises(ValueError):
        accuracy_at_coverage([0.5], [1], 0)
