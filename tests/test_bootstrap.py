"""Bootstrap intervals: seeded, paired, and close to their nominal coverage."""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.bootstrap import bootstrap, paired_difference, resamples


def test_the_same_seed_gives_the_same_interval():
    x = np.random.default_rng(1).random(50)

    def stat(i):
        return x[i].mean()

    a = bootstrap(stat, 50, count=500, seed=9)
    b = bootstrap(stat, 50, count=500, seed=9)
    assert a == b
    assert a.low < a.estimate < a.high and a.estimate == pytest.approx(x.mean())


def test_resamples_draw_with_replacement_from_every_question():
    draws = list(resamples(5, 2500, seed=0))
    assert len(draws) == 2500 and all(d.shape == (5,) for d in draws)
    assert set(np.concatenate(draws).tolist()) == set(range(5))


def test_a_constant_statistic_has_no_width():
    iv = bootstrap(lambda i: 3.0, 20, count=200)
    assert (iv.estimate, iv.low, iv.high) == (3.0, 3.0, 3.0)


def test_paired_difference_of_the_same_run_is_zero():
    x = np.random.default_rng(2).integers(0, 2, 80)

    def stat(i):
        return x[i].mean()

    iv = paired_difference(stat, stat, 80, count=300)
    assert (iv.estimate, iv.low, iv.high) == (0.0, 0.0, 0.0)


def test_pairing_removes_the_shared_question_difficulty():
    # b is a with one extra error: unpaired intervals overlap widely, the paired one is tight
    rng = np.random.default_rng(3)
    a = rng.integers(0, 2, 100).astype(float)
    b = a.copy()
    b[np.flatnonzero(a)[0]] = 0
    iv = paired_difference(lambda i: a[i].mean(), lambda i: b[i].mean(), 100, count=2000)
    assert iv.estimate == pytest.approx(0.01)
    assert 0 <= iv.low <= iv.high <= 0.04


def test_undefined_resamples_are_left_out_and_counted():
    iv = bootstrap(lambda i: float("nan") if 0 in i else 1.0, 10, count=400)
    assert 0 < iv.undefined < 400 and iv.low == iv.high == 1.0
    d = bootstrap(lambda i: float("nan"), 3, count=10).to_dict()
    assert d["low"] is None and d["high"] is None and d["estimate"] is None


def test_coverage_is_near_nominal():
    rng = np.random.default_rng(4)
    covered = 0
    trials = 200
    for t in range(trials):
        x = (rng.random(100) < 0.7).astype(float)
        iv = bootstrap(lambda i, x=x: x[i].mean(), 100, count=1000, seed=t)
        covered += iv.low <= 0.7 <= iv.high
    assert 0.88 <= covered / trials <= 0.99
