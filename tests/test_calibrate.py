"""Calibrators and the decline threshold on data with known answers."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from src.eval import selective
from src.eval.bootstrap import bootstrap
from src.eval.calibrate import (
    Isotonic,
    Logistic,
    Platt,
    answered_at,
    calibrated,
    choose_threshold,
    logistic_fit,
    threshold_outcome,
)


def logistic_sample(a: float, b: float, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-(a * x + b)))).astype(int)
    return x, y


class TestPlatt:
    def test_recovers_the_generating_parameters(self):
        x, y = logistic_sample(4.0, -2.0, 20_000)
        f = Platt.fit(x, y)
        assert f.slope == pytest.approx(4.0, abs=0.15)
        assert f.intercept == pytest.approx(-2.0, abs=0.1)

    def test_is_the_maximum_likelihood_solution(self):
        # the score equations of an unpenalised logistic regression hold at the fit
        x = np.array([0.6, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.9, 0.95, 0.95])
        y = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 1])
        f = Platt.fit(x, y)
        p = f(x)
        assert (y - p).sum() == pytest.approx(0, abs=1e-9)
        assert (x * (y - p)).sum() == pytest.approx(0, abs=1e-9)

    def test_separable_answers_have_no_fit(self):
        with pytest.raises(ValueError, match="separable|singular|converge"):
            Platt.fit([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])

    def test_one_class_has_no_fit(self):
        with pytest.raises(ValueError, match="both correct and wrong"):
            Platt.fit([0.5, 0.7], [1, 1])

    def test_keeps_the_ranking_so_aurc_is_unchanged(self):
        x, y = logistic_sample(3.0, -1.0, 300, seed=1)
        x = np.round(x, 1)  # ties, as stated confidences have
        f = Platt.fit(x, y)
        assert f.slope > 0
        assert selective.aurc(f(x), y) == pytest.approx(selective.aurc(x, y), abs=1e-12)

    def test_to_dict(self):
        assert Platt(0.5, 2.0).to_dict() == {"method": "platt", "intercept": 0.5, "slope": 2.0}


class TestIsotonic:
    def test_pools_adjacent_violators(self):
        f = Isotonic.fit([1, 2, 3, 4], [1, 0, 1, 1])
        assert f.x == (1, 2, 3, 4)
        assert f.y == pytest.approx((0.5, 0.5, 1.0, 1.0))

    def test_weights_each_answer(self):
        # two answers at 0.5 (one right), one at 0.7 (wrong): pooled to 1/3
        f = Isotonic.fit([0.5, 0.5, 0.7], [1, 0, 0])
        assert f.y == pytest.approx((1 / 3, 1 / 3))

    def test_interpolates_between_and_clips_outside(self):
        f = Isotonic.fit([0.2, 0.4, 0.8], [0, 1, 1])
        assert f([0.3, 0.1, 0.9]) == pytest.approx([0.5, 0.0, 1.0])

    def test_non_decreasing_and_mean_preserving(self):
        x, y = logistic_sample(5.0, -2.5, 500, seed=2)
        x = np.round(x, 2)
        f = Isotonic.fit(x, y)
        assert np.all(np.diff(f.y) >= -1e-12)
        assert f(x).mean() == pytest.approx(y.mean())  # PAV keeps the overall mean


class TestLogistic:
    def test_recovers_two_weights(self):
        rng = np.random.default_rng(3)
        a, b = rng.uniform(0, 1, 20_000), rng.uniform(0, 1, 20_000)
        p = 1 / (1 + np.exp(-(-3.0 + 2.0 * a + 3.0 * b)))
        y = (rng.uniform(0, 1, a.size) < p).astype(int)
        f = Logistic.fit(["stated", "critic"], [a, b], y)
        assert f.intercept == pytest.approx(-3.0, abs=0.15)
        assert f.weights == pytest.approx((2.0, 3.0), abs=0.15)
        assert f.to_dict()["weights"] == dict(zip(("stated", "critic"), f.weights, strict=True))
        assert f([a[:3], b[:3]]).shape == (3,)

    def test_logistic_fit_rejects_a_single_class(self):
        with pytest.raises(ValueError):
            logistic_fit(np.array([[0.1], [0.2]]), np.array([0, 0]))


def test_declined_answers_are_calibrated_to_zero():
    assert calibrated(np.array([0.8, 0.9]), [False, True]).tolist() == [0.8, 0.0]


class TestThreshold:
    # three confidence groups of 10: 9, 8 and 5 correct
    p = np.repeat([0.9, 0.8, 0.6], 10)
    y = np.array([1] * 9 + [0] + [1] * 8 + [0] * 2 + [1] * 5 + [0] * 5)
    d = np.zeros(30, dtype=bool)

    def test_largest_coverage_reaching_the_target(self):
        r = choose_threshold(self.p, self.y, self.d, 0.85, 15)
        assert r["threshold"] == 0.8 and r["reached_target"] and not r["fallback_used"]
        assert r["chosen"] == {
            "threshold": 0.8,
            "answered": 20,
            "coverage": 2 / 3,
            "accuracy": 0.85,
        }

    def test_stricter_target(self):
        r = choose_threshold(self.p, self.y, self.d, 0.9, 15)
        assert r["threshold"] == 0.9 and r["chosen"]["answered"] == 10

    def test_ties_are_never_split(self):
        r = choose_threshold(self.p, self.y, self.d, 0.9, 15)
        assert [c["threshold"] for c in r["candidates"]] == [0.9, 0.8, 0.6]
        assert [c["answered"] for c in r["candidates"]] == [10, 20, 30]

    def test_fallback_when_no_threshold_reaches_the_target(self):
        r = choose_threshold(self.p, self.y, self.d, 0.95, 5)
        assert r["fallback_used"] and not r["reached_target"] and r["threshold"] == 0.9
        # with at least 11 answered, the 0.9 group alone is too small
        r = choose_threshold(self.p, self.y, self.d, 0.95, 11)
        assert r["threshold"] == 0.8 and r["fallback_used"]

    def test_fallback_needs_enough_answers(self):
        with pytest.raises(ValueError, match="at least 31"):
            choose_threshold(self.p, self.y, self.d, 0.95, 31)

    def test_declined_answers_are_never_answered_but_count_in_coverage(self):
        d = self.d.copy()
        d[:10] = True  # the 0.9 group declined (calibrated to 0 by then)
        p = calibrated(self.p, d)
        r = choose_threshold(p, self.y, d, 0.8, 5)
        assert [c["threshold"] for c in r["candidates"]] == [0.8, 0.6]
        assert r["chosen"]["answered"] == 10 and r["chosen"]["coverage"] == pytest.approx(1 / 3)
        assert not answered_at(p, d, 0.0)[:10].any()

    def test_outcome_at_a_fixed_threshold(self):
        boot = partial(bootstrap, count=500, seed=0)
        r = threshold_outcome(self.p, self.y, self.d, 0.8, boot)
        assert r["answered"] == 20
        assert r["coverage"]["estimate"] == pytest.approx(2 / 3)
        assert r["accuracy"]["estimate"] == pytest.approx(0.85)
        assert r["accuracy"]["low"] <= 0.85 <= r["accuracy"]["high"]

    def test_outcome_counts_resamples_with_nothing_answered(self):
        boot = partial(bootstrap, count=200, seed=0)
        p, y, d = np.array([0.9, 0.1]), np.array([1, 0]), np.zeros(2, dtype=bool)
        r = threshold_outcome(p, y, d, 0.9, boot)
        assert r["accuracy"]["undefined"] > 0  # resamples holding only the second question
