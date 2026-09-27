"""Brier score, ECE, reliability bins and AUROC on cases with known answers."""

from __future__ import annotations

import random

import pytest

from src.eval.calibration import auroc, brier, ece, reliability


def test_brier():
    assert brier([1.0, 0.0], [1, 0]) == 0
    assert brier([0.5, 0.5], [1, 0]) == pytest.approx(0.25)
    assert brier([0.0], [1]) == 1


def test_ece_equal_width_hand_worked():
    # one bin: confidence 0.95, accuracy 0.9
    assert ece([0.95] * 10, [1] * 9 + [0]) == pytest.approx(0.05)
    # two bins of four: (0.25, 1/4 right) is calibrated; (0.85, 3/4 right) is 0.1 over
    conf = [0.25] * 4 + [0.85] * 4
    ok = [1, 0, 0, 0] + [1, 1, 1, 0]
    assert ece(conf, ok) == pytest.approx(0.5 * 0 + 0.5 * 0.1)


def test_a_perfectly_calibrated_set_has_no_error():
    conf = [0.2] * 5 + [0.6] * 5 + [1.0] * 5
    ok = [1, 0, 0, 0, 0] + [1, 1, 1, 0, 0] + [1] * 5
    assert ece(conf, ok) == pytest.approx(0)
    assert ece(conf, ok, strategy="mass") == pytest.approx(0)


def test_confidence_one_falls_in_the_last_bin():
    rows = reliability([1.0, 0.95], [1, 1])
    assert len(rows) == 1 and rows[0]["range"] == [0.9, 1.0] and rows[0]["n"] == 2


def test_equal_mass_bins_keep_ties_together():
    # the cut after position 2 of 6 would split the four 0.1s: it moves after them
    rows = reliability([0.1] * 4 + [0.9] * 2, [0, 0, 0, 1, 1, 1], bins=3, strategy="mass")
    assert [r["n"] for r in rows] == [4, 2]
    assert [r["accuracy"] for r in rows] == [0.25, 1.0]


def test_equal_mass_bins_are_equal_without_ties():
    rows = reliability([i / 100 for i in range(100)], [0, 1] * 50, bins=10, strategy="mass")
    assert [r["n"] for r in rows] == [10] * 10


def pairs_auroc(conf, ok):
    pos = [c for c, y in zip(conf, ok, strict=True) if y]
    neg = [c for c, y in zip(conf, ok, strict=True) if not y]
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def test_auroc_known_values():
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1
    assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0
    assert auroc([0.5] * 4, [1, 0, 1, 0]) == 0.5
    assert auroc([0.5, 0.7], [1, 1]) is None
    assert auroc([0.5, 0.7], [0, 0]) is None


def test_auroc_counts_pairs_with_ties_as_half():
    rng = random.Random(5)
    for _ in range(300):
        n = rng.randint(2, 30)
        conf = [rng.choice([0.1, 0.3, 0.5, 0.7, 0.9]) for _ in range(n)]
        ok = [rng.randint(0, 1) for _ in range(n)]
        if 0 < sum(ok) < n:
            assert auroc(conf, ok) == pytest.approx(pairs_auroc(conf, ok))


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        brier([1.2], [1])
    with pytest.raises(ValueError):
        ece([0.5], [1], strategy="quantile")
    with pytest.raises(ValueError):
        auroc([0.5, 0.5], [1])
