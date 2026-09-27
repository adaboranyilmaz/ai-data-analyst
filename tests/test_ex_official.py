"""The project's EX and Soft-F1 against the official evaluator's own comparison functions, on
random results. Skipped unless the official evaluator is downloaded (scripts/30_fetch_bird_eval.py)
and the `bird-eval` dependency group is installed (`uv run --group bird-eval pytest ...`); CI has
neither. The database-level check of the same claim is scripts/31_validate_ex.py."""

from __future__ import annotations

import random
from decimal import Decimal

import pytest

from src.eval.config import official_evaluator_dir
from src.eval.ex import PredictionStream, ex, soft_f1

for module in ("psycopg2", "func_timeout", "pymysql"):
    pytest.importorskip(module, reason="the bird-eval dependency group is not installed")


@pytest.fixture(scope="module")
def official():
    from src.eval import official as o

    try:
        return o.load(official_evaluator_dir())
    except FileNotFoundError as e:
        pytest.skip(str(e))


VALUES = [0, 1, 2, 3, 1.0, 2.5, -0.0, Decimal("2.5"), Decimal("0.10"), 0.1, "a", "b", "", None]
VALUES += [True, False, float("nan")]


def rows(rng: random.Random, width: int, n: int) -> list[tuple]:
    return [tuple(rng.choice(VALUES) for _ in range(width)) for _ in range(n)]


def outcome(f, *args):
    try:
        return ("value", f(*args))
    except Exception as e:
        return ("raises", type(e).__name__)


def test_ex_and_soft_f1_agree_with_the_official_functions(official):
    rng = random.Random(11)
    for _ in range(20_000):
        width = rng.randint(0, 3)
        gold = rows(rng, width, rng.randint(0, 6))
        pred = rows(
            rng, rng.choice([width, width, max(width - 1, 0), width + 1]), rng.randint(0, 6)
        )
        if rng.random() < 0.3 and gold:  # a reordered or repeated copy of the gold result
            pred = rng.sample(gold, len(gold)) + rng.choices(gold, k=rng.randint(0, 2))
        if rng.random() < 0.02:
            pred = pred + [([1, 2],) * max(width, 1)]  # a value that cannot be hashed
        assert outcome(ex, pred, gold) == outcome(official.ex.calculate_ex, pred, gold)
        ours = outcome(soft_f1, pred, gold)
        theirs = outcome(official.f1.calculate_f1_score, pred, gold)
        assert ours == theirs, (pred, gold)
        if ours[0] == "value":  # the same floating-point number, not merely close
            assert repr(ours[1]) == repr(theirs[1])


def test_the_stream_agrees_with_the_official_functions(official):
    rng = random.Random(12)
    for _ in range(5_000):
        width = rng.randint(1, 3)
        gold = rows(rng, width, rng.randint(0, 6))
        pred = rows(rng, width, rng.randint(0, 6))
        try:
            stream = PredictionStream(gold, max_distinct=1_000)
            stream.feed(pred)
        except TypeError:
            pytest.fail("no unhashable values are generated here")
        assert stream.ex() == official.ex.calculate_ex(pred, gold)
        f1 = stream.soft_f1()
        try:
            expected = official.f1.calculate_f1_score(pred, gold)
        except ZeroDivisionError:
            continue
        assert repr(f1) == repr(expected)
