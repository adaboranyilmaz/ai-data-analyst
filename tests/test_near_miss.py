"""The near-miss rules on hand-made results."""

from __future__ import annotations

import importlib.util
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("near_miss", ROOT / "scripts/58_near_miss.py")
near_miss = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(near_miss)
classify = near_miss.classify


def test_an_extra_column():
    assert classify([("a", 1), ("b", 2)], [("a",), ("b",)], 1) == "columns"


def test_columns_in_another_order():
    assert classify([(1, "a"), (2, "b")], [("a", 1), ("b", 2)], 2) == "columns"


def test_rounding_only():
    assert classify([(33.3333,)], [(Decimal("33.33"),)], 1) == "rounding"


def test_extra_column_and_rounding():
    assert classify([("x", 0.12345)], [(0.12,)], 1) == "columns_and_rounding"


def test_wrong_values_are_other():
    assert classify([("a",), ("c",)], [("a",), ("b",)], 1) == "other"


def test_fewer_columns_than_gold_is_other():
    assert classify([("a",)], [("a", 1)], 2) == "other"


def test_large_results_are_not_checked():
    big = [(i,) for i in range(near_miss.MAX_ROWS_CHECKED + 1)]
    assert classify(big, [(1,)], 1) == "not_checked"
