"""Query results survive the JSON encoding exactly, and the result hashes mean what they say."""

from __future__ import annotations

import datetime as dt
import json
import math
import uuid
from decimal import Decimal

import pytest

from src.db.values import decode_row, decode_value, encode_row, encode_value, ordered_hash, set_hash

VALUES = [
    None,
    True,
    False,
    0,
    -(2**63),
    "PRIJEM",
    "",
    " ",
    "příjem",
    0.1,
    -0.0,
    1e300,
    float("inf"),
    float("-inf"),
    Decimal("12.50"),
    Decimal("-0.000001"),
    dt.date(1993, 1, 1),
    dt.datetime(1998, 12, 31, 23, 59, 59, 999999),
    dt.datetime(1998, 12, 31, 12, 0, tzinfo=dt.UTC),
    dt.time(13, 45, 1, 5),
    dt.timedelta(days=-1, seconds=5, microseconds=7),
    b"\x00\xffbytes",
    uuid.UUID("12345678-1234-5678-1234-567812345678"),
    [1, None, "a"],
]


@pytest.mark.parametrize("v", VALUES, ids=repr)
def test_round_trip_through_json_keeps_value_and_type(v):
    back = decode_value(json.loads(json.dumps(encode_value(v))))
    assert back == v
    assert type(back) is type(v) or (isinstance(v, list) and isinstance(back, list))


def test_nan_round_trips():
    assert math.isnan(decode_value(json.loads(json.dumps(encode_value(float("nan"))))))


def test_numbers_that_compare_equal_stay_distinct():
    """1, 1.0 and Decimal('1') are equal in Python but are different results to record."""
    encoded = {json.dumps(encode_value(v)) for v in (1, 1.0, Decimal("1"))}
    assert len(encoded) == 3
    assert decode_row(encode_row((1, 1.0, Decimal("1")))) == (1, 1.0, Decimal("1"))


def test_unknown_type_is_an_error():
    with pytest.raises(TypeError):
        encode_value(object())
    with pytest.raises(ValueError):
        decode_value({"complex": "1+2j"})


def test_set_hash_ignores_order_and_repeats():
    rows = [(1, "a"), (2, "b")]
    assert set_hash(rows) == set_hash(list(reversed(rows))) == set_hash(rows + rows)
    assert set_hash(rows) != set_hash([(1, "a")])


def test_ordered_hash_sees_order():
    rows = [(1, "a"), (2, "b")]
    assert ordered_hash(rows) == ordered_hash(list(rows))
    assert ordered_hash(rows) != ordered_hash(list(reversed(rows)))


def test_hashes_are_type_exact():
    assert set_hash([(1,)]) != set_hash([(1.0,)])
    assert set_hash([(0.1,)]) != set_hash([(0.1000000000000001,)])
    assert set_hash([(None,)]) != set_hash([("",)])
