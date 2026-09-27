"""Lossless JSON encoding of query results, and hashes that identify a result.

Execution accuracy compares result rows as Python values from the database driver, where
`1`, `1.0` and `Decimal("1")` are different types, and a date is not a string. Plain JSON
would collapse those differences, so every value JSON cannot carry exactly is stored as a
one-key object naming its type: `{"decimal": "12.50"}`, `{"float": "0.1"}`,
`{"date": "1996-01-31"}`. Integers, strings, booleans and NULL stay plain JSON.
`decode_value(encode_value(v)) == v` for every type psycopg returns for the benchmark's columns.

The hashes compare two executions of the same query, e.g. gold SQL before and after the
tables moved schema. They are type-exact (`1` and `1.0` hash differently), so equal hashes
mean identical results; they are not the benchmark's comparison, which the evaluation code
implements separately.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import uuid
from decimal import Decimal
from typing import Any

JSONValue = Any


def encode_value(v: Any) -> JSONValue:
    if v is None or isinstance(v, bool | str):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        # repr round-trips every float, including nan and the infinities JSON cannot hold
        return {"float": repr(v)}
    if isinstance(v, Decimal):
        return {"decimal": str(v)}
    if isinstance(v, dt.datetime):  # before date: a datetime is a date
        return {"timestamp": v.isoformat()}
    if isinstance(v, dt.date):
        return {"date": v.isoformat()}
    if isinstance(v, dt.time):
        return {"time": v.isoformat()}
    if isinstance(v, dt.timedelta):
        return {"interval": [v.days, v.seconds, v.microseconds]}
    if isinstance(v, bytes | bytearray | memoryview):
        return {"bytes": base64.b64encode(bytes(v)).decode("ascii")}
    if isinstance(v, uuid.UUID):
        return {"uuid": str(v)}
    if isinstance(v, list | tuple):  # PostgreSQL arrays
        return {"array": [encode_value(x) for x in v]}
    raise TypeError(f"no lossless encoding for {type(v).__name__}: {v!r}")


def decode_value(j: JSONValue) -> Any:
    if not isinstance(j, dict):
        return j
    ((tag, x),) = j.items()
    match tag:
        case "float":
            return float(x)
        case "decimal":
            return Decimal(x)
        case "timestamp":
            return dt.datetime.fromisoformat(x)
        case "date":
            return dt.date.fromisoformat(x)
        case "time":
            return dt.time.fromisoformat(x)
        case "interval":
            return dt.timedelta(days=x[0], seconds=x[1], microseconds=x[2])
        case "bytes":
            return base64.b64decode(x)
        case "uuid":
            return uuid.UUID(x)
        case "array":
            return [decode_value(y) for y in x]
    raise ValueError(f"unknown value tag {tag!r}")


def encode_row(row: tuple | list) -> list[JSONValue]:
    return [encode_value(v) for v in row]


def decode_row(row: list[JSONValue]) -> tuple:
    return tuple(decode_value(v) for v in row)


def _row_key(row: tuple | list) -> str:
    return json.dumps(encode_row(row), ensure_ascii=False, separators=(",", ":"))


def _digest(keys: list[str]) -> str:
    h = hashlib.sha256()
    for k in keys:
        h.update(k.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def ordered_hash(rows: list) -> str:
    """Identifies the rows in their order: equal only if every row and the order match."""
    return _digest([_row_key(r) for r in rows])


def set_hash(rows: list) -> str:
    """Identifies the set of distinct rows, ignoring order and repeats."""
    return _digest(sorted({_row_key(r) for r in rows}))
