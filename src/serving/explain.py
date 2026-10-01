"""What the right answer was, and why an answer was scored wrong.

For a wrong answer on the page: the expected result (the expert query's rows, run now on the
same database), the expert's query, a comparison of the two results that is computed (never
written by hand), the automatic classification of the first place the queries differ, and a
reviewer's note in plain words. The analyst never saw any of this: it is added after the fact,
for the reader.

The comparison reads rows, not intent. A reviewer's note says what the rows cannot: for example
that two queries answer different readings of the question, or that the expert's query departs
from its own hint.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

SHOWN_ROWS = 20

# The automatic classification (src/eval/errors.py), in words. It reads the structure of the
# queries, not their intent.
CATEGORY_TEXT = {
    "format_only": "the same information in a different shape",
    "tables": "it read different tables",
    "join": "it joined the tables differently",
    "filter": "it filtered the rows differently",
    "computation": "it computed the value differently",
    "output": "it returned different columns",
    "order_limit": "it ordered or limited the rows differently",
    "other": "a difference the automatic comparison could not classify",
    "no_result": "it returned no result",
}


def reference(run_sql: Callable[[str], dict[str, Any]], sql: str) -> dict[str, Any] | None:
    """The expected result: the expert query's columns and first rows, or None if it cannot run."""
    out = run_sql(sql)
    if not out.get("ok"):
        return None
    rows = out["rows"]
    total = out.get("total_rows")
    return {
        "columns": [c["name"] for c in out["columns"]],
        "rows": rows[:SHOWN_ROWS],
        "total_rows": len(rows) if total is None else total,
        "truncated": len(rows) > SHOWN_ROWS or bool(out.get("truncated")),
    }


def _key(row: list[Any]) -> str:
    return repr(row)


def compare(got: dict[str, Any] | None, expected: dict[str, Any] | None) -> dict[str, Any] | None:
    """How the analyst's result relates to the expected one, from the rows alone. `got` has
    `columns` and `rows` (the first rows, with `truncated` and `fetched_rows`). None when either
    result is missing or too long to compare in full."""
    if not got or not expected or not got.get("ok", True):
        return None
    if got.get("truncated") or expected["truncated"]:
        return None
    if len(got["rows"]) > SHOWN_ROWS or len(expected["rows"]) > SHOWN_ROWS:
        return None
    gc = [c.lower() for c in got["columns"]]
    ec = [c.lower() for c in expected["columns"]]
    grows, erows = got["rows"], expected["rows"]
    if len(grows) == 1 and len(erows) == 1 and len(gc) == 1 and len(ec) == 1:
        a, b = grows[0][0], erows[0][0]
        return {
            "relation": "single_value",
            "got": a,
            "expected": b,
            "text": f"It returned {a}; the expected answer is {b}.",
        }
    if gc == ec:
        if sorted(map(_key, grows)) == sorted(map(_key, erows)):
            return {
                "relation": "same_rows",
                "text": "The rows are the same; they come in a different order."
                if grows != erows
                else "The rows are identical.",
            }
        return {"relation": "different", "text": "The rows differ from the expected rows."}
    if set(ec) <= set(gc):
        idx = [gc.index(c) for c in ec]
        projected = [[r[i] for i in idx] for r in grows]
        if sorted(map(_key, projected)) == sorted(map(_key, erows)):
            extra = [got["columns"][i] for i in range(len(gc)) if i not in idx]
            return {
                "relation": "extra_columns",
                "extra": extra,
                "text": "Every expected row is there. The analyst returned the extra column"
                + ("s " if len(extra) > 1 else " ")
                + ", ".join(extra)
                + ", and the benchmark compares every column.",
            }
    return {"relation": "different", "text": "The rows differ from the expected rows."}


def build(
    *,
    got: dict[str, Any] | None,
    expected: dict[str, Any] | None,
    expected_sql: str | None,
    note: str | None,
    category: str | None,
    hint: str | None,
) -> dict[str, Any]:
    """The explanation block of a wrong answer's evidence record."""
    return {
        "expected": expected,
        "expected_sql": expected_sql,
        "comparison": compare(got, expected),
        "category": category,
        "category_text": CATEGORY_TEXT.get(category or ""),
        "note": note,
        "hint": hint,
    }
