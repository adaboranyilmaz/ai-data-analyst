"""One plain-English line per step of a run, from what happened (no model writes them), so the
same run always reads the same way. The UI's step timeline and the labelling page reuse them."""

from __future__ import annotations

from typing import Any


def _rows(n: Any) -> str:
    return "1 row" if n == 1 else f"{n} rows"


def tool_line(name: str, args: Any, result: dict) -> str:
    args = args if isinstance(args, dict) else {}
    if not result.get("ok"):
        err = result.get("error") or {}
        why = err.get("message") or "; ".join(err.get("reasons") or []) or err.get("kind", "error")
        what = {
            "list_tables": "Listing the tables",
            "describe_table": f"Looking up the table {args.get('table', '?')}",
            "sample_rows": f"Sampling rows of {args.get('table', '?')}",
            "run_sql": "A query",
        }.get(name, f"The tool {name}")
        return f"{what} failed: {why}"
    if name == "list_tables":
        return f"Listed the {len(result.get('tables', []))} tables."
    if name == "describe_table":
        return (
            f"Looked up the table {result.get('table')} ({len(result.get('columns', []))} columns)."
        )
    if name == "sample_rows":
        return f"Looked at {_rows(result.get('row_count'))} of the table {args.get('table')}."
    if name == "run_sql":
        total = result.get("total_rows")
        return f"Ran a query: {_rows(total if total is not None else result.get('row_count'))}."
    return f"Used the tool {name}."


def submit_line(declined: bool, confidence: float | None) -> str:
    if declined:
        return "Declined to answer."
    pct = f"{round(100 * confidence)}%" if confidence is not None else "no stated"
    return f"Submitted an answer ({pct} confidence)."


def resubmit_line(outcome: str) -> str:
    return f"The submitted query {outcome}; asked to fix it."
