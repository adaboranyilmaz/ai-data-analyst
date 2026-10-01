"""The evidence record: everything the UI shows about one answer, in one JSON document.

A recorded run (a results record plus its trace) and a live run are turned into the same
record, so the same stream and the same page serve both. The record holds the question, the
steps in plain English, the SQL and its outline, the rows the answer rests on, the automatic
checks, the answer and its confidence block (src/serving/meter.py). A recorded run also holds
the verdict the evaluation gave it. It never holds an expert query: the verdict is all the
evaluation leaves in.
"""

from __future__ import annotations

from typing import Any

from src.explain.outline import outline
from src.serving.meter import Meter

SCHEMA_VERSION = 1
SUBMIT_PREFIXES = ("Submitted", "Declined")


def check_lines(checks: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The automatic checks as one line each: whether it passed, and in words."""
    if not checks or not checks.get("applicable"):
        return [{"name": "applicable", "ok": True, "text": "No result to check."}]
    out_of_range = checks.get("out_of_range") or []
    return [
        {
            "name": "not_empty",
            "ok": not checks.get("empty"),
            "text": "The query returned no rows."
            if checks.get("empty")
            else "The query returned rows.",
        },
        {
            "name": "no_repeated_rows",
            "ok": not checks.get("repeated_rows"),
            "text": "Two rows are identical."
            if checks.get("repeated_rows")
            else "No row is repeated.",
        },
        {
            "name": "values_in_range",
            "ok": not out_of_range,
            "text": "; ".join(
                f"{o['column']} has a {o['kind']} of {o['value']}, outside its possible range"
                for o in out_of_range
            )
            if out_of_range
            else "No count, amount or share is outside its possible range.",
        },
        {
            "name": "result_size",
            "ok": not checks.get("too_many_rows"),
            "text": "The result is larger than the rows shown."
            if checks.get("too_many_rows")
            else "The whole result fits.",
        },
    ]


def status_of(answer: dict[str, Any], confidence: dict[str, Any]) -> str:
    """`declined` (the analyst said the data cannot answer), `clarify` (it asked), `withheld`
    (its calibrated confidence is under the decline threshold) or `answered`."""
    if answer.get("declined"):
        return "declined"
    if answer.get("clarifying_question"):
        return "clarify"
    if confidence.get("withheld"):
        return "withheld"
    return "answered"


def model_step_text(db: str) -> str:
    return f"Read the question and the {db} database's tables and columns."


def _step_events(lines: list[str], model_ms: float | None, db: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [
        {
            "type": "step",
            "kind": "model",
            "text": model_step_text(db),
            "recorded_ms": model_ms or None,
        }
    ]
    for line in lines:
        kind = "submit" if line.startswith(SUBMIT_PREFIXES) else "tool"
        events.append({"type": "step", "kind": kind, "text": line, "recorded_ms": None})
    return events


def from_run(
    *,
    run_id: str,
    kind: str,
    record: dict[str, Any],
    trace: dict[str, Any],
    meter: Meter | None,
    source_run: str,
    larger: bool = False,
) -> dict[str, Any]:
    """The evidence record of one recorded benchmark or banking-set run, or of a live one
    (`meter` None: the confidence is shown as the model stated it, not calibrated; `larger`: a
    router's larger model answered, so its own calibration curve applies)."""
    final = trace["final"]
    answer = final["answer"]
    result = final["result"]
    if meter is None:  # a database the calibration was not measured on
        confidence = {
            "stated": final["confidence"] if not answer["declined"] else None,
            "calibrated": None,
            "band": None,
            "threshold": None,
            "withheld": False,
            "not_calibrated": True,
        }
    else:
        confidence = meter.describe(final["confidence"], bool(answer["declined"]), larger)
    sql = None if answer["declined"] else final["sql"]
    requests = trace.get("requests") or []
    model_ms = requests[0]["latency_ms"] if requests else None
    return {
        "schema_version": SCHEMA_VERSION,
        "id": run_id,
        "kind": kind,
        "source_run": source_run,
        "question": trace["question"]["question"],
        "hint": trace["question"].get("evidence"),
        "db_id": trace["question"]["db_id"],
        "category": record.get("category"),
        "difficulty": record.get("difficulty"),
        "design": trace["design"],
        "model": trace["model"],
        "steps": _step_events(final["steps"], model_ms, trace["question"]["db_id"]),
        "answer": {
            "text": answer["answer"],
            "sql": sql,
            "declined": answer["declined"],
            "decline_reason": answer["decline_reason"],
            "clarifying_question": answer["clarifying_question"],
            "assumptions": answer["assumptions"],
            "premise_correction": answer["premise_correction"],
        },
        "sql_outline": outline(sql) if sql else None,
        "result": {
            "ok": result["ok"],
            "error": result["error"],
            "columns": result["columns"],
            "rows": result["preview"],
            "fetched_rows": result["rows"],
            "truncated": result["truncated"],
        },
        "checks": check_lines(final["checks"]),
        "chart_spec": final["chart_spec"],
        "statistics": None,
        "confidence": confidence,
        "status": status_of(answer, confidence),
        "evaluation": _evaluation(record),
        "explanation": None,
        "cost_usd": record["cost_usd"],
    }


def _evaluation(record: dict[str, Any]) -> dict[str, Any] | None:
    """The recorded verdict: execution accuracy for a benchmark question, the category's rule
    for a banking-set question (a clarifying question, a decline, a premise correction)."""
    if record["source"] == "live":  # no expert answer exists for a question asked just now
        return None
    if record["source"] == "own":
        from src.eval.summary import own_success  # needs numpy: only the build of the demo set does

        verdict = own_success(record)
        return {
            "correct": None if verdict is None else bool(verdict),
            "rule": f"banking set, category {record['category']}",
        }
    return {"correct": bool(record["correct"]), "rule": "execution accuracy"}


def from_guardrail(
    *,
    run_id: str,
    rec: dict[str, Any],
    before: dict[str, Any] | None,
    review: dict[str, Any] | None,
    source_run: str,
) -> dict[str, Any]:
    """The evidence record of a guarded comparative or causal question: the pull, the analysis in
    the sandbox and the answer written from its result. `before` is the evidence record of the
    winning design's answer to the same question, which the guardrail leaves unchanged."""
    guarded = rec["guarded"]
    plan = rec["plan"]
    analysis = rec["analysis"]
    n = analysis.get("n")
    steps = [
        {
            "type": "step",
            "kind": "model",
            "text": "Recognized a comparative or causal question: it needs an interval.",
            "recorded_ms": None,
        },
        {
            "type": "step",
            "kind": "tool",
            "text": f"Planned the comparison: {plan['analysis'].replace('_', ' ')} of "
            f"{plan['outcome']} by {plan.get('group') or plan.get('x')}, "
            f"one row per {plan['unit']}.",
            "recorded_ms": None,
        },
        {
            "type": "step",
            "kind": "tool",
            "text": f"Pulled {n:,} rows through the read-only guard."
            if n
            else "Pulled the rows through the read-only guard.",
            "recorded_ms": None,
        },
        {
            "type": "step",
            "kind": "tool",
            "text": "Ran the analysis in the sandbox: no network, no files, a time limit.",
            "recorded_ms": None,
        },
        {
            "type": "step",
            "kind": "submit",
            "text": "Wrote the answer from the computed result.",
            "recorded_ms": None,
        },
    ]
    verdict = review or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "id": run_id,
        "kind": "guardrail",
        "source_run": source_run,
        "question": rec["question"],
        "hint": None,
        "db_id": rec["db_id"],
        "category": rec.get("category"),
        "difficulty": None,
        "design": "d1+guardrail",
        "model": "claude-sonnet-5",
        "steps": steps,
        "answer": {
            "text": guarded["delivered"],
            "sql": plan["sql"],
            "declined": False,
            "decline_reason": None,
            "clarifying_question": None,
            "assumptions": [],
            "premise_correction": None,
        },
        "sql_outline": outline(plan["sql"]),
        "result": None,
        "checks": [],
        "chart_spec": None,
        "statistics": analysis,
        "confidence": {
            "stated": None,
            "calibrated": None,
            "band": None,
            "threshold": None,
            "withheld": False,
            "not_calibrated": True,
        },
        "status": "answered",
        "evaluation": {
            "reviewed_success": verdict.get("reviewed_success"),
        },
        "explanation": None,
        "before": None
        if before is None
        else {"text": before["answer"]["text"], "sql": before["answer"]["sql"]},
        "cost_usd": rec.get("cost_usd"),
    }


def final_events(ev: dict[str, Any]) -> list[dict[str, Any]]:
    """The events after the steps: the SQL, the rows, the checks, the chart, the statistics, the
    answer and its confidence. Replay and live runs end with the same events."""
    out: list[dict[str, Any]] = []
    a = ev["answer"]
    if a["sql"]:
        out.append({"type": "sql", "sql": a["sql"], "outline": ev["sql_outline"]})
    if ev.get("result") is not None:
        r = ev["result"]
        out.append(
            {
                "type": "rows",
                "ok": r["ok"],
                "error": r["error"],
                "columns": r["columns"],
                "rows": r["rows"],
                "fetched_rows": r["fetched_rows"],
                "truncated": r["truncated"],
            }
        )
    if ev["checks"]:
        out.append({"type": "checks", "checks": ev["checks"]})
    if ev.get("statistics") is not None:
        out.append({"type": "statistics", "statistics": ev["statistics"]})
    if ev.get("chart_spec") is not None:
        out.append({"type": "chart", "spec": ev["chart_spec"], "columns": ev["result"]["columns"]})
    out.append({"type": "answer", "status": ev["status"], **a})
    out.append({"type": "confidence", **ev["confidence"]})
    return out
