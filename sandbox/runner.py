"""The sandbox's entry point: analysis jobs as JSON on stdin, their results as JSON on stdout.

Input: {"version": 1, "jobs": [{"id": ..., "spec": {...}, "columns": [...], "rows": [[...]]}]}.
Output: {"version": 1, "results": [{"id": ..., "ok": true, "result": {...}} or
{"id": ..., "ok": false, "error": {"kind": ..., "message": ...}}]}, keys sorted, so the same
input always gives the same bytes.

The input is data from the database, so it is untrusted: it is parsed as JSON only (no NaN or
Infinity, a size limit, a nesting limit), and a job that cannot be analyzed returns an error
instead of stopping the others. The container this runs in has no network, a read-only file
system, no secrets and hard limits on time, memory and processes (src/stats/sandbox.py).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analysis  # noqa: E402

VERSION = 1
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_DEPTH = 8  # the deepest valid input: document, jobs, job, rows, row


def _reject_constant(name: str):
    raise ValueError(f"{name} is not allowed")


def _depth(obj, limit: int) -> int:
    stack = [(obj, 1)]
    deepest = 0
    while stack:
        o, d = stack.pop()
        deepest = max(deepest, d)
        if d > limit:
            return d
        if isinstance(o, dict):
            stack.extend((v, d + 1) for v in o.values())
        elif isinstance(o, list):
            stack.extend((v, d + 1) for v in o)
    return deepest


def fail(kind: str, message: str) -> None:
    sys.stdout.write(
        json.dumps(
            {"version": VERSION, "error": {"kind": kind, "message": message}}, sort_keys=True
        )
    )
    sys.exit(2)


def run_job(job) -> dict:
    job_id = job.get("id") if isinstance(job, dict) else None
    try:
        if not isinstance(job, dict):
            raise analysis.AnalysisError("invalid_job", "a job must be an object")
        columns, rows = job.get("columns"), job.get("rows")
        if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
            raise analysis.AnalysisError("invalid_job", "columns must be a list of names")
        if not isinstance(rows, list):
            raise analysis.AnalysisError("invalid_job", "rows must be a list")
        result = analysis.run(job.get("spec"), columns, rows)
        return {"id": job_id, "ok": True, "result": analysis.clean(result)}
    except analysis.AnalysisError as e:
        return {"id": job_id, "ok": False, "error": {"kind": e.kind, "message": str(e)}}
    except (ValueError, TypeError, ArithmeticError, IndexError, KeyError) as e:
        return {
            "id": job_id,
            "ok": False,
            "error": {"kind": "analysis_failed", "message": f"{type(e).__name__}: {e}"},
        }


def main() -> None:
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        fail("input_too_large", f"the input exceeds {MAX_INPUT_BYTES} bytes")
    try:
        doc = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (ValueError, RecursionError) as e:
        fail("invalid_input", f"not valid JSON: {type(e).__name__}")
    if _depth(doc, MAX_DEPTH) > MAX_DEPTH:
        fail("invalid_input", f"nested deeper than {MAX_DEPTH} levels")
    if not isinstance(doc, dict) or doc.get("version") != VERSION:
        fail("invalid_input", f"expected an object with version {VERSION}")
    jobs = doc.get("jobs")
    if not isinstance(jobs, list):
        fail("invalid_input", "jobs must be a list")
    results = [run_job(job) for job in jobs]
    sys.stdout.write(
        json.dumps({"version": VERSION, "results": results}, sort_keys=True, allow_nan=False)
    )


if __name__ == "__main__":
    main()
