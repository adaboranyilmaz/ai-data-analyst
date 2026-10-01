"""The canary: the same requests, sent again with no cache, compared with the recorded answers.

A provider can change what a model does without a version change. The canary asks a fixed set of
questions on demand: each request is rebuilt exactly as it was evaluated (src/tracking/gate.py),
sent to the model, and the response read as the agent reads it. What it records per question: did
the response parse to an answer, is the SQL the recorded one, how far the stated confidence moved,
how many prompt tokens the same request counted now, and what it cost. Optionally (`score`, where
the benchmark database is loaded) whether the new SQL returns the recorded SQL's rows.

Runs are appended to a history; a run that crosses a flag in configs/canary.yaml says so. The flags
are prompts to look, not verdicts: model outputs are samples.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.tracking import gate

ROOT = gate.ROOT
CONFIG = "configs/canary.yaml"
BENCH_RUN = "main/all-d1-claude-sonnet-5-evidence"
OWN_RUN = "own/own-d1-claude-sonnet-5-no-evidence"
ROUTER_RUN = "router/held_out-d1-claude-opus-5-5-evidence"


def config(root: Path = ROOT) -> dict[str, Any]:
    return yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))


def _demo(root: Path, name: str) -> dict[str, Any]:
    return json.loads((root / "results/demo/runs" / f"{name}.json").read_text(encoding="utf-8"))


def _own_question(root: Path, qid: str) -> dict[str, Any]:
    """A banking question's text. Only the text is read: its expert query never leaves the file."""
    data = yaml.safe_load((root / "own_set/questions.yaml").read_text(encoding="utf-8"))
    q = next(q for q in data["questions"] if q["id"] == qid)
    return {"db_id": data["database"], "question": " ".join(q["question"].split())}


def items(cfg: dict[str, Any], root: Path = ROOT) -> list[dict[str, Any]]:
    """The calls of a run: entries shaped as the gate's, in a fixed order."""
    primary, larger = cfg["primary"], cfg["larger"]
    out: list[dict[str, Any]] = []
    for n, qid in enumerate(cfg["benchmark"]):
        run = _demo(root, f"bench-{qid}")
        base = {
            "question": {
                "question_id": qid,
                "source": "bird",
                "db_id": run["db_id"],
                "question": run["question"],
                "evidence": run["hint"],
            },
            "design": primary["design"],
            "evidence": True,
        }
        out.append(
            {
                "id": f"bench-{qid}/{primary['model']}",
                "model": primary["model"],
                "source_run": BENCH_RUN,
                **base,
            }
        )
        if n < larger["limit"]:
            out.append(
                {
                    "id": f"bench-{qid}/{larger['model']}",
                    "model": larger["model"],
                    "source_run": ROUTER_RUN,
                    **base,
                }
            )
    for qid in cfg["banking"]:
        q = _own_question(root, qid)
        out.append(
            {
                "id": f"bank-{qid}/{primary['model']}",
                "model": primary["model"],
                "source_run": OWN_RUN,
                "question": {"question_id": qid, "source": "own", **q, "evidence": None},
                "design": primary["design"],
                "evidence": False,
            }
        )
    return out


def _end_span(run, answer) -> None:
    """Close the run's span. A finished run closes it itself (src/agent/run.py `finish`), but that
    executes the SQL; the canary needs no database, so it closes the span with what it knows."""
    span = getattr(run, "_span", None)
    if span is None:
        return
    span.set_attributes(
        {
            "confidence": float(answer.confidence or 0.0) if answer else 0.0,
            "declined": bool(answer.declined) if answer else False,
            "errors": len(answer.problems) if answer else 1,
        }
    )
    span.end()


def normalized(sql: str | None) -> str | None:
    return None if sql is None else " ".join(sql.split()).rstrip(";").lower()


def run_item(
    entry: dict[str, Any],
    call: Callable[[Any], Any],
    root: Path = ROOT,
    tracer=None,
    score: Callable[[str, str | None, str | None], bool | None] | None = None,
) -> dict[str, Any]:
    """One call: build the request, send it through `call` (a function from request to response),
    read the response and compare it with the recorded answer."""
    run, box = gate._run(entry, root, tracer)
    try:
        ((conv, request),) = run.pending()
        started = time.perf_counter()
        response = call(request)
        seconds = time.perf_counter() - started
        run.feed(conv, request.cache_key, response)
        answer = conv.answer
        recorded = gate._records(root, entry["source_run"])[entry["question"]["question_id"]]
        t = response.tokens
        prompt = t.input + t.cache_write_5m + t.cache_write_1h + t.cache_read
        r = recorded["tokens"]
        was = r["input"] + r["cache_write_5m"] + r["cache_write_1h"] + r["cache_read"]
        new_sql = None if answer is None or answer.declined else answer.sql
        _end_span(run, answer)
        out = {
            "id": entry["id"],
            "model": entry["model"],
            "stop_reason": response.stop_reason,
            "valid_answer": answer is not None and not answer.problems and not conv.errors,
            "declined": bool(answer.declined) if answer else None,
            "recorded_declined": recorded["declined"],
            "sql_unchanged": normalized(new_sql) == normalized(recorded["final_sql"]),
            "confidence": answer.confidence if answer else None,
            "recorded_confidence": recorded["confidence"],
            "prompt_tokens": prompt,
            "recorded_prompt_tokens": was,
            "output_tokens": t.output,
            "latency_s": round(seconds, 3),
        }
        if score is not None:
            out["rows_match_recorded"] = score(
                entry["question"]["db_id"], new_sql, recorded["final_sql"]
            )
        return out
    finally:
        box.close()


def summarize(results: list[dict[str, Any]], flags: dict[str, Any]) -> dict[str, Any]:
    n = len(results)
    valid = [r for r in results if r["valid_answer"]]
    shifts = [
        abs(r["confidence"] - r["recorded_confidence"])
        for r in valid
        if r["confidence"] is not None and not r["declined"] and not r["recorded_declined"]
    ]
    token_changes = [
        abs(r["prompt_tokens"] - r["recorded_prompt_tokens"]) / r["recorded_prompt_tokens"]
        for r in results
        if r["recorded_prompt_tokens"]
    ]
    summary = {
        "calls": n,
        "valid_answers": len(valid) / n if n else None,
        "sql_unchanged_rate": sum(r["sql_unchanged"] for r in results) / n if n else None,
        "mean_confidence_shift": sum(shifts) / len(shifts) if shifts else None,
        "max_prompt_token_change": max(token_changes) if token_changes else None,
        "declined_changed": sum(1 for r in valid if r["declined"] != r["recorded_declined"]),
        "stop_reasons": {
            k: sum(r["stop_reason"] == k for r in results)
            for k in sorted({r["stop_reason"] for r in results})
        },
    }
    scored = [r["rows_match_recorded"] for r in results if r.get("rows_match_recorded") is not None]
    if scored:
        summary["rows_match_recorded_rate"] = sum(scored) / len(scored)
    raised = []
    if n and summary["valid_answers"] < flags["min_valid_answers"]:
        raised.append("a response did not parse to an answer")
    if n and summary["sql_unchanged_rate"] < flags["min_sql_unchanged_rate"]:
        raised.append("fewer questions than usual give the recorded SQL")
    if shifts and summary["mean_confidence_shift"] > flags["max_mean_confidence_shift"]:
        raised.append("the stated confidence moved")
    if token_changes and summary["max_prompt_token_change"] > flags["max_prompt_token_change"]:
        raised.append("the same request counts different tokens")
    summary["flags"] = raised
    return summary


def read_history(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "note": "canary runs, oldest first; each re-sends the same fixed requests with no response "
        "cache and compares with the recorded answers (src/tracking/canary.py). Model outputs "
        "are samples: a flag is a prompt to look, not a verdict.",
        "runs": [],
    }


def append(path: Path, run: dict[str, Any]) -> dict[str, Any]:
    history = read_history(path)
    if any(r["utc"] == run["utc"] for r in history["runs"]):
        raise ValueError("this run is already in the history")
    history["runs"].append(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(history, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return history


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
