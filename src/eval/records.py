"""The per-question record: one line of a JSON-lines file per question an agent answered.

Every evaluated answer, on the benchmark or the hand-written banking set, is recorded with
what it cost and how it was scored, so that every reported number can be recomputed from the
records alone:

- what was asked: `run` (one design, model and setting), `question_id`, `source` (`bird` or
  `own`), `db_id`, `difficulty` (benchmark) or `category` (own set: a-f), `design`, `model`,
  `evidence` (whether the benchmark's hint was given);
- what came back: `final_sql`, `answer`, `confidence` (0-1), `declined` and
  `decline_reason`, `clarifying_question`, `assumptions`, `premise_correction`;
- how it scored: `correct` (EX of `final_sql`, 0 or 1; for an ambiguous own-set question, 1 if
  it matches any accepted reading; null where the category has no gold result), `soft_f1`,
  `score_outcome` (src/eval/execution.py, or `not_scored`);
- what it took: `tokens`, `cost_usd`, `latency_s`, `steps`, `tool_calls`, `errors`, `trace`
  (the replayable trace file), `evaluated_at` (when it was scored: eleven gold queries read
  the clock).

`correct` scores the SQL; whether the answer counts is `answered_correct`: a declined question
is never a correct answer, whatever its SQL would have returned.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import jsonschema

SCORE_OUTCOMES = [
    "ok",
    "no_prediction",
    "timeout",
    "error",
    "comparison_error",
    "gold_error",
    "not_scored",
]
_STRING_OR_NULL = {"type": ["string", "null"]}
_COUNT = {"type": "integer", "minimum": 0}

RECORD_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "run",
        "question_id",
        "source",
        "db_id",
        "difficulty",
        "category",
        "design",
        "model",
        "evidence",
        "final_sql",
        "answer",
        "confidence",
        "declined",
        "decline_reason",
        "clarifying_question",
        "assumptions",
        "premise_correction",
        "correct",
        "soft_f1",
        "score_outcome",
        "tokens",
        "cost_usd",
        "latency_s",
        "steps",
        "tool_calls",
        "errors",
        "trace",
        "evaluated_at",
    ],
    "properties": {
        "run": {"type": "string", "minLength": 1},
        "question_id": {"type": ["integer", "string"]},
        "source": {"enum": ["bird", "own"]},
        "db_id": {"type": "string", "minLength": 1},
        "difficulty": {"enum": ["simple", "moderate", "challenging", None]},
        "category": {"enum": ["a", "b", "c", "d", "e", "f", None]},
        "design": {"type": "string", "minLength": 1},
        "model": {"type": "string", "minLength": 1},
        "evidence": {"type": "boolean"},
        "final_sql": _STRING_OR_NULL,
        "answer": _STRING_OR_NULL,
        "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "declined": {"type": "boolean"},
        "decline_reason": _STRING_OR_NULL,
        "clarifying_question": _STRING_OR_NULL,
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "premise_correction": _STRING_OR_NULL,
        "correct": {"enum": [0, 1, None]},
        "soft_f1": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "score_outcome": {"enum": SCORE_OUTCOMES},
        "tokens": {
            "type": "object",
            "additionalProperties": False,
            "required": ["input", "output", "cache_write_5m", "cache_write_1h", "cache_read"],
            "properties": {
                k: _COUNT
                for k in ("input", "output", "cache_write_5m", "cache_write_1h", "cache_read")
            },
        },
        "cost_usd": {"type": "number", "minimum": 0},
        "latency_s": {"type": "number", "minimum": 0},
        "steps": _COUNT,
        "tool_calls": _COUNT,
        "errors": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["kind", "message"],
                "properties": {"kind": {"type": "string"}, "message": {"type": "string"}},
            },
        },
        "trace": _STRING_OR_NULL,
        "evaluated_at": {"type": "string"},  # an ISO 8601 time with its offset: checked below
    },
    "allOf": [
        # a benchmark question has a difficulty and no category; an own-set one the reverse
        {
            "if": {"properties": {"source": {"const": "bird"}}},
            "then": {
                "properties": {
                    "difficulty": {"enum": ["simple", "moderate", "challenging"]},
                    "category": {"const": None},
                }
            },
            "else": {
                "properties": {
                    "difficulty": {"const": None},
                    "category": {"enum": ["a", "b", "c", "d", "e", "f"]},
                }
            },
        }
    ],
}

_VALIDATOR = jsonschema.Draft202012Validator(RECORD_SCHEMA)


def problems(record: dict) -> list[str]:
    found = [
        f"{'/'.join(map(str, e.absolute_path)) or '(record)'}: {e.message}"
        for e in sorted(_VALIDATOR.iter_errors(record), key=lambda e: list(e.absolute_path))
    ]
    when = record.get("evaluated_at")
    if isinstance(when, str):
        try:
            if dt.datetime.fromisoformat(when).utcoffset() is None:
                found.append("evaluated_at: no time zone offset")
        except ValueError:
            found.append(f"evaluated_at: not an ISO 8601 time: {when!r}")
    return found


def validate(record: dict) -> None:
    if found := problems(record):
        raise ValueError(f"invalid record {record.get('question_id')!r}: " + "; ".join(found))


def answered_correct(record: dict) -> int:
    """1 if the question was answered, not declined, and its SQL scored correct."""
    return int(record["correct"] == 1 and not record["declined"])


def write_records(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            validate(r)
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_records(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            validate(r)
            out.append(r)
    return out
