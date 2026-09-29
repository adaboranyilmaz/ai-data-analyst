"""The agent's structured answer: the `submit_answer` tool, and design 5's `select_schema`.

Every design ends with the same answer, given as the input of one `submit_answer` call. The tool
is strict, so the API guarantees an input that matches its schema on the models that support
strict tools; the local model's input is checked here all the same, and a malformed answer is
recorded as an error, never repaired silently. Strict schemas cannot state a numeric range, so a
confidence outside 0-1 is clipped here and the clipping recorded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

SUBMIT = "submit_answer"
SELECT_SCHEMA = "select_schema"

_TEXT_OR_NULL = {"anyOf": [{"type": "string"}, {"type": "null"}]}

SUBMIT_TOOL: dict[str, Any] = {
    "name": SUBMIT,
    "description": (
        "Submit your final answer: the SQL whose result answers the question, the answer in "
        "words, your confidence that the SQL's result is correct, and whether you decline, ask "
        "for clarification, assume something or correct a false premise."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "sql": {**_TEXT_OR_NULL, "description": "One PostgreSQL SELECT query, or null."},
            "answer": {"type": "string", "description": "The answer in plain words."},
            "confidence": {
                "type": "number",
                "description": "Probability from 0 to 1 that the SQL's result is correct.",
            },
            "declined": {"type": "boolean", "description": "True if the data cannot answer."},
            "decline_reason": {**_TEXT_OR_NULL, "description": "What is missing, if declined."},
            "clarifying_question": {
                **_TEXT_OR_NULL,
                "description": "The question to ask the user if it is ambiguous, else null.",
            },
            "assumptions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Assumptions the SQL makes that the question does not state.",
            },
            "premise_correction": {
                **_TEXT_OR_NULL,
                "description": "What the data shows, if the question's premise is false.",
            },
            "chart_spec": {
                **_TEXT_OR_NULL,
                "description": "Optional Vega-Lite v6 spec as JSON text, or null.",
            },
        },
        "required": [
            "sql",
            "answer",
            "confidence",
            "declined",
            "decline_reason",
            "clarifying_question",
            "assumptions",
            "premise_correction",
            "chart_spec",
        ],
        "additionalProperties": False,
    },
}

SELECT_SCHEMA_TOOL: dict[str, Any] = {
    "name": SELECT_SCHEMA,
    "description": "Give the tables, and the columns in each, the analyst may need.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "tables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string"},
                        "columns": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["table", "columns"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["tables"],
        "additionalProperties": False,
    },
}


@dataclass
class Answer:
    sql: str | None
    answer: str | None
    confidence: float | None
    declined: bool
    decline_reason: str | None = None
    clarifying_question: str | None = None
    assumptions: list[str] = field(default_factory=list)
    premise_correction: str | None = None
    chart_spec: Any = None  # the parsed Vega-Lite spec, or None
    problems: list[str] = field(default_factory=list)

    @classmethod
    def none(cls) -> Answer:
        """No answer was given (budget spent, refusal, no submit call); the conversation records
        why among its errors. Confidence 0: nothing was claimed."""
        return cls(None, None, 0.0, False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sql": self.sql,
            "answer": self.answer,
            "confidence": self.confidence,
            "declined": self.declined,
            "decline_reason": self.decline_reason,
            "clarifying_question": self.clarifying_question,
            "assumptions": list(self.assumptions),
            "premise_correction": self.premise_correction,
            "chart_spec": self.chart_spec,
            "problems": list(self.problems),
        }


def _text(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def parse_answer(args: Any) -> Answer:
    """A `submit_answer` input as an Answer; problems are recorded, not raised."""
    if not isinstance(args, dict):
        return Answer(None, None, 0.0, False, problems=["submit_answer input is not an object"])
    problems: list[str] = []
    missing = [k for k in SUBMIT_TOOL["input_schema"]["required"] if k not in args]
    if missing:
        problems.append(f"submit_answer is missing {missing}")
    extra = sorted(set(args) - set(SUBMIT_TOOL["input_schema"]["properties"]))
    if extra:
        problems.append(f"submit_answer has unexpected fields {extra}")

    conf = args.get("confidence")
    if isinstance(conf, str):
        try:
            conf = float(conf)
        except ValueError:
            conf = None
    if isinstance(conf, bool) or not isinstance(conf, int | float) or conf != conf:
        problems.append(f"confidence is not a number: {args.get('confidence')!r}")
        conf = 0.0
    elif not 0.0 <= conf <= 1.0:
        problems.append(f"confidence {conf} outside 0-1, clipped")
        conf = min(1.0, max(0.0, float(conf)))

    declined = args.get("declined")
    if not isinstance(declined, bool):
        problems.append(f"declined is not a boolean: {declined!r}")
        declined = declined == "true" or (declined == 1 and not isinstance(declined, float))

    assumptions = args.get("assumptions") or []
    if not isinstance(assumptions, list):
        problems.append("assumptions is not a list")
        assumptions = [str(assumptions)]

    chart = None
    raw_chart = args.get("chart_spec")
    if isinstance(raw_chart, dict):
        chart = raw_chart
    elif isinstance(raw_chart, str) and raw_chart.strip():
        try:
            chart = json.loads(raw_chart)
        except json.JSONDecodeError:
            problems.append("chart_spec is not valid JSON; dropped")

    sql = _text(args.get("sql"))
    return Answer(
        sql=sql,
        answer=_text(args.get("answer")),
        confidence=float(conf),
        declined=declined,
        decline_reason=_text(args.get("decline_reason")),
        clarifying_question=_text(args.get("clarifying_question")),
        assumptions=[s for s in (_text(a) for a in assumptions) if s],
        premise_correction=_text(args.get("premise_correction")),
        chart_spec=chart,
        problems=problems,
    )


def parse_selection(args: Any, known: dict[str, list[str]]) -> tuple[dict[str, list[str]], list]:
    """A `select_schema` input as {table: [columns]}, keeping only real tables and columns (in
    the schema's order). Unknown names are reported. A table given with no known column keeps
    all its columns: it was chosen, and a join needs its keys."""
    problems: list[str] = []
    chosen: dict[str, set[str]] = {}
    items = args.get("tables") if isinstance(args, dict) else None
    if not isinstance(items, list):
        return {}, ["select_schema input has no `tables` list"]
    lower = {t.lower(): t for t in known}
    for item in items:
        if not isinstance(item, dict):
            problems.append(f"not a table entry: {item!r}")
            continue
        name = lower.get(str(item.get("table", "")).lower())
        if name is None:
            problems.append(f"unknown table {item.get('table')!r}")
            continue
        cols = {c.lower(): c for c in known[name]}
        picked = chosen.setdefault(name, set())
        for c in item.get("columns") or []:
            if str(c).lower() in cols:
                picked.add(cols[str(c).lower()])
            else:
                problems.append(f"unknown column {name}.{c}")
    out = {}
    for t in known:  # the schema's order, so the result does not depend on the model's order
        if t in chosen:
            out[t] = [c for c in known[t] if c in chosen[t]] or list(known[t])
    return out, problems
