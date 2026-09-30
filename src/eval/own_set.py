"""The hand-written banking test set: questions on the Czech bank database, pre-registered.

own_set/questions.yaml holds 40-60 questions written for this project (none paraphrases one of
the benchmark's financial questions), in six categories. Per category, what a question
carries and what a good answer does:

  a  standard analyst question   `gold_sql`                  returns the gold result
  b  multi-step or multi-join    `gold_sql`                  returns the gold result
  c  ambiguous                   `interpretations`: accepted asks a clarifying question, or
                                 readings, each an           states an assumption and returns
                                 `assumption` and `gold_sql` the result of an accepted reading
  d  unanswerable from the data  `missing`: what the data    declines, saying what is missing
                                 lacks
  e  false premise               `premise`, `correction`,    corrects the premise
                                 `premise_sql` (shows the
                                 premise is false)
  f  comparative or causal       `gold_sql` (the numbers     gives a difference with an
                                 the comparison rests on),   interval, and no causal claim
                                 `check` (what a sound       from observational data
                                 answer must include)

Every question has a `review`: `approved` (as drafted), `edited` (changed by the author),
`written` (written by the author), or `pending`. The set is frozen by its hash
(configs/eval.yaml `own_set.sha256`) only when none is pending, before any agent sees it.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import jsonschema
import yaml

CATEGORIES = ("a", "b", "c", "d", "e", "f")
REVIEWS = ("approved", "edited", "written", "pending")
MIN_QUESTIONS, MAX_QUESTIONS = 40, 60
_TEXT = {"type": "string", "minLength": 1}

_COMMON = {
    "id": {"type": "string", "pattern": "^own-[a-f][0-9]{2}$"},
    "category": {"enum": list(CATEGORIES)},
    "question": _TEXT,
    "review": {"enum": list(REVIEWS)},
    "notes": {"type": "string"},
}


def _question(category: str, required: dict[str, Any]) -> dict[str, Any]:
    return {
        "if": {"properties": {"category": {"const": category}}},
        "then": {
            "required": ["id", "category", "question", "review", *required],
            "properties": {**_COMMON, "category": {"const": category}, **required},
            "additionalProperties": False,
        },
    }


QUESTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["id", "category", "question", "review"],
    "properties": {"category": {"enum": list(CATEGORIES)}},
    "allOf": [
        _question("a", {"gold_sql": _TEXT}),
        _question("b", {"gold_sql": _TEXT}),
        _question(
            "c",
            {
                "interpretations": {
                    "type": "array",
                    "minItems": 2,
                    "items": {
                        "type": "object",
                        "required": ["assumption", "gold_sql"],
                        "additionalProperties": False,
                        "properties": {"assumption": _TEXT, "gold_sql": _TEXT},
                    },
                }
            },
        ),
        _question("d", {"missing": _TEXT}),
        _question("e", {"premise": _TEXT, "correction": _TEXT, "premise_sql": _TEXT}),
        _question("f", {"gold_sql": _TEXT, "check": _TEXT}),
    ],
}
_VALIDATOR = jsonschema.Draft202012Validator(QUESTION_SCHEMA)


def load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def problems(data: dict[str, Any]) -> list[str]:
    qs = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(qs, list):
        return ["no `questions` list"]
    found = []
    for i, q in enumerate(qs, 1):
        label = q.get("id", f"question {i}") if isinstance(q, dict) else f"question {i}"
        found += [f"{label}: {e.message}" for e in _VALIDATOR.iter_errors(q)]
        if isinstance(q, dict) and isinstance(q.get("id"), str) and q.get("category"):
            if q["id"][4] != q["category"]:
                found.append(f"{label}: the id's letter is not its category")
    ids = Counter(q.get("id") for q in qs if isinstance(q, dict))
    found += [f"{i}: appears {n} times" for i, n in ids.items() if n > 1]
    if not MIN_QUESTIONS <= len(qs) <= MAX_QUESTIONS:
        found.append(f"{len(qs)} questions, not {MIN_QUESTIONS}-{MAX_QUESTIONS}")
    return found


def queries(q: dict[str, Any]) -> list[tuple[str, str]]:
    """Every SQL a question carries, labeled: `gold`, `reading-<n>`, `premise`."""
    if q["category"] in ("a", "b", "f"):
        return [("gold", q["gold_sql"])]
    if q["category"] == "c":
        return [(f"reading-{i}", r["gold_sql"]) for i, r in enumerate(q["interpretations"], 1)]
    if q["category"] == "e":
        return [("premise", q["premise_sql"])]
    return []
