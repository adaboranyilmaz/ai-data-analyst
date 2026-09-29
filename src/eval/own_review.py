"""The hand check of the banking set's answers that the automatic rules pass on form alone.

The pre-registered rules count an ambiguous question as a success when a clarifying question is
asked, and a false-premise question when a premise correction is given, whatever either says.
Those answers are checked by hand against what the frozen set records: an ambiguous question's
accepted readings (a clarifying question succeeds if it asks the user to choose between them, or
between readings like them), a false premise's recorded correction (a correction succeeds if it
says what that correction says). Every other answer's success is already decided by its rule:
a failure stays a failure, and an assumption-based answer is checked by its SQL.

The review file (results/reviews/own_set_review.yaml) lists each such answer with what was
expected and what was given, and a verdict, `correct` or `incorrect`, with a note. The reviewed
success of a category is its rule's success, and for an answer that needed a check, a `correct`
verdict as well. It is reported beside the pre-registered success, which stays the headline.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from src.eval.summary import own_success

VERDICTS = ("correct", "incorrect")
CRITERIA = (
    "clarifying_question: correct if it asks the user to choose between the accepted readings, "
    "or readings like them, of what the question leaves open; premise_correction: correct if it "
    "states what the recorded correction states (the premise is false, and what the data shows)"
)


def check_kind(record: dict) -> str | None:
    """What of an answer needs a hand check, if anything."""
    if record["category"] == "c" and record["clarifying_question"]:
        return "clarifying_question"
    if record["category"] == "e" and record["premise_correction"]:
        return "premise_correction"
    return None


def template(records: Sequence[dict], questions: dict[str, dict], run: str) -> dict[str, Any]:
    """The review file to fill in: one item per answer that needs a check, verdict empty.
    `questions`: the banking set's questions by id."""
    items = []
    for r in records:
        kind = check_kind(r)
        if kind is None:
            continue
        q = questions[r["question_id"]]
        expected: Any = (
            [i["assumption"] for i in q["interpretations"]]
            if kind == "clarifying_question"
            else {"premise": q["premise"], "correction": q["correction"]}
        )
        items.append(
            {
                "id": r["question_id"],
                "check": kind,
                "question": q["question"],
                "expected": expected,
                "given": r[kind],
                "verdict": None,
                "note": "",
            }
        )
    return {"run": run, "criteria": CRITERIA, "reviewer": None, "reviewed_on": None, "items": items}


def load(path: Path) -> dict[str, Any] | None:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None


def reviewed_success(records: Sequence[dict], review: dict[str, Any]) -> dict[str, list[int]]:
    """Per category (c, e): each answer's success once the hand check is applied. Refuses a
    review that misses an answer needing a check, or gives a verdict that is not one of
    `VERDICTS`, or checks something the answers do not contain."""
    verdicts = {i["id"]: i for i in review["items"]}
    needed = {r["question_id"]: check_kind(r) for r in records if check_kind(r)}
    missing = sorted(set(needed) - set(verdicts))
    extra = sorted(set(verdicts) - set(needed))
    bad = sorted(i for i, v in verdicts.items() if v.get("verdict") not in VERDICTS)
    wrong_kind = sorted(i for i in set(needed) & set(verdicts) if verdicts[i]["check"] != needed[i])
    if missing or extra or bad or wrong_kind:
        raise ValueError(
            f"the review does not match the answers: missing {missing}, not needed {extra}, "
            f"no valid verdict {bad}, other check {wrong_kind}"
        )
    out: dict[str, list[int]] = {"c": [], "e": []}
    for r in records:
        if r["category"] not in out:
            continue
        ok = own_success(r)
        if r["question_id"] in needed:
            ok = int(ok == 1 and verdicts[r["question_id"]]["verdict"] == "correct")
        out[r["category"]].append(ok)
    return out
