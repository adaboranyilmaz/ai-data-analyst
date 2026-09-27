"""The per-question record: what it must carry, and what it may not."""

from __future__ import annotations

import pytest

from src.eval.records import (
    answered_correct,
    problems,
    read_records,
    validate,
    write_records,
)


def record(**changes) -> dict:
    r = {
        "run": "design-1/sonnet/evidence",
        "question_id": 1471,
        "source": "bird",
        "db_id": "debit_card_specializing",
        "difficulty": "simple",
        "category": None,
        "design": "design-1",
        "model": "claude-sonnet-5",
        "evidence": True,
        "final_sql": "SELECT 1",
        "answer": "one",
        "confidence": 0.8,
        "declined": False,
        "decline_reason": None,
        "clarifying_question": None,
        "assumptions": [],
        "premise_correction": None,
        "correct": 1,
        "soft_f1": 1.0,
        "score_outcome": "ok",
        "tokens": {"input": 10, "output": 5, "cache_write_5m": 0, "cache_write_1h": 0,
                   "cache_read": 0},
        "cost_usd": 0.001,
        "latency_s": 2.5,
        "steps": 3,
        "tool_calls": 2,
        "errors": [],
        "trace": "results/traces/x.json",
        "evaluated_at": "2026-09-28T10:00:00+00:00",
    }  # fmt: skip
    r.update(changes)
    return r


def test_a_complete_record_is_valid():
    assert problems(record()) == []


def test_missing_and_unknown_fields_are_refused():
    r = record()
    del r["confidence"]
    assert any("confidence" in p for p in problems(r))
    assert problems(record(extra=1))


@pytest.mark.parametrize(
    "changes",
    [
        {"confidence": 1.2},
        {"correct": 2},
        {"score_outcome": "fine"},
        {"steps": -1},
        {"category": "a"},  # a benchmark question has no category
        {"difficulty": None},
        {"evaluated_at": "yesterday"},
        {"evaluated_at": "2026-09-28T10:00:00"},  # no offset
        {"errors": [{"kind": "timeout"}]},
    ],
)
def test_bad_values_are_refused(changes):
    assert problems(record(**changes))


def test_own_set_records_carry_a_category_not_a_difficulty():
    own = record(source="own", question_id="own-d01", db_id="financial", difficulty=None,
                 category="d", correct=None, soft_f1=None, score_outcome="not_scored")  # fmt: skip
    assert problems(own) == []
    assert problems({**own, "category": None})


def test_a_declined_question_is_never_a_correct_answer():
    assert answered_correct(record()) == 1
    assert answered_correct(record(declined=True, decline_reason="unsure")) == 0
    assert answered_correct(record(correct=0)) == 0
    assert answered_correct(record(correct=None)) == 0


def test_round_trip_and_validation_on_write(tmp_path):
    path = tmp_path / "run.jsonl"
    write_records(path, [record(), record(question_id=2)])
    assert [r["question_id"] for r in read_records(path)] == [1471, 2]
    with pytest.raises(ValueError):
        write_records(path, [record(confidence=-1)])
    with pytest.raises(ValueError):
        validate(record(source="web"))
