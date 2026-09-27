"""The hand-written banking set: its format, and its freeze."""

from __future__ import annotations

import pytest

from src.eval.config import ROOT, config
from src.eval.own_set import load, problems, queries
from src.eval.preregistration import file_sha256


def q(qid: str, **fields) -> dict:
    return {"id": qid, "category": qid[4], "question": "How many?", "review": "approved", **fields}


VALID = [
    q("own-a01", gold_sql="SELECT 1"),
    q("own-b01", gold_sql="SELECT 1"),
    q("own-c01", interpretations=[{"assumption": "x", "gold_sql": "SELECT 1"},
                                  {"assumption": "y", "gold_sql": "SELECT 2"}]),
    q("own-d01", missing="satisfaction scores"),
    q("own-e01", premise="p", correction="c", premise_sql="SELECT 0"),
    q("own-f01", gold_sql="SELECT 1", check="an interval; no causal claim"),
]  # fmt: skip


def padded(questions: list[dict]) -> dict:
    """Enough extra (a) questions to reach the minimum size."""
    extra = [q(f"own-a{i:02d}", gold_sql="SELECT 1") for i in range(2, 2 + 40 - len(questions))]
    return {"questions": questions + extra}


def test_one_valid_question_per_category():
    assert problems(padded(VALID)) == []


@pytest.mark.parametrize(
    "bad",
    [
        q("own-a50"),  # no gold SQL
        q("own-c50", interpretations=[{"assumption": "x", "gold_sql": "SELECT 1"}]),  # one only
        q("own-d50", missing="x", gold_sql="SELECT 1"),  # a field its category does not have
        q("own-e50", premise="p", correction="c"),  # no premise SQL
        q("own-a51", gold_sql="SELECT 1", review="looked at"),
        {**q("own-a52", gold_sql="SELECT 1"), "category": "b"},  # id letter differs
        q("own-x01", gold_sql="SELECT 1"),
    ],
)
def test_malformed_questions_are_found(bad):
    assert problems(padded([*VALID, bad]))


def test_duplicates_and_size_are_checked():
    assert any("appears 2 times" in p for p in problems(padded([*VALID, VALID[0]])))
    assert any("not 40-60" in p for p in problems({"questions": VALID}))


def test_every_sql_of_a_question_is_listed():
    assert [label for label, _ in queries(VALID[2])] == ["reading-1", "reading-2"]
    assert queries(VALID[3]) == []
    assert queries(VALID[4]) == [("premise", "SELECT 0")]


def test_the_committed_set_is_well_formed_and_frozen_correctly():
    cfg = config()["own_set"]
    path = ROOT / cfg["path"]
    if not path.exists():
        pytest.skip("the banking set is not written yet")
    data = load(path)
    assert problems(data) == []
    if cfg["sha256"] is not None:  # frozen: reviewed throughout, and unchanged
        assert all(x["review"] != "pending" for x in data["questions"])
        assert file_sha256(path) == cfg["sha256"]
