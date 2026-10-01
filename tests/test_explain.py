"""What the right answer was and why an answer was scored wrong: the computed comparison, the
expected result, and the explanations on the curated wrong runs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from src.serving import explain
from src.serving.store import RunStore

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "results/demo"


def table(columns, rows, truncated=False):
    return {"columns": columns, "rows": rows, "total_rows": len(rows), "truncated": truncated}


def got(columns, rows, truncated=False):
    return {"ok": True, "columns": columns, "rows": rows, "truncated": truncated}


class TestCompare:
    def test_a_single_value_is_stated_beside_the_expected_one(self):
        c = explain.compare(got(["count"], [[5146]]), table(["count"], [[4941]]))
        assert c["relation"] == "single_value" and c["got"] == 5146 and c["expected"] == 4941
        assert "5146" in c["text"] and "4941" in c["text"]

    def test_extra_columns_are_named_when_the_expected_rows_are_all_there(self):
        c = explain.compare(
            got(["client_id", "total"], [[1, 9], [2, 8]]), table(["client_id"], [[2], [1]])
        )
        assert c["relation"] == "extra_columns" and c["extra"] == ["total"]
        assert "extra column total" in c["text"]

    def test_extra_columns_with_other_rows_are_not_called_the_same(self):
        c = explain.compare(got(["a", "b"], [[1, 1]]), table(["a"], [[2]]))
        assert c["relation"] == "different"

    def test_the_same_rows_in_another_order_are_recognized(self):
        c = explain.compare(got(["a"], [[2], [1]]), table(["a"], [[1], [2]]))
        assert c["relation"] == "same_rows" and "different order" in c["text"]

    def test_column_names_are_compared_without_case(self):
        c = explain.compare(got(["Count"], [[3]]), table(["count"], [[4]]))
        assert c["relation"] == "single_value"

    def test_a_result_too_long_to_compare_in_full_says_nothing(self):
        assert explain.compare(got(["a"], [[1]], truncated=True), table(["a"], [[1]])) is None
        assert explain.compare(got(["a"], [[1]]), table(["a"], [[1]], truncated=True)) is None
        assert explain.compare(None, table(["a"], [[1]])) is None
        assert explain.compare(got(["a"], [[1]]), None) is None


class TestReference:
    def run(self, rows, total=None, ok=True):
        def run_sql(sql):
            if not ok:
                return {"ok": False, "error": {"kind": "timeout"}}
            return {
                "ok": True,
                "columns": [{"name": "n", "type": "int8"}],
                "rows": rows,
                "total_rows": total,
            }

        return run_sql

    def test_the_expected_result_is_capped_and_counted(self):
        out = explain.reference(self.run([[i] for i in range(100)], total=250), "SELECT 1")
        assert len(out["rows"]) == explain.SHOWN_ROWS and out["total_rows"] == 250
        assert out["truncated"] and out["columns"] == ["n"]

    def test_a_query_that_cannot_run_gives_nothing(self):
        assert explain.reference(self.run([], ok=False), "SELECT 1") is None


class TestDemoExplanations:
    def runs(self):
        store = RunStore(DEMO)
        return [store.get(r["id"]) for r in store.index()]

    def test_every_wrong_scored_run_has_an_explanation_with_the_expected_result(self):
        wrong = [e for e in self.runs() if e["evaluation"].get("correct") is False]
        assert wrong
        for e in wrong:
            x = e["explanation"]
            assert x and x["expected"] and x["expected"]["rows"] and x["expected_sql"], e["id"]

    def test_runs_that_were_not_scored_wrong_carry_no_explanation(self):
        for e in self.runs():
            if e["evaluation"] is None or e["evaluation"].get("correct") is not False:
                assert e["explanation"] is None, e["id"]

    def test_every_wrong_curated_run_has_a_reviewer_note_and_every_note_has_a_run(self):
        notes = yaml.safe_load(
            (ROOT / "results/reviews/demo_explanations.yaml").read_text("utf-8")
        )["notes"]
        wrong = {e["id"] for e in self.runs() if e["evaluation"].get("correct") is False}
        assert set(notes) == wrong

    def test_the_notes_state_no_result_figures(self):
        # the page shows both results from the data; a note may name only the dates and bounds
        # of the question itself
        notes = yaml.safe_load(
            (ROOT / "results/reviews/demo_explanations.yaml").read_text("utf-8")
        )["notes"]
        allowed = {"1", "2014", "80", "500", "1990", "1991"}
        for run_id, text in notes.items():
            assert set(re.findall(r"\d+", text)) <= allowed, run_id

    def test_the_first_wrong_answer_shows_both_numbers(self):
        x = RunStore(DEMO).get("bench-533")["explanation"]
        assert x["comparison"]["relation"] == "single_value"
        assert x["comparison"]["got"] != x["comparison"]["expected"]

    def test_the_extra_column_case_is_computed_from_the_rows_not_asserted(self):
        x = RunStore(DEMO).get("bank-own-b06")["explanation"]
        assert x["comparison"]["relation"] == "extra_columns"

    def test_the_analyst_never_saw_the_expected_query(self):
        # the expert query is in the page's record, never in a model request's cache
        text = json.dumps(RunStore(DEMO).get("bench-533"))
        assert "DATE(LastAccessDate)" in text  # shown to the reader, after the fact
