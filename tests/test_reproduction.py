"""The reproduction checker: what it lets through and what it calls a failure."""

from __future__ import annotations

import json

from src.tracking import reproduction as rp

RULES = [
    ("*", r"\.(updated_utc)$", "when it ran", None),
    ("m/t.json", r"\.score$", "stored to six decimals", 1e-5),
]


def write(root, rel, data, raw=False):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(data if raw else json.dumps(data), encoding="utf-8")


def test_identical_trees_pass(tmp_path):
    for side in ("a", "b"):
        write(tmp_path / side, "m/x.json", {"ex": 0.6})
    out = rp.check(tmp_path / "a", tmp_path / "b", RULES)
    assert out["passed"] and out["counts"] == {"identical": 1}


def test_a_field_that_records_the_run_may_differ(tmp_path):
    write(tmp_path / "a", "m/x.json", {"ex": 0.6, "updated_utc": "1"})
    write(tmp_path / "b", "m/x.json", {"ex": 0.6, "updated_utc": "2"})
    out = rp.check(tmp_path / "a", tmp_path / "b", RULES)
    assert out["passed"] and out["counts"] == {"exempt_only": 1}


def test_a_result_that_differs_is_a_failure(tmp_path):
    write(tmp_path / "a", "m/x.json", {"ex": 0.6, "updated_utc": "1"})
    write(tmp_path / "b", "m/x.json", {"ex": 0.61, "updated_utc": "2"})
    out = rp.check(tmp_path / "a", tmp_path / "b", RULES)
    assert not out["passed"] and ".ex" in out["failures"]["m/x.json"]["detail"][0]


def test_a_tolerance_covers_only_what_it_names(tmp_path):
    write(tmp_path / "a", "m/t.json", [{"score": 0.5}, {"score": 0.5}])
    write(tmp_path / "b", "m/t.json", [{"score": 0.500004}, {"score": 0.51}])
    out = rp.check(tmp_path / "a", tmp_path / "b", RULES)
    assert not out["passed"] and out["failures"]["m/t.json"]["n"] == 1  # only the 0.01 one


def test_a_missing_or_new_file_is_a_failure(tmp_path):
    write(tmp_path / "a", "m/gone.json", {})
    write(tmp_path / "b", "m/new.json", {})
    out = rp.check(tmp_path / "a", tmp_path / "b", RULES)
    assert out["counts"] == {"missing_after_rebuild": 1, "new_after_rebuild": 1}
    assert not out["passed"]


def test_other_files_are_compared_byte_for_byte(tmp_path):
    write(tmp_path / "a", "p/x.txt", "one", raw=True)
    write(tmp_path / "b", "p/x.txt", "two", raw=True)
    assert not rp.check(tmp_path / "a", tmp_path / "b", RULES)["passed"]


def test_a_number_and_a_string_are_different_values(tmp_path):
    write(tmp_path / "a", "m/x.json", {"n": 1})
    write(tmp_path / "b", "m/x.json", {"n": "1"})
    assert not rp.check(tmp_path / "a", tmp_path / "b", RULES)["passed"]


def test_the_committed_rules_name_only_run_records():
    """The shipped exemptions cover when and how long, never a measured result."""
    for _, pattern, reason, tol in rp.EXEMPT:
        assert reason and tol is None
        assert not any(w in pattern for w in ("ex", "aurc", "accuracy", "cost", "ece"))


def test_a_known_cause_is_reported_apart_and_only_inside_its_bounds(tmp_path):
    soft_f1 = "runs/ablation/ablation-d2-qwen2.5_3b-instruct-evidence.jsonl"
    rows = [{"soft_f1": 0.0}] * 51
    write(tmp_path / "a", soft_f1, rows, raw=False)
    rebuilt = [dict(r) for r in rows]
    rebuilt[50]["soft_f1"] = 0.4
    write(tmp_path / "b", soft_f1, rebuilt)
    # the file is a JSON array here, so write it as JSON lines the way the checker reads it
    for side, data in (("a", rows), ("b", rebuilt)):
        (tmp_path / side / soft_f1).write_text(
            "\n".join(json.dumps(r) for r in data), encoding="utf-8"
        )
    report = rp.check(tmp_path / "a", tmp_path / "b")
    assert report["passed"] and not report["reproduced_exactly"]
    seen = report["known_nondeterministic"][soft_f1]["seen"][0]
    assert (seen["committed"], seen["rebuilt"]) == (0.0, 0.4)
    rebuilt[3]["soft_f1"] = 0.4  # a record the cause does not name
    (tmp_path / "b" / soft_f1).write_text(
        "\n".join(json.dumps(r) for r in rebuilt), encoding="utf-8"
    )
    assert not rp.check(tmp_path / "a", tmp_path / "b")["passed"]


def test_time_inside_a_recorded_string_is_masked_and_nothing_else_in_it():
    rel, path = "metrics/security_suite.json", ".records[3].modes.parser.detail"
    assert rp.exemption(rel, path, "row limit: 100 rows in 2.80 s", "row limit: 100 rows in 2.57 s")
    assert not rp.exemption(
        rel, path, "row limit: 100 rows in 2.80 s", "row limit: 99 rows in 2.80 s"
    )


def test_every_known_cause_names_one_file_and_says_why():
    for glob, pattern, reason, _ in rp.NONDETERMINISTIC:
        assert "*" not in glob and pattern and len(reason) > 20
