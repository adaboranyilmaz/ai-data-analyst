"""The store of clock-reading query results (src/agent/clock.py), without a database."""

from __future__ import annotations

import json

import pytest

from src.agent.clock import ClockStore, key, seed_from_traces

AT = ("request-1", "toolu_1")
CLOCK_SQL = "SELECT AGE(NOW(), birthday) FROM player"


class Runs:
    """A query that returns a new result on every run, as a clock-reading one does."""

    def __init__(self, result=None):
        self.n = 0
        self.result = result

    def __call__(self):
        self.n += 1
        return self.result or {"ok": True, "rows": [[f"run {self.n}"]], "seconds": 0.01 * self.n}


def test_a_clock_query_is_stored_at_its_first_run(tmp_path):
    store, run = ClockStore(tmp_path), Runs()
    first = store.through("db", ["player"], CLOCK_SQL, AT, run)
    again = store.through("db", ["player"], CLOCK_SQL, AT, run)
    assert run.n == 1 and first["rows"] == again["rows"] == [["run 1"]]
    assert "seconds" not in again  # timings are never stored


def test_other_queries_and_calls_outside_a_conversation_always_run(tmp_path):
    store, run = ClockStore(tmp_path), Runs()
    store.through("db", ["player"], "SELECT COUNT(*) FROM player", AT, run)
    store.through("db", ["player"], "SELECT COUNT(*) FROM player", AT, run)
    store.through("db", ["player"], CLOCK_SQL, None, run)
    store.through("db", ["player"], CLOCK_SQL, None, run)
    assert run.n == 4


def test_each_call_keeps_its_own_result(tmp_path):
    """Two samples that ran the same query at different moments saw different results."""
    store, run = ClockStore(tmp_path), Runs()
    a = store.through("db", ["player"], CLOCK_SQL, ("request-1", "toolu_1"), run)
    b = store.through("db", ["player"], CLOCK_SQL, ("request-2", "toolu_2"), run)
    assert a["rows"] != b["rows"]
    again = store.through("db", ["player"], CLOCK_SQL, ("request-2", "toolu_2"), run)
    assert again["rows"] == b["rows"] and run.n == 2
    # a narrowed guard (other tables) is another key too
    assert key("db", ["player"], CLOCK_SQL, AT) != key("db", ["team"], CLOCK_SQL, AT)


@pytest.mark.parametrize(
    ("error", "stored"),
    [("sql_error", True), ("timeout", False), ("connection", False), ("refused", False)],
)
def test_only_what_came_from_the_database_is_stored(tmp_path, error, stored):
    store = ClockStore(tmp_path)
    run = Runs({"ok": False, "error": {"kind": error, "message": "m"}})
    store.through("db", ["player"], CLOCK_SQL, AT, run)
    store.through("db", ["player"], CLOCK_SQL, AT, run)
    assert run.n == (1 if stored else 2)


def test_a_stage_reads_the_results_of_the_stages_before_it(tmp_path):
    earlier = ClockStore(tmp_path / "ablation")
    earlier.through("db", ["player"], CLOCK_SQL, AT, Runs())
    later, run = ClockStore(tmp_path / "main", [tmp_path / "ablation"]), Runs()
    assert later.through("db", ["player"], CLOCK_SQL, AT, run)["rows"] == [["run 1"]]
    assert run.n == 0


def tool_use(id_, name, sql):
    return {"type": "tool_use", "id": id_, "name": name, "input": {"sql": sql}}


def test_results_are_taken_from_traces_with_the_calls_they_answered(tmp_path):
    """Two samples of one question each ran the same clock query; the trace keeps both results
    in order, and the cached responses give each call's id."""
    cache = {
        "k0a": {
            "request": {"tools": [{"name": "run_sql"}], "params": {}},
            "response": {"content": [tool_use("t0", "run_sql", "SELECT 1")]},
        },
        "k0b": {
            "request": {"tools": [{"name": "run_sql"}], "params": {}},
            "response": {"content": [tool_use("t1", "run_sql", CLOCK_SQL)]},
        },
        "k1a": {
            "request": {"tools": [{"name": "run_sql"}], "params": {"_sample": 1}},
            "response": {"content": [tool_use("t2", "run_sql", CLOCK_SQL)]},
        },
    }
    trace = {
        "question": {"db_id": "db"},
        "narrowing": None,
        "requests": [{"cache_key": k} for k in ("k0a", "k1a", "k0b")],  # samples interleave
        "samples": [
            {
                "tool_results": [
                    {"tool": "run_sql", "input": {"sql": "SELECT 1"}, "result": {"ok": True}},
                    {
                        "tool": "run_sql",
                        "input": {"sql": CLOCK_SQL},
                        "result": {"ok": True, "rows": [["sample 0"]], "seconds": 0.2},
                    },
                ]
            },
            {
                "tool_results": [
                    {
                        "tool": "run_sql",
                        "input": {"sql": CLOCK_SQL},
                        "result": {"ok": True, "rows": [["sample 1"]], "seconds": 0.3},
                    }
                ]
            },
        ],
    }
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace), encoding="utf-8")
    store = ClockStore(tmp_path / "cache")
    counts = seed_from_traces([path], store, lambda db: ["player"], cache.__getitem__)
    assert counts == {"stored": 2, "already": 0, "not_storable": 0}
    never = Runs()
    s0 = store.through("db", ["player"], CLOCK_SQL, ("k0b", "t1"), never)
    s1 = store.through("db", ["player"], CLOCK_SQL, ("k1a", "t2"), never)
    assert (s0["rows"], s1["rows"], never.n) == ([["sample 0"]], [["sample 1"]], 0)
    # seeding again changes nothing
    counts = seed_from_traces([path], store, lambda db: ["player"], cache.__getitem__)
    assert counts == {"stored": 0, "already": 2, "not_storable": 0}
