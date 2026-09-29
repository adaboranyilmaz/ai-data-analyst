"""The replay check's comparison of committed and rebuilt records."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("replay", ROOT / "scripts/44_replay_check.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def record(qid, **kw):
    base = {
        "question_id": qid,
        "final_sql": "SELECT 1",
        "correct": 1,
        "cost_usd": 0.01,
        "latency_s": 1.0,
        "evaluated_at": "2026-09-28T10:00:00+00:00",
        "trace": f"data/traces/x/{qid}.json",
    }
    return {**base, **kw}


def test_timings_and_trace_paths_are_ignored():
    before = [record(1), record(2)]
    after = [
        record(1, latency_s=9.9, evaluated_at="2026-09-29T00:00:00+00:00"),
        record(2, trace="/tmp/elsewhere/2.json"),
    ]
    out = replay.compare(before, after)
    assert out["ok"] and out["identical"] == 2 and out["differing"] == 0


def test_a_changed_answer_or_cost_is_reported():
    before = [record(1), record(2), record(3)]
    after = [record(1, final_sql="SELECT 2", correct=0), record(2, cost_usd=0.02), record(3)]
    out = replay.compare(before, after)
    assert not out["ok"] and out["differing"] == 2 and out["identical"] == 1
    assert out["examples"] == [
        {"question_id": 1, "fields": ["correct", "final_sql"]},
        {"question_id": 2, "fields": ["cost_usd"]},
    ]


def test_missing_and_extra_questions_fail():
    out = replay.compare([record(1), record(2)], [record(1), record(3)])
    assert not out["ok"]
    assert out["missing_in_replay"] == [2] and out["extra_in_replay"] == [3]
