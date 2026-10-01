"""The product's evidence layer: the confidence meter, the curated draw, the replay stream and the
evidence records served from results/demo. None of it needs a database or a model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.serving import curated, evidence, replay
from src.serving.meter import Meter, config
from src.serving.store import RunStore, valid_id

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "results/demo"


@pytest.fixture(scope="module")
def meter() -> Meter:
    return Meter.load()


# --------------------------------------------------------------------------- the meter


class TestMeter:
    def test_threshold_is_the_stated_confidence_the_calibration_chose(self, meter):
        # the calibration file records the stated confidence at the threshold as 0.92
        assert meter.calibrate(0.92) == pytest.approx(meter.threshold, abs=1e-3)
        assert not meter.describe(0.95, False)["withheld"]
        assert meter.describe(0.90, False)["withheld"]

    def test_calibrated_is_monotone_and_in_unit_interval(self, meter):
        values = [meter.calibrate(c / 20) for c in range(21)]
        assert values == sorted(values) and 0 < values[0] < values[-1] < 1

    def test_band_is_the_held_out_row_with_its_interval(self, meter):
        b = meter.describe(0.95, False)["band"]
        assert b["range"] == [0.8, 0.9] and b["n"] == 20
        low, high = b["interval"]
        assert low < b["accuracy"] < high and not b["thin"]

    def test_a_calibrated_value_above_every_held_out_band_has_no_record(self, meter):
        # the calibrator never reaches 0.9 (a stated 1.0 maps to about 0.87), and the held-out
        # table stops at the band below it, so a higher value has no record to quote
        assert meter.describe(1.0, False)["calibrated"] < 0.9
        assert meter.band(0.95) is None and meter.band(0.85) is not None

    def test_a_declined_answer_carries_no_calibrated_confidence(self, meter):
        d = meter.describe(0.95, True)
        assert d["calibrated"] is None and d["band"] is None and not d["withheld"]

    def test_thin_bands_say_so(self, meter):
        thin = [b for b in meter.bands if b.n < meter.min_n]
        assert thin and all(b.to_dict(meter.min_n)["thin"] for b in thin)

    def test_the_summary_states_the_held_out_numbers_it_rests_on(self, meter):
        s = meter.summary()
        assert s["held_out"]["questions"] == 320 and s["held_out"]["answered"] == 23
        assert sum(b["n"] for b in s["bands"]) == 320

    def test_the_summary_carries_what_the_page_needs_to_explain_the_calibration(self, meter):
        c = meter.summary()["calibration"]
        assert c["method"] == "platt" and c["fitted_on"] == 150 and c["target_accuracy"] == 0.9
        # the numbers reproduce the page's worked example: stated 0.90 -> the calibrated value
        z = c["slope"] * 0.9 + c["intercept"]
        assert 1 / (1 + 2.718281828459045**-z) == pytest.approx(meter.calibrate(0.9))


# --------------------------------------------------------------------------- the curated draw


def rec(i, correct=1, declined=False, confidence=0.9, category="a"):
    return {
        "question_id": i,
        "correct": correct,
        "declined": declined,
        "confidence": confidence,
        "category": category,
        "score_outcome": "ok",
        "clarifying_question": None,
        "assumptions": [],
        "premise_correction": None,
        "decline_reason": None,
    }


class TestCuratedDraw:
    def test_the_same_seed_gives_the_same_runs_whatever_the_order(self):
        pool = [rec(i) for i in range(50)]
        a = curated.draw(7, "slot", pool, 5, lambda r: str(r["question_id"]))
        b = curated.draw(7, "slot", pool[::-1], 5, lambda r: str(r["question_id"]))
        assert [r["question_id"] for r in a] == [r["question_id"] for r in b]
        c = curated.draw(8, "slot", pool, 5, lambda r: str(r["question_id"]))
        assert [r["question_id"] for r in a] != [r["question_id"] for r in c]

    def test_a_short_slot_takes_what_exists_and_reports_the_shortfall(self):
        records = [
            rec(1, 0, confidence=0.99),
            rec(2, 1, confidence=0.99),
            rec(3, 1, confidence=0.1),
        ]
        picks, short = curated.held_out_slots(
            records,
            {1, 2, 3},
            lambda c: c,
            0.5,
            {"wrong_above_threshold": 2, "correct_below_threshold": 1},
            1,
        )
        assert {p.key for p in picks} == {"1", "3"}
        assert short == {"wrong_above_threshold": 1}

    def test_only_held_out_questions_and_never_a_declined_one(self):
        records = [rec(1), rec(2), rec(3, declined=True)]
        picks, _ = curated.held_out_slots(
            records, {1, 3}, lambda c: c, 0.5, {"correct_above_threshold": 5}, 1
        )
        assert [p.key for p in picks] == ["1"]

    def test_banking_takes_one_per_category_then_a_wrong_one(self):
        records = [rec(f"a{i}", category="a") for i in range(3)]
        records += [rec("c0", category="c") | {"clarifying_question": "which?"}]
        records += [rec(f"w{i}", correct=0, category="a") for i in range(4)]
        picks, short = curated.banking_slots(records, ["a", "c", "d"], 1, 3)
        slots = {p.slot for p in picks}
        assert {"category_a", "category_c", "wrong_extra"} <= slots
        assert short == {"category_d": 1}
        assert sum(p.slot == "category_a" for p in picks) == 1

    def test_the_guardrail_draw_follows_the_authors_reading(self):
        records = [{"id": f"own:own-f0{i}", "guarded": {"delivered": "x"}} for i in range(1, 5)]
        review = [
            {"id": "own-f01", "reviewed_success": True},
            {"id": "own-f02", "reviewed_success": False},
            {"id": "own-f03", "reviewed_success": False},
            {"id": "own-f04", "reviewed_success": True},
        ]
        picks, short = curated.guardrail_slots(records, review, 1, 2, 5)
        by = {p.slot: [] for p in picks}
        for p in picks:
            by[p.slot].append(p.key)
        assert len(by["reviewed_success"]) == 1 and sorted(by["reviewed_failure"]) == [
            "own-f02",
            "own-f03",
        ]
        assert short == {}


# --------------------------------------------------------------------------- the demo records


def demo_runs():
    return json.loads((DEMO / "index.json").read_text(encoding="utf-8"))["runs"]


class TestDemoSet:
    def test_the_set_has_the_slots_the_rule_asks_for(self):
        index = json.loads((DEMO / "index.json").read_text(encoding="utf-8"))
        cfg = config()["curated"]
        wanted = sum(cfg["held_out"]["slots"].values()) + len(cfg["banking"]["one_per_category"])
        wanted += cfg["banking"]["wrong_extra"] + cfg["guardrail"]["reviewed_success"]
        wanted += cfg["guardrail"]["reviewed_failure"]
        assert index["count"] == wanted == len(index["runs"]) and index["short"] == {}

    def test_wrong_and_withheld_answers_are_in_the_set(self):
        store = RunStore(DEMO)
        evs = [store.get(r["id"]) for r in demo_runs()]
        assert any(e["evaluation"].get("correct") is False for e in evs)
        assert any(e["status"] == "withheld" for e in evs)
        assert any(e["status"] == "declined" for e in evs)

    def test_no_expert_query_or_gold_field_is_in_any_record(self):
        def keys(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    yield k
                    yield from keys(v)
            elif isinstance(node, list):
                for v in node:
                    yield from keys(v)

        for r in demo_runs():
            ev = json.loads((DEMO / "runs" / f"{r['id']}.json").read_text(encoding="utf-8"))
            assert not [k for k in keys(ev) if "gold" in k.lower() or "expert" in k.lower()]

    def test_every_record_streams_a_complete_run(self):
        store = RunStore(DEMO)
        for r in demo_runs():
            ev = store.get(r["id"])
            stream = [e for e, _ in replay.events(ev, config()["replay"])]
            types = [e["type"] for e in stream]
            assert types[0] == "start" and types[-1] == "done"
            assert types.count("answer") == 1 and types.count("confidence") == 1
            assert replay.assemble(stream)["answer"]["text"] == ev["answer"]["text"]


# --------------------------------------------------------------------------- the stream


class TestReplayStream:
    def test_recorded_durations_are_kept_and_capped(self):
        cfg = {"default_ms": 700, "cap_ms": 4000, "speed": 1.0}
        assert replay.step_delay_ms({"recorded_ms": 1200}, cfg) == 1200
        assert replay.step_delay_ms({"recorded_ms": 90_000}, cfg) == 4000
        assert replay.step_delay_ms({"recorded_ms": None}, cfg) == 700

    def test_speed_scales_every_delay(self):
        ev = RunStore(DEMO).get(demo_runs()[0]["id"])
        slow = sum(d for _, d in replay.events(ev, {"default_ms": 700, "cap_ms": 4000}))
        fast = sum(
            d for _, d in replay.events(ev, {"default_ms": 700, "cap_ms": 4000, "speed": 10})
        )
        assert fast == pytest.approx(slow / 10)

    def test_the_evidence_events_follow_the_steps(self):
        ev = RunStore(DEMO).get("bench-782")
        types = [e["type"] for e, _ in replay.events(ev, config()["replay"])]
        first_final = types.index("sql")
        assert set(types[1:first_final]) == {"step"}
        assert types[first_final:] == [
            "sql",
            "rows",
            "checks",
            "chart",
            "answer",
            "confidence",
            "done",
        ] or (types[first_final:][-3:] == ["answer", "confidence", "done"])


class TestStore:
    def test_ids_are_checked_before_they_touch_the_file_system(self):
        for bad in ("../x", "a/b", "a\\b", "", "..", ".hidden", "x" * 80, "a b"):
            assert not valid_id(bad)
        assert valid_id("bench-782") and valid_id("live-0123456789ab")

    def test_an_unknown_or_malformed_id_is_none(self):
        s = RunStore(DEMO)
        assert s.get("nope") is None and s.get("../index") is None

    def test_live_runs_round_trip_and_bad_ids_are_refused(self, tmp_path):
        s = RunStore(DEMO, tmp_path)
        ev = {"id": s.new_live_id(), "question": "q"}
        s.save_live(ev)
        assert s.get(ev["id"]) == ev
        with pytest.raises(ValueError):
            s.save_live({"id": "../escape"})


class TestEvidence:
    def test_check_lines_state_each_check(self):
        lines = evidence.check_lines(
            {
                "applicable": True,
                "empty": True,
                "repeated_rows": False,
                "out_of_range": [],
                "too_many_rows": False,
                "failed": True,
            }
        )
        assert [(x["name"], x["ok"]) for x in lines] == [
            ("not_empty", False),
            ("no_repeated_rows", True),
            ("values_in_range", True),
            ("result_size", True),
        ]

    def test_status_priority(self):
        none = {"withheld": False}
        assert evidence.status_of({"declined": True}, {"withheld": True}) == "declined"
        assert (
            evidence.status_of({"clarifying_question": "which?"}, {"withheld": True}) == "clarify"
        )
        assert evidence.status_of({}, {"withheld": True}) == "withheld"
        assert evidence.status_of({}, none) == "answered"


# --------------------------------------------------------------------------- the serving check


class TestServingCheck:
    def evaluated(self, run_id="bench-782"):
        key = run_id.split("-", 1)[1]
        path = ROOT / "results/runs/main/all-d1-claude-sonnet-5-evidence.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if str(r["question_id"]) == key:
                return r
        raise AssertionError(run_id)

    def test_a_served_record_that_matches_its_evaluation_has_no_differences(self, meter):
        from src.serving import check

        ev = RunStore(DEMO).get("bench-782")
        assert check.against_record(ev, self.evaluated(), meter) == {}

    def test_a_changed_answer_sql_confidence_or_verdict_is_named(self, meter):
        from src.serving import check

        ev = RunStore(DEMO).get("bench-782")
        changed = json.loads(json.dumps(ev))
        changed["answer"]["text"] += " (edited)"
        changed["answer"]["sql"] = "SELECT 1"
        changed["confidence"]["stated"] = 0.5
        changed["evaluation"]["correct"] = not ev["evaluation"]["correct"]
        diffs = check.against_record(changed, self.evaluated(), meter)
        assert {"answer_text", "sql", "stated_confidence", "verdict"} <= set(diffs)

    def test_a_stream_that_drops_or_alters_an_event_is_named(self):
        from src.serving import check

        ev = RunStore(DEMO).get("bench-782")
        stream = [e for e, _ in replay.events(ev, config()["replay"])]
        assert check.stream_against_record(stream, ev) == {}
        no_sql = [e for e in stream if e["type"] != "sql"]
        assert "sql" in check.stream_against_record(no_sql, ev)
        altered = [dict(e, text="other") if e["type"] == "answer" else e for e in stream]
        assert "answer_text" in check.stream_against_record(altered, ev)

    def test_a_meter_that_drifted_from_the_recorded_calibration_is_caught(self, meter):
        from src.serving import check

        ev = RunStore(DEMO).get("bench-782")
        drifted = Meter(
            json.loads((ROOT / "results/metrics/calibration.json").read_text(encoding="utf-8"))
        )
        drifted.slope += 1.0
        assert "calibrated" in check.against_record(ev, self.evaluated(), drifted)


class TestWilson:
    def test_the_services_interval_is_the_evaluations(self):
        from src.serving.meter import wilson
        from src.stats.report import wilson as evaluated

        assert all(wilson(k, n) == evaluated(k, n) for n in range(0, 80) for k in range(n + 1))
