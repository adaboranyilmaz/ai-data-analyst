"""The agent registry and the evaluation gate: what they accept and what they refuse."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from src.tracking import gate, registry

ROOT = registry.ROOT
OPUS_RUN = "results/runs/escalation/ablation-d1-claude-opus-5-5-evidence.jsonl"


@pytest.fixture
def tree(tmp_path) -> Path:
    """A copy of what the registry reads: configs, prompts, the calibration, the registry itself
    and the larger model's calibration-split run."""
    for name in ("configs", "prompts", "results/registry"):
        shutil.copytree(ROOT / name, tmp_path / name)
    (tmp_path / "results/metrics").mkdir(parents=True)
    shutil.copy(ROOT / "results/metrics/calibration.json", tmp_path / "results/metrics")
    (tmp_path / OPUS_RUN).parent.mkdir(parents=True)
    shutil.copy(ROOT / OPUS_RUN, tmp_path / OPUS_RUN)
    return tmp_path


def test_the_committed_registry_has_no_problems():
    assert registry.problems() == []
    assert gate.promotion_problems() == []
    assert gate.coverage_problems() == []


def test_resolving_is_deterministic_and_matches_the_registered_version():
    state = registry.read_state()
    for name in registry.config_names():
        a, b = registry.resolve(name), registry.resolve(name)
        assert a == b == state["versions"][name]["config"]
        assert registry.sha256_of(a) == state["versions"][name]["config_sha256"]


def test_a_resolved_configuration_names_everything_that_decides_its_behavior():
    c = registry.resolve("d1-sonnet-5-opus-router")
    assert c["design"] == "d1" and c["model"] == "claude-sonnet-5" and c["samples"] == 1
    assert c["request_settings"]["params"] == {"thinking": {"type": "disabled"}}
    assert set(c["prompts"]) == {"single_shot", "classify_v1", "stat_plan_v1", "stat_answer_v1"}
    assert all(len(p["sha256"]) == 64 for p in c["prompts"].values())
    assert c["router"]["model"] == "claude-opus-5-5"
    assert c["router"]["request_settings"]["params"] == {"output_config": {"effort": "low"}}
    assert c["router"]["calibrator"]["questions"] == 150
    assert c["decline_threshold"] != c["calibration"]["decline_threshold"]  # the system's own
    assert registry.resolve("d1-sonnet-5")["router"] is None


def test_a_changed_prompt_is_a_new_version_and_the_registry_says_so(tree):
    path = tree / "prompts/single_shot_v3.md"
    path.write_text(path.read_text(encoding="utf-8") + "\nBe brief.\n", encoding="utf-8")
    found = registry.problems(tree)
    assert len(found) == 2 and all("changed since it was registered" in p for p in found)


def test_a_configuration_that_was_never_registered_is_refused(tree):
    cfg = (tree / "configs/agents/d1-sonnet-5.yaml").read_text(encoding="utf-8")
    (tree / "configs/agents/d1-haiku.yaml").write_text(
        cfg.replace("name: d1-sonnet-5", "name: d1-haiku"), encoding="utf-8"
    )
    assert any("d1-haiku: not registered" in p for p in registry.problems(tree))


def test_an_alias_cannot_move_without_a_promotion(tree):
    state = registry.read_state(tree)
    state["aliases"]["champion"] = "d1-sonnet-5"  # back to the first, with no log entry for it
    registry.write_state(state, tree)
    assert any("an alias moves only through a promotion" in p for p in registry.problems(tree))


def test_an_evaluation_of_another_version_does_not_count(tree):
    path = tree / registry.EVALUATIONS_DIR / "d1-sonnet-5.json"
    ev = json.loads(path.read_text(encoding="utf-8"))
    ev["config_sha256"] = "0" * 64
    path.write_text(json.dumps(ev), encoding="utf-8")
    assert any("another version" in p for p in registry.problems(tree))


def test_a_missing_evaluation_file_is_refused(tree):
    (tree / registry.EVALUATIONS_DIR / "d1-sonnet-5-opus-router.json").unlink()
    assert any("no evaluation file" in p for p in registry.problems(tree))


def test_a_promotion_under_another_rule_does_not_count(tree):
    (tree / "configs/promotion.yaml").write_text(
        (tree / "configs/promotion.yaml").read_text(encoding="utf-8") + "\n# looser\n",
        encoding="utf-8",
    )
    assert any("another version of the promotion rule" in p for p in gate.promotion_problems(tree))


def test_a_promotion_about_another_version_does_not_count(tree):
    log = tree / registry.PROMOTIONS
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    entry["challenger_sha256"] = "f" * 64
    log.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    assert any("another version of" in p for p in gate.promotion_problems(tree))


def test_a_logged_decision_that_the_results_do_not_support_is_found(tree, monkeypatch):
    from src.eval import promotion

    router_run = "results/runs/router/held_out-d1-claude-opus-5-5-evidence.jsonl"
    (tree / router_run).parent.mkdir(parents=True)
    shutil.copy(ROOT / router_run, tree / router_run)
    monkeypatch.setattr(promotion, "decide", lambda *a: {"checks": {}, "promote": False})
    found = gate.promotion_problems(tree)
    assert any("not the logged one" in p for p in found)


# --------------------------------------------------------------------------- the gate


@pytest.fixture(scope="module")
def entries():
    return json.loads((ROOT / gate.BUNDLE).read_text(encoding="utf-8"))["entries"]


def test_the_bundle_covers_both_models_and_replays_unchanged(entries):
    by_model = {e["model"] for e in entries}
    assert by_model == {"claude-sonnet-5", "claude-opus-5-5"} and len(entries) == 26
    assert all(gate.replay_entry(e)["request_key_ok"] for e in entries[:6])
    report = gate.replay()
    assert report["ok"] and report["unchanged"] == report["entries"] == 26


def test_a_changed_request_is_found_before_the_answer_is_compared(entries):
    e = copy.deepcopy(entries[0])
    e["request_key"] = "0" * 64
    out = gate.replay_entry(e)
    assert out["request_key_ok"] is False and out["differences"] == []


def test_a_changed_reading_of_a_response_is_found(entries):
    e = copy.deepcopy(next(x for x in entries if x["model"] == "claude-sonnet-5"))
    call = next(c for c in e["response"]["content"] if c["type"] == "tool_use")
    call["input"]["confidence"] = 0.123
    call["input"]["sql"] = "SELECT 1"
    out = gate.replay_entry(e)
    assert out["request_key_ok"] and set(out["differences"]) == {"final_sql", "confidence"}


def test_a_model_the_bundle_does_not_cover_is_refused(tmp_path, monkeypatch, entries):
    thin = {"entries": [e for e in entries if e["model"] == "claude-sonnet-5"]}
    (tmp_path / "results/registry").mkdir(parents=True)
    shutil.copy(ROOT / registry.STATE, tmp_path / registry.STATE)
    (tmp_path / "results/regression").mkdir()
    (tmp_path / gate.BUNDLE).write_text(json.dumps(thin), encoding="utf-8")
    found = gate.coverage_problems(tmp_path)
    assert found and all("claude-opus-5-5" in p for p in found)
