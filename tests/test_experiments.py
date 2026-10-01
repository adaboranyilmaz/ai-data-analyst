"""The MLflow experiments are a faithful, repeatable view of the committed results files."""

from __future__ import annotations

import json
import os

import pytest

from src.tracking import experiments

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


@pytest.fixture(scope="module")
def specs():
    return experiments.collect()


def test_every_evaluated_run_is_collected(specs):
    by_exp = {}
    for s in specs:
        by_exp[s.experiment] = by_exp.get(s.experiment, 0) + 1
    assert by_exp == {
        "analyst-ablation": 10,
        "analyst-benchmark": 5,
        "analyst-own-set": 1,
        "analyst-escalation": 2,
        "analyst-router": 3,
        "analyst-calibration": 5,
        "analyst-framework": 1,
    }
    assert len({s.key for s in specs}) == len(specs)


def test_numbers_come_from_the_results_files(specs):
    ab = json.loads((experiments.ROOT / "results/metrics/ablation.json").read_text("utf-8"))
    run = ab["runs"]["claude-sonnet-5/d1"]
    s = next(s for s in specs if s.key == "analyst-ablation/claude-sonnet-5/d1")
    assert s.metrics["ex"] == run["execution_accuracy"]["estimate"]
    assert s.metrics["aurc"] == run["selective"]["aurc"]["estimate"]
    assert s.metrics["ece"] == run["calibration"]["ece"]["estimate"]
    assert s.metrics["cost_total_usd"] == run["cost"]["total_usd"]
    assert s.params["design"] == "d1" and s.params["k"] == "1" and s.params["evidence"] == "true"
    d4 = next(s for s in specs if s.key == "analyst-ablation/claude-sonnet-5/d4")
    assert d4.params["k"] == "3"


def test_logging_is_repeatable_and_read_back_matches(specs, tmp_path):
    uri = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    experiments.log(specs, uri)
    experiments.log(specs, uri)  # a second pass replaces the first, never adds to it
    stored = experiments.read_back(uri)
    assert set(stored) == {s.key for s in specs}
    for s in specs:
        got = stored[s.key]
        assert got["params"] == s.params and got["source"] == s.source
        assert got["metrics"] == pytest.approx(s.metrics, abs=1e-12)


def test_manifest_counts(specs):
    m = experiments.manifest(specs)
    assert m["runs_logged"] == len(specs) == sum(m["by_experiment"].values())
