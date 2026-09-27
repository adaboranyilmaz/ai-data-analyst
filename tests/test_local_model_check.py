"""The local-model check (scripts/00_local_model_check.py): what counts as a valid tool call,
and the selection rule applied to synthetic measurements."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("check", ROOT / "scripts/00_local_model_check.py")
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)
CFG = yaml.safe_load((ROOT / "configs/local_models.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "call, valid",
    [
        ({"name": "list_tables", "input": {}}, True),
        ({"name": "run_sql", "input": {"sql": "SELECT count(*) FROM customers"}}, True),
        ({"name": "describe_table", "input": {"table": "orders"}}, True),
        ({"name": "run_query", "input": {"sql": "SELECT 1"}}, False),  # unknown tool
        ({"name": "run_sql", "input": {}}, False),  # required argument missing
        ({"name": "run_sql", "input": {"sql": "  "}}, False),  # empty
        ({"name": "run_sql", "input": {"sql": 1}}, False),  # not a string
        ({"name": "run_sql", "input": {"sql": "SELECT 1", "limit": "5"}}, False),  # unknown key
        ({"name": "run_sql", "input": "SELECT 1"}, False),  # arguments not an object
    ],
)
def test_valid_call(call, valid):
    assert check.valid_call(call) is valid


def measured(rate: float, max_ctx: int, gpu_mib: int) -> dict:
    return {
        "status": "measured",
        "capabilities": ["completion", "tools"],
        "valid_call_rate": rate,
        "max_context_on_gpu": max_ctx,
        "gpu_fit": [{"num_ctx": CFG["min_context"], "gpu_mib": gpu_mib, "on_gpu": True}],
    }


def test_highest_valid_call_rate_wins():
    s = check.select(CFG, {"a": measured(0.8, 8192, 2000), "b": measured(1.0, 8192, 3000)})
    assert (s["selected"], s["provisional"]) == ("b", False)


def test_ties_go_to_larger_context_then_less_memory():
    s = check.select(CFG, {"a": measured(1.0, 8192, 2000), "b": measured(1.0, 16384, 3000)})
    assert s["selected"] == "b"
    s = check.select(CFG, {"a": measured(1.0, 8192, 2000), "b": measured(1.0, 8192, 3000)})
    assert s["selected"] == "a"


def test_a_model_that_spills_off_the_gpu_at_min_context_is_not_eligible():
    s = check.select(CFG, {"a": measured(1.0, 4096, 2000), "b": measured(0.2, 8192, 3000)})
    assert s["selected"] == "b" and s["eligible"] == ["b"]


def test_undownloaded_candidates_make_the_choice_provisional():
    s = check.select(CFG, {"a": measured(1.0, 8192, 2000), "b": {"status": "not_downloaded"}})
    assert (s["selected"], s["provisional"], s["not_downloaded"]) == ("a", True, ["b"])
