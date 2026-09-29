"""Pieces shared by the stage reports (scripts/43_ablation_report.py, 45_main_report.py,
46_own_set_report.py): which records belong to which question set, a batched run's latency,
and a row of the cost-per-correct-answer table."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.eval.config import config

ROOT = Path(__file__).resolve().parent.parent.parent

# A batched model call has no latency of its own (it waits in the batch queue), so a batched
# run's latency would count only its tool calls: it is reported as not measured.
NOT_MEASURED = {
    "p50": None,
    "p95": None,
    "not_measured": "model calls were batched; a batched call has no latency",
}


def split_ids(name: str) -> set:
    """The question ids of one benchmark split (pilot, ablation, held_out)."""
    splits = json.loads((ROOT / config()["splits_file"]).read_text(encoding="utf-8"))
    return set(splits["sets"][name])


def subset(records: Sequence[dict], ids: set) -> list[dict]:
    return [r for r in records if r["question_id"] in ids]


def cost_row(summary: dict[str, Any], **labels: Any) -> dict[str, Any]:
    """One row of the cost-per-correct-answer table, from a run's summary."""
    cost = summary["cost"]
    return {
        **labels,
        "questions": summary["questions"],
        "execution_accuracy": summary["execution_accuracy"]["estimate"],
        "cost_per_question_usd": cost["per_question_usd"],
        "cost_per_correct_answer_usd": cost.get("per_correct_answer_usd", {}).get("estimate"),
        "total_usd": cost["total_usd"],
    }


def with_predictions(observed: dict[str, dict], predictions: dict[str, dict]) -> dict:
    """Each checked prediction's claim and prediction beside what was observed."""
    return {
        k: {"claim": predictions[k]["claim"], "prediction": predictions[k]["prediction"], **v}
        for k, v in observed.items()
    }
