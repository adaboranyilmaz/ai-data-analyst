"""The confidence arms' configuration and the critic's record files, without the agent's runtime.

configs/confidence.yaml configures the critic, the escalation model and the decline threshold;
the reports that read their results need only the configuration, the path of the answers
reviewed and the critic's record format, so these live here, importing nothing of the agent's
loop, tools or model backends (and the pipeline stages of those reports depend on this module
alone).

A critic record is one line per reviewed answer (src/agent/critic.py writes them): which answer
(`question_id`, `answer_run`), whether it was reviewed (`reviewed`, and `not_reviewed` saying why
not), the verdict, the confidence (0 without a review, null for a review that ended without a
verdict), the problems found, the result's size and checks, and what the review cost and took.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIDENCE_CONFIG = ROOT / "configs/confidence.yaml"
ABLATION = ROOT / "results/metrics/ablation.json"
WINNER = "winner"
VERDICTS = ("correct", "incorrect", "unsure")

RECORD_FIELDS = (
    "run",
    "question_id",
    "source",
    "db_id",
    "difficulty",
    "answer_run",
    "model",
    "reviewed",
    "not_reviewed",
    "verdict",
    "confidence",
    "problems",
    "result_rows",
    "checks_failed",
    "tokens",
    "cost_usd",
    "latency_s",
    "errors",
    "trace",
    "evaluated_at",
)


def confidence_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIDENCE_CONFIG.read_text(encoding="utf-8"))


def winning_design(path: Path = ABLATION) -> str:
    """The design chosen on the ablation set by the pre-registered rule."""
    return json.loads(Path(path).read_text(encoding="utf-8"))["selection"]["winner"]


def run_name(
    set_name: str, design: str, model: str, evidence: bool, limit: int | None = None
) -> str:
    """A run's name as the agent's stages write it (src/agent/stages.py `run_id`)."""
    ev = "evidence" if evidence else "no-evidence"
    rid = f"{set_name}-{design}-{model.replace(':', '_')}-{ev}"
    return rid + (f"-first{limit}" if limit else "")


def answers_path(conf: dict[str, Any]) -> tuple[Path, str]:
    """The records of the answers the critic reviews, and their run name."""
    a = conf["answers"]
    design = winning_design() if a["design"] == WINNER else a["design"]
    rid = run_name(a["set"], design, a["model"], a["evidence"])
    return ROOT / "results/runs" / a["stage"] / f"{rid}.jsonl", f"{a['stage']}/{rid}"


def validate_record(record: dict) -> None:
    missing = [k for k in RECORD_FIELDS if k not in record]
    extra = [k for k in record if k not in RECORD_FIELDS]
    if missing or extra:
        raise ValueError(f"critic record {record.get('question_id')!r}: {missing=} {extra=}")
    c = record["confidence"]
    if c is not None and not 0 <= c <= 1:
        raise ValueError(f"critic record {record['question_id']!r}: confidence {c}")


def write_records(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            validate_record(r)
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_records(path: Path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            validate_record(r)
            out.append(r)
    return out


def routed_ids(answers: Sequence[dict], calibration: dict[str, Any]) -> set:
    """The questions the router sends to the escalation model: those whose calibrated confidence
    (the Platt calibration fitted on the calibration split, in calibration.json) is below the
    decline threshold chosen there. A declined answer has calibrated confidence 0, so it is
    routed too."""
    import math

    platt = calibration["calibration_split"]["calibrators"]["platt"]
    threshold = calibration["decline"]["chosen_on_calibration_split"]["threshold"]
    out = set()
    for a in answers:
        if a["declined"]:
            p = 0.0
        else:
            p = 1 / (1 + math.exp(-(platt["slope"] * a["confidence"] + platt["intercept"])))
        if p < threshold:
            out.add(a["question_id"])
    return out
