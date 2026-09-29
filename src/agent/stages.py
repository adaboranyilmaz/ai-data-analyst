"""Where a stage's runs live, and how a run is named.

A stage (pilot, ablation, main, own) is a group of runs over one question set. Its model
responses go to its own cache directory and may be read from earlier stages' (the winner's run
on all questions reuses the design comparison's answers); every run writes, for run <id> =
<set>-<design>-<model>-<evidence|no-evidence>[-first<N>]:

  results/runs/<stage>/<id>.jsonl      one record per question (src/eval/records.py)
  data/traces/<stage>/<id>/<q>.json    one trace per question
  data/spans/<stage>/<id>.jsonl        the run's spans
  data/cache/llm_<stage>/              the stage's model responses

An arm may name its design `winner`: the design chosen on the ablation set
(results/metrics/ablation.json), so the stages after it are configured before it is known.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.agent.evaluate import RunSpec, banking_set, benchmark_set

ROOT = Path(__file__).resolve().parent.parent.parent
ABLATION = ROOT / "results/metrics/ablation.json"
WINNER = "winner"
# a stage may read the responses stored by earlier stages (never write them)
READS = {"main": ("ablation",), "own": ("ablation", "main")}


def winner(path: Path = ABLATION) -> str:
    """The design chosen on the ablation set by the pre-registered rule."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} is missing: run scripts/43_ablation_report.py before the stages that "
            "use the winning design"
        )
    return json.loads(path.read_text(encoding="utf-8"))["selection"]["winner"]


def resolve_arms(
    stage: dict, ablation: dict, path: Path = ABLATION
) -> tuple[list[dict], list[str]]:
    """A stage's arms, each with its set and evidence (the arm's own, else the stage's), and
    `winner` replaced by the design. An arm that the ablation stage already ran (Claude Haiku
    4.5 on a winning design it ran there: same model, design, set and evidence) is left out with
    a note, since its answers are the ablation's. Returns (arms, notes)."""
    chosen = winner(path) if any(a["design"] == WINNER for a in stage["arms"]) else None
    ran = {
        (a["model"], a["design"], ablation["set"], ablation.get("evidence", True))
        for a in ablation["arms"]
    }
    arms, notes = [], []
    for arm in stage["arms"]:
        arm = {"set": stage["set"], "evidence": stage.get("evidence", True), **arm}
        if arm["design"] == WINNER:
            arm["design"] = chosen
            if (arm["model"], chosen, arm["set"], arm["evidence"]) in ran:
                notes.append(f"{arm['model']} on {chosen}: run in the ablation stage already")
                continue
        arms.append(arm)
    return arms, notes


def run_id(set_name: str, design: str, model: str, evidence: bool, limit: int | None) -> str:
    ev = "evidence" if evidence else "no-evidence"
    rid = f"{set_name}-{design}-{model.replace(':', '_')}-{ev}"
    return rid + (f"-first{limit}" if limit else "")


def make_spec(
    stage: str,
    set_name: str,
    design: str,
    model: str,
    evidence: bool = True,
    mode: str = "direct",
    limit: int | None = None,
    phase: str = "phase4",
) -> RunSpec:
    items = banking_set() if set_name == "own" else benchmark_set(set_name)
    if limit:
        items = items[:limit]
    rid = run_id(set_name, design, model, evidence, limit)
    return RunSpec(
        name=f"{stage}/{rid}",
        items=items,
        design=design,
        model=model,
        evidence=evidence,
        phase=phase,
        cache_dir=ROOT / f"data/cache/llm_{stage}",
        read_caches=tuple(ROOT / f"data/cache/llm_{s}" for s in READS.get(stage, ())),
        mode=mode,
        records=ROOT / f"results/runs/{stage}/{rid}.jsonl",
        traces_dir=ROOT / f"data/traces/{stage}/{rid}",
        spans=ROOT / f"data/spans/{stage}/{rid}.jsonl",
    )
