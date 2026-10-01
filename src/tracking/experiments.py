"""MLflow experiments rebuilt from the committed results files.

MLflow here is a view, not a source: every number in it is read from a results file, so the
store can be deleted and rebuilt at any time and no number exists only in MLflow. Each
evaluated run of the project becomes one MLflow run with its design, model, evidence setting and
number of samples as parameters, and its execution accuracy, area under the risk-coverage curve,
calibration error and cost as metrics.

`collect` reads the files and is pure; `log` writes the runs to a tracking store, replacing
runs it logged before, so running it twice leaves one copy of each.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
KEY_TAG = "analyst.key"


@dataclass(frozen=True)
class RunSpec:
    experiment: str
    name: str
    params: dict[str, str]
    metrics: dict[str, float]
    source: str  # the results file the numbers come from

    @property
    def key(self) -> str:
        return f"{self.experiment}/{self.name}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "run": self.name,
            "source": self.source,
            "params": self.params,
            "metrics": self.metrics,
        }


def _num(x: Any) -> float | None:
    return float(x) if isinstance(x, int | float) and not isinstance(x, bool) else None


def _estimate(block: Any) -> dict[str, float]:
    """An interval block ({estimate, low, high}) as three numbers; nothing if it is not one."""
    if not isinstance(block, dict) or _num(block.get("estimate")) is None:
        return {}
    return {
        k: float(block[k]) for k in ("estimate", "low", "high") if _num(block.get(k)) is not None
    }


def block_metrics(b: dict[str, Any]) -> dict[str, float]:
    """The metrics a run block reports: execution accuracy, AURC, ECE and cost, whichever it has."""
    out: dict[str, float] = {}
    for key, blk in (("ex", b.get("execution_accuracy")),):
        for part, v in _estimate(blk).items():
            out[key if part == "estimate" else f"{key}_{part}"] = v
    selective = b.get("selective") or {}
    for part, v in _estimate(selective.get("aurc")).items():
        out["aurc" if part == "estimate" else f"aurc_{part}"] = v
    calibration = b.get("calibration") or {}
    for part, v in _estimate(calibration.get("ece")).items():
        out["ece" if part == "estimate" else f"ece_{part}"] = v
    cost = b.get("cost") or {}
    for src, dst in (
        ("total_usd", "cost_total_usd"),
        ("per_question_usd", "cost_per_question_usd"),
    ):
        if (v := _num(cost.get(src))) is not None:
            out[dst] = v
    for src in ("questions", "declined"):
        if (v := _num(b.get(src))) is not None:
            out[src] = v
    return out


def _samples(design: str, designs: dict[str, Any]) -> str:
    return str(designs.get(design, {}).get("samples", ""))


def collect(root: Path = ROOT) -> list[RunSpec]:
    """Every evaluated run in the committed results files, as MLflow runs to log."""
    metrics_dir = root / "results/metrics"
    designs = yaml.safe_load((root / "configs/agent.yaml").read_text(encoding="utf-8"))["designs"]

    def load(name: str) -> dict[str, Any]:
        return json.loads((metrics_dir / name).read_text(encoding="utf-8"))

    specs: list[RunSpec] = []

    def add(exp, name, source, block, *, design, model, evidence, subset, extra=None):
        params = {
            "design": design,
            "model": model,
            "evidence": str(evidence).lower(),
            "k": _samples(design, designs),
            "questions_set": subset,
            "single_run": "true",
        } | (extra or {})
        specs.append(RunSpec(exp, name, params, block_metrics(block), source))

    ab = load("ablation.json")
    for key, block in ab["runs"].items():
        model, design = key.rsplit("/", 1)
        add(
            "analyst-ablation",
            key,
            "ablation.json",
            block,
            design=design,
            model=model,
            evidence=ab["evidence"],
            subset="ablation",
        )

    main = load("benchmark_main.json")
    win = main["winner"]
    for subset in ("all", "held_out", "ablation"):
        add(
            "analyst-benchmark",
            f"{win['design']}/claude-sonnet-5/evidence/{subset}",
            "benchmark_main.json",
            win[subset],
            design=win["design"],
            model="claude-sonnet-5",
            evidence=True,
            subset=subset,
        )
    add(
        "analyst-benchmark",
        f"{win['design']}/claude-sonnet-5/no-evidence/ablation",
        "benchmark_main.json",
        main["without_evidence"]["ablation"],
        design=win["design"],
        model="claude-sonnet-5",
        evidence=False,
        subset="ablation",
    )
    add(
        "analyst-benchmark",
        f"{win['design']}/claude-haiku-4-5/evidence/ablation",
        "benchmark_main.json",
        main["haiku"]["ablation"],
        design=win["design"],
        model="claude-haiku-4-5",
        evidence=True,
        subset="ablation",
    )

    own = load("own_set.json")
    add(
        "analyst-own-set",
        f"{own['design']}/{own['model']}/no-evidence/banking",
        "own_set.json",
        own["run"],
        design=own["design"],
        model=own["model"],
        evidence=False,
        subset="banking",
    )

    esc = load("escalation.json")
    for who, model in (("sonnet", "claude-sonnet-5"), ("opus", esc["model"])):
        add(
            "analyst-escalation",
            f"{esc['design']}/{model}/evidence/ablation",
            "escalation.json",
            esc[who],
            design=esc["design"],
            model=model,
            evidence=True,
            subset="ablation",
        )

    router = load("router.json")
    for label, block, model in (
        ("routed", router["routed_system"], f"claude-sonnet-5 -> {router['escalation_model']}"),
        ("sonnet-alone", router["sonnet_alone"], "claude-sonnet-5"),
        (
            "opus-alone",
            router["exploratory"]["opus_alone"],
            router["escalation_model"],
        ),
    ):
        add(
            "analyst-router",
            f"{router['design']}/{label}/evidence/held_out",
            "router.json",
            block,
            design=router["design"],
            model=model,
            evidence=True,
            subset="held_out",
        )

    cal = load("calibration.json")
    held = cal["held_out"]
    arms = [
        ("stated-raw", held["stated_raw"]),
        ("stated-platt", held["stated_platt"]),
        ("stated-isotonic", held["stated_isotonic"]),
        ("critic-raw", cal["critic"]["held_out"]["critic_raw"]),
        ("critic-platt", cal["critic"]["held_out"]["critic_platt"]),
    ]
    for label, arm in arms:
        out = {}
        for part, v in _estimate(arm.get("ece")).items():
            out["ece" if part == "estimate" else f"ece_{part}"] = v
        for name in ("brier", "auroc"):
            for part, v in _estimate(arm.get(name)).items():
                out[name if part == "estimate" else f"{name}_{part}"] = v
        out["questions"] = float(held["questions"])
        specs.append(
            RunSpec(
                "analyst-calibration",
                f"d1/claude-sonnet-5/{label}/held_out",
                {
                    "design": win["design"],
                    "model": "claude-sonnet-5",
                    "confidence": label,
                    "questions_set": "held_out",
                    "single_run": "true",
                },
                out,
                "calibration.json",
            )
        )

    fw = load("framework_comparison.json")
    fw_metrics = {
        "ex_graph_minus_own_loop": fw["graph_minus_own_loop"]["execution_accuracy"]["estimate"],
        "cost_per_correct_graph_usd": fw["cost_per_correct_answer_usd"]["graph"],
        "cost_per_correct_own_loop_usd": fw["cost_per_correct_answer_usd"]["own_loop"],
        "code_lines_graph": float(fw["code_lines"]["graph_arm"]["lines"]),
        "code_lines_own_loop": float(fw["code_lines"]["own_loop"]["lines"]),
        "questions": float(fw["questions"]),
    }
    specs.append(
        RunSpec(
            "analyst-framework",
            "graph-vs-own-loop/ablation",
            {"design": "d1", "model": "claude-sonnet-5", "questions_set": "ablation"},
            fw_metrics,
            "framework_comparison.json",
        )
    )
    return specs


def log(specs: list[RunSpec], tracking_uri: str) -> dict[str, int]:
    """Write the runs to a tracking store, replacing the ones this function logged before."""
    import mlflow
    from mlflow import MlflowClient

    from src.tracking.mlflow_setup import tolerant_console

    tolerant_console()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)
    counts: dict[str, int] = {}
    for spec in specs:
        exp = client.get_experiment_by_name(spec.experiment)
        exp_id = exp.experiment_id if exp else client.create_experiment(spec.experiment)
        for old in client.search_runs([exp_id], f"tags.`{KEY_TAG}` = '{spec.key}'"):
            client.delete_run(old.info.run_id)
        run = client.create_run(exp_id, run_name=spec.name, tags={KEY_TAG: spec.key})
        rid = run.info.run_id
        for k, v in spec.params.items():
            client.log_param(rid, k, v)
        client.log_param(rid, "results_file", spec.source)
        for k, v in spec.metrics.items():
            client.log_metric(rid, k, v)
        client.set_terminated(rid)
        counts[spec.experiment] = counts.get(spec.experiment, 0) + 1
    return counts


def read_back(tracking_uri: str) -> dict[str, dict[str, Any]]:
    """The runs a store holds, by key, shaped like `RunSpec.as_dict`, to compare with `collect`."""
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri)
    out: dict[str, dict[str, Any]] = {}
    for exp in client.search_experiments():
        for r in client.search_runs([exp.experiment_id]):
            key = r.data.tags.get(KEY_TAG)
            if key:
                params = dict(r.data.params)
                source = params.pop("results_file", "")
                out[key] = {
                    "experiment": exp.name,
                    "run": r.info.run_name,
                    "source": source,
                    "params": params,
                    "metrics": dict(r.data.metrics),
                }
    return out


def manifest(specs: list[RunSpec]) -> dict[str, Any]:
    by_exp: dict[str, int] = {}
    for s in specs:
        by_exp[s.experiment] = by_exp.get(s.experiment, 0) + 1
    return {
        "note": "every evaluated run as logged to MLflow; each number is read from the results "
        "file it names, so the store is a view that can be rebuilt (scripts/84_mlflow_runs.py)",
        "runs_logged": len(specs),
        "by_experiment": by_exp,
        "runs": [s.as_dict() for s in specs],
    }
