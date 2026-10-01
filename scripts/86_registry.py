"""The agent registry: evaluate and register agent configurations, mirror them in MLflow, check.

  evaluate   score every configuration in configs/agents/ on the held-out questions (from the
             committed run files, no model call) and write one evaluation file per configuration
             to results/registry/evaluations/. Deterministic: a pipeline stage.
  register   record each configuration's resolved form and evaluation file in
             results/registry/state.json. The first registration needs --champion and
             --challenger; later ones keep the aliases (they move only through a promotion).
  sync       mirror the committed registry in MLflow's model registry (versions and aliases).
  check      list what is wrong with the registry as committed; exit 1 if anything is.

Usage:
    uv run python scripts/86_registry.py evaluate
    uv run python scripts/86_registry.py register [--champion NAME --challenger NAME]
    uv run python scripts/86_registry.py sync [--tracking-uri URI]
    uv run python scripts/86_registry.py check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import yaml  # noqa: E402

from src.data.bird import write_json  # noqa: E402
from src.eval import promotion  # noqa: E402
from src.llm.ledger import load_ledger  # noqa: E402
from src.tracking import registry  # noqa: E402

NOTE = (
    "one run; the held-out questions; the system's answers come from the committed run files and "
    "its confidence is calibrated on the calibration split only; intervals are 95% bootstrap over "
    "questions; costs at the direct price from the recorded tokens"
)


def evaluation(name: str) -> dict:
    resolved = registry.resolve(name)
    records, routed = promotion.system_records(resolved, ROOT, load_ledger("phase5").cost)
    rule = yaml.safe_load((ROOT / "configs/confidence.yaml").read_text(encoding="utf-8"))["decline"]
    choice = promotion.decline_threshold(resolved, ROOT, rule)
    return {
        "config_name": name,
        "config_sha256": registry.sha256_of(resolved),
        "note": NOTE,
        "split": "held_out",
        "routed_to_larger_model": routed,
        "system": promotion.summary(records),
        # what a confidence means for this system: the threshold chosen on the calibration split
        # and, on the held-out questions, what answers at and above it achieve and how reliable
        # each band of calibrated confidence was
        "confidence": {
            "decline_threshold": resolved["decline_threshold"],
            "calibration_split": {k: v for k, v in choice.items() if k != "candidates"},
            "held_out": promotion.held_out_confidence(records, resolved["decline_threshold"]),
        },
    }


def evaluate() -> None:
    for name in registry.config_names():
        out = ROOT / registry.EVALUATIONS_DIR / f"{name}.json"
        write_json(out, evaluation(name))
        ex = json.loads(out.read_text(encoding="utf-8"))["system"]["execution_accuracy"]
        print(f"  {name}: EX {ex['estimate']:.3f} -> {out.relative_to(ROOT)}")


def register(champion: str | None, challenger: str | None) -> None:
    try:
        state = registry.read_state()
    except FileNotFoundError:
        if not (champion and challenger):
            sys.exit("the first registration needs --champion and --challenger")
        state = {"initial_champion": champion, "aliases": {}, "versions": {}}
        state["aliases"] = {"champion": champion, "challenger": challenger}
    state["model_name"] = registry.MODEL_NAME
    for name in registry.config_names():
        resolved = registry.resolve(name)
        ev = f"{registry.EVALUATIONS_DIR}/{name}.json"
        if not (ROOT / ev).exists():
            sys.exit(f"evaluate first: no {ev}")
        state["versions"][name] = {
            "config": resolved,
            "config_sha256": registry.sha256_of(resolved),
            "evaluation": ev,
        }
    registry.write_state(state)
    print(f"registered {len(state['versions'])} configurations; aliases {state['aliases']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("evaluate")
    r = sub.add_parser("register")
    r.add_argument("--champion")
    r.add_argument("--challenger")
    s = sub.add_parser("sync")
    s.add_argument("--tracking-uri", default=None)
    sub.add_parser("check")
    a = p.parse_args()

    if a.cmd == "evaluate":
        evaluate()
    elif a.cmd == "register":
        register(a.champion, a.challenger)
    elif a.cmd == "sync":
        from src.tracking.mlflow_setup import tracking_uri

        print(json.dumps(registry.sync_mlflow(a.tracking_uri or tracking_uri()), indent=1))
    else:
        found = registry.problems()
        for line in found:
            print("  -", line)
        print("registry ok" if not found else f"{len(found)} problem(s)")
        sys.exit(1 if found else 0)


if __name__ == "__main__":
    main()
