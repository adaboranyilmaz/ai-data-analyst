"""Decide whether the challenger agent configuration replaces the champion, by the fixed rule.

Scores the champion and the challenger on the held-out questions (committed run files, no model
call), compares them paired, applies configs/promotion.yaml and prints the decision. With
`--apply` the decision is appended to results/registry/promotions.jsonl, and a promotion also
moves the aliases in results/registry/state.json (the challenger becomes the champion, and no
configuration is the challenger until another is registered). Without it, nothing is written.

Usage:
    uv run python scripts/87_promotion.py [--apply]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval import promotion  # noqa: E402
from src.llm.ledger import load_ledger  # noqa: E402
from src.tracking import registry  # noqa: E402

RULE = ROOT / "configs/promotion.yaml"
COMPARED = ("execution_accuracy", "aurc", "accuracy_at_80", "accuracy_at_90")
NOTE = (
    "the challenger's execution accuracy on this split was known before the rule was fixed (its "
    "run was reported in an earlier phase); its AURC and cost under the rule were computed after"
)


def decision_entry() -> dict:
    state = registry.read_state()
    if found := registry.problems():
        sys.exit("the registry has problems, so no decision is made:\n  - " + "\n  - ".join(found))
    aliases = state["aliases"]
    if not aliases.get("challenger"):
        sys.exit("there is no challenger")
    champ_name, chall_name = aliases["champion"], aliases["challenger"]
    rule = yaml.safe_load(RULE.read_text(encoding="utf-8"))
    cost = load_ledger("phase5").cost
    champ, _ = promotion.system_records(state["versions"][champ_name]["config"], ROOT, cost)
    chall, routed = promotion.system_records(state["versions"][chall_name]["config"], ROOT, cost)
    comparison = promotion.compare_systems(chall, champ)
    chall_summary = promotion.summary(chall)
    return {
        "utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "champion": champ_name,
        "challenger": chall_name,
        "champion_sha256": state["versions"][champ_name]["config_sha256"],
        "challenger_sha256": state["versions"][chall_name]["config_sha256"],
        "rule_sha256": registry.file_sha256(RULE),
        "split": rule["split"],
        "questions": comparison["questions"],
        "routed_to_larger_model": routed,
        "challenger_minus_champion": {
            k: comparison[k] for k in (*COMPARED, "cost_per_question_usd")
        },
        "champion_system": {
            k: promotion.summary(champ)[k] for k in (*COMPARED[:2], "ece", "cost_per_question_usd")
        },
        "challenger_system": {
            k: chall_summary[k] for k in (*COMPARED[:2], "ece", "cost_per_question_usd")
        },
        "decision": promotion.decide(rule, comparison, chall_summary),
        "note": NOTE,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--apply", action="store_true", help="write the decision (and the promotion)")
    a = p.parse_args()

    entry = decision_entry()
    d = entry["challenger_minus_champion"]
    print(f"{entry['challenger']} against {entry['champion']} on {entry['questions']} questions")
    for key in COMPARED:
        v = d[key]
        print(f"  {key:<20} {v['estimate']:+.4f}  [{v['low']:+.4f}, {v['high']:+.4f}]")
    for name, c in entry["decision"]["checks"].items():
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {name}")
    verdict = "PROMOTE" if entry["decision"]["promote"] else "do not promote"
    print(f"decision: {verdict}")
    if not a.apply:
        print("(not written: pass --apply)")
        return
    with (ROOT / registry.PROMOTIONS).open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    if entry["decision"]["promote"]:
        state = registry.read_state()
        state["aliases"] = {"champion": entry["challenger"]}
        registry.write_state(state)
        print(f"{entry['challenger']} is now the champion")


if __name__ == "__main__":
    main()
