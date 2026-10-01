"""The evaluation gate: replay the regression bundle and check the registry, at no cost.

Fails (exit 1) if any of these is true:
- a recorded request, rebuilt from the repository, differs from the recorded one: a prompt, the
  schema description, a model setting or a design changed;
- a recorded response no longer parses to the answer that was evaluated;
- an agent configuration changed without a fresh registered version and evaluation, an alias moved
  without a promotion, or a logged promotion does not follow from the held-out results under the
  rule in force;
- the regression bundle does not cover a model the registered configurations use.

Writes results/metrics/eval_gate.json (what was checked and found, without timings).

Usage:
    uv run python scripts/89_eval_gate.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.bird import write_json  # noqa: E402
from src.tracking import gate, registry  # noqa: E402

OUT = ROOT / "results/metrics/eval_gate.json"


def main() -> None:
    os.environ["ANALYST_REPLAY_ONLY"] = "1"  # nothing here may call a model
    replay = gate.replay()
    problems = [*registry.problems(), *gate.coverage_problems(), *gate.promotion_problems()]
    state = registry.read_state()
    report = {
        "note": "recorded model calls rebuilt from the repository and compared; the registry "
        "checked against the committed results; no model, database or key",
        "regression": replay,
        "registry": {
            "champion": registry.champion_name(state),
            "challenger": state["aliases"].get("challenger"),
            "versions": sorted(state["versions"]),
            "promotions_logged": len(registry.read_promotions()),
        },
        "problems": problems,
        "ok": replay["ok"] and not problems,
    }
    write_json(OUT, report)
    print(
        f"regression: {replay['unchanged']}/{replay['entries']} recorded calls unchanged; "
        f"champion {report['registry']['champion']}"
    )
    for r in replay["changed_request"]:
        print(f"  request changed: {r}")
    for r, fields in replay["changed_answer"].items():
        print(f"  answer changed: {r}: {fields}")
    for p in problems:
        print(f"  problem: {p}")
    print("gate " + ("passed" if report["ok"] else "FAILED"))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
