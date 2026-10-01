"""Run the sandbox security suite: every attack in configs/sandbox_attacks.yaml under the
guardrail's container flags, and each protection's control.

The attacks and outcomes are explained in the attack file, the runner in src/stats/security.py.
Needs Docker and the sandbox image (scripts/70_build_sandbox.py). Writes
results/metrics/sandbox_security.json, and exits with an error if any attack was not blocked or
any control failed to show its attack is real.

Usage:
    uv run python scripts/71_sandbox_security.py [--only ID ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.bird import write_json  # noqa: E402
from src.stats.sandbox import image_present, image_tag  # noqa: E402
from src.stats.security import run_suite  # noqa: E402

OUT = ROOT / "results/metrics/sandbox_security.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--only", nargs="+", help="run these attacks only (nothing is written)")
    args = p.parse_args()
    if not image_present():
        sys.exit(f"the sandbox image {image_tag()} is not built (scripts/70_build_sandbox.py)")

    result = run_suite(only=set(args.only) if args.only else None)
    for r in result["records"]:
        mark = "ok " if r["passed"] else "FAIL"
        print(f"{mark} {r['id']:28} {r['outcome']:10} {r['seconds']:6.2f}s  {r['detail'][:90]}")
    for c in result["control_records"]:
        mark = "ok " if c["achieved"] else "FAIL"
        print(f"{mark} control {c['id']:20} {c['relaxed']}  {c['detail'][:80]}")
    print(f"outcomes {result['outcomes']}; controls {result['controls']}")
    if args.only:
        return
    write_json(OUT, result)
    if not (result["every_attack_blocked"] and result["every_control_achieved"]):
        sys.exit("an attack was not blocked or a control failed: see " + str(OUT.relative_to(ROOT)))


if __name__ == "__main__":
    main()
