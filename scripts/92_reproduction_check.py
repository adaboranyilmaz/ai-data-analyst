"""Check that a rebuild of the pipeline reproduced the committed results.

A rebuild is `dvc repro --force` in a clean checkout (data pulled with `dvc pull`, the compose
PostgreSQL up, ANALYST_REPLAY_ONLY=1 so that no model can be called). Before it starts, copy the
committed results; afterwards this compares the rebuilt tree with the copy, file by file
(src/tracking/reproduction.py): JSON semantically, everything else byte for byte, differing only
where a rule names a field that records the run rather than a result.

Writes the report to --out (default results/metrics/reproduction_check.json) and exits 1 if
anything differs outside the rules.

Usage:
    cp -r results /tmp/baseline
    ANALYST_REPLAY_ONLY=1 uv run dvc repro --force
    uv run python scripts/92_reproduction_check.py /tmp/baseline [--current results]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.bird import write_json  # noqa: E402
from src.tracking import reproduction  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "baseline", type=Path, help="a copy of the results directory from before the rebuild"
    )
    p.add_argument("--current", type=Path, default=ROOT / "results")
    p.add_argument("--out", type=Path, default=ROOT / "results/metrics/reproduction_check.json")
    a = p.parse_args()

    report = reproduction.check(a.baseline, a.current)
    out = {
        "note": "results/ as committed, copied before a forced rebuild (dvc repro --force with "
        "ANALYST_REPLAY_ONLY=1, no model call), compared with the rebuilt results/",
        "exempt": [
            {"files": g, "field": f, "reason": r, "abs_tol": t}
            for g, f, r, t in reproduction.EXEMPT
        ],
        "masked": [
            {"files": g, "field": f, "reason": r} for g, f, _p, _r, r in reproduction.MASKED
        ],
        "known_nondeterministic_rules": [
            {"files": g, "field": f, "reason": r, "abs_tol": tol}
            for g, f, r, tol in reproduction.NONDETERMINISTIC
        ],
        **report,
    }
    write_json(a.out, out)
    c = report["counts"]
    print(
        f"{report['files_compared']} files: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items()))
    )
    if not report["reproduced_exactly"]:
        print("known non-deterministic (a result, with its cause; see the report):")
        for rel in report["known_nondeterministic"]:
            print(f"  {rel}")
    for rel, f in report["failures"].items():
        print(f"  {rel}: {f['status']} {f.get('detail', '')[:3] if f.get('detail') else ''}")
    print("reproduction " + ("passed" if report["passed"] else "FAILED"))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
