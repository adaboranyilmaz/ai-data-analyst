"""Run the security suite: every attack against the query guard alone, the database alone,
and the role's privileges alone.

The attacks are in configs/security_attacks.yaml, the runner in src/db/security_suite.py
(both explain the configurations and outcomes). Needs the hardened databases
(scripts/20_harden_db.py) and, for the attacks on the Czech bank tables, the benchmark
loaded; without it those attacks are listed as skipped.
Writes results/metrics/security_suite.json, and exits with an error if any attack is not
stopped by every layer on its own (the documented object-name exception aside).

Usage:
    uv run python scripts/21_security_suite.py [--require-bird]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.bird import write_json  # noqa: E402
from src.db.security_suite import MODES, bird_available, run_suite  # noqa: E402

OUT = ROOT / "results/metrics/security_suite.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--require-bird", action="store_true", help="fail if the benchmark is absent")
    args = p.parse_args()
    if args.require_bird and not bird_available():
        sys.exit("the bird database is not loaded (scripts/11_load_bird.py)")

    result = run_suite()
    write_json(OUT, result)
    for mode in MODES:
        print(f"{mode:10} {result['summary'][mode]['outcomes']}")
    print(f"skipped without bird: {result['skipped_without_bird'] or 'none'}")
    if not result["every_attack_stopped_by_each_layer"]:
        sys.exit("an attack got through a layer: see " + str(OUT.relative_to(ROOT)))


if __name__ == "__main__":
    main()
