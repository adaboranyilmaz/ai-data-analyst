"""Log every evaluated run to MLflow, rebuilt from the committed results files.

One MLflow run per evaluated run (design, model, evidence setting and number of samples as
parameters; execution accuracy, AURC, ECE and cost as metrics), grouped in experiments. The store
is a view: the numbers are read from results/metrics/, so deleting the store loses nothing. The
runs are read back and compared with the results before the manifest is written.

Writes results/metrics/mlflow_runs.json. The tracking store is `MLFLOW_TRACKING_URI`, else the
local `mlflow.db`.

Usage:
    uv run python scripts/84_mlflow_runs.py [--tracking-uri URI]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from src.data.bird import write_json  # noqa: E402
from src.tracking import experiments  # noqa: E402
from src.tracking.mlflow_setup import tracking_uri  # noqa: E402

OUT = ROOT / "results/metrics/mlflow_runs.json"


def matches(spec: experiments.RunSpec, got: dict | None) -> bool:
    """The stored run holds exactly what the results file says."""
    if got is None:
        return False
    want = spec.as_dict()
    if any(got[k] != want[k] for k in ("experiment", "run", "source", "params")):
        return False
    return got["metrics"].keys() == want["metrics"].keys() and all(
        abs(got["metrics"][k] - v) <= 1e-12 for k, v in want["metrics"].items()
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tracking-uri", default=None)
    a = p.parse_args()
    uri = a.tracking_uri or tracking_uri()

    specs = experiments.collect()
    counts = experiments.log(specs, uri)
    stored = experiments.read_back(uri)
    wrong = [s.key for s in specs if not matches(s, stored.get(s.key))]
    if wrong:
        sys.exit(f"the store does not hold what the results files say for: {wrong}")
    write_json(OUT, experiments.manifest(specs))
    for exp, n in counts.items():
        print(f"  {exp}: {n} runs")
    print(f"logged {len(specs)} runs to {uri}; wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
