"""Split the benchmark's questions into the pilot, ablation and held-out sets.

Sizes and seed from configs/eval.yaml `splits`; the method in src/eval/splits.py (each set in
proportion to every database x difficulty stratum). The sets are fixed here, before any agent
runs, and named by their hash in the pre-registration.
Writes results/metrics/splits.json: per set, its question ids and its count per stratum.

Usage:
    uv run python scripts/32_splits.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import bird  # noqa: E402
from src.data.bird import write_json  # noqa: E402
from src.eval.config import config  # noqa: E402
from src.eval.splits import sets_sha256, split, stratum_counts  # noqa: E402


def main() -> None:
    cfg = config()
    s = cfg["splits"]
    questions = bird.questions()
    sets = split(questions, {"pilot": s["pilot"], "ablation": s["ablation"]}, s["seed"])
    by_id = {q["question_id"]: q for q in questions}
    out = {
        "questions": len(questions),
        "questions_sha256": bird.config()["bird_minidev"]["questions"]["sha256"],
        "seed": s["seed"],
        "sizes": {name: len(ids) for name, ids in sets.items()},
        "sets_sha256": sets_sha256(sets),
        "difficulty": {
            name: dict(sorted(Counter(by_id[i]["difficulty"] for i in ids).items()))
            for name, ids in sets.items()
        },
        "strata": stratum_counts(questions, sets),
        "sets": sets,
    }
    path = ROOT / cfg["splits_file"]
    write_json(path, out)
    print(out["sizes"], out["difficulty"], out["sets_sha256"])
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
