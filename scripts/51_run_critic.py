"""Run the critic's reviews of the winning design's answers (configs/confidence.yaml `critic`).

Each configured run reviews the answers on one question set: the pilot's first ten (the format
check before the critic's prompt is frozen), the ablation set (the calibration split) and the
held-out set (the test split). Every review goes through the critic stage's response cache and
the spend ledger's phase cap; with ANALYST_REPLAY_ONLY=1 the stage is rebuilt from the cache at no
cost. The reviews of the calibration and held-out sets refuse to run until the prompt is frozen
by its hash in the config.

Writes results/runs/critic/<run>.jsonl, and traces and spans under data/.

Usage:
    uv run python scripts/51_run_critic.py [--sets pilot ablation held_out]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from src.agent.confidence import answers_path, confidence_config  # noqa: E402
from src.agent.critic import execute, items_for, load_prompt, make_spec  # noqa: E402
from src.eval.records import read_records  # noqa: E402

PROBE_SET = "pilot"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sets", nargs="*", help="only the runs on these question sets")
    a = p.parse_args()

    load_dotenv(ROOT / ".env")  # the API key, for live calls
    conf = confidence_config()
    crit = conf["critic"]
    runs = [r for r in crit["runs"] if not a.sets or r["set"] in a.sets]
    _, sha = load_prompt(ROOT / crit["prompt"])
    if any(r["set"] != PROBE_SET for r in runs) and crit.get("prompt_sha256") != sha:
        sys.exit(
            f"the critic's prompt is not frozen at its current hash ({sha}): the calibration and "
            "held-out reviews wait until configs/confidence.yaml critic.prompt_sha256 is set"
        )
    path, answer_run = answers_path(conf)
    answers = read_records(path)
    specs = [
        make_spec(
            r["set"],
            r.get("limit"),
            crit["model"],
            items_for(r["set"], answers, r.get("limit")),
            answer_run,
            conf["phase"],
            crit["mode"],
            conf["answers"]["evidence"],
        )
        for r in runs
    ]
    for s in specs:
        if s.spans.exists():
            s.spans.unlink()  # spans describe this execution only
    print(f"critic: {len(specs)} runs of {answer_run}", flush=True)
    for s, records in zip(specs, execute(specs, crit), strict=True):
        reviewed = [r for r in records if r["reviewed"]]
        confs = [r["confidence"] for r in reviewed if r["confidence"] is not None]
        print(
            f"{s.name}: {len(reviewed)}/{len(records)} reviewed, "
            f"{len(reviewed) - len(confs)} without a verdict, "
            f"mean confidence {statistics.mean(confs) if confs else float('nan'):.3f}, "
            f"cost ${sum(r['cost_usd'] for r in records):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
