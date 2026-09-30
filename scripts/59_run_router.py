"""Run the router's model: the escalation model on the held-out questions.

Refuses to run unless the escalation decision (results/metrics/escalation.json) adopted the
escalation model. The router sends the held-out questions whose calibrated confidence (the Platt
calibration fitted on the calibration split) is below the decline threshold chosen there
(results/metrics/calibration.json) to the escalation model. Its answers do not depend on that
choice, so it answers every held-out question with the winning design, exactly as in the
escalation arm, and the router's choice is applied in its report; the questions the router keeps
also give the escalation model alone on the whole set, an exploratory comparison. Every call goes
through the router stage's response cache and the spend ledger's phase cap; with
ANALYST_REPLAY_ONLY=1 the stage is rebuilt from the cache at no cost.

Writes results/runs/router/<run>.jsonl, and traces and spans under data/.

Usage:
    uv run python scripts/59_run_router.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from src.agent.confidence import (  # noqa: E402
    answers_path,
    confidence_config,
    routed_ids,
    winning_design,
)
from src.agent.escalation import agent_config_with, execute, router_spec  # noqa: E402
from src.eval.records import answered_correct, read_records  # noqa: E402

ESCALATION = ROOT / "results/metrics/escalation.json"
CALIBRATION = ROOT / "results/metrics/calibration.json"


def main() -> None:
    load_dotenv(ROOT / ".env")  # the API key, for live calls
    if not json.loads(ESCALATION.read_text(encoding="utf-8"))["rule"]["adopted"]:
        sys.exit("the escalation model was not adopted: there is no router to run")
    conf = confidence_config()
    esc = conf["escalation"]
    cfg = agent_config_with(esc["model"], esc["settings"])
    path, _ = answers_path(conf)
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    spec = router_spec(conf, winning_design())
    routed = routed_ids(read_records(path), calibration)
    if spec.spans.exists():
        spec.spans.unlink()  # spans describe this execution only
    print(
        f"router: {len(spec.items)} questions to {esc['model']}, "
        f"{len([q for q, _ in spec.items if q.question_id in routed])} of them routed",
        flush=True,
    )
    (records,) = execute([spec], cfg)
    ex = sum(answered_correct(r) for r in records) / len(records)
    print(
        f"{spec.name}: EX {ex:.3f} on {len(records)} questions (one run), "
        f"cost ${sum(r['cost_usd'] for r in records):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
