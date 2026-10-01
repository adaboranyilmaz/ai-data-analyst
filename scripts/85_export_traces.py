"""Send the recorded spans of evaluated runs to the tracing servers (MLflow and Langfuse).

The agent's instrumentation wrote a span for every run, model call and tool call when the run
was evaluated (data/spans/). This sends the spans of the chosen questions, unchanged and with
their original ids and timestamps, to whichever servers the environment names, so the same run
can be looked at in both tools. No model call and no database.

Servers (environment): ANALYST_TRACE_MLFLOW_URI and ANALYST_TRACE_MLFLOW_EXPERIMENT_ID for
MLflow; LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY for Langfuse. `--mlflow-uri`
looks the MLflow experiment up (and creates it) by name.

Usage:
    uv run python scripts/85_export_traces.py --spans data/spans/ablation/<run>.jsonl \\
        --question 1017 [--question 1090 ...] [--mlflow-uri http://127.0.0.1:5000]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from src.tracking import otlp, spanfile  # noqa: E402

EXPERIMENT = "analyst-traces"


def main() -> None:
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--spans", type=Path, required=True)
    p.add_argument("--question", action="append", default=[], help="a question id; repeatable")
    p.add_argument("--first", type=int, default=0, help="or the first N questions in the file")
    p.add_argument("--mlflow-uri", default=None)
    a = p.parse_args()

    env = dict(os.environ)
    if a.mlflow_uri:
        import mlflow

        mlflow.set_tracking_uri(a.mlflow_uri)
        env[otlp.MLFLOW_URI] = a.mlflow_uri
        env[otlp.MLFLOW_EXPERIMENT_ID] = mlflow.set_experiment(EXPERIMENT).experiment_id
    exporters = otlp.exporters_from_env(env)
    if not exporters:
        sys.exit("no tracing server named: set the MLflow or Langfuse variables (see --help)")

    records = spanfile.read(a.spans if a.spans.is_absolute() else ROOT / a.spans)
    questions = list(a.question) or spanfile.question_ids(records)[: a.first]
    if not questions:
        sys.exit("name a question (--question ID) or --first N")
    fan = otlp.FanOutExporter(exporters)
    for q in questions:
        spans = [spanfile.to_span(r) for r in spanfile.run_trace(records, q)]
        ok = fan.export(spans).name
        print(f"question {q}: {len(spans)} spans, export {ok}")
    time.sleep(0.5)
    fan.shutdown()


if __name__ == "__main__":
    main()
