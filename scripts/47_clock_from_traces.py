"""Store the results of clock-reading agent queries from the traces of runs made before such
results were stored (src/agent/clock.py), so those runs replay exactly.

For each stage given, every trace under data/traces/<stage>/ is read; each `run_sql` result
whose query reads the clock is stored in the stage's cache (data/cache/llm_<stage>/clock/),
keyed by the call it answered, found through the cached responses the trace's requests name.
A result already stored is left as it is. Free: no model is called, no query is run.

Usage:
    uv run python scripts/47_clock_from_traces.py --stages pilot ablation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.clock import ClockStore, seed_from_traces  # noqa: E402
from src.agent.stages import READS  # noqa: E402
from src.db.guard import QueryGuard  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stages", nargs="+", required=True)
    a = p.parse_args()
    tables: dict[str, frozenset[str]] = {}

    def tables_of(db: str) -> frozenset[str]:
        if db not in tables:
            tables[db] = QueryGuard.for_benchmark_db(db).tables
        return tables[db]

    for stage in a.stages:
        roots = [ROOT / f"data/cache/llm_{s}" for s in (stage, *READS.get(stage, ()))]

        def entry(key: str, roots=roots, stage=stage) -> dict:
            for root in roots:
                path = root / key[:2] / f"{key}.json"
                if path.exists():
                    return json.loads(path.read_text(encoding="utf-8"))
            raise FileNotFoundError(f"request {key[:12]} is in no cache of stage {stage}")

        store = ClockStore(roots[0], roots[1:])
        traces = sorted((ROOT / f"data/traces/{stage}").glob("*/*.json"))
        counts = seed_from_traces(traces, store, tables_of, entry)
        print(f"{stage}: {len(traces)} traces; {counts}")


if __name__ == "__main__":
    main()
