"""Build the regression bundle the evaluation gate replays, from the local response caches.

For each curated benchmark and banking question the demo serves, the Claude Sonnet 5 call that
answered it, and for the benchmark questions the Claude Opus 5.5 call of the router's run: the
request's cache key (the request is rebuilt from the repository's prompts, schema snapshot and
settings) and the response. The caches are not committed; the bundle, a few kilobytes per call,
is. Run again only when the curated set changes: the gate (scripts/89_eval_gate.py) is what
checks the bundle against the repository.

Writes results/regression/bundle.json.

Usage:
    uv run python scripts/88_regression_bundle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.bird import write_json  # noqa: E402
from src.tracking import gate  # noqa: E402


def main() -> None:
    dirs = sorted(p for p in (ROOT / "data/cache").glob("llm*") if p.name != "llm_batches")
    bundle = gate.build(ROOT, dirs)
    out = ROOT / gate.BUNDLE
    write_json(out, bundle)
    by_model: dict[str, int] = {}
    for e in bundle["entries"]:
        by_model[e["model"]] = by_model.get(e["model"], 0) + 1
    print(f"{len(bundle['entries'])} recorded calls {by_model} -> {out.relative_to(ROOT)}")
    print(f"{out.stat().st_size / 1000:.0f} KB")


if __name__ == "__main__":
    main()
